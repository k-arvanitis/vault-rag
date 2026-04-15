"""Generate a synthetic QA evaluation dataset from ingested Qdrant chunks.

Usage:
    python -m eval.generate_dataset [--n-pairs 200] [--output eval/dataset.json]

How it works:
1. Connects to Qdrant and scrolls ALL chunks (documents + tables).
2. Stratified-samples up to --n-pairs diverse chunks (mix of file types,
   chunk types, and content length buckets).
3. For each sampled chunk, calls the local judge LLM to produce one focused
   factual question + expected answer grounded in that chunk only.
4. Saves eval/dataset.json — a list of dicts with keys:
       input           the generated question
       expected_output the expected answer
       context         [the chunk text the answer was derived from]
       metadata        file/sheet/chunk_type for traceability

The judge LLM is the same Qwen3-32B-AWQ used for generation and evaluation,
configured via JUDGE_API_BASE / JUDGE_MODEL env vars (see eval/judge_llm.py).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import textwrap
import time
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "eval" / "dataset.json"

QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:7333")
COLLECTION: str = os.getenv("QDRANT_COLLECTION", "documents_chunks")

# Prompt sent to the judge LLM for each chunk
QA_GENERATION_PROMPT = textwrap.dedent("""\
    You are creating an evaluation dataset for a RAG (Retrieval-Augmented \
Generation) system specialised in Greek environmental policy, greenhouse gas \
inventories, and water quality standards.

    Below is a document chunk. Write ONE focused factual question that can be \
answered ONLY from this chunk, and then write the correct answer.

    Rules:
    - The question must be answerable from the chunk text alone.
    - The answer must be a complete sentence or short paragraph.
    - Do not reference "the chunk" or "the document" in the question — phrase it \
naturally.
    - If the chunk contains numbers or specific data, prefer questions about them.
    - Output ONLY valid JSON, no markdown fences, no extra text:

    {{"question": "<question here>", "answer": "<answer here>"}}

    CHUNK:
    {chunk}
""")


def _qdrant_scroll(qdrant_url: str, collection: str, limit: int = 100) -> list[dict]:
    """Scroll all points from a Qdrant collection (no vector, payload only)."""
    all_points: list[dict] = []
    offset = None

    while True:
        body: dict = {"limit": limit, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset

        req = Request(
            f"{qdrant_url}/collections/{collection}/points/scroll",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        result = data.get("result", {})
        points = result.get("points", [])
        all_points.extend(points)

        next_offset = result.get("next_page_offset")
        if not next_offset:
            break
        offset = next_offset

    return all_points


def _stratified_sample(points: list[dict], n: int) -> list[dict]:
    """Return up to n points via round-robin across chunk_type × file buckets.

    Round-robin guarantees exactly min(n, total) samples with good diversity,
    regardless of how many buckets there are.
    """
    buckets: dict[str, list[dict]] = {}
    for p in points:
        payload = p.get("payload", {})
        meta = payload.get("metadata", {}) or {}
        key = f"{meta.get('chunk_type', 'doc')}::{meta.get('file_name', meta.get('source_file', 'unknown'))}"
        buckets.setdefault(key, []).append(p)

    for bucket in buckets.values():
        random.shuffle(bucket)

    # Round-robin: cycle through buckets taking one at a time
    iterators = {k: iter(v) for k, v in buckets.items()}
    keys = list(iterators.keys())
    random.shuffle(keys)
    sampled: list[dict] = []

    while len(sampled) < n and iterators:
        exhausted = []
        for key in keys:
            if len(sampled) >= n:
                break
            try:
                sampled.append(next(iterators[key]))
            except StopIteration:
                exhausted.append(key)
        for k in exhausted:
            del iterators[k]
            keys.remove(k)

    random.shuffle(sampled)
    return sampled


def _extract_json_object(text: str) -> str | None:
    """Extract the first complete {...} JSON object using brace matching.

    Unlike a lazy regex, this handles nested braces and values that contain
    quotes or curly braces without breaking early.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


_BAD_QUESTION_PATTERNS = [
    # Self-referential — question references the source chunk/table/document directly
    r"\bthis table\b",
    r"\bthe following table\b",
    r"\bthe table\b",
    r"\bthe chunk\b",
    r"\bthe document\b",
    r"\bthe text\b",
    r"\bthe passage\b",
    r"\bthe context\b",
    r"\bthe data\b",
    r"\bthis dataset\b",
    r"\bthe dataset\b",
    r"\bthe file\b",
    r"\bthe report\b",
    r"\bthe sheet\b",
    # Garbage synthetic column names from poorly-parsed tables
    r"\bcol_\d+\b",
    # Questions that are purely meta / unanswerable without the source
    r"^what (kind of |type of |sort of )?information is (included|provided|contained|described)",
    r"^what (are the |is the )?fields? (included|provided|listed|described)",
    r"^what (are the |is the )?(different )?fields? in (this|the) (table|dataset|data)",
    r"^what (are the |is the )?structure of",
    r"^what (does|is) (this|the) (table|dataset|data) (contain|include|describe|represent)",
    # Analytical/essay questions — too complex for real RAG users
    r"^analyze (how|the|what|why|which)",
    r"^explain how",
    r"^determine (how|the|what|why|which)",
    r"^how do(es)? .{0,60} (correlate|relate|interact|collectively|interrelat)",
    r"^how do(es)? .{0,60} ensure",
    r"^how do(es)? .{0,60} support",
    r"^discuss ",
    r"^compare ",
    r"^evaluate ",
]
_BAD_RE = re.compile("|".join(_BAD_QUESTION_PATTERNS), re.IGNORECASE)


def _is_bad_question(question: str) -> bool:
    """Return True if the question is self-referential or unanswerable without its source."""
    return bool(_BAD_RE.search(question))


def _generate_qa(chunk_text: str, judge, retries: int = 2) -> dict | None:
    """Call the judge LLM and parse the JSON question+answer pair.

    Retries up to `retries` times on parse failure before giving up.
    """
    prompt = QA_GENERATION_PROMPT.format(chunk=chunk_text[:2000])
    for attempt in range(1, retries + 2):
        try:
            raw = judge.generate(prompt)
        except Exception as exc:
            print(f"  [WARN] LLM call failed (attempt {attempt}): {exc}", file=sys.stderr)
            continue

        # Strip accidental markdown fences
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip()

        candidate = _extract_json_object(raw)
        if not candidate:
            print(f"  [WARN] No JSON object found (attempt {attempt}): {raw[:80]}", file=sys.stderr)
            continue

        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as exc:
            print(f"  [WARN] JSON decode error (attempt {attempt}): {exc}", file=sys.stderr)
            continue

        q = obj.get("question", "").strip()
        a = obj.get("answer", "").strip()
        if q and a:
            return {"question": q, "answer": a}

    return None


def main(n_pairs: int, output: Path, cross_doc_ratio: float = 0.15, num_evolutions: int = 1) -> None:
    from deepeval.synthesizer import Synthesizer
    from deepeval.synthesizer.config import EvolutionConfig, Evolution

    from eval.judge_llm import OpenAICompatibleJudge

    judge = OpenAICompatibleJudge()
    print(f"Judge LLM : {judge.get_model_name()}")
    print(f"Qdrant    : {QDRANT_URL}  collection={COLLECTION}")
    print(f"Target    : {n_pairs} QA pairs (cross-doc ratio={cross_doc_ratio:.0%}) → {output}")

    # 1. Fetch all chunks
    print("\nScrolling Qdrant…")
    try:
        points = _qdrant_scroll(QDRANT_URL, COLLECTION)
    except Exception as exc:
        sys.exit(f"[ERROR] Cannot reach Qdrant: {exc}")

    if not points:
        sys.exit("[ERROR] No points found in collection. Run ingestion first.")

    usable = [
        p for p in points
        if len((p.get("payload", {}).get("content") or "").strip()) >= 80
    ]
    print(f"  Found {len(points)} chunks total.")
    print(f"  Usable  {len(usable)} chunks (≥80 chars).")

    # 2. Build contexts
    # Oversample by ~45% to account for ~30% filter rate and still hit n_pairs exactly
    _oversample = max(int(n_pairs * 1.45), n_pairs + 20)
    n_cross = int(_oversample * cross_doc_ratio)
    n_single = _oversample - n_cross

    sampled_single = _stratified_sample(usable, n_single)
    single_contexts = [
        [(p.get("payload", {}).get("content") or "").strip()]
        for p in sampled_single
    ]
    single_meta = [
        {
            "file_name": (p.get("payload", {}).get("metadata") or {}).get("file_name", "unknown"),
            "chunk_type": (p.get("payload", {}).get("metadata") or {}).get("chunk_type", "doc"),
            "sheet_name": (p.get("payload", {}).get("metadata") or {}).get("sheet_name", ""),
            "qa_type": "single",
        }
        for p in sampled_single
    ]

    cross_contexts: list[list[str]] = []
    cross_meta: list[dict] = []
    by_file: dict[str, list] = {}
    for p in usable:
        fn = (p.get("payload", {}).get("metadata") or {}).get("file_name", "unknown")
        by_file.setdefault(fn, []).append(p)

    file_names = [f for f, pts in by_file.items() if len(pts) >= 1]
    random.shuffle(file_names)
    while len(cross_contexts) < n_cross and len(file_names) >= 2:
        f1, f2 = random.sample(file_names, 2)
        p1 = random.choice(by_file[f1])
        p2 = random.choice(by_file[f2])
        c1 = (p1.get("payload", {}).get("content") or "").strip()
        c2 = (p2.get("payload", {}).get("content") or "").strip()
        if not c1 or not c2:
            continue
        cross_contexts.append([c1, c2])
        cross_meta.append({
            "file_name": f"{f1} + {f2}",
            "chunk_type": "cross_doc",
            "sheet_name": "",
            "qa_type": "cross",
        })

    all_contexts = single_contexts + cross_contexts
    all_meta = single_meta + cross_meta
    print(f"  Built {len(single_contexts)} single-chunk + {len(cross_contexts)} cross-doc contexts.")

    # 3. Configure Synthesizer: REASONING for single, MULTICONTEXT for cross-doc
    evolutions: dict = {Evolution.REASONING: 1.0 - cross_doc_ratio}
    if cross_doc_ratio > 0:
        evolutions[Evolution.MULTICONTEXT] = cross_doc_ratio

    evolution_config = EvolutionConfig(evolutions=evolutions, num_evolutions=num_evolutions)
    synthesizer = Synthesizer(model=judge, async_mode=False, evolution_config=evolution_config)

    # 4. Generate one context at a time to avoid DeepEval IndexError on failed contexts
    print(f"\nGenerating goldens with DeepEval Synthesizer…")
    goldens = []
    for i, ctx in enumerate(all_contexts):
        try:
            synthesizer.synthetic_goldens = []
            synthesizer.generate_goldens_from_contexts(contexts=[ctx], max_goldens_per_context=1)
            goldens.extend(synthesizer.synthetic_goldens)
        except Exception as exc:
            print(f"  [WARN] Context {i} failed: {exc}")

    print(f"  Synthesizer produced {len(goldens)} goldens.")

    # 5. Convert to dataset format
    dataset: list[dict] = []
    skipped = 0
    for i, golden in enumerate(goldens):
        meta = all_meta[i] if i < len(all_meta) else {}
        q = (golden.input or "").strip()
        a = (golden.expected_output or "").strip()
        if not q or not a:
            continue
        if _is_bad_question(q):
            skipped += 1
            continue
        dataset.append({
            "input": q,
            "expected_output": a,
            "context": list(golden.context) if golden.context else all_contexts[i],
            "metadata": meta,
        })

    # Truncate to exactly n_pairs if we got more (due to oversampling)
    dataset = dataset[:n_pairs]

    print(f"\nGenerated {len(dataset)}/{n_pairs} QA pairs ({skipped} filtered as self-referential/unanswerable).")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation QA dataset from Qdrant chunks.")
    parser.add_argument("--n-pairs", type=int, default=200, help="Target number of QA pairs (default 200).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--cross-doc", type=float, default=0.15,
                        help="Fraction of QA pairs as cross-document MULTICONTEXT (0.0–1.0). Default: 0.15")
    parser.add_argument("--num-evolutions", type=int, default=1,
                        help="Number of evolution steps (0 = no evolution, plain QA). Default: 1")
    args = parser.parse_args()

    random.seed(args.seed)
    main(n_pairs=args.n_pairs, output=args.output, cross_doc_ratio=args.cross_doc, num_evolutions=args.num_evolutions)
