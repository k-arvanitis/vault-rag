from __future__ import annotations

import argparse
import base64
import html
import io
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Literal, Union

import pypdfium2 as pdfium
import requests
from PIL import Image

from src.parser.chandra_parser import (
    normalize_footnote_prefixes,
    normalize_image_annotations,
    normalize_subscripts,
    normalize_superscripts,
    replace_html_tables_with_titles,
)
from src.parser.input_utils import resolve_to_pdf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/output/lightonocr"

DEFAULT_ENDPOINT = "http://127.0.0.1:8002/v1/chat/completions"
DEFAULT_MODEL = "lightonocr-2-1b-ocr-soup"
def _build_image_analysis_endpoint() -> str:
    base = os.getenv("IMAGE_ANALYSIS_API_BASE", os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")).rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/chat/completions"

DEFAULT_IMAGE_ANALYSIS_ENDPOINT = _build_image_analysis_endpoint()
DEFAULT_IMAGE_ANALYSIS_MODEL = os.getenv("IMAGE_ANALYSIS_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
DEFAULT_PROMPT = (
    "Transcribe this page into Markdown. If there are images, figures, or diagrams, "
    "provide a brief description of what they contain within the text flow."
)
DEFAULT_IMAGE_ANALYSIS_PROMPT = (
    "You are analyzing a document page image. The OCR output detected visual elements. "
    "Return JSON only with key 'descriptions' containing exactly {count} strings in reading order. "
    "Each string should describe one visual element. Follow these rules per element type:\n"
    "- Chart/graph/plot: state the chart type, axes and their ranges, all series/models shown, "
    "which series performs best and at what value, notable trends or crossover points, and one key takeaway.\n"
    "- Table: state the columns, value ranges, and which row/column stands out.\n"
    "- Diagram/figure: describe the structure and what it illustrates.\n"
    "- Logo/icon: name or describe it briefly.\n"
    "Be specific with numbers where readable. Do not use generic filler like 'the graph shows performance'."
)
HTML_IMG_RE = re.compile(r"(?is)<img\b(?P<attrs>[^>]*)>")
IMG_ALT_RE = re.compile(r"""(?is)\balt\s*=\s*(?:"(?P<d>[^"]*)"|'(?P<s>[^']*)')""")
IMAGE_MARKER_RE = re.compile(r"\[IMAGE\](?:\s+[^\n]+)?")


def _image_to_data_url(image: Image.Image, max_side: int = 1568) -> str:
    """Encode image as base64 JPEG data URL, resizing to max_side if needed.

    Groq vision models require images ≤ 1568px on the longest side.
    JPEG is used instead of PNG to keep payload size small.
    """
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _post_process_markdown(
    markdown_content: str,
    markdown_tables: bool = True,
    table_mode: Literal["pipe", "grid"] = "grid",
) -> str:
    # LightOn often returns escaped HTML (e.g. &lt;table&gt;, &lt;sup&gt;), so unescape first.
    markdown_content = html.unescape(markdown_content)

    def _replace_html_img(match: re.Match) -> str:
        attrs = match.group("attrs") or ""
        alt_match = IMG_ALT_RE.search(attrs)
        alt_text = ""
        if alt_match:
            alt_text = (alt_match.group("d") or alt_match.group("s") or "").strip()
            alt_text = " ".join(alt_text.split())
        return "[IMAGE]" if not alt_text else f"[IMAGE] {alt_text}"

    markdown_content = HTML_IMG_RE.sub(_replace_html_img, markdown_content)

    if markdown_tables:
        markdown_content = replace_html_tables_with_titles(markdown_content, mode=table_mode)
    markdown_content = normalize_superscripts(markdown_content)
    markdown_content = normalize_subscripts(markdown_content)
    markdown_content = normalize_footnote_prefixes(markdown_content)
    markdown_content = normalize_image_annotations(markdown_content)
    return markdown_content


def _analyze_page_images_with_qwen(
    image: Image.Image,
    marker_count: int,
    endpoint: str,
    model_name: str,
    timeout_sec: int = 120,
    prompt_template: str = DEFAULT_IMAGE_ANALYSIS_PROMPT,
    api_key: str | None = None,
) -> list[str]:
    if marker_count <= 0:
        return []

    prompt = prompt_template.format(count=marker_count)
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout_sec)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    text = str(content).strip()
    # Strip markdown code fences (model may wrap JSON in ```json ... ```)
    text = re.sub(r"^\s*```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?\s*```\s*$", "", text).strip()

    try:
        parsed = json.loads(text)
        descriptions = parsed.get("descriptions", []) if isinstance(parsed, dict) else []
        descriptions = [str(x).strip() for x in descriptions if str(x).strip()]
    except json.JSONDecodeError:
        descriptions = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]

    if not descriptions:
        descriptions = ["Visual element detected, but detailed description was not produced."] * marker_count

    if len(descriptions) < marker_count:
        descriptions.extend([descriptions[-1]] * (marker_count - len(descriptions)))

    return descriptions[:marker_count]


def _inject_image_blocks(markdown_content: str, descriptions: list[str]) -> str:
    idx = 0

    def _replace(_: re.Match) -> str:
        nonlocal idx
        if idx < len(descriptions):
            desc = descriptions[idx]
        else:
            desc = "Visual element detected."
        idx += 1
        return f"[IMAGE START]\n{desc}\n[IMAGE END]"

    return IMAGE_MARKER_RE.sub(_replace, markdown_content)


def _ocr_image_markdown(
    image: Image.Image,
    endpoint: str,
    model_name: str,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout_sec: int = 120,
) -> str:
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = requests.post(endpoint, json=payload, timeout=timeout_sec)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return str(content)


def pdf_to_markdown(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    endpoint: str = DEFAULT_ENDPOINT,
    model_name: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    render_scale: float = 2.77,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout_sec: int = 120,
    markdown_tables: bool = True,
    table_mode: Literal["pipe", "grid"] = "grid",
    enrich_images: bool = True,
    image_analysis_endpoint: str = DEFAULT_IMAGE_ANALYSIS_ENDPOINT,
    image_analysis_model: str = DEFAULT_IMAGE_ANALYSIS_MODEL,
    image_analysis_timeout_sec: int = 120,
    image_analysis_api_key: str | None = None,
) -> Path:
    pdf_path = resolve_to_pdf(pdf_path)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    doc = pdfium.PdfDocument(str(pdf_path))
    page_blocks: list[str] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pil_image = page.render(scale=render_scale).to_pil().convert("RGB")
        page_md = _ocr_image_markdown(
            image=pil_image,
            endpoint=endpoint,
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_sec=timeout_sec,
        )
        page_md = _post_process_markdown(
            markdown_content=page_md,
            markdown_tables=markdown_tables,
            table_mode=table_mode,
        )

        marker_count = len(IMAGE_MARKER_RE.findall(page_md))
        if enrich_images and marker_count > 0:
            descriptions = _analyze_page_images_with_qwen(
                image=pil_image,
                marker_count=marker_count,
                endpoint=image_analysis_endpoint,
                model_name=image_analysis_model,
                timeout_sec=image_analysis_timeout_sec,
                api_key=image_analysis_api_key,
            )
            page_md = _inject_image_blocks(page_md, descriptions)

        page_blocks.append(f"\n\n<!-- PAGE {page_idx + 1} -->\n\n{page_md}")

    markdown_content = "".join(page_blocks)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pdf_path.stem}.md"
    out_path.write_text(markdown_content, encoding="utf-8")
    return out_path


def png_to_markdown(
    image_path: Union[str, Path],
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    endpoint: str = DEFAULT_ENDPOINT,
    model_name: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout_sec: int = 120,
    markdown_tables: bool = True,
    table_mode: Literal["pipe", "grid"] = "grid",
    enrich_images: bool = True,
    image_analysis_endpoint: str = DEFAULT_IMAGE_ANALYSIS_ENDPOINT,
    image_analysis_model: str = DEFAULT_IMAGE_ANALYSIS_MODEL,
    image_analysis_timeout_sec: int = 120,
    image_analysis_api_key: str | None = None,
) -> Path:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        markdown_content = _ocr_image_markdown(
            image=img_rgb,
            endpoint=endpoint,
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_sec=timeout_sec,
        )

    markdown_content = _post_process_markdown(
        markdown_content=markdown_content,
        markdown_tables=markdown_tables,
        table_mode=table_mode,
    )
    marker_count = len(IMAGE_MARKER_RE.findall(markdown_content))
    if enrich_images and marker_count > 0:
        descriptions = _analyze_page_images_with_qwen(
            image=img_rgb,
            marker_count=marker_count,
            endpoint=image_analysis_endpoint,
            model_name=image_analysis_model,
            timeout_sec=image_analysis_timeout_sec,
            api_key=image_analysis_api_key,
        )
        markdown_content = _inject_image_blocks(markdown_content, descriptions)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}.md"
    out_path.write_text(markdown_content, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LightOn OCR (vLLM OpenAI API) on a PDF/Office file or PNG and save markdown."
    )
    parser.add_argument("input_path", help="Path to a PDF/Office file (.doc/.docx/.ppt/.pptx) or PNG.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where output markdown will be saved.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="OpenAI-compatible chat completions endpoint.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Model name exposed by the endpoint.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="OCR instruction prompt sent for each page/image.",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=2.77,
        help="PDF page render scale (2.77 is around 200 DPI).",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument(
        "--no-markdown-tables",
        action="store_true",
        help="Disable conversion of HTML tables to [TABLE_START]/[TABLE_END] markdown blocks.",
    )
    parser.add_argument(
        "--table-mode",
        choices=["pipe", "grid"],
        default="grid",
        help="Markdown table rendering mode when --markdown-tables is enabled.",
    )
    parser.add_argument(
        "--no-enrich-images",
        action="store_true",
        help="Disable image analysis and [IMAGE START]/[IMAGE END] injection via Qwen.",
    )
    parser.add_argument(
        "--image-analysis-endpoint",
        default=DEFAULT_IMAGE_ANALYSIS_ENDPOINT,
        help="Endpoint for image analysis model (OpenAI-compatible chat completions).",
    )
    parser.add_argument(
        "--image-analysis-model",
        default=DEFAULT_IMAGE_ANALYSIS_MODEL,
        help="Model name for image analysis endpoint.",
    )
    parser.add_argument("--image-analysis-timeout-sec", type=int, default=120)

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix in {".pdf", ".docx", ".doc", ".ppt", ".pptx"}:
        out_path = pdf_to_markdown(
            pdf_path=input_path,
            output_dir=Path(args.output_dir),
            endpoint=args.endpoint,
            model_name=args.model_name,
            prompt=args.prompt,
            render_scale=args.render_scale,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
            markdown_tables=not args.no_markdown_tables,
            table_mode=args.table_mode,
            enrich_images=not args.no_enrich_images,
            image_analysis_endpoint=args.image_analysis_endpoint,
            image_analysis_model=args.image_analysis_model,
            image_analysis_timeout_sec=args.image_analysis_timeout_sec,
        )
    elif suffix == ".png":
        out_path = png_to_markdown(
            image_path=input_path,
            output_dir=Path(args.output_dir),
            endpoint=args.endpoint,
            model_name=args.model_name,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
            markdown_tables=not args.no_markdown_tables,
            table_mode=args.table_mode,
            enrich_images=not args.no_enrich_images,
            image_analysis_endpoint=args.image_analysis_endpoint,
            image_analysis_model=args.image_analysis_model,
            image_analysis_timeout_sec=args.image_analysis_timeout_sec,
        )
    else:
        raise ValueError(f"Unsupported input type: {suffix}. Use .pdf/.doc/.docx/.ppt/.pptx or .png")

    print(f"Saved markdown to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
