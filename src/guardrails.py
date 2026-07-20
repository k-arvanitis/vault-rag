"""Lightweight input/output guardrails for /query and /query/stream.

Hand-rolled rather than a third-party framework -- both alternatives evaluated
(Guardrails Hub's no-LLM-call injection validator, then LLM Guard) turned out
to be dead ends: the former's validator package no longer exists on the Hub
registry, the latter's repo was archived by its owner shortly before this was
written. A small local heuristic has no dependency on either project's
maintenance lifecycle, adds no latency, and needs no model download.

Three independent checks:
- `check_prompt_injection`: input-side, pattern match against known
  instruction-override/jailbreak phrasings.
- `check_system_prompt_leak`: output-side, n-gram overlap against the real
  system prompt -- catches the model being tricked into echoing its own
  instructions. Deliberately narrow: a PII/refusal output scanner would
  redact or flag this corpus's actual correct answers (business records --
  supplier names, departments, amounts), so this checks only for leaking
  internals, not for content classification.
- `check_corpus_enumeration`: input-side, pattern match for questions about
  the corpus itself rather than its content ("how many documents", "what was
  last uploaded") -- meta/inventory information a non-admin viewer shouldn't
  get from the chat, applied only when the caller isn't an admin (see api.py).
"""

from __future__ import annotations

import re

from src.rag_agent import SYSTEM_PROMPT

# Classic instruction-override / jailbreak phrasings. Deliberately pattern-based
# and reviewable in a diff, not an ML classifier -- see module docstring for why.
_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above|earlier) instructions",
    r"disregard (all |any )?(previous|prior|above|earlier) instructions",
    r"forget (all |any )?(previous|prior|above|earlier) instructions",
    r"you are (now|no longer) (a|an) ",
    r"act as (if you are |a |an )?(dan|jailbreak|unrestricted|unfiltered)",
    r"pretend (you are|to be) (not |)an? ai",
    r"reveal (your |the )?(system prompt|instructions|prompt)",
    r"(print|show|repeat|output) (your |the )?(system prompt|instructions)",
    r"what (is|are) your (system prompt|instructions)",
    r"new instructions?:",
    r"\bsystem\s*:\s*you are\b",
    r"developer mode",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def check_prompt_injection(question: str) -> bool:
    """True if `question` matches a known instruction-override/jailbreak pattern."""
    return bool(_INJECTION_RE.search(question))


# Meta/inventory questions about the corpus itself, not its content -- "how
# many documents" and "what's in the corpus" ask about the collection; "what
# does the procurement policy say about X" asks about content and must not
# match here.
_ENUMERATION_PATTERNS = [
    r"how many (documents|files|sources|pdfs|records)( are| do you)? (there|in|indexed)",
    r"(list|name) (all |the )?(documents|files|sources)",
    r"what (documents|files|sources) (do you have|are (there|available|indexed))",
    r"(last|most recent(ly)?|latest|newest) (uploaded|added|indexed) (document|file)",
    r"what('s| is| was) the (last|latest|newest|most recent) (document|file|upload)",
    r"what('s| is) in (the |your )?(corpus|collection|knowledge base|database|vector (store|database))",
    r"how many .{0,15}(vector (store|database)|database)",
    r"(full |entire )?(list|inventory) of (documents|files|sources)",
]
_ENUMERATION_RE = re.compile("|".join(_ENUMERATION_PATTERNS), re.IGNORECASE)


def check_corpus_enumeration(question: str) -> bool:
    """True if `question` asks about the corpus's inventory/metadata rather
    than its content -- see the module docstring for the content/meta split."""
    return bool(_ENUMERATION_RE.search(question))


_COUNT_RE = re.compile(
    r"how many (documents|files|sources|pdfs|records)( are| do you)? (there|in|indexed)",
    re.IGNORECASE,
)


def check_document_count_question(question: str) -> bool:
    """True for the specific "how many documents" sub-case of corpus
    enumeration -- an admin asking this gets a real count (see api.py),
    not a RAG-agent guess: the agent has no counting tool, so it was
    answering from whatever text chunk happened to be retrieved."""
    return bool(_COUNT_RE.search(question))


_WORD_RE = re.compile(r"[a-z0-9]+")
_LEAK_NGRAM = 8


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


_SYSTEM_PROMPT_NGRAMS = _ngrams(_WORD_RE.findall(SYSTEM_PROMPT.lower()), _LEAK_NGRAM)


def check_system_prompt_leak(answer: str) -> bool:
    """True if `answer` echoes an `_LEAK_NGRAM`-word run verbatim from the real
    system prompt -- a normal answer to a question about this corpus has no
    reason to share an 8-word run with the agent's own instructions."""
    if len(_SYSTEM_PROMPT_NGRAMS) == 0:
        return False
    answer_ngrams = _ngrams(_WORD_RE.findall(answer.lower()), _LEAK_NGRAM)
    return not _SYSTEM_PROMPT_NGRAMS.isdisjoint(answer_ngrams)


REFUSAL_MESSAGE = "This request can't be processed."
