"""Utility to inspect and reset ingested table data in Qdrant.

Usage:
  # List table chunks currently in Qdrant
  python scripts/reset_tables.py --list

  # Delete all table chunks for a specific file
  python scripts/reset_tables.py --file GRC_2023.xlsx

  # Wipe all table chunks (rows + summaries) from documents_chunks
  python scripts/reset_tables.py --all
"""

from __future__ import annotations

import argparse
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

COLLECTION = "documents_chunks"


def _qdrant_url() -> str:
    return os.getenv("QDRANT_URL", "http://localhost:7333").rstrip("/")


def _request(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    req = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except HTTPError as e:
        raise RuntimeError(f"Qdrant {e.code}: {e.read().decode()}") from e
    except URLError as e:
        raise RuntimeError(f"Cannot connect to Qdrant: {e}") from e


def list_ingested() -> None:
    base = _qdrant_url()
    # Count all table chunks
    result = _request("POST", f"{base}/collections/{COLLECTION}/points/count",
                       {"filter": {"must": [{"key": "source_type", "match": {"value": "table"}}]}})
    total = result["result"]["count"]
    print(f"Table chunks in '{COLLECTION}': {total}")

    # Count sheet summaries
    result = _request("POST", f"{base}/collections/{COLLECTION}/points/count",
                       {"filter": {"must": [{"key": "metadata.chunk_type", "match": {"value": "sheet_summary"}}]}})
    summaries = result["result"]["count"]
    print(f"  of which sheet summaries: {summaries}")
    print(f"  of which row chunks:      {total - summaries}")

    # List distinct source files via scroll (sample)
    scroll = _request("POST", f"{base}/collections/{COLLECTION}/points/scroll", {
        "filter": {"must": [{"key": "metadata.chunk_type", "match": {"value": "sheet_summary"}}]},
        "limit": 100,
        "with_payload": True,
        "with_vector": False,
    })
    points = scroll["result"]["points"]
    if points:
        files: dict[str, list[str]] = {}
        for p in points:
            meta = p["payload"].get("metadata", {})
            f = meta.get("source_file", "?")
            s = meta.get("sheet_name", "?")
            files.setdefault(f, []).append(s)
        print("\nIngested files:")
        for f, sheets in sorted(files.items()):
            print(f"  {f}: {len(sheets)} sheets")


def delete_file(file_name: str) -> None:
    base = _qdrant_url()
    result = _request("POST", f"{base}/collections/{COLLECTION}/points/count", {
        "filter": {"must": [
            {"key": "source_type", "match": {"value": "table"}},
            {"key": "metadata.source_file", "match": {"value": file_name}},
        ]}
    })
    count = result["result"]["count"]
    if count == 0:
        print(f"No table chunks found for '{file_name}'.")
        return

    _request("POST", f"{base}/collections/{COLLECTION}/points/delete?wait=true", {
        "filter": {"must": [
            {"key": "source_type", "match": {"value": "table"}},
            {"key": "metadata.source_file", "match": {"value": file_name}},
        ]}
    })
    print(f"Deleted {count} table chunks for '{file_name}'.")


def delete_all() -> None:
    base = _qdrant_url()
    result = _request("POST", f"{base}/collections/{COLLECTION}/points/count",
                       {"filter": {"must": [{"key": "source_type", "match": {"value": "table"}}]}})
    total = result["result"]["count"]

    _request("POST", f"{base}/collections/{COLLECTION}/points/delete?wait=true",
             {"filter": {"must": [{"key": "source_type", "match": {"value": "table"}}]}})
    print(f"Deleted {total} table chunks from '{COLLECTION}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and reset ingested table data in Qdrant.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List all ingested files and chunk counts")
    group.add_argument("--file", metavar="FILE_NAME", help="Delete all table chunks for a specific file")
    group.add_argument("--all", action="store_true", help="Delete ALL table chunks from documents_chunks")
    args = parser.parse_args()

    if args.list:
        list_ingested()
    elif args.file:
        delete_file(args.file)
    elif args.all:
        confirm = input("This will delete ALL table chunks from Qdrant. Type 'yes' to confirm: ")
        if confirm.strip().lower() == "yes":
            delete_all()
        else:
            print("Aborted.")


if __name__ == "__main__":
    main()
