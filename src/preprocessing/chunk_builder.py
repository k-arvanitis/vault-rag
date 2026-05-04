"""
Convert cleaned SheetResult objects into text chunks for vector stores.
"""

from typing import List, Dict, Any
import pandas as pd
from .excel_cleaner import SheetResult

def build_chunks(sheet_result: SheetResult) -> List[Dict[str, Any]]:
    """Build a list of chunks from a cleaned SheetResult."""
    chunks = []
    
    meta = sheet_result.metadata
    
    prefix_lines = []
    if meta.table_description:
        prefix_lines.append(f"Description: {meta.table_description}")
    if meta.notes:
        prefix_lines.append("Notes:")
        for n in meta.notes:
            prefix_lines.append(f"- {n}")
            
    prefix_str = "\n".join(prefix_lines)
    
    df = sheet_result.df
    columns = df.columns.tolist()
    
    for i, row in df.iterrows():
        row_pairs = []
        for col_idx, col in enumerate(columns):
            val = row.iloc[col_idx]
            if pd.isna(val) or str(val).strip() == "":
                continue
            row_pairs.append(f"{col}: {val}")
            
        if not row_pairs:
            continue
            
        row_str = " | ".join(row_pairs)
        
        chunk_text = f"[{meta.source_file} - {meta.sheet_name}]\n{prefix_str}\n\nData Row: {row_str}".strip()
        
        chunk_dict = {
            "text": chunk_text,
            "metadata": {
                "sheet_name": meta.sheet_name,
                "source_file": meta.source_file,
                "row_index": i,
                "table_description": meta.table_description,
                "notes": meta.notes,
            }
        }
        chunks.append(chunk_dict)
        
    return chunks
