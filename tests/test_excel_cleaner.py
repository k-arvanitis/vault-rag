import pytest
import json
import openpyxl
from src.preprocessing.excel_cleaner import process_file, heuristic_fallback, _parse_llm_json, find_data_end_row

@pytest.fixture
def mock_llm_fn():
    def _llm(prompt: str) -> str:
        return json.dumps({
            "metadata_rows": [0, 1],
            "header_rows": [2, 3],
            "units_row": 4,
            "data_start_row": 5,
            "notes": ["Note 1"],
            "table_description": "Test table"
        })
    return _llm

def test_excel_cleaner_mock(tmp_path, mock_llm_fn):
    # Create a dummy excel file
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "TestSheet"
    
    rows = [
        ["Report 2026", None],
        ["Subtitle", None],
        ["Group A", "Group A"],
        ["Sub 1", "Sub 2"],
        ["(kg)", "(kg)"],
        [10, 20],
        [30, 40],
        ["", ""], # Empty row
        ["Footer note", None] # Footer
    ]
    
    for r in rows:
        ws.append(r)
        
    excel_path = tmp_path / "test.xlsx"
    wb.save(excel_path)
    
    res = process_file(str(excel_path), mock_llm_fn)
    assert "TestSheet" in res
    sheet_res = res["TestSheet"]
    
    # Check DataFrame
    df = sheet_res.df
    assert len(df) == 2 # Only rows 5 and 6 should be included
    
    # Check headers
    assert list(df.columns) == ["Group A > Sub 1", "Group A > Sub 2"]
    
    # Check values
    assert df.iloc[0, 0] == 10
    
    # Check metadata, including stripped footer notes.
    assert sheet_res.metadata.notes == ["Note 1", "Footer note"]
    
def test_fallback():
    rows = [
        ["Report", None],
        ["Col1", "Col2"],
        [1, 2]
    ]
    res = heuristic_fallback(rows)
    assert res["header_rows"] == [1]
    assert res["data_start_row"] == 2
    
def test_footer_detection():
    rows = [
        ["H1", "H2"],
        [1, 2],
        [3, 4],
        [None, None],
        ["Note", None]
    ]
    end_row = find_data_end_row(rows, 1)
    assert end_row == 3

def test_invalid_json():
    rows = [
        ["Report", None],
        ["Col1", "Col2"],
        [1, 2]
    ]
    res = _parse_llm_json("Invalid JSON { something", rows)
    assert res["header_rows"] == [1]
