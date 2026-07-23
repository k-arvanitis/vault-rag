from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.ingestion.unstructured_ocr import call_unstructured_ocr


class _FakePix:
    """Minimal fitz.Pixmap stand-in: only what call_unstructured_ocr touches."""

    width = 3000
    height = 3900

    def pdfocr_save(self, path):
        # Real fitz would write a searchable PDF; we never read it back since
        # partition_pdf itself is mocked, so a no-op is sufficient.
        pass


class _FakeElement:
    def __init__(self, category, text, coordinates=None):
        self.category = category
        self._text = text
        self.metadata = SimpleNamespace(coordinates=coordinates, text_as_html=None)

    def __str__(self):
        return self._text


def _fake_element(category, text, points, system_w, system_h):
    coords = SimpleNamespace(
        points=points,
        system=SimpleNamespace(width=system_w, height=system_h),
    )
    return _FakeElement(category, text, coords)


def test_call_unstructured_ocr_scales_bbox_to_pdf_points():
    # unstructured's hi_res analysis re-renders our pixmap-derived PDF at its
    # own internal resolution (system 1500x1950 here, half our 3000x3900
    # pixmap) -- the returned bbox must be in PDF points (72 dpi) on the
    # *original* page, not unstructured's analysis-image pixels.
    # Real unstructured coordinates are numpy float64, not plain Python
    # floats -- a prior version of this code produced "np.float64(1.2)" in
    # the comment instead of "1.2", silently failing the downstream regex.
    el = _fake_element(
        "NarrativeText",
        "2. Term. This Lease shall commence on...",
        points=(
            (np.float64(100), np.float64(200)),
            (np.float64(100), np.float64(250)),
            (np.float64(400), np.float64(250)),
            (np.float64(400), np.float64(200)),
        ),
        system_w=1500,
        system_h=1950,
    )

    with patch(
        "unstructured.partition.pdf.partition_pdf", return_value=[el]
    ):
        result = call_unstructured_ocr(_FakePix(), dpi=300)

    assert "2. Term." in result
    # scale_x = pix.width/system.width * 72/dpi = (3000/1500) * (72/300) = 0.48
    # x: [100, 400] * 0.48 = [48.0, 192.0]; y: [200, 250] * 0.48 = [96.0, 120.0]
    assert "<!-- ocr_bbox:[48.0, 96.0, 192.0, 120.0] -->" in result


def test_call_unstructured_ocr_omits_bbox_comment_when_no_coordinates():
    el = _FakeElement("NarrativeText", "Some text with no layout info")

    with patch(
        "unstructured.partition.pdf.partition_pdf", return_value=[el]
    ):
        result = call_unstructured_ocr(_FakePix(), dpi=300)

    assert "ocr_bbox" not in result
    assert "Some text with no layout info" in result
