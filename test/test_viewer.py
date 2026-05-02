"""Smoke tests for AFE viewer HTML generation."""

from pathlib import Path

import numpy as np
import pytest

from afe.viewer import save_viewer


def test_save_viewer_creates_html_file(tmp_path):
    """save_viewer produces a standalone HTML file with expected markers."""
    rng = np.random.RandomState(42)
    spatial = rng.randn(30, 3).astype(np.float32)
    arrows = rng.randn(30, 2, 3).astype(np.float32)
    labels = np.array([0] * 15 + [1] * 15)

    out_path = tmp_path / "viewer.html"
    save_viewer(spatial, arrows, labels=labels, path=str(out_path), open_browser=False)

    assert out_path.exists()
    content = out_path.read_text()

    # Should contain the core Three.js scene setup
    assert "<html" in content
    assert "</html>" in content
    assert "THREE" in content
    assert "scene" in content
    assert "camera" in content
    assert "renderer" in content

    # Should contain the data arrays
    assert "DATA.spatial" in content
    assert "arrow_" in content
    assert "DATA.labels" in content


def test_save_viewer_without_labels(tmp_path):
    """save_viewer works without explicit labels."""
    rng = np.random.RandomState(42)
    spatial = rng.randn(20, 3).astype(np.float32)
    arrows = rng.randn(20, 1, 3).astype(np.float32)

    out_path = tmp_path / "viewer_no_labels.html"
    save_viewer(spatial, arrows, path=str(out_path), open_browser=False)

    assert out_path.exists()
    content = out_path.read_text()
    assert "THREE" in content
