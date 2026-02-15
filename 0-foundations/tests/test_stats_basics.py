from __future__ import annotations

from pathlib import Path

import pytest


def test_stats_basics_main_saves_plots(tmp_path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    stats_basics = pytest.importorskip("src.stats_basics")

    monkeypatch.setattr(stats_basics, "ensure_plots_dir", lambda: Path(tmp_path))

    stats_basics.main()

    assert (tmp_path / "hist_heights.png").exists()
    assert (tmp_path / "hist_weights.png").exists()
    assert (tmp_path / "scatter_height_vs_weight.png").exists()
