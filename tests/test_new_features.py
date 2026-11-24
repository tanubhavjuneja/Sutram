"""Tests for the newest feature round only: meta fusion, deep models
(LSTM autoencoder + TGN-lite GNN) and the new report views."""

# aml first: it eagerly imports torch, which is only stable before the
# heavy pandas/sklearn stack has loaded on Windows.
from aml.pipeline import Pipeline
from aml.report import generate_report

import pandas as pd
import pytest


def _run(paths, **kw):
    return Pipeline(min_events_per_entity=8, **kw).run(paths, with_models=True)


def _dense_df(n_entities: int = 20, n_events: int = 30, seed: int = 3) -> pd.DataFrame:
    """Dense synthetic unified frame: every entity has >= n_events rows."""
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_entities):
        t0 = pd.Timestamp("2024-01-01") + pd.Timedelta(days=seed)
        for i in range(n_events):
            rows.append({
                "entity_id": f"E{e:04d}",
                "counterparty_id": f"C{(e + i) % n_entities:04d}",
                "timestamp": t0 + pd.Timedelta(hours=i * 3),
                "amount": float(rng.lognormal(7, 0.6)),
                "source": ["bank", "cdr", "social"][i % 3],
                "event_type": ["transaction", "call", "post"][i % 3],
                "location": f"T{e % 8}",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# meta fusion
# ---------------------------------------------------------------------------
def test_meta_fusion_returns_ranking_and_falls_back(synthetic_dir, monkeypatch):
    paths = [str(p) for p in sorted(synthetic_dir.glob("sources/*"))]
    res = _run(paths)
    assert res.rankings

    from aml import fusion_meta
    from aml.fusion_meta import fuse_rank_meta
    meta = fuse_rank_meta(res.model_scores, res.unified)
    assert len(meta.rank) == len(res.rankings)
    assert meta.rank

    monkeypatch.setattr(fusion_meta, "_ASSET",
                        synthetic_dir / "does_not_exist.joblib")
    fusion_meta._loaded.update({"clf": None, "path": None})
    fallback = fuse_rank_meta(res.model_scores, res.unified)
    assert fallback.rank, "Borda fallback must still rank"
    assert getattr(fallback, "fallback", False)


def test_pipeline_meta_fusion_end_to_end(synthetic_dir):
    paths = [str(p) for p in sorted(synthetic_dir.glob("sources/*"))]
    res = _run(paths, fusion="meta")
    assert res.rankings
    assert all(0.0 <= r["score"] <= 1.0 for r in res.rankings)


# ---------------------------------------------------------------------------
# deep models (torch optional — skip silently if unavailable)
# ---------------------------------------------------------------------------
@pytest.fixture
def torch_ok():
    from aml.models import _torch
    torch = _torch()
    return torch is not None


def test_statml_lstm_scores_entities(dense_df, torch_ok):
    if not torch_ok:
        pytest.skip("torch not installed")
    from aml.models import fit_statml_lstm
    scores = fit_statml_lstm(dense_df, epochs=4)
    assert len(scores) == dense_df["entity_id"].nunique()
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_network_tgn_scores_entities(dense_df, torch_ok):
    if not torch_ok:
        pytest.skip("torch not installed")
    from aml.models import fit_network_tgn
    scores = fit_network_tgn(dense_df, epochs=10)
    # the graph includes counterparty nodes, not just the reporting entities
    expected = (dense_df["entity_id"].nunique()
                + dense_df["counterparty_id"].nunique())
    assert len(scores) == expected
    assert all(0.0 <= v <= 1.0 for v in scores.values())


# ---------------------------------------------------------------------------
# report views (network ego-graph, activity map, per-person drill-down)
# ---------------------------------------------------------------------------
def test_report_contains_new_views(synthetic_dir, tmp_path):
    paths = [str(p) for p in sorted(synthetic_dir.glob("sources/*"))]
    res = _run(paths)
    from aml.config import PipelineConfig
    written = generate_report(res, PipelineConfig(), tmp_path)

    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Suspect correlation graph" in md
    assert "Activity map" in md
    assert "A closer look at the people who stand out" in md

    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Suspect correlation graph" in html
    assert "Activity map" in html

    charts = {p.name for p in (tmp_path / "charts").glob("*.png")}
    assert "network.png" in charts
    assert "map.png" in charts
    assert any(name.startswith("entity_") for name in charts)

    assert (tmp_path / "report.pdf").stat().st_size > 0


def test_entity_timeline_renders_any_eid(synthetic_dir, tmp_path):
    from aml.visuals import render_entity_timeline
    paths = [str(p) for p in sorted(synthetic_dir.glob("sources/*"))]
    res = _run(paths)
    eid = res.rankings[0]["entity_id"]
    img = render_entity_timeline(res, tmp_path, eid)
    assert img.exists() and img.stat().st_size > 0


def test_map_label_places_beside_single_marker():
    """Regression: annotation extents include the leader line, so measuring the
    annotation (rather than its text) made every label collide with its own
    marker and silently placed nothing."""
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    from aml.visuals import THEMES, _place_map_labels

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    point = {"label": "T1", "lon": 5.0, "lat": 5.0, "count": 20,
             "flagged": True}

    placed = _place_map_labels(ax, fig, [point], [point], THEMES["dark"],
                               max_count=20)
    assert placed == 1
    (label,) = ax.texts
    fig.canvas.draw()
    box = Text.get_window_extent(label, renderer=fig.canvas.get_renderer())
    px, py = ax.transData.transform((point["lon"], point["lat"]))
    assert not (box.x0 <= px <= box.x1 and box.y0 <= py <= box.y1), \
        "label must not cover its own marker"
    plt.close(fig)


def test_map_labels_dense_do_not_overlap():
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    from aml.visuals import THEMES, _place_map_labels

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    points = [
        {"label": f"P{i}", "lon": 5 + (i % 12) * 8, "lat": 5 + (i // 12) * 8,
         "count": 3, "flagged": False}
        for i in range(120)
    ]
    flagged = []
    for i, (lon, lat) in enumerate([(50, 50), (20, 70), (75, 25),
                                    (35, 20), (80, 70)]):
        point = {"label": f"F{i}", "lon": lon, "lat": lat,
                 "count": 90 - i * 10, "flagged": True}
        points.append(point)
        flagged.append(point)

    placed = _place_map_labels(ax, fig, points, flagged, THEMES["dark"],
                               max_count=90)
    assert placed >= 4
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [t for t in ax.texts if t.get_text().startswith("F")]
    assert len(labels) == placed
    boxes = [Text.get_window_extent(t, renderer=renderer) for t in labels]
    for i, box in enumerate(boxes):
        assert ax.bbox.contains(*box.p0) and ax.bbox.contains(*box.p1)
        for other in boxes[i + 1:]:
            assert not box.overlaps(other)
    plt.close(fig)


@pytest.fixture
def dense_df():
    return _dense_df()
