"""Report charts — terminal-style matplotlib renders for the Sutram console.

All functions render PNGs into a report directory and are safe on empty /
tiny inputs. Charts support dark (default) and light themes. Palette is
strictly semantic: amber = primary data, red = risk/alert, green = ok,
sky = info. Text-overlap guards: collision-checked map labels, spaced
ego-graph layouts, callout badges instead of in-circle text.
"""
from __future__ import annotations

import hashlib
import io
import math
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

THEMES: Dict[str, Dict[str, str]] = {
    "dark": {
        "bg_fig": "#020617",
        "bg_ax": "#0F172A",
        "bg_card": "#1E293B",
        "border": "#334155",
        "text": "#F8FAFC",
        "title": "#FFB000",
        "sub": "#94A3B8",
        "muted": "#64748B",
        "amber": "#FFB000",
        "red": "#F87171",
        "green": "#34D399",
        "sky": "#38BDF8",
        "yellow": "#FDE047",
        "violet": "#A78BFA",
        "edge": "#334155",
        "grid": "#1E293B",
        "node_edge": "#94A3B8",
    },
    "light": {
        "bg_fig": "#F5F5F0",
        "bg_ax": "#FFFFFF",
        "bg_card": "#FAFAF7",
        "border": "#D4D4D8",
        "text": "#18181B",
        "title": "#B45309",
        "sub": "#3F3F46",
        "muted": "#71717A",
        "amber": "#B45309",
        "red": "#DC2626",
        "green": "#047857",
        "sky": "#0369A1",
        "yellow": "#A16207",
        "violet": "#6D28D9",
        "edge": "#A1A1AA",
        "grid": "#E4E4E7",
        "node_edge": "#52525B",
    },
}


def _t(theme: str = "dark") -> Dict[str, str]:
    return THEMES.get(str(theme).lower(), THEMES["dark"])


def _apply(fig, ax, t: Dict[str, str]) -> None:
    """Terminal chrome: hard border box, mono ticks, no ornament."""
    fig.patch.set_facecolor(t["bg_fig"])
    ax.set_facecolor(t["bg_ax"])
    ax.tick_params(colors=t["sub"], labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_color(t["border"])
        ax.spines[s].set_linewidth(0.8)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(t["border"])
        ax.spines[s].set_linewidth(0.8)
    if ax.xaxis.label:
        ax.xaxis.label.set_color(t["sub"])
    if ax.yaxis.label:
        ax.yaxis.label.set_color(t["sub"])


def _style_axes_text(ax, t: Dict[str, str]) -> None:
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(t["sub"])
    ax.grid(axis="y", color=t["grid"], linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def _title(ax, t: Dict[str, str], text: str) -> None:
    ax.set_title(text, fontsize=10.5, fontweight="bold", color=t["title"],
                 pad=10, loc="left")


def _legend(ax, t: Dict[str, str], loc: str = "upper left") -> None:
    leg = ax.legend(frameon=True, facecolor=t["bg_card"], edgecolor=t["border"],
                    fontsize=8, loc=loc)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    leg.get_frame().set_linewidth(0.8)


def _headroom(ax, top: float = 1.18) -> None:
    """Leave space above the tallest mark so peak callouts never clip."""
    lo, hi = ax.get_ylim()
    if hi > lo:
        ax.set_ylim(lo, hi * top if hi > 0 else hi)


def _legend_below(ax, t: Dict[str, str], ncol: int = 4) -> None:
    leg = ax.legend(frameon=True, facecolor=t["bg_card"], edgecolor=t["border"],
                    fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                    ncol=ncol)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    leg.get_frame().set_linewidth(0.8)


def _boxes_overlap(a, b) -> bool:
    return not (a[1] < b[0] or a[0] > b[1] or a[3] < b[2] or a[2] > b[3])


def _box_hits_circle(box, cx: float, cy: float, radius: float) -> bool:
    closest_x = min(max(cx, box[0]), box[1])
    closest_y = min(max(cy, box[2]), box[3])
    return math.hypot(cx - closest_x, cy - closest_y) < radius


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(),
                edgecolor="none")
    plt.close(fig)
    return path


def _empty(ax, t: Dict[str, str], msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10,
            color=t["muted"], family="monospace", transform=ax.transAxes)
    ax.axis("off")


def _src_colors(t: Dict[str, str]) -> Dict[str, str]:
    return {"bank": t["amber"], "cdr": t["sky"], "social": t["yellow"]}


def _mono_labels(ax, t: Dict[str, str]) -> None:
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_family("monospace")


def render_timeline(result, out_dir: str | Path, name: str = "timeline.png",
                    theme: str = "dark") -> Path:
    """Daily traffic by modality as a line graph (not a stacked area)."""
    t = _t(theme)
    src = _src_colors(t)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df = result.unified
    fig, ax = plt.subplots(figsize=(10.2, 3.6), dpi=170)
    _apply(fig, ax, t)
    if df is None or df.empty or "timestamp" not in df.columns or df["timestamp"].dropna().empty:
        _empty(ax, t, "no timestamped events to plot")
        return _save(fig, path)
    ts = df.dropna(subset=["timestamp"]).copy()
    ts["date"] = pd.to_datetime(ts["timestamp"]).dt.floor("D")
    pivot = ts.pivot_table(index="date", columns="source", values="entity_id",
                           aggfunc="count", fill_value=0)
    pivot = pivot.resample("D").sum().fillna(0)
    pivot["total"] = pivot.sum(axis=1)

    for col in [c for c in pivot.columns if c != "total"]:
        ax.plot(pivot.index, pivot[col], marker="o", markersize=3.8,
                linewidth=1.9, label=str(col).upper(),
                color=src.get(str(col), t["sky"]), alpha=0.95)
    ax.plot(pivot.index, pivot["total"], color=t["muted"], linestyle="--",
            linewidth=1.5, label="TOTAL", alpha=0.9)

    ax.set_ylabel("records / day", fontsize=9.5, family="monospace")
    ax.set_xlabel("date", fontsize=9.5, family="monospace")
    _title(ax, t, "WHEN IT HAPPENED — daily traffic by modality")
    _apply(fig, ax, t)
    _style_axes_text(ax, t)
    ax.grid(axis="x", color=t["grid"], linewidth=0.6, alpha=0.45)
    _mono_labels(ax, t)
    _legend(ax, t, loc="upper left")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _headroom(ax, 1.15)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0)
        lbl.set_fontsize(8.5)

    if not pivot.empty and pivot["total"].max() > 0:
        peak_day = pivot["total"].idxmax()
        peak_val = int(pivot["total"].max())
        ax.annotate(f"peak {peak_val:,}", xy=(peak_day, peak_val),
                    xytext=(8, 10), textcoords="offset points",
                    fontsize=8.5, family="monospace", color=t["text"],
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=t["bg_card"],
                              edgecolor=t["border"], alpha=0.92, linewidth=0.6),
                    arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.7))
    return _save(fig, path)


def render_model_scores(result, out_dir: str | Path, name: str = "models.png",
                        theme: str = "dark") -> Path:
    """Per-model peak/coverage bars."""
    t = _t(theme)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    scores = result.model_scores or {}
    fig, ax = plt.subplots(figsize=(9.6, 3.2), dpi=160)
    _apply(fig, ax, t)
    if not scores:
        _empty(ax, t, "no model scores to plot")
        return _save(fig, path)
    models = sorted(scores)
    peaks = [max(scores[m].values()) if scores[m] else 0.0 for m in models]
    flagged = [sum(1 for v in scores[m].values() if v > 0.6) for m in models]
    x = np.arange(len(models))
    ax.bar(x - 0.19, peaks, 0.38, color=t["amber"], alpha=0.9)
    ax.set_xticks(x, [m.replace("_", "\n") for m in models], fontsize=7)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("peak score", fontsize=8.5, family="monospace")
    _title(ax, t, "DETECTION ENGINES — peak score and flagged count")
    _apply(fig, ax, t)
    _style_axes_text(ax, t)
    _mono_labels(ax, t)
    # Flagged counts are integer totals, not scores; a second axis prevents
    # them from being clipped above the 0-1 score scale.
    ax2 = ax.twinx()
    ax2.bar(x + 0.19, flagged, 0.38, color=t["red"], alpha=0.9)
    ax2.set_ylim(0, max(flagged + [1]) * 1.30)
    ax2.set_ylabel("flagged entities", fontsize=8.5, family="monospace",
                   color=t["red"])
    ax2.tick_params(axis="y", colors=t["red"], labelsize=8)
    ax2.spines["right"].set_color(t["red"])
    ax2.spines["right"].set_linewidth(0.8)
    for side in ("top", "left", "bottom"):
        ax2.spines[side].set_visible(False)
    handles = [
        mpatches.Patch(color=t["amber"], label="peak score (left)"),
        mpatches.Patch(color=t["red"], label="flagged >0.6 (right)"),
    ]
    leg = ax.legend(handles=handles, frameon=True, facecolor=t["bg_card"],
                    edgecolor=t["border"], fontsize=8, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    leg.get_frame().set_linewidth(0.8)
    return _save(fig, path)


def render_person_attention(result, out_dir: str | Path, name: str = "people.png",
                            theme: str = "dark") -> Path:
    """Who stands out — evidence-count bars, semantic risk colors."""
    from .insights import build_insights

    t = _t(theme)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig, ax = plt.subplots(figsize=(9.6, 3.8), dpi=160)
    _apply(fig, ax, t)
    top = build_insights(result, top_k=10)
    if not top:
        _empty(ax, t, "no people stood out this run")
        return _save(fig, path)
    labels = [f'{x["rank"]:02d}  {x["entity_id"]}' for x in reversed(top)]
    names = [x["name"] for x in reversed(top)]
    vals = [max(1, len(x["evidence"])) for x in reversed(top)]
    colors = [t["red"] if x["risk"] == "HIGH"
              else (t["yellow"] if x["risk"] == "MEDIUM" else t["sky"])
              for x in reversed(top)]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, alpha=0.92, height=0.62)
    ax.set_yticks(y, [f"{lab}\n{nm}" for lab, nm in zip(labels, names)],
                  fontsize=8.5)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlim(0, max(vals) * 1.26)
    ax.set_xlabel("evidence factors", fontsize=8.5, family="monospace")
    _title(ax, t, "A CLOSER LOOK AT THE PEOPLE WHO STAND OUT")
    _apply(fig, ax, t)
    _style_axes_text(ax, t)
    ax.grid(axis="x", color=t["grid"], linewidth=0.6, alpha=0.6)
    ax.grid(axis="y", visible=False)
    _mono_labels(ax, t)
    for i, v in enumerate(vals):
        ax.text(v + 0.12, i, f"{v}", va="center", fontsize=7.5,
                color=t["sub"], family="monospace")
    risk_handles = [
        mpatches.Patch(color=t["red"], label="HIGH risk"),
        mpatches.Patch(color=t["yellow"], label="MEDIUM risk"),
        mpatches.Patch(color=t["sky"], label="LOW risk"),
    ]
    leg = ax.legend(handles=risk_handles, frameon=True, facecolor=t["bg_card"],
                    edgecolor=t["border"], fontsize=8, loc="lower right")
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    leg.get_frame().set_linewidth(0.8)
    return _save(fig, path)


def render_sources(result, out_dir: str | Path, name: str = "sources.png",
                   theme: str = "dark") -> Path:
    """Per-source row / entity counts."""
    t = _t(theme)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    per_source = getattr(result, "per_source", None) or []
    fig, ax = plt.subplots(figsize=(9.6, 2.9), dpi=160)
    _apply(fig, ax, t)
    if not per_source:
        _empty(ax, t, "no source files loaded")
        return _save(fig, path)
    names = [f"{rs.source}\n{Path(rs.file).suffix[1:].upper()}" for rs in per_source]
    rows = [rs.n_rows for rs in per_source]
    ents = [rs.n_entities for rs in per_source]
    x = np.arange(len(names))
    ax.bar(x, rows, 0.52, label="records (left axis)", color=t["sky"], alpha=0.9)
    ax.set_xticks(x, names, fontsize=7.5)
    _title(ax, t, "WHERE THE RECORDS CAME FROM — ingestion volumes")
    _apply(fig, ax, t)
    _style_axes_text(ax, t)
    _mono_labels(ax, t)
    # Entities are typically 1-4 per file while records are thousands; a
    # second axis keeps both readable instead of hiding the entity counts.
    ax2 = ax.twinx()
    # Markers only: files are categorical, so a connecting line would imply a
    # trend between unrelated sources.
    ax2.scatter(x, ents, color=t["green"], marker="o", s=42, zorder=5,
                edgecolors=t["text"], linewidths=0.6)
    ax2.set_ylabel("entities", fontsize=8.5, family="monospace",
                   color=t["green"])
    ax2.tick_params(axis="y", colors=t["green"], labelsize=8)
    ax2.set_ylim(0, max(ents + [1]) * 1.45)
    ax2.spines["right"].set_color(t["green"])
    ax2.spines["right"].set_linewidth(0.8)
    for side in ("top", "left", "bottom"):
        ax2.spines[side].set_visible(False)
    handles = [mpatches.Patch(color=t["sky"], label="records (left axis)"),
               Line2D([0], [0], color=t["green"], marker="o", markersize=6,
                      linewidth=0, label="entities (right axis)")]
    leg = ax.legend(handles=handles, frameon=True, facecolor=t["bg_card"],
                    edgecolor=t["border"], fontsize=8, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    leg.get_frame().set_linewidth(0.8)
    return _save(fig, path)


def _entity_display_name(result, entity_id: str) -> str:
    from .insights import display_name
    return display_name(result, entity_id)


def _build_correlation_graph(result, top_k: int = 12, tol_minutes: int = 25):
    """Build the suspect-suspect connection graph used by the report.

    Connections are only drawn when the ensemble can support them:

    * direct events between two flagged people (bank / CDR / social)
    * call-or-post followed by a transfer to the same party (colocation)
    * two flagged people seen at the same location
    * two flagged people dealing with the same counterparty
    """
    import networkx as nx
    from .insights import build_insights

    top = build_insights(result, top_k=top_k)
    if not top:
        return None
    suspects = {str(it["entity_id"]): it for it in top}
    df = result.unified.copy()
    if df is None or df.empty:
        return None
    for col in ("entity_id", "counterparty_id", "source", "timestamp", "amount", "location"):
        if col not in df.columns:
            df[col] = ""
    df["entity_id"] = df["entity_id"].astype(str)
    df["counterparty_id"] = df["counterparty_id"].astype(str)
    df = df[df["entity_id"].isin(suspects) | df["counterparty_id"].isin(suspects)]
    if df.empty:
        return None

    actor_counts = df[df["entity_id"].isin(suspects)].groupby("entity_id").size().to_dict()
    G = nx.Graph()
    for eid, info in suspects.items():
        G.add_node(eid, name=info.get("name", eid), risk=info.get("risk", "LOW"),
                   n_events=int(actor_counts.get(eid, 0)), flagged=True)

    def add(a: str, b: str, kind: str, count: int = 1, detail: str = "") -> None:
        if a == b or a not in suspects or b not in suspects:
            return
        if not G.has_edge(a, b):
            G.add_edge(a, b, types=set(), counts=Counter(), details=set(), weight=0,
                       sources=set())
        edge = G[a][b]
        edge["types"].add(kind)
        edge["counts"][kind] += int(count)
        edge["weight"] += int(count)
        if detail:
            edge["details"].add(str(detail))

    edges = df.dropna(subset=["entity_id", "counterparty_id"])
    for (a, b), grp in edges.groupby(["entity_id", "counterparty_id"]):
        if a in suspects and b in suspects:
            sources = sorted(set(grp["source"].astype(str)))
            add(a, b, "direct", len(grp), "/".join(sources) or "event")
            if G.has_edge(a, b):
                G[a][b]["sources"] |= set(sources)

    # call/post shortly before a transfer to the same counterparty
    bank = edges[edges["source"] == "bank"].copy()
    comm = edges[edges["source"].isin(("cdr", "social"))].copy()
    if not bank.empty and not comm.empty:
        bank["ts"] = pd.to_datetime(bank["timestamp"], errors="coerce", utc=True)
        comm["ts"] = pd.to_datetime(comm["timestamp"], errors="coerce", utc=True)
        tol = pd.Timedelta(minutes=max(1, int(tol_minutes)))
        for a in suspects:
            for b in suspects:
                if a == b:
                    continue
                b_events = bank[(bank["entity_id"] == a) & (bank["counterparty_id"] == b)]
                c_events = comm[(comm["entity_id"] == a) & (comm["counterparty_id"] == b)]
                b_events = b_events.dropna(subset=["ts"]).sort_values("ts")
                c_events = c_events.dropna(subset=["ts"]).sort_values("ts")
                if b_events.empty or c_events.empty:
                    continue
                matches = 0
                for _, brow in b_events.iterrows():
                    near = c_events[(c_events["ts"] <= brow["ts"])
                                    & (c_events["ts"] >= brow["ts"] - tol)]
                    if not near.empty:
                        matches += 1
                if matches:
                    add(a, b, "colocation", matches, "call/post before transfer")

    # shared locations
    loc = edges[edges["location"].astype(str).str.len() > 0]
    loc_candidates = []
    for loc_name, grp in loc.groupby("location"):
        ids = sorted(set(grp["entity_id"]) & set(suspects))
        # Keep the full shared-location connection set so the suspect graph
        # stays connected; the drawing layer limits which edges get labels.
        if len(ids) >= 2:
            loc_candidates.append((str(loc_name), ids, len(grp)))
    loc_candidates.sort(key=lambda x: (-len(x[1]), -x[2]))
    for loc_name, ids, _ in loc_candidates[:8]:
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                add(ids[i], ids[j], "shared_location", 1, loc_name)

    # shared counterparties (connector accounts / phones)
    cp = edges[(~edges["counterparty_id"].isin(suspects))
               & (edges["counterparty_id"].astype(str).str.len() > 0)]
    cp_candidates = []
    for cp_id, grp in cp.groupby("counterparty_id"):
        ids = sorted(set(grp["entity_id"]) & set(suspects))
        # Keep all shared-counterparty connections so the graph does not fall
        # apart into isolated suspects; labels are limited in the drawing layer.
        if len(ids) >= 2:
            cp_candidates.append((str(cp_id), ids, len(grp)))
    cp_candidates.sort(key=lambda x: (-len(x[1]), -x[2]))
    for cp_id, ids, _ in cp_candidates[:8]:
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                add(ids[i], ids[j], "shared_counterparty", 1, cp_id)

    return G


def _edge_style(edge: dict, t: Dict[str, str]) -> Tuple[str, str]:
    types = edge.get("types", set())
    if "colocation" in types:
        return t["red"], "colocation"
    if "direct" in types:
        srcs = edge.get("sources") or set()
        if "bank" in srcs:
            return t["amber"], "transfer"
        if "cdr" in srcs:
            return t["sky"], "call"
        if "social" in srcs:
            return t["yellow"], "social"
        return t["edge"], "direct"
    if "shared_location" in types:
        return t["green"], "shared location"
    if "shared_counterparty" in types:
        return t["violet"], "shared counterparty"
    return t["edge"], "connection"


def _edge_label(edge: dict) -> str:
    parts = []
    types = edge.get("types", set())
    counts = edge.get("counts", {})
    has_direct = "direct" in types
    if has_direct:
        srcs = "/".join(sorted(edge.get("sources") or [])) or "event"
        parts.append(f"{counts.get('direct', 1)} {srcs}")
    if "colocation" in types:
        parts.append(f"colocation x{counts.get('colocation', 1)}")
    # When an edge already has a direct event count, the shared-location /
    # shared-counterparty detail is secondary; dropping it keeps the label
    # readable.  Shared-only edges still say what the connection is.
    if not has_direct and "shared_location" in types:
        locs = sorted(x for x in edge.get("details", set()) if x)
        parts.append(f"shared {locs[0]}" if locs else "shared location")
    if not has_direct and "shared_counterparty" in types:
        cps = sorted(x for x in edge.get("details", set()) if x)
        parts.append(f"shared {cps[0]}" if cps else "shared counterparty")
    return " | ".join(parts[:2])


def render_correlation_graph(result, out_dir: str | Path, top_k: int = 12,
                             name: str = "network.png", theme: str = "dark",
                             tol_minutes: int = 25) -> Path:
    """Single connected suspect-correlation graph with labelled edges.

    This replaces the old per-suspect ego panels: all flagged people are on one
    canvas, and every edge is a connection the ensemble actually found.
    """
    import networkx as nx

    t = _t(theme)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig, ax = plt.subplots(figsize=(13.0, 8.6), dpi=170)
    _apply(fig, ax, t)
    G = _build_correlation_graph(result, top_k=top_k, tol_minutes=tol_minutes)
    if G is None or G.number_of_nodes() == 0:
        _empty(ax, t, "no flagged people to correlate")
        return _save(fig, path)
    if G.number_of_edges() == 0:
        _empty(ax, t, "flagged people exist but no connection was found between them")
        ax.set_title("SUSPECT CORRELATION GRAPH", fontsize=14, fontweight="bold",
                     color=t["title"], pad=12)
        return _save(fig, path)

    n = G.number_of_nodes()
    # A spring layout with generous spacing keeps labels apart better than
    # Kamada-Kawai on dense suspect graphs.  The axes are fixed to [-1.35, 1.35]
    # so labels always have room outside the node cluster.
    pos = nx.spring_layout(G, seed=42, k=6.5 / math.sqrt(max(n, 1)),
                           iterations=1400)
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    pos = {k: ((v[0] - xmin) / span * 2 - 1, (v[1] - ymin) / span * 2 - 1)
           for k, v in pos.items()}
    ax.set_xlim(-1.36, 1.36)
    ax.set_ylim(-1.36, 1.36)
    ax.set_aspect("equal")

    node_sizes: Dict[str, float] = {}
    node_radii: Dict[str, float] = {}
    for node in G.nodes():
        info = G.nodes[node]
        size = min(1700.0, 520.0 + 26.0 * math.sqrt(max(1, info.get("n_events", 1))))
        node_sizes[node] = size
        node_radii[node] = 0.105 if size >= 1000 else 0.085

    edge_items = sorted(G.edges(data=True), key=lambda e: -e[2].get("weight", 0))
    # Dense suspect graphs have more meaningful edges than label-sized gaps.
    # Label only the strongest links; the legend still explains every colour.
    label_budget = 7 if n > 10 else 12
    for idx, (u, v, edge) in enumerate(edge_items):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        color, _ = _edge_style(edge, t)
        width = 1.15 + 0.75 * math.log1p(max(1, edge.get("weight", 1)))
        types = edge.get("types", set())
        alpha = 0.42 if "shared_counterparty" in types else 0.70
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width,
                alpha=alpha, zorder=1, solid_capstyle="round")

    # Numbered nodes plus a side register: a dense 12-suspect graph has no
    # room for collision-free name boxes on the canvas.
    node_order = sorted(
        G.nodes(),
        key=lambda nd: ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(
            G.nodes[nd].get("risk", "LOW"), 3), -G.nodes[nd].get("n_events", 0)),
    )
    rank_of = {node: i + 1 for i, node in enumerate(node_order)}
    for node in G.nodes():
        x, y = pos[node]
        info = G.nodes[node]
        risk = info.get("risk", "LOW")
        color = t["red"] if risk == "HIGH" else (t["amber"] if risk == "MEDIUM" else t["sky"])
        size = node_sizes[node]
        ax.scatter([x], [y], s=size, color=color, alpha=0.96,
                   edgecolors=t["text"], linewidths=1.1, zorder=5)
        ax.scatter([x], [y], s=size * 1.32, facecolors="none",
                   edgecolors=color, linewidths=1.2, alpha=0.45, zorder=4)
        rank_color = "#0B1220" if (theme == "dark" and risk == "LOW") else "#FFFFFF"
        ax.text(x, y, str(rank_of[node]), ha="center", va="center",
                fontsize=8.8, fontweight="bold", color=rank_color, zorder=6)

    # Edge labels are secondary: place only the strongest links, on either
    # side of the edge, with generous collision padding.  It is better to omit
    # a secondary label than to stack unreadable boxes in the graph centre.
    placed_edge_labels = []
    labelled_edges = 0
    for idx, (u, v, edge) in enumerate(edge_items):
        if labelled_edges >= label_budget:
            break
        types = edge.get("types", set())
        weight = edge.get("weight", 1)
        label_worthy = ("colocation" in types or "shared_location" in types
                        or ("direct" in types and weight >= 2))
        if not label_worthy:
            continue
        label = _edge_label(edge)
        if not label:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        color, _ = _edge_style(edge, t)
        dx, dy = x2 - x1, y2 - y1
        norm = math.hypot(dx, dy) or 1.0
        nxo, nyo = -dy / norm, dx / norm
        w = 0.060 + 0.013 * len(label)
        h = 0.100
        placed = False
        for side in (1, -1):
            if placed:
                break
            ox, oy = nxo * 0.065 * side, nyo * 0.065 * side
            for frac in (0.50, 0.38, 0.62, 0.28, 0.72):
                mx = x1 + dx * frac + ox
                my = y1 + dy * frac + oy
                box = (mx - w / 2, mx + w / 2, my - h / 2, my + h / 2)
                if any(_box_hits_circle(box, pos[nd][0], pos[nd][1],
                                        node_radii.get(nd, 0.09) + 0.030)
                       for nd in G.nodes()):
                    continue
                if any(_boxes_overlap(box, b) for b in placed_edge_labels):
                    continue
                ax.text(
                    mx, my, label, fontsize=7.4, family="monospace",
                    color=color, ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.20", facecolor=t["bg_card"],
                              edgecolor=color, alpha=0.96, linewidth=0.6),
                )
                placed_edge_labels.append(box)
                labelled_edges += 1
                placed = True
                break

    # ---- side register ---------------------------------------------------
    panel = fig.add_axes([0.735, 0.12, 0.25, 0.74])
    panel.set_facecolor(t["bg_fig"])
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    panel.set_xticks([])
    panel.set_yticks([])
    for spine in panel.spines.values():
        spine.set_visible(False)
    panel.text(0.0, 1.0, "PEOPLE IN THIS GRAPH", ha="left", va="top",
               fontsize=9.0, fontweight="bold", color=t["title"],
               family="monospace")
    panel.text(0.0, 0.955, "numbered nodes · risk · events", ha="left", va="top",
               fontsize=7.2, color=t["muted"], family="monospace")
    row_step = 0.078
    for i, node in enumerate(node_order):
        info = G.nodes[node]
        risk = info.get("risk", "LOW")
        color = t["red"] if risk == "HIGH" else (t["amber"] if risk == "MEDIUM" else t["sky"])
        y = 0.90 - i * row_step
        panel.scatter([0.018], [y - 0.006], s=42, color=color,
                      edgecolors=t["text"], linewidths=0.6, zorder=3)
        panel.text(0.065, y, f"{rank_of[node]:02d}  {info.get('name', node)}",
                   ha="left", va="center", fontsize=8.0, fontweight="bold",
                   color=t["text"], family="sans-serif")
        panel.text(0.065, y - 0.030,
                   f"{node} · {risk} · {int(info.get('n_events', 0)):,} events",
                   ha="left", va="center", fontsize=6.9, color=t["sub"],
                   family="monospace")

    edge_handles = [
        Line2D([0], [0], color=t["amber"], lw=2.6, label="transfer"),
        Line2D([0], [0], color=t["sky"], lw=2.6, label="call"),
        Line2D([0], [0], color=t["yellow"], lw=2.6, label="social"),
        Line2D([0], [0], color=t["red"], lw=2.6, label="colocation"),
        Line2D([0], [0], color=t["green"], lw=2.6, label="shared location"),
        Line2D([0], [0], color=t["violet"], lw=2.6, label="shared counterparty"),
    ]
    node_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=t["red"],
               markersize=10, label="HIGH risk"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=t["amber"],
               markersize=10, label="MEDIUM risk"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=t["sky"],
               markersize=10, label="LOW risk"),
    ]
    legend = fig.legend(handles=edge_handles + node_handles, loc="lower center",
                        ncol=3, frameon=True, facecolor=t["bg_card"],
                        edgecolor=t["border"], fontsize=9.0,
                        bbox_to_anchor=(0.36, 0.005))
    for txt in legend.get_texts():
        txt.set_color(t["text"])
        txt.set_family("monospace")

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(t["border"])
        s.set_linewidth(0.8)
    fig.suptitle("SUSPECT CORRELATION GRAPH", fontsize=15.5, fontweight="bold",
                 color=t["title"], y=0.985, family="sans-serif")
    fig.text(0.5, 0.945,
             "Connections found by the ensemble: direct events, call/post -> transfer "
             "colocation, shared locations and shared counterparties · numbered nodes, "
             "register at right · edge labels show the strongest links",
             ha="center", va="top", fontsize=9.5, color=t["sub"],
             family="sans-serif")
    components = list(nx.connected_components(G))
    if len(components) > 1:
        names = [" / ".join(G.nodes[n].get("name", n) for n in comp)
                 for comp in components[:3]]
        ax.text(0.5, -0.055, "Disconnected groups (no connection found): " + " | ".join(names),
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
                color=t["muted"], family="monospace")
    fig.subplots_adjust(left=0.03, right=0.70, bottom=0.10, top=0.90)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return path


def render_ego_graph(result, out_dir: str | Path, top_k: int = 3,
                     name: str = "network.png", theme: str = "dark") -> Path:
    """Backward-compatible alias for the single correlation graph."""
    return render_correlation_graph(result, out_dir, top_k=max(8, top_k),
                                    name=name, theme=theme)


def render_entity_timeline(result, out_dir: str | Path, entity_id: str,
                           name: str | None = None, theme: str = "dark") -> Path:
    """Per-person daily activity as a line graph (traffic pattern)."""
    t = _t(theme)
    src = _src_colors(t)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or f"entity_{entity_id}.png"
    path = out_dir / name
    eid = str(entity_id)
    df = result.unified[result.unified["entity_id"].astype(str) == eid].copy()
    fig, ax = plt.subplots(figsize=(10.2, 3.5), dpi=170)
    _apply(fig, ax, t)
    if df.empty or "timestamp" not in df.columns or df["timestamp"].dropna().empty:
        _empty(ax, t, "no events recorded for this person")
        return _save(fig, path)
    sub = df.dropna(subset=["timestamp"]).copy()
    sub["t"] = pd.to_datetime(sub["timestamp"], errors="coerce")
    sub = sub.dropna(subset=["t"]).sort_values("t")
    if sub.empty:
        _empty(ax, t, "no dated events recorded for this person")
        return _save(fig, path)
    sub["date"] = sub["t"].dt.floor("D")
    pivot = sub.pivot_table(index="date", columns="source", values="entity_id",
                            aggfunc="count", fill_value=0)
    pivot = pivot.resample("D").sum().fillna(0)
    pivot["total"] = pivot.sum(axis=1)

    for col in [c for c in pivot.columns if c != "total"]:
        ax.plot(pivot.index, pivot[col], marker="o", markersize=3.6,
                linewidth=1.8, label=str(col).upper(),
                color=src.get(str(col), t["sky"]), alpha=0.95)
    ax.plot(pivot.index, pivot["total"], color=t["muted"], linestyle="--",
            linewidth=1.4, label="TOTAL", alpha=0.9)

    ax.set_ylabel("records / day", fontsize=9.5, family="monospace")
    ax.set_xlabel("date", fontsize=9.5, family="monospace")
    _title(ax, t, f"ACTIVITY TIMELINE — {_entity_display_name(result, eid)} ({eid})")
    _apply(fig, ax, t)
    _style_axes_text(ax, t)
    ax.grid(axis="x", color=t["grid"], linewidth=0.6, alpha=0.45)
    _mono_labels(ax, t)
    _legend_below(ax, t, ncol=4)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _headroom(ax, 1.16)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0)
        lbl.set_fontsize(8.5)

    if not pivot.empty and pivot["total"].max() > 0:
        peak_day = pivot["total"].idxmax()
        peak_val = int(pivot["total"].max())
        ax.annotate(f"peak {peak_val:,}", xy=(peak_day, peak_val),
                    xytext=(8, 10), textcoords="offset points",
                    fontsize=8.5, family="monospace", color=t["text"],
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=t["bg_card"],
                              edgecolor=t["border"], alpha=0.92, linewidth=0.6),
                    arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.7))
    return _save(fig, path)


_GAZETTEER = {
    "mumbai": (19.0760, 72.8777), "bombay": (19.0760, 72.8777),
    "delhi": (28.6139, 77.2090), "new delhi": (28.6139, 77.2090),
    "bengaluru": (12.9716, 77.5946), "bangalore": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867), "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639), "pune": (18.5204, 73.8567),
    "london": (51.5074, -0.1278), "new york": (40.7128, -74.0060),
    "dubai": (25.2048, 55.2708), "singapore": (1.3521, 103.8198),
    "toronto": (43.6532, -79.3832), "sydney": (-33.8688, 151.2093),
    "paris": (48.8566, 2.3522), "berlin": (52.5200, 13.4050),
}

_TILE_CACHE: Dict[Tuple[int, int, int], bytes] = {}
_COORD_RE = __import__("re").compile(
    r"(-?\d{1,3}(?:\.\d+)?)\s*[,;/ ]\s*(-?\d{1,3}(?:\.\d+)?)"
)


def _geocode(location: str) -> Tuple[float, float, bool]:
    """Resolve a location label to (lat, lon, synthetic).

    Real coordinates are used when the label contains them or matches a known
    city.  Synthetic tower/branch codes are projected deterministically into a
    city-scale bounding box so the report still has a usable map; the chart
    labels this as a synthetic projection.
    """
    raw = str(location or "").strip()
    low = raw.lower()
    match = _COORD_RE.search(raw)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon, False
    for key, (lat, lon) in _GAZETTEER.items():
        if key in low:
            return lat, lon, False
    digest = hashlib.sha256(low.encode("utf-8")).digest()
    lat = 18.90 + int.from_bytes(digest[:4], "big") / 2**32 * 0.34
    lon = 72.78 + int.from_bytes(digest[4:8], "big") / 2**32 * 0.28
    return lat, lon, True


def location_points(result, top_n: int = 200) -> list:
    """Return map-ready points for the static chart and the HTML map."""
    from .insights import build_insights

    df = result.unified
    if df is None or df.empty or "location" not in df.columns:
        return []
    df = df.dropna(subset=["location"]).copy()
    if df.empty:
        return []
    flagged = {str(it["entity_id"]) for it in build_insights(result, top_k=8)}
    df["loc"] = df["location"].astype(str)
    agg = df.groupby("loc").agg(
        n=("entity_id", "size"),
        flagged=("entity_id", lambda s: int(s.astype(str).isin(flagged).any())),
        sources=("source", lambda s: sorted(set(s.astype(str)))),
    ).reset_index()
    agg = agg.sort_values("n", ascending=False).head(top_n)
    points = []
    for _, row in agg.iterrows():
        lat, lon, synthetic = _geocode(row["loc"])
        points.append({
            "label": str(row["loc"]),
            "lat": float(lat),
            "lon": float(lon),
            "count": int(row["n"]),
            "flagged": bool(row["flagged"]),
            "synthetic": bool(synthetic),
            "sources": list(row["sources"]),
        })
    return points


def _tile_xy(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def _tile_lat(y: int, zoom: int) -> float:
    n = 2 ** zoom
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))


def _fetch_osm_mosaic(lat_min: float, lon_min: float, lat_max: float,
                      lon_max: float, zoom: int = 12):
    """Fetch a small OSM tile mosaic; returns (image, extent) or raises."""
    x0, y0 = _tile_xy(lat_max, lon_min, zoom)
    x1, y1 = _tile_xy(lat_min, lon_max, zoom)
    while (x1 - x0 > 4 or y1 - y0 > 4) and zoom > 8:
        zoom -= 1
        x0, y0 = _tile_xy(lat_max, lon_min, zoom)
        x1, y1 = _tile_xy(lat_min, lon_max, zoom)
    rows = []
    for y in range(y0, y1 + 1):
        row = []
        for x in range(x0, x1 + 1):
            key = (zoom, x, y)
            data = _TILE_CACHE.get(key)
            if data is None:
                url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
                req = urllib.request.Request(
                    url, headers={"User-Agent": "SutramInvestigator/0.1 (local demo)"}
                )
                with urllib.request.urlopen(req, timeout=3) as resp:
                    data = resp.read()
                _TILE_CACHE[key] = data
            img = plt.imread(io.BytesIO(data), format="png")
            if img.ndim == 2:
                img = np.dstack([img] * 3)
            if img.shape[2] == 4:
                img = img[:, :, :3]
            row.append(img)
        rows.append(np.hstack(row))
    mosaic = np.vstack(rows)
    lon_left = x0 / 2 ** zoom * 360.0 - 180.0
    lon_right = (x1 + 1) / 2 ** zoom * 360.0 - 180.0
    lat_top = _tile_lat(y0, zoom)
    lat_bottom = _tile_lat(y1 + 1, zoom)
    return mosaic, [lon_left, lon_right, lat_bottom, lat_top]


def _stylized_map(ax, bbox, t: Dict[str, str]) -> None:
    lon_min, lon_max, lat_min, lat_max = bbox
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.grid(color=t["grid"], linewidth=0.6, alpha=0.5, zorder=0)
    xs = np.linspace(lon_min, lon_max, 120)
    ys = lat_min + (lat_max - lat_min) * (
        0.48 + 0.10 * np.sin((xs - lon_min) / max(1e-9, lon_max - lon_min) * 2 * np.pi)
    )
    ax.plot(xs, ys, color=t["sky"], linewidth=9, alpha=0.16, zorder=0)
    districts = [
        (0.04, 0.06, 0.28, 0.26, "NORTH"),
        (0.38, 0.58, 0.26, 0.30, "CENTRAL"),
        (0.68, 0.18, 0.27, 0.28, "EAST"),
        (0.18, 0.62, 0.20, 0.22, "WEST"),
    ]
    for fx, fy, fw, fh, label in districts:
        x = lon_min + fx * (lon_max - lon_min)
        y = lat_min + fy * (lat_max - lat_min)
        w = fw * (lon_max - lon_min)
        h = fh * (lat_max - lat_min)
        ax.add_patch(mpatches.Rectangle(
            (x, y), w, h, facecolor=t["bg_card"], edgecolor=t["border"],
            alpha=0.32, linewidth=0.6, zorder=0,
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=8, color=t["muted"], family="monospace", zorder=1)
    ax.text(0.99, 0.02, "synthetic geo-projection (source has location codes, not coordinates)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.8,
            color=t["muted"], family="monospace")


def _place_map_labels(ax, fig, points, flagged, t: Dict[str, str],
                      max_count: float, obstacles=None) -> int:
    """Place map labels without clipping or overlapping other labels/markers.

    Matplotlib has no built-in label de-confliction.  Candidate positions sit on
    rings around each marker and are tried nearest-first, so labels use the
    closest clear space and a leader line keeps them attached to their point.
    Every candidate is measured with the renderer and kept only when it is
    fully inside the axes and clear of the legend, the attribution, previously
    placed labels and marker circles.  A first strict pass avoids every marker;
    a relaxed pass may cover small unflagged markers but still never covers a
    flagged marker or another label.  A missing label is better than an
    unreadable pile of callouts.
    """
    if not flagged:
        return 0
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_box = ax.bbox
    px_per_pt = fig.dpi / 72.0
    fontsize = 7.9

    def marker_circle(point):
        size = 55 + 260 * (point["count"] / max_count)
        px, py = ax.transData.transform((point["lon"], point["lat"]))
        return px, py, math.sqrt(size) / 2.0 * px_per_pt + 1.0

    def hits_circle(box, circle) -> bool:
        cx, cy, radius = circle
        nx = min(max(cx, box.x0), box.x1)
        ny = min(max(cy, box.y0), box.y1)
        return (nx - cx) ** 2 + (ny - cy) ** 2 < radius * radius

    hard = list(obstacles or [])
    hard_circles = [marker_circle(p) for p in points if p["flagged"]]
    soft_circles = [marker_circle(p) for p in points if not p["flagged"]]

    # Offsets are in points (1 pt ~ 2.36 px at 170 dpi).  Rings clear the
    # largest marker bubble, then step outwards; cardinals keep text level.
    candidates = []
    for radius in (13, 22, 34, 48, 64):
        diag = radius * 0.72
        for dx, dy, ha, va in (
            (radius, 0, "left", "center"), (-radius, 0, "right", "center"),
            (0, radius, "center", "bottom"), (0, -radius, "center", "top"),
            (diag, diag, "left", "bottom"), (-diag, diag, "right", "bottom"),
            (diag, -diag, "left", "top"), (-diag, -diag, "right", "top"),
        ):
            candidates.append((dx, dy, ha, va))

    placed = 0
    for point in flagged:
        label = f"{point['label']} ({point['count']:,})"
        for relaxed in (False, True):
            chosen = None
            for dx, dy, ha, va in candidates:
                text = ax.annotate(
                    label, xy=(point["lon"], point["lat"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=fontsize, family="monospace", fontweight="bold",
                    color=t["red"], ha=ha, va=va, zorder=7,
                    annotation_clip=False,
                    bbox=dict(boxstyle="round,pad=0.22",
                              facecolor=t["bg_card"], edgecolor=t["red"],
                              alpha=0.96, linewidth=0.7),
                    arrowprops=dict(arrowstyle="-", color=t["red"], lw=0.7,
                                    alpha=0.7, shrinkA=1, shrinkB=2),
                )
                text.update_positions(renderer)
                raw = Text.get_window_extent(text, renderer=renderer)
                pad = 0.30 * fontsize * px_per_pt + 1.0
                box = Bbox.from_extents(raw.x0 - pad, raw.y0 - pad,
                                        raw.x1 + pad, raw.y1 + pad)
                inside = (box.x0 >= ax_box.x0 + 8
                          and box.x1 <= ax_box.x1 - 8
                          and box.y0 >= ax_box.y0 + 8
                          and box.y1 <= ax_box.y1 - 8)
                blocked = (not inside
                           or any(box.overlaps(other) for other in hard)
                           or any(hits_circle(box, c) for c in hard_circles))
                if not blocked and not relaxed:
                    blocked = any(hits_circle(box, c) for c in soft_circles)
                if not blocked:
                    chosen = box
                    break
                text.remove()
            if chosen is not None:
                hard.append(chosen)
                placed += 1
                break
    return placed


def render_location_map(result, out_dir: str | Path, name: str = "map.png",
                        theme: str = "dark") -> Path:
    """Activity map with a real map background when online and a zoom inset."""
    t = _t(theme)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig, ax = plt.subplots(figsize=(12.6, 7.6), dpi=170)
    _apply(fig, ax, t)
    points = location_points(result)
    if not points:
        _empty(ax, t, "no location data in these files")
        return _save(fig, path)

    lats = np.array([p["lat"] for p in points])
    lons = np.array([p["lon"] for p in points])
    lat_pad = max(0.01, (lats.max() - lats.min()) * 0.14)
    lon_pad = max(0.01, (lons.max() - lons.min()) * 0.14)
    lat_min, lat_max = float(lats.min() - lat_pad), float(lats.max() + lat_pad)
    lon_min, lon_max = float(lons.min() - lon_pad), float(lons.max() + lon_pad)
    bbox = [lon_min, lon_max, lat_min, lat_max]

    mosaic = extent = None
    attribution = "synthetic geo-projection"
    try:
        mosaic, extent = _fetch_osm_mosaic(lat_min, lon_min, lat_max, lon_max)
        ax.imshow(mosaic, extent=extent, aspect="auto", alpha=0.92, zorder=0)
        attribution = "(c) OpenStreetMap contributors"
    except Exception:
        _stylized_map(ax, bbox, t)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    mean_lat = float(np.mean(lats))
    ax.set_aspect(1.0 / max(0.25, math.cos(math.radians(mean_lat))))

    counts = np.array([p["count"] for p in points], dtype=float)
    max_count = max(1.0, counts.max())
    for point in points:
        size = 55 + 260 * (point["count"] / max_count)
        color = t["red"] if point["flagged"] else t["amber"]
        ax.scatter(point["lon"], point["lat"], s=size, color=color,
                   alpha=0.82, edgecolors=t["node_edge"], linewidths=0.7, zorder=3)

    flagged = sorted([p for p in points if p["flagged"]],
                     key=lambda p: -p["count"])[:10]

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=t["amber"],
               markersize=9, label="active locations"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=t["red"],
               markersize=9, label="flagged activity"),
    ]
    leg = ax.legend(handles=handles, frameon=True, facecolor=t["bg_card"],
                    edgecolor=t["border"], fontsize=9, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(t["text"])
        txt.set_family("monospace")
    leg.get_frame().set_linewidth(0.8)
    attribution_text = ax.text(
        0.005, 0.008, attribution, transform=ax.transAxes,
        ha="left", va="bottom", fontsize=7.6, color=t["muted"],
        family="monospace")

    inset_used = False
    if len(points) >= 5:
        anchor = max(points, key=lambda p: p["count"])
        cluster = [
            p for p in points
            if abs(p["lat"] - anchor["lat"]) < (lat_max - lat_min) * 0.32
            and abs(p["lon"] - anchor["lon"]) < (lon_max - lon_min) * 0.32
        ]
        if len(cluster) >= 3:
            cx = [p["lon"] for p in cluster]
            cy = [p["lat"] for p in cluster]
            x0 = min(cx) - (lon_max - lon_min) * 0.06
            x1 = max(cx) + (lon_max - lon_min) * 0.06
            y0 = min(cy) - (lat_max - lat_min) * 0.06
            y1 = max(cy) + (lat_max - lat_min) * 0.06
            # Dedicated panel outside the map: no label/inset collisions.
            fig.subplots_adjust(left=0.06, right=0.60, bottom=0.10, top=0.86)
            axins = fig.add_axes([0.655, 0.30, 0.30, 0.31])
            if mosaic is not None and extent is not None:
                axins.imshow(mosaic, extent=extent, aspect="auto", alpha=0.92)
            else:
                axins.set_facecolor(t["bg_ax"])
                axins.grid(color=t["grid"], linewidth=0.5, alpha=0.5)
            for point in cluster:
                axins.scatter(point["lon"], point["lat"],
                              s=36 + 160 * (point["count"] / max_count),
                              color=t["red"] if point["flagged"] else t["amber"],
                              alpha=0.85, edgecolors=t["node_edge"], linewidths=0.5)
            axins.set_xlim(x0, x1)
            axins.set_ylim(y0, y1)
            axins.set_xticks([])
            axins.set_yticks([])
            axins.set_title("DENSEST CLUSTER — ZOOM", fontsize=8.5,
                            color=t["title"], loc="left", pad=5,
                            family="monospace", fontweight="bold")
            for spine in axins.spines.values():
                spine.set_color(t["amber"])
                spine.set_linewidth(0.9)
            ax.add_patch(mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=t["amber"],
                linewidth=1.1, linestyle="--", alpha=0.9, zorder=9,
            ))
            ax.annotate("zoom", xy=(x0, y1), xytext=(4, 4),
                        textcoords="offset points", fontsize=7.6,
                        family="monospace", color=t["amber"], zorder=10)
            inset_used = True

    if not inset_used:
        fig.tight_layout(rect=(0, 0, 1, 0.90))

    # Place labels only after the axes geometry is final.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # The dashed zoom locator is a guide, not an obstacle: blocking its whole
    # area would suppress every label in the densest part of the map.
    obstacles = []
    for artist in (leg, attribution_text):
        try:
            box = artist.get_window_extent(renderer=renderer)
            obstacles.append(Bbox.from_extents(
                box.x0 - 3, box.y0 - 3, box.x1 + 3, box.y1 + 3))
        except Exception:
            pass
    labelled = _place_map_labels(ax, fig, points, flagged, t, max_count,
                                 obstacles=obstacles)
    if labelled < len(flagged):
        fig.text(0.5, 0.912,
                 f"{len(flagged) - labelled} of {len(flagged)} flagged "
                 "locations unlabelled (too dense to place)",
                 ha="center", va="center", fontsize=7.6, color=t["muted"],
                 family="monospace")

    ax.set_xlabel("longitude", fontsize=9.5, family="monospace")
    ax.set_ylabel("latitude", fontsize=9.5, family="monospace")
    ax.tick_params(labelsize=8.5)
    fig.suptitle("ACTIVITY MAP - locations & tower telemetry", fontsize=15.0,
                 fontweight="bold", color=t["title"], y=0.985,
                 family="sans-serif")
    fig.text(0.5, 0.945,
             "OpenStreetMap background when online | synthetic projection otherwise | "
             "scroll/drag to zoom in report.html",
             ha="center", va="top", fontsize=8.8, color=t["sub"],
             family="sans-serif")

    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(),
                edgecolor="none")
    plt.close(fig)
    return path
