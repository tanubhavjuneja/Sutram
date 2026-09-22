"""Report charts — terminal-style matplotlib renders for the Sutram console.

All functions render PNGs into a report directory and are safe on empty /
tiny inputs. Charts support dark (default) and light themes. Palette is
strictly semantic: amber = primary data, red = risk/alert, green = ok,
sky = info. Text-overlap guards: collision-checked map labels, spaced
ego-graph layouts, callout badges instead of in-circle text.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    """Event volume over time, stacked by source."""
    t = _t(theme)
    src = _src_colors(t)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df = result.unified
    fig, ax = plt.subplots(figsize=(9.6, 3.2), dpi=160)
    _apply(fig, ax, t)
    if df is None or df.empty or "timestamp" not in df.columns or df["timestamp"].dropna().empty:
        _empty(ax, t, "no timestamped events to plot")
        return _save(fig, path)
    ts = df.dropna(subset=["timestamp"]).copy()
    ts["date"] = pd.to_datetime(ts["timestamp"]).dt.floor("D")
    pivot = ts.pivot_table(index="date", columns="source", values="entity_id",
                           aggfunc="count", fill_value=0)
    pivot = pivot.resample("D").sum().fillna(0)
    colors = [src.get(c, t["sky"]) for c in pivot.columns]
    pivot.plot.area(ax=ax, color=colors, alpha=0.85, linewidth=0)
    ax.set_ylabel("records", fontsize=8.5, family="monospace")
    _title(ax, t, "WHEN IT HAPPENED — event volume by modality")
    _apply(fig, ax, t)
    _style_axes_text(ax, t)
    _mono_labels(ax, t)
    _legend(ax, t)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(25)
        lbl.set_fontsize(7.5)
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
    ax.bar(x - 0.19, peaks, 0.38, label="peak score", color=t["amber"], alpha=0.9)
    ax.bar(x + 0.19, flagged, 0.38, label="flagged >0.6", color=t["red"], alpha=0.9)
    ax.set_xticks(x, [m.replace("_", "\n") for m in models], fontsize=7)
    ax.set_ylim(0, max(1.0, max(peaks + [0.0]) * 1.15))
    _title(ax, t, "DETECTION ENGINES — peak score and flagged count")
    _apply(fig, ax, t)
    _style_axes_text(ax, t)
    _mono_labels(ax, t)
    if x.size:
        _legend(ax, t, loc="upper right")
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
    ax.set_yticks(y, [f"{lab}  ·  {nm}" for lab, nm in zip(labels, names)],
                  fontsize=8)
    ax.set_xlim(0, max(vals) * 1.22)
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
    ax.bar(x - 0.19, rows, 0.38, label="records", color=t["sky"], alpha=0.9)
    ax.bar(x + 0.19, ents, 0.38, label="entities", color=t["green"], alpha=0.9)
    ax.set_xticks(x, names, fontsize=7.5)
    _title(ax, t, "WHERE THE RECORDS CAME FROM — ingestion volumes")
    _apply(fig, ax, t)
    _style_axes_text(ax, t)
    _mono_labels(ax, t)
    _legend(ax, t, loc="upper right")
    return _save(fig, path)


def _entity_display_name(result, entity_id: str) -> str:
    from .insights import display_name
    return display_name(result, entity_id)


def render_ego_graph(result, out_dir: str | Path, top_k: int = 3,
                     name: str = "network.png", theme: str = "dark") -> Path:
    """Ego-graph of the top people. Top-12 connections, spaced layout,
    badge labels outside node circles — no text overlap."""
    import networkx as nx
    from .insights import build_insights

    t = _t(theme)
    src = _src_colors(t)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    top = build_insights(result, top_k=top_k)

    if not top:
        fig, ax = plt.subplots(figsize=(9.6, 3.0), dpi=160)
        _apply(fig, ax, t)
        _empty(ax, t, "no flagged people to map")
        return _save(fig, path)

    df = result.unified.dropna(subset=["entity_id", "counterparty_id"]).copy()
    df["entity_id"] = df["entity_id"].astype(str)
    df["counterparty_id"] = df["counterparty_id"].astype(str)
    wgt = df.groupby(["entity_id", "counterparty_id", "source"]).size()

    n_panels = len(top)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.6, 3.3 * nrows), dpi=160)
    fig.patch.set_facecolor(t["bg_fig"])
    axes = np.atleast_1d(axes).ravel()

    for ax, it in zip(axes, top):
        ax.set_facecolor(t["bg_ax"])
        eid = str(it["entity_id"])
        sub = wgt.loc[eid] if eid in wgt.index.get_level_values(0) else None
        G = nx.Graph()
        G.add_node(eid)
        if sub is not None:
            for (cp, s), w in sub.items():
                if cp == eid or w <= 0:
                    continue
                G.add_edge(eid, cp, weight=float(w), source=str(s))
        if G.number_of_edges() == 0:
            _empty(ax, t, "no contacts recorded")
            ax.set_title(_entity_display_name(result, eid), fontsize=9,
                         family="monospace", color=t["title"], pad=6)
            continue
        if G.number_of_nodes() > 13:
            keep = sorted([n for n in G.nodes() if n != eid],
                          key=lambda n: G[eid][n]["weight"], reverse=True)[:12]
            G = G.subgraph([eid] + keep).copy()
        pos = nx.spring_layout(G, seed=42, k=1.6 / np.sqrt(G.number_of_nodes()))
        ews = [G[u][v]["weight"] for u, v in G.edges()]
        elw = 1.0 + 2.8 * (np.asarray(ews) - min(ews)) / max(1e-9, max(ews) - min(ews))
        nx.draw_networkx_edges(G, pos, ax=ax, width=elw, edge_color=t["edge"], alpha=0.55)
        for node in G.nodes():
            if node == eid:
                continue
            w = G[eid][node]["weight"]
            color = src.get(G[eid][node]["source"], t["sky"])
            ax.scatter([pos[node][0]], [pos[node][1]],
                       s=min(340, 80 + 20 * int(w)), color=color, alpha=0.95,
                       edgecolors=t["node_edge"], linewidths=0.8, zorder=3)
            dy = pos[node][1] - pos[eid][1]
            ax.annotate(str(node), xy=pos[node], xytext=(0, 10 if dy >= 0 else -12),
                        textcoords="offset points", fontsize=6,
                        family="monospace", color=t["text"], ha="center",
                        va="bottom" if dy >= 0 else "top",
                        bbox=dict(boxstyle="square,pad=0.18",
                                  facecolor=t["bg_card"], edgecolor=t["border"],
                                  alpha=0.9, linewidth=0.5), zorder=4)
        ax.scatter([pos[eid][0]], [pos[eid][1]], s=560, color=t["red"],
                   alpha=1.0, edgecolors=t["text"], linewidths=1.2, zorder=5)
        ax.annotate("FLAGGED", xy=pos[eid], fontsize=6.5, fontweight="bold",
                    family="monospace", color=t["bg_fig"] if theme == "dark" else "#FFFFFF",
                    ha="center", va="center", zorder=6)
        ax.set_title(f'{_entity_display_name(result, eid)}  ·  {eid}', fontsize=8.5,
                     family="monospace", color=t["title"], pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(t["border"])
            s.set_linewidth(0.8)
    for ax in axes[len(top):]:
        ax.axis("off")
    fig.suptitle("WHO THE FLAGGED PEOPLE ARE CONNECTED TO", fontsize=10.5,
                 fontweight="bold", color=_t(theme)["title"], y=1.0,
                 family="DejaVu Sans")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(),
                edgecolor="none")
    plt.close(fig)
    return path


def render_entity_timeline(result, out_dir: str | Path, entity_id: str,
                           name: str | None = None, theme: str = "dark") -> Path:
    """Per-person drill-down event strip."""
    t = _t(theme)
    src = _src_colors(t)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or f"entity_{entity_id}.png"
    path = out_dir / name
    eid = str(entity_id)
    df = result.unified[result.unified["entity_id"].astype(str) == eid].copy()
    fig, ax = plt.subplots(figsize=(9.6, 3.2), dpi=160)
    _apply(fig, ax, t)
    if df.empty or "timestamp" not in df.columns or df["timestamp"].dropna().empty:
        _empty(ax, t, "no events recorded for this person")
        return _save(fig, path)
    sub = df.dropna(subset=["timestamp"]).copy()
    sub["t"] = pd.to_datetime(sub["timestamp"])
    sub = sub.sort_values("t")
    amt = pd.to_numeric(sub["amount"], errors="coerce")
    size = 14 + 44 * np.clip(amt.fillna(0).abs() / max(1.0, amt.abs().max()), 0, 1)
    colors = [src.get(s, t["sky"]) for s in sub["source"]]
    ax.scatter(sub["t"], np.zeros(len(sub)) + 0.5, s=size, color=colors,
               alpha=0.9, edgecolors=t["node_edge"], linewidths=0.5, zorder=3)
    daily = sub.groupby(sub["t"].dt.floor("D")).size()
    ax.bar(daily.index, daily.values, width=0.8, color=t["amber"], alpha=0.3, zorder=2)
    ax.set_ylim(0, 1.6)
    ax.set_yticks([])
    ax.set_xlabel("time", fontsize=8.5, family="monospace")
    _title(ax, t, f"ACTIVITY STRIP — {_entity_display_name(result, eid)} ({eid})")
    _apply(fig, ax, t)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                          color=src.get(s, t["sky"]), label=s.upper())
               for s in sorted(set(sub["source"]))]
    ax.legend(handles=handles, frameon=True, facecolor=t["bg_card"],
              edgecolor=t["border"], fontsize=7.5, loc="upper right")
    for txt in ax.get_legend().get_texts():
        txt.set_color(t["text"])
        txt.set_family("monospace")
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(25)
        lbl.set_fontsize(7.5)
        lbl.set_color(t["sub"])
    return _save(fig, path)


def render_location_map(result, out_dir: str | Path, name: str = "map.png",
                        theme: str = "dark") -> Path:
    """Activity map — collision-checked directional callouts with leader lines."""
    t = _t(theme)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df = result.unified.dropna(subset=["location"]).copy()
    fig, ax = plt.subplots(figsize=(9.6, 5.0), dpi=160)
    _apply(fig, ax, t)
    if df.empty:
        _empty(ax, t, "no location data in these files")
        return _save(fig, path)
    from .insights import build_insights
    flagged = {str(it["entity_id"]) for it in build_insights(result, top_k=8)}
    df["loc"] = df["location"].astype(str)
    agg = df.groupby("loc").agg(
        n=("entity_id", "size"),
        flagged=("entity_id", lambda s: int(s.astype(str).isin(flagged).any())),
    ).reset_index()
    agg = agg.sort_values("n", ascending=False).head(200).reset_index(drop=True)
    rng = np.random.default_rng(11)
    xs = rng.uniform(0.04, 0.96, size=len(agg)) * agg["n"].rank(pct=True) * 1.6
    ys = rng.uniform(0.06, 0.94, size=len(agg))
    sizes = 30 + 150 * (agg["n"] / agg["n"].max())
    for _, row in agg.iterrows():
        ax.scatter(xs[row.name], ys[row.name], s=sizes[row.name],
                   color=t["red"] if row["flagged"] else t["amber"],
                   alpha=0.8, edgecolors=t["node_edge"], linewidths=0.6, zorder=3)
    flagged_locs = agg[agg["flagged"] == 1].sort_values("n", ascending=False)
    placed: list = []
    dirs = [(14, 12), (-14, 12), (14, -14), (-14, -14),
            (0, 18), (0, -18), (20, 0), (-20, 0)]
    ha_va = [("left", "bottom"), ("right", "bottom"), ("left", "top"),
             ("right", "top"), ("center", "bottom"), ("center", "top"),
             ("left", "center"), ("right", "center")]
    count = 0
    for _, row in flagged_locs.iterrows():
        if count >= 10:
            break
        px, py = xs[row.name], ys[row.name]
        if any(np.hypot(px - ox, py - oy) < 0.11 for ox, oy in placed) and count >= 4:
            continue
        dx, dy = dirs[count % 8]
        ha, va = ha_va[count % 8]
        label = f"{row['loc']} ({row['n']:,})" if row["n"] > 1 else str(row["loc"])
        ax.annotate(label, xy=(px, py), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.5, family="monospace",
                    fontweight="bold", color=t["red"], ha=ha, va=va,
                    bbox=dict(boxstyle="square,pad=0.25", facecolor=t["bg_card"],
                              edgecolor=t["red"], alpha=0.92, linewidth=0.7),
                    arrowprops=dict(arrowstyle="-", color=t["red"], lw=0.7,
                                    alpha=0.7), zorder=7)
        placed.append((px, py))
        count += 1
    ax.scatter([], [], s=70, color=t["amber"], label="active locations", alpha=0.8)
    ax.scatter([], [], s=70, color=t["red"], label="flagged activity", alpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(t["border"])
        s.set_linewidth(0.8)
    _title(ax, t, "THE ACTIVITY MAP — locations & tower telemetry")
    leg = ax.legend(frameon=True, facecolor=t["bg_card"], edgecolor=t["border"],
                    fontsize=8, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(t["text"])
        txt.set_family("monospace")
    leg.get_frame().set_linewidth(0.8)
    return _save(fig, path)
