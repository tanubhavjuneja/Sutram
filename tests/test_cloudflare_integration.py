"""Live Cloudflare Workers AI integration tests.

Skipped by default so normal CI runs stay offline and deterministic.  Run with:

    $env:RUN_CLOUDFLARE_TESTS="1"   # PowerShell
    python -m pytest tests/test_cloudflare_integration.py -q

Requires the ``.env`` credentials (``Account ID`` / ``API KEY``).
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from aml.config import InvestigationConfig
from aml.llm import LLMClient, LLMConfig
from aml.service import InvestigationService

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_CLOUDFLARE_TESTS") != "1",
    reason="set RUN_CLOUDFLARE_TESTS=1 to call the live Cloudflare API",
)


def _make_chart(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    ax.axis("off")
    ax.text(0.05, 0.7, "SUSPECT: ALICE", fontsize=16, color="black")
    ax.text(0.05, 0.35, "AMOUNT: 9,500", fontsize=16, color="black")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


@pytest.fixture(scope="module")
def client() -> LLMClient:
    cfg = LLMConfig.from_env()
    if not cfg.enabled or cfg.provider != "cloudflare":
        pytest.skip("Cloudflare credentials are not configured in .env")
    client = LLMClient(cfg)
    if not client.available:
        pytest.skip("Cloudflare Workers AI client is not available")
    return client


def test_cloudflare_text_model(client):
    out = client.complete("Reply with exactly: OK", max_tokens=20)
    assert out
    assert "OK" in out.upper()


def test_cloudflare_vision_model(client, tmp_path):
    chart = _make_chart(tmp_path / "chart.png")
    out = client.describe_image(chart.read_bytes(),
                                "What text is visible? Answer briefly.", lang="en")
    assert out
    upper = out.upper()
    assert "ALICE" in upper or "SUSPECT" in upper or "9,500" in out or "9500" in out


def test_cloudflare_image_generation(client):
    image = client.generate_image(
        "a simple blue circle on a white background, minimal illustration", steps=4
    )
    assert image
    assert len(image) > 1000
    assert image[:2] == b"\xff\xd8" or image[:4] == b"\x89PNG"


def test_service_vision_and_generation_use_cloudflare(client, tmp_path):
    cfg = InvestigationConfig(case_root=str(tmp_path / "cases"), min_events_per_entity=2)
    svc = InvestigationService(case_root=cfg.case_root, config=cfg)
    case = svc.create_case("Cloudflare Integration", language="en")
    chart = _make_chart(tmp_path / "chart.png")
    data_url = "data:image/png;base64," + base64.b64encode(chart.read_bytes()).decode("ascii")

    vision = svc.ask(case.case_id, "What is in this image? Answer briefly.",
                     images=[data_url])
    assert vision.intent == "vision"
    assert vision.images and Path(vision.images[0]["path"]).exists()
    assert "Image analysis" in vision.answer

    generated = svc.ask(
        case.case_id,
        "Generate an image of a clean forensic case board with connected suspect cards.",
    )
    assert generated.intent == "generate_image"
    assert generated.images and Path(generated.images[0]["path"]).exists()
