"""Pipeline configuration loaded from YAML (no hardcoded demo values in code)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PipelineConfig:
    data_dir: str = "data"
    sources_dir: str = "data/sources"
    report_dir: str = "data/report"

    top_k: int = 10
    min_events_per_entity: int = 30
    tol_minutes: int = 30
    structuring_threshold: float = 10000.0
    with_models: bool = True
    seeds: List[int] = field(default_factory=lambda: [42, 99])

    # Stage-A content-probe thresholds (see aml.schema_detect.ProbeConfig)
    probe_date_parse_ratio: float = 0.9
    probe_phone_ratio: float = 0.7
    probe_numeric_ratio: float = 0.9
    probe_id_ratio: float = 0.7
    probe_amount_median_abs: float = 100.0
    probe_amount_frac_ratio: float = 0.3
    probe_amount_marker_ratio: float = 0.5
    probe_big_code_magnitude: float = 10000.0
    probe_amount_integer_median: float = 100.0

    def __post_init__(self) -> None:
        self._path: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        p = Path(path)
        cfg = cls()
        if not p.exists():
            return cfg
        with open(p, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        for key, val in raw.get("pipeline", {}).items():
            if key in cfg.__dataclass_fields__:
                setattr(cfg, key, val)
        cfg._path = str(p)
        return cfg

    def probe_config(self):
        """Build the Stage-A content-probe thresholds from this config."""
        from .schema_detect import ProbeConfig
        return ProbeConfig(
            date_parse_ratio=self.probe_date_parse_ratio,
            phone_ratio=self.probe_phone_ratio,
            numeric_ratio=self.probe_numeric_ratio,
            id_ratio=self.probe_id_ratio,
            amount_median_abs=self.probe_amount_median_abs,
            amount_frac_ratio=self.probe_amount_frac_ratio,
            amount_marker_ratio=self.probe_amount_marker_ratio,
            big_code_magnitude=self.probe_big_code_magnitude,
            amount_integer_median=self.probe_amount_integer_median,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump({"pipeline": asdict(self)}, sort_keys=False),
            encoding="utf-8",
        )


@dataclass
class InvestigationConfig:
    """Case-workspace / investigation-layer settings.

    Kept in the same config module as :class:`PipelineConfig` so the analytics
    and investigation halves of the product share one configuration surface.
    """
    case_root: str = "data/cases"
    language: str = "en"
    min_events_per_entity: int = 30
    tol_minutes: int = 25
    structuring_threshold: float = 10000.0
    top_k: int = 10
    max_findings: int = 25
    llm_provider: str = "none"
    llm_model: str = ""
    llm_text_model: str = ""
    llm_image_model: str = ""
    llm_base_url: str = ""
    llm_timeout: int = 120

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InvestigationConfig":
        p = Path(path)
        cfg = cls()
        if not p.exists():
            return cfg
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return cfg
        section = raw.get("investigation", raw) or {}
        for key, value in section.items():
            if key in cfg.__dataclass_fields__:
                setattr(cfg, key, value)
        return cfg

    def llm_config(self):
        """Resolve the configured provider, letting ``.env`` enable Cloudflare.

        The dashboard-style ``.env`` keys (``Account ID`` / ``API KEY``) are
        normalized by :func:`aml.llm._load_dotenv`; a config value of ``none``
        still yields the env-detected provider so dropping the key into
        ``.env`` is enough to connect the assistant.
        """
        import os
        from .llm import LLMConfig

        env_cfg = LLMConfig.from_env()
        configured = (self.llm_provider or "").lower()
        env_provider = (env_cfg.provider or "").lower()
        if configured in ("", "none") and env_provider not in ("", "none"):
            provider = env_provider
        else:
            provider = configured or env_provider or "none"

        if provider in ("cloudflare", "cf"):
            account_id = (env_cfg.account_id
                          or os.getenv("CLOUDFLARE_ACCOUNT_ID")
                          or os.getenv("ACCOUNT_ID")
                          or os.getenv("CF_ACCOUNT_ID")
                          or "")
            api_key = (env_cfg.api_key
                       or os.getenv("CLOUDFLARE_API_TOKEN")
                       or os.getenv("CLOUDFLARE_API_KEY")
                       or os.getenv("API_KEY")
                       or os.getenv("CF_API_TOKEN")
                       or "")
            base_url = self.llm_base_url or env_cfg.base_url
            if not base_url and account_id:
                base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"
            return LLMConfig(
                provider="cloudflare",
                model=self.llm_model or env_cfg.model
                or "@cf/meta/llama-3.2-11b-vision-instruct",
                text_model=self.llm_text_model or env_cfg.text_model
                or "@cf/meta/llama-3.1-8b-instruct",
                image_model=self.llm_image_model or env_cfg.image_model
                or "@cf/black-forest-labs/flux-1-schnell",
                base_url=base_url,
                api_key=api_key,
                account_id=account_id,
                timeout=int(self.llm_timeout or env_cfg.timeout or 120),
                enabled=bool(account_id and api_key),
            )
        if provider == "none":
            return env_cfg
        return LLMConfig(
            provider=provider,
            model=self.llm_model or env_cfg.model,
            text_model=self.llm_text_model or env_cfg.text_model,
            image_model=self.llm_image_model or env_cfg.image_model,
            base_url=self.llm_base_url or env_cfg.base_url,
            api_key=env_cfg.api_key or os.getenv("INVESTIGATION_LLM_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", ""),
            timeout=int(self.llm_timeout or env_cfg.timeout or 30),
            enabled=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_investigation_config(path: Optional[str] = None,
                              case_root: Optional[str] = None,
                              lang: Optional[str] = None) -> InvestigationConfig:
    """Load ``config/investigation.yaml`` and apply CLI overrides."""
    cfg = InvestigationConfig()
    if path and Path(path).exists():
        cfg = InvestigationConfig.from_yaml(path)
    if case_root:
        cfg.case_root = case_root
    if lang:
        cfg.language = lang
    return cfg


# Backward-compatible alias for the old investigation CLI/web imports.
load_config = load_investigation_config