"""LLM adapters — Cloudflare Workers AI, OpenAI-compatible and Ollama.

The assistant works fully offline with deterministic templates.  When an LLM is
configured it can:

* polish a grounded draft answer (text model),
* read an uploaded image and answer questions about it (vision model),
* generate a synthetic illustration from a text prompt (text-to-image model).

Grounding rules for the polish path are unchanged: the model must preserve a
valid citation label and may not invent evidence IDs.  Vision answers are
explicitly labelled as model observations, not case evidence.

Cloudflare Workers AI is configured through ``.env``::

    Account ID=...
    API KEY=...

or the conventional ``CLOUDFLARE_ACCOUNT_ID`` / ``CLOUDFLARE_API_TOKEN`` names.
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .case import Citation

_CF_VISION_DEFAULT = "@cf/meta/llama-3.2-11b-vision-instruct"
_CF_TEXT_DEFAULT = "@cf/meta/llama-3.1-8b-instruct"
_CF_IMAGE_DEFAULT = "@cf/black-forest-labs/flux-1-schnell"
_CF_API = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"


def _load_dotenv() -> None:
    """Load project-root ``.env`` without adding a python-dotenv dependency.

    Keys are normalized so dashboard-style names such as ``Account ID`` and
    ``API KEY`` become ``ACCOUNT_ID`` and ``API_KEY``.
    """
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    seen = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = re.sub(r"[^A-Za-z0-9]+", "_", key.strip()).strip("_").upper()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip('"').strip("'")
    return default


@dataclass
class LLMConfig:
    provider: str = "none"           # none | cloudflare | openai | ollama
    model: str = ""                  # primary/vision model
    text_model: str = ""
    image_model: str = ""
    base_url: str = ""
    api_key: str = ""
    account_id: str = ""
    timeout: int = 30
    temperature: float = 0.1
    max_tokens: int = 700
    enabled: bool = False

    @classmethod
    def from_env(cls) -> "LLMConfig":
        _load_dotenv()
        provider = (_first_env("INVESTIGATION_LLM_PROVIDER") or "none").lower()
        account_id = _first_env(
            "CLOUDFLARE_ACCOUNT_ID", "ACCOUNT_ID", "CF_ACCOUNT_ID"
        )
        api_key = _first_env(
            "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY", "API_KEY", "CF_API_TOKEN"
        )
        if provider in ("cloudflare", "cf") or (
            provider in ("", "none") and account_id and api_key
        ):
            return cls(
                provider="cloudflare",
                model=_first_env("INVESTIGATION_LLM_MODEL", "CLOUDFLARE_VISION_MODEL",
                                 default=_CF_VISION_DEFAULT),
                text_model=_first_env("INVESTIGATION_LLM_TEXT_MODEL",
                                      "CLOUDFLARE_TEXT_MODEL",
                                      default=_CF_TEXT_DEFAULT),
                image_model=_first_env("INVESTIGATION_LLM_IMAGE_MODEL",
                                       "CLOUDFLARE_IMAGE_MODEL",
                                       default=_CF_IMAGE_DEFAULT),
                base_url=_CF_API.format(account_id=account_id) if account_id else "",
                api_key=api_key,
                account_id=account_id,
                timeout=int(_first_env("INVESTIGATION_LLM_TIMEOUT", default="120") or 120),
                enabled=True,
            )
        if provider == "openai":
            model = _first_env("INVESTIGATION_LLM_MODEL", "OPENAI_MODEL",
                               default="gpt-4o-mini")
            return cls(
                provider="openai",
                model=model,
                text_model=model,
                base_url=_first_env("INVESTIGATION_LLM_BASE_URL", "OPENAI_BASE_URL",
                                    default="https://api.openai.com/v1"),
                api_key=_first_env("INVESTIGATION_LLM_API_KEY", "OPENAI_API_KEY"),
                enabled=True,
            )
        if provider == "ollama":
            model = _first_env("INVESTIGATION_LLM_MODEL", default="llama3.1")
            return cls(
                provider="ollama",
                model=model,
                text_model=model,
                base_url=_first_env("INVESTIGATION_LLM_BASE_URL", "OLLAMA_HOST",
                                    default="http://localhost:11434"),
                enabled=True,
            )
        return cls()

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "text_model": self.text_model,
            "image_model": self.image_model,
            "base_url": self.base_url,
            "enabled": self.enabled,
            "timeout": self.timeout,
            "available": bool(self.enabled and self.provider != "none"),
        }


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()

    @property
    def available(self) -> bool:
        if not self.config.enabled or self.config.provider == "none":
            return False
        if self.config.provider == "cloudflare":
            return bool(self.config.account_id and self.config.api_key)
        if self.config.provider == "openai":
            return bool(self.config.api_key)
        return True

    # -- generic text ------------------------------------------------------
    def complete(self, prompt: str, system: str = "",
                 max_tokens: Optional[int] = None) -> Optional[str]:
        if not self.available:
            return None
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            if self.config.provider == "cloudflare":
                return self._cloudflare_chat(
                    messages, max_tokens,
                    model=self.config.text_model or self.config.model,
                )
            if self.config.provider == "openai":
                return self._openai(messages, max_tokens)
            if self.config.provider == "ollama":
                return self._ollama(messages, max_tokens)
        except Exception:
            return None
        return None

    # -- Cloudflare Workers AI --------------------------------------------
    def _cloudflare_post(self, model: str, payload: dict,
                         timeout: Optional[int] = None) -> Tuple[str, bytes]:
        if not self.available:
            raise RuntimeError("Cloudflare Workers AI is not configured")
        url = self.config.base_url.rstrip("/") + "/" + model
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout or self.config.timeout) as resp:
            return (resp.headers.get("Content-Type", ""), resp.read())

    def _cloudflare_chat(self, messages: List[dict], max_tokens: Optional[int],
                         model: str) -> Optional[str]:
        payload = {
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        _, body = self._cloudflare_post(model, payload)
        data = json.loads(body.decode("utf-8", errors="replace"))
        result = data.get("result")
        if isinstance(result, dict):
            choices = result.get("choices")
            if choices:
                return str(choices[0].get("message", {}).get("content", "")).strip()
            if "response" in result:
                return str(result["response"]).strip()
        if isinstance(result, str):
            return result.strip()
        return None

    def _cloudflare_agree(self) -> None:
        """Accept the Meta vision-model licence if Cloudflare asks for it."""
        try:
            self._cloudflare_post(self.config.model, {"prompt": "agree"}, timeout=30)
        except Exception:
            pass

    @staticmethod
    def _as_data_uri(image) -> str:
        if isinstance(image, str):
            if image.startswith("data:"):
                return image
            return "data:image/png;base64," + image
        raw = bytes(image)
        if raw.startswith(b"\xff\xd8"):
            mime = "image/jpeg"
        elif raw.startswith(b"\x89PNG"):
            mime = "image/png"
        else:
            mime = "image/png"
        return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")

    def describe_image(self, image, question: str, lang: str = "en",
                       max_tokens: int = 500) -> Optional[str]:
        """Ask the Cloudflare vision model about an uploaded image."""
        if not self.available or self.config.provider != "cloudflare":
            return None
        data_uri = self._as_data_uri(image)
        prompt = (
            "You are a forensic investigation assistant. Answer the question using "
            "ONLY what is visible in the image. Do not guess identities, names, "
            "amounts, locations or events that are not visible. If the image is "
            "unclear or the answer is not visible, say so. "
            f"Answer in language code '{lang}'.\n\nQuestion: {question}"
        )
        payload = {
            "prompt": prompt,
            "image": data_uri,
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }
        try:
            _, body = self._cloudflare_post(self.config.model, payload)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 403 and ("agree" in error_body.lower()
                                    or "license" in error_body.lower()):
                self._cloudflare_agree()
                try:
                    _, body = self._cloudflare_post(self.config.model, payload)
                except Exception:
                    return None
            else:
                return None
        except Exception:
            return None
        try:
            data = json.loads(body.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return None
        result = data.get("result")
        if isinstance(result, dict):
            if "response" in result:
                return str(result["response"]).strip()
            choices = result.get("choices")
            if choices:
                return str(choices[0].get("message", {}).get("content", "")).strip()
        if isinstance(result, str):
            return result.strip()
        return None

    def generate_image(self, prompt: str, steps: int = 4) -> Optional[bytes]:
        """Generate an image with FLUX.1 [schnell] and return raw bytes."""
        if not self.available or self.config.provider != "cloudflare":
            return None
        if not self.config.image_model:
            return None
        payload = {"prompt": prompt[:2048], "steps": max(1, min(8, int(steps)))}
        try:
            content_type, body = self._cloudflare_post(
                self.config.image_model, payload, timeout=max(120, self.config.timeout)
            )
        except Exception:
            return None
        if "application/json" in content_type.lower() or body[:1] == b"{":
            try:
                data = json.loads(body.decode("utf-8", errors="replace"))
                image_b64 = (data.get("result") or {}).get("image")
                if image_b64:
                    return base64.b64decode(image_b64)
            except Exception:
                return None
        if body[:2] == b"\xff\xd8" or body[:4] == b"\x89PNG":
            return body
        return None

    # -- OpenAI / Ollama ---------------------------------------------------
    def _openai(self, messages: List[dict], max_tokens: Optional[int]) -> Optional[str]:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.config.text_model or self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()

    def _ollama(self, messages: List[dict], max_tokens: Optional[int]) -> Optional[str]:
        url = self.config.base_url.rstrip("/") + "/api/chat"
        body = {
            "model": self.config.text_model or self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            },
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("message") or {}).get("content", "").strip()

    # -- grounded polish ---------------------------------------------------
    def polish_answer(self, question: str, draft: str, citations: List[Citation],
                      lang: str = "en", max_citations: int = 6) -> str:
        """Rewrite ``draft`` in ``lang`` while preserving citation labels."""
        if not self.available or not citations:
            return draft
        labels = [f"C{i}" for i in range(1, min(len(citations), max_citations) + 1)]
        citation_lines = []
        for i, citation in enumerate(citations[:max_citations], 1):
            citation_lines.append(
                f"[C{i}] evidence={citation.evidence_id} file={citation.file} "
                f"time={citation.timestamp} snippet={citation.snippet[:220]}"
            )
        prompt = (
            f"Question: {question}\n\n"
            f"Draft answer:\n{draft}\n\n"
            f"Available citations:\n" + "\n".join(citation_lines) + "\n\n"
            f"Rewrite the draft answer in the language with code '{lang}'. "
            "Use ONLY facts present in the draft and citations. Do not invent names, "
            "amounts, dates, evidence ids or citation labels. Keep the citation labels "
            f"({', '.join(labels)}) exactly as they appear. Be concise and professional. "
            "If the draft says no evidence was found, keep that meaning."
        )
        system = (
            "You are a forensic investigation assistant. You must be strictly grounded: "
            "never add facts that are not in the draft or citations. Never invent citations."
        )
        out = self.complete(prompt, system=system)
        if not out:
            return draft
        if citations and not any(re.search(rf"\b{re.escape(label)}\b", out) for label in labels):
            return draft
        invented = [label for label in re.findall(r"\bC\d+\b", out) if label not in labels]
        if invented:
            return draft
        return out


def llm_status(config: Optional[LLMConfig] = None) -> dict:
    cfg = config or LLMConfig.from_env()
    return {
        "available": LLMClient(cfg).available,
        "provider": cfg.provider,
        "model": cfg.model,
        "text_model": cfg.text_model,
        "image_model": cfg.image_model,
        "enabled": cfg.enabled,
    }
