"""Optional LLM adapter — grounded polish only, never the source of truth.

The assistant works fully offline with deterministic templates.  If an LLM is
configured (OpenAI-compatible endpoint or a local Ollama server), it may rewrite
a draft answer into more natural prose **in the requested language**, but only
under strict grounding rules:

* the prompt contains the draft and the exact citation list;
* the model must keep at least one citation label from that list;
* if the model fails, times out, or invents citations, the deterministic draft
  is returned unchanged.

No evidence text is sent anywhere unless the operator explicitly enables a
provider.  The default is ``none``.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import Citation


@dataclass
class LLMConfig:
    provider: str = "none"           # none | openai | ollama
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    timeout: int = 30
    temperature: float = 0.1
    max_tokens: int = 700
    enabled: bool = False

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = (os.getenv("INVESTIGATION_LLM_PROVIDER") or "none").lower()
        if provider == "none":
            return cls()
        if provider == "openai":
            return cls(
                provider="openai",
                model=os.getenv("INVESTIGATION_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
                base_url=os.getenv("INVESTIGATION_LLM_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.openai.com/v1",
                api_key=os.getenv("INVESTIGATION_LLM_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or "",
                enabled=True,
            )
        if provider == "ollama":
            return cls(
                provider="ollama",
                model=os.getenv("INVESTIGATION_LLM_MODEL") or "llama3.1",
                base_url=os.getenv("INVESTIGATION_LLM_BASE_URL")
                or os.getenv("OLLAMA_HOST")
                or "http://localhost:11434",
                enabled=True,
            )
        return cls()

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "enabled": self.enabled,
            "timeout": self.timeout,
        }


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()

    @property
    def available(self) -> bool:
        if not self.config.enabled or self.config.provider == "none":
            return False
        if self.config.provider == "openai" and not self.config.api_key:
            return False
        return True

    def complete(self, prompt: str, system: str = "", max_tokens: Optional[int] = None) -> Optional[str]:
        if not self.available:
            return None
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            if self.config.provider == "openai":
                return self._openai(messages, max_tokens)
            if self.config.provider == "ollama":
                return self._ollama(messages, max_tokens)
        except Exception:
            return None
        return None

    def _openai(self, messages: List[dict], max_tokens: Optional[int]) -> Optional[str]:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.config.model,
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
            "model": self.config.model,
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
        """Rewrite ``draft`` in ``lang`` while preserving citation labels.

        Returns the draft unchanged if the LLM is unavailable, fails, or does
        not preserve at least one valid citation label.
        """
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
        # grounding guard: at least one citation label must survive
        if citations and not any(re.search(rf"\b{re.escape(label)}\b", out) for label in labels):
            return draft
        # reject invented labels such as C99 when only C1..C6 exist
        invented = [
            label for label in re.findall(r"\bC\d+\b", out)
            if label not in labels
        ]
        if invented:
            return draft
        return out


def llm_status(config: Optional[LLMConfig] = None) -> dict:
    cfg = config or LLMConfig.from_env()
    return {
        "available": LLMClient(cfg).available,
        "provider": cfg.provider,
        "model": cfg.model,
        "enabled": cfg.enabled,
    }
