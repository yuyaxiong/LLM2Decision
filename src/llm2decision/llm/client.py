from __future__ import annotations

from typing import List, Optional, Sequence

import httpx

from ..core.config import ModelSettings
from ..core.providers import get_profile


class ChatError(RuntimeError):
    pass


class ChatClient:
    """OpenAI-compatible chat client that adapts to vendor differences via settings.provider.

    Only three things are vendor-specific, all from the ProviderProfile in core/providers.py:
    the disable-thinking param name, whether an online tokenizer exists, and the vendor name in error
    messages.
    """

    def __init__(self, settings: ModelSettings) -> None:
        self._settings = settings
        self.profile = get_profile(settings.provider)
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            headers=headers,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat_completions(
        self,
        messages: Sequence[dict],
        max_tokens: int,
        logit_bias: Optional[dict] = None,
    ) -> dict:
        if not self._settings.api_key:
            raise ChatError(f"No api_key configured for {self.profile.display_name}")

        payload = {
            "model": self._settings.model,
            "messages": list(messages),
            "stream": False,
            "max_tokens": max_tokens,
            # Sampling temperature is fixed at 0 (greedy): the first token is necessarily the argmax,
            # so the answer position deterministically lands at position 0. Measured: the returned
            # logprobs/top_logprobs are computed before sampling and are unaffected by this value, so
            # the readout distribution stays complete.
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": True,
            "top_logprobs": self._settings.top_logprobs,
        }
        if logit_bias:
            payload["logit_bias"] = logit_bias
        # Disable-thinking param: Ark uses thinking={"type":"disabled"}, DashScope uses enable_thinking=false.
        # Measured: DashScope returns null logprobs unless thinking is explicitly disabled, so this field is a hard prerequisite.
        payload.update(self.profile.thinking_payload(self._settings.disable_thinking))

        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise ChatError(f"Request to {self.profile.display_name} failed: {exc}") from exc

        if response.status_code >= 400:
            raise ChatError(
                f"{self.profile.display_name} returned {response.status_code}: {response.text[:500]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ChatError(f"{self.profile.display_name} returned a body that is not valid JSON") from exc

    async def tokenize(self, text: str) -> List[int]:
        """Tokenization endpoint, used to get a handle's token id (for logit_bias) or to check whether a label is a single token.

        Only available for vendors that support this endpoint (Ark); DashScope has no online tokenizer,
        so callers should check profile.supports_tokenize before deciding to use logit_bias.
        """
        if not self.profile.supports_tokenize:
            raise ChatError(
                f"{self.profile.display_name} has no online tokenizer, so token ids cannot be resolved"
                f" (logit_bias needs them)"
            )
        if not self._settings.api_key:
            raise ChatError(f"No api_key configured for {self.profile.display_name}")
        try:
            response = await self._client.post(
                "/tokenization", json={"model": self._settings.model, "text": text}
            )
        except httpx.HTTPError as exc:
            raise ChatError(f"Tokenization request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ChatError(f"The tokenization endpoint returned {response.status_code}: {response.text[:300]}")
        data = response.json().get("data") or []
        if not data:
            return []
        return [int(token_id) for token_id in (data[0].get("token_ids") or [])]
