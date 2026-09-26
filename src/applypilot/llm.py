"""
Unified LLM client for ApplyPilot.

Supported providers:
  - OpenRouter / any OpenAI-compatible endpoint via LLM_URL + LLM_API_KEY
  - Google Gemini via GEMINI_API_KEY
  - OpenAI via OPENAI_API_KEY
  - DeepSeek via DEEPSEEK_API_KEY
  - Anthropic via ANTHROPIC_API_KEY

Primary provider:
  LLM_URL + LLM_API_KEY

For OpenRouter:
  LLM_URL=https://openrouter.ai/api/v1
  LLM_API_KEY=<your OpenRouter key>
  LLM_MODEL=openrouter/free

LLM_MODEL env var overrides the default fast model.
LLM_MODEL_QUALITY env var sets the model used for quality-critical tasks
such as resume tailoring and cover letters.

The client automatically falls back to other configured providers/models
when a model is unavailable, rate-limited, exhausted, or otherwise fails.
"""

import logging
import os
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelEntry:
    """A model with everything needed to call it."""

    name: str
    provider: str
    base_url: str
    api_key: str


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

def _build_fallback_chain(
    primary_model: str,
    quality: bool = False,
) -> list[ModelEntry]:
    """
    Build a cross-provider fallback chain.

    The explicitly configured LLM_URL + LLM_API_KEY provider is always
    placed first. This allows OpenRouter or another OpenAI-compatible
    service to be the primary provider.

    Additional configured providers are then added as fallbacks.
    """

    # API keys
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("DEEPSEEK_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "") or os.environ.get("GROQ_KEY", "")

    # Generic OpenAI-compatible provider.
    #
    # For this project this will normally be:
    # https://openrouter.ai/api/v1
    local_url = os.environ.get("LLM_URL", "").rstrip("/")
    local_key = os.environ.get("LLM_API_KEY", "")
    if not local_url and os.environ.get("OPENROUTER_KEY"):
        local_url = "https://openrouter.ai/api/v1"
        local_key = os.environ.get("OPENROUTER_KEY", "")

    # Provider endpoints
    gemini_url = (
        "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    openai_url = "https://api.openai.com/v1"
    anthropic_url = "https://api.anthropic.com"
    deepseek_url = "https://api.deepseek.com/v1"
    groq_url = "https://api.groq.com/openai/v1"

    chain: list[ModelEntry] = []

    # ------------------------------------------------------------------
    # 1. Primary generic OpenAI-compatible provider
    # ------------------------------------------------------------------
    #
    # This is where OpenRouter enters the chain.
    #
    # Example:
    #   LLM_URL=https://openrouter.ai/api/v1
    #   LLM_API_KEY=...
    #   LLM_MODEL=openrouter/free
    #
    if local_url and local_key:
        chain.append(
            ModelEntry(
                primary_model,
                "openai-compatible",
                local_url,
                local_key,
            )
        )

    # ------------------------------------------------------------------
    # 2. Gemini fallbacks
    # ------------------------------------------------------------------

    if gemini_key:
        if quality:
            gemini_models = [
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.0-flash",
            ]
        else:
            gemini_models = [
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
            ]

        for model in gemini_models:
            chain.append(
                ModelEntry(
                    model,
                    "gemini",
                    gemini_url,
                    gemini_key,
                )
            )

    # ------------------------------------------------------------------
    # 3. OpenAI fallbacks
    # ------------------------------------------------------------------

    if openai_key:
        if quality:
            openai_models = [
                "gpt-4.1-mini",
                "gpt-4.1-nano",
            ]
        else:
            openai_models = [
                "gpt-4.1-nano",
                "gpt-4.1-mini",
            ]

        for model in openai_models:
            chain.append(
                ModelEntry(
                    model,
                    "openai",
                    openai_url,
                    openai_key,
                )
            )

    # ------------------------------------------------------------------
    # 4. DeepSeek fallback
    # ------------------------------------------------------------------

    if deepseek_key:
        chain.append(
            ModelEntry(
                "deepseek-chat",
                "deepseek",
                deepseek_url,
                deepseek_key,
            )
        )

    if groq_key:
        for model in ("llama-3.3-70b-versatile", "qwen/qwen3-32b"):
            chain.append(ModelEntry(model, "groq", groq_url, groq_key))

    # ------------------------------------------------------------------
    # 5. Anthropic fallback
    # ------------------------------------------------------------------

    if anthropic_key:
        if quality:
            anthropic_models = [
                "claude-sonnet-4-5-20250514",
                "claude-haiku-4-5-20251001",
            ]
        else:
            anthropic_models = [
                "claude-haiku-4-5-20251001",
            ]

        for model in anthropic_models:
            chain.append(
                ModelEntry(
                    model,
                    "anthropic",
                    anthropic_url,
                    anthropic_key,
                )
            )

    # ------------------------------------------------------------------
    # No providers configured
    # ------------------------------------------------------------------

    if not chain:
        raise RuntimeError(
            "No LLM provider configured. "
            "Set LLM_URL + LLM_API_KEY, GEMINI_API_KEY, "
            "OPENAI_API_KEY, DEEPSEEK_API_KEY, or ANTHROPIC_API_KEY."
        )

    return chain


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider(
    quality: bool = False,
) -> tuple[str, str, str]:
    """
    Return:
        (base_url, model, api_key)

    Priority:

    1. Explicit LLM_URL + LLM_API_KEY
       Example: OpenRouter

    2. Gemini

    3. OpenAI

    4. Error
    """

    # Generic OpenAI-compatible provider
    local_url = os.environ.get("LLM_URL", "").strip()
    local_key = os.environ.get("LLM_API_KEY", "").strip()

    # Other providers
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

    # Model overrides
    model_override = os.environ.get("LLM_MODEL", "").strip()
    quality_model = os.environ.get("LLM_MODEL_QUALITY", "").strip()

    if quality and quality_model:
        chosen_model = quality_model
    else:
        chosen_model = model_override

    # ---------------------------------------------------------------
    # Explicit generic provider first.
    #
    # This is important because our GitHub Actions workflow uses
    # OpenRouter through LLM_URL.
    # ---------------------------------------------------------------

    if local_url:
        return (
            local_url.rstrip("/"),
            chosen_model or "openrouter/free",
            local_key,
        )

    # ---------------------------------------------------------------
    # Gemini
    # ---------------------------------------------------------------

    if gemini_key:
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            chosen_model or "gemini-2.5-flash",
            gemini_key,
        )

    # ---------------------------------------------------------------
    # OpenAI
    # ---------------------------------------------------------------

    if openai_key:
        return (
            "https://api.openai.com/v1",
            chosen_model or "gpt-4.1-nano",
            openai_key,
        )

    # ---------------------------------------------------------------
    # Nothing configured
    # ---------------------------------------------------------------

    raise RuntimeError(
        "No LLM provider configured. "
        "Set LLM_URL + LLM_API_KEY, GEMINI_API_KEY, "
        "or OPENAI_API_KEY."
    )


# ---------------------------------------------------------------------------
# Client configuration
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_TIMEOUT = 300


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Multi-provider LLM client with automatic fallback."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        quality: bool = False,
    ) -> None:

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.quality = quality

        self._fallback_chain = _build_fallback_chain(
            model,
            quality=quality,
        )

        self._client = httpx.Client(
            timeout=_TIMEOUT,
        )

        # Models temporarily unavailable because of rate limits,
        # quota exhaustion, payment issues, etc.
        self._exhausted: dict[str, float] = {}

        chain_names = [
            f"{entry.name} ({entry.provider})"
            for entry in self._fallback_chain
        ]

        log.info(
            "Fallback chain (%s): %s",
            "quality" if quality else "fast",
            " -> ".join(chain_names),
        )

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a chat completion request.

        Automatically tries configured fallback models if the current
        model is unavailable.
        """

        # Qwen3 optimization.
        #
        # Some Qwen models benefit from explicitly disabling reasoning
        # for short operational tasks.
        if "qwen" in self.model.lower() and messages:
            first = messages[0]

            if (
                first.get("role") == "user"
                and not first["content"].startswith("/no_think")
            ):
                messages = [
                    {
                        "role": first["role"],
                        "content": f"/no_think\n{first['content']}",
                    }
                ] + messages[1:]

        # Skip models that were recently exhausted.
        now = time.time()

        entries_to_try = [
            entry
            for entry in self._fallback_chain
            if (
                entry.name not in self._exhausted
                or (now - self._exhausted[entry.name]) > 300
            )
        ]

        # If every model is marked exhausted, reset the temporary
        # exhaustion state and try again.
        if not entries_to_try:
            self._exhausted.clear()
            entries_to_try = list(self._fallback_chain)

        for index, entry in enumerate(entries_to_try):

            is_last = index == len(entries_to_try) - 1

            result = self._try_entry(
                entry,
                messages,
                temperature,
                max_tokens,
                is_last,
            )

            if result is not None:
                return result

        raise RuntimeError(
            "All models exhausted after trying: "
            f"{[entry.name for entry in entries_to_try]}. "
            "Wait a few minutes for rate limits to reset."
        )

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------

    def _try_entry(
        self,
        entry: ModelEntry,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        is_last: bool = False,
    ) -> str | None:

        if entry.provider == "anthropic":
            return self._try_anthropic(
                entry,
                messages,
                temperature,
                max_tokens,
                is_last,
            )

        return self._try_openai_compat(
            entry,
            messages,
            temperature,
            max_tokens,
            is_last,
        )

    # ------------------------------------------------------------------
    # OpenAI-compatible providers
    # ------------------------------------------------------------------

    def _try_openai_compat(
        self,
        entry: ModelEntry,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        is_last: bool = False,
    ) -> str | None:
        """
        Try an OpenAI-compatible endpoint.

        Works with:
          - OpenRouter
          - Gemini OpenAI-compatible API
          - OpenAI
          - DeepSeek
          - local OpenAI-compatible servers
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {entry.api_key}",
        }

        # DeepSeek has an output-token limit.
        if entry.provider == "deepseek":
            max_tokens = min(max_tokens, 8192)

        payload = {
            "model": entry.name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(_MAX_RETRIES):

            try:
                response = self._client.post(
                    f"{entry.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )

                # ------------------------------------------------------
                # 402 Payment Required
                # ------------------------------------------------------

                if response.status_code == 402:

                    log.warning(
                        "%s/%s payment required (402), "
                        "marking exhausted for 1 hour",
                        entry.provider,
                        entry.name,
                    )

                    self._exhausted[entry.name] = (
                        time.time() + 3600 - 300
                    )

                    return None

                # ------------------------------------------------------
                # 400 Bad Request
                # ------------------------------------------------------

                if response.status_code == 400:

                    body = response.text.lower()

                    if (
                        "api_key_invalid" in body
                        or "api key expired" in body
                        or "invalid api key" in body
                    ):
                        log.warning(
                            "%s/%s API key invalid/expired, "
                            "trying next provider",
                            entry.provider,
                            entry.name,
                        )

                        self._exhausted[entry.name] = time.time()

                        return None

                    # Other 400 errors are usually request/model-specific.
                    if not is_last:

                        log.warning(
                            "%s/%s 400 Bad Request, trying next: %.120s",
                            entry.provider,
                            entry.name,
                            response.text,
                        )

                        return None

                # ------------------------------------------------------
                # 404 Model not found
                # ------------------------------------------------------

                if response.status_code == 404:

                    log.warning(
                        "%s/%s model not found (404), trying next",
                        entry.provider,
                        entry.name,
                    )

                    self._exhausted[entry.name] = time.time()

                    return None

                # ------------------------------------------------------
                # 429 Rate limit / quota
                # ------------------------------------------------------

                if response.status_code == 429:

                    body = response.text.lower()

                    quota_error = (
                        "resource has been exhausted" in body
                        or "quota" in body
                        or "rate_limit" in body
                        or "rate limit" in body
                        or "too many requests" in body
                    )

                    if quota_error:

                        log.warning(
                            "%s/%s hit quota/rate limit, "
                            "trying next model",
                            entry.provider,
                            entry.name,
                        )

                        self._exhausted[entry.name] = time.time()

                        return None

                    if attempt < _MAX_RETRIES - 1:

                        wait = 2 ** attempt + 1

                        log.warning(
                            "%s/%s 429, retry in %ds (%d/%d)",
                            entry.provider,
                            entry.name,
                            wait,
                            attempt + 1,
                            _MAX_RETRIES,
                        )

                        time.sleep(wait)
                        continue

                    if not is_last:

                        log.warning(
                            "%s/%s still 429, trying next model",
                            entry.provider,
                            entry.name,
                        )

                        return None

                    response.raise_for_status()

                # ------------------------------------------------------
                # 503 Temporary service failure
                # ------------------------------------------------------

                if (
                    response.status_code == 503
                    and attempt < _MAX_RETRIES - 1
                ):

                    wait = 2 ** attempt

                    log.warning(
                        "%s/%s 503, retry in %ds",
                        entry.provider,
                        entry.name,
                        wait,
                    )

                    time.sleep(wait)
                    continue

                # ------------------------------------------------------
                # Normal successful response
                # ------------------------------------------------------

                response.raise_for_status()

                data = response.json()

                # Guard against malformed responses.
                if (
                    not isinstance(data, dict)
                    or not data.get("choices")
                ):

                    if not is_last:

                        log.warning(
                            "%s/%s malformed response "
                            "(no choices), trying next",
                            entry.provider,
                            entry.name,
                        )

                        return None

                    raise RuntimeError(
                        f"Malformed response from "
                        f"{entry.provider}/{entry.name}: "
                        f"no choices in "
                        f"{type(data).__name__}"
                    )

                message = data["choices"][0].get(
                    "message",
                    {},
                )

                text = message.get("content")

                # Some providers may return null content.
                if text is None:

                    if not is_last:

                        log.warning(
                            "%s/%s null content in response, "
                            "trying next",
                            entry.provider,
                            entry.name,
                        )

                        return None

                    raise RuntimeError(
                        f"Null content from "
                        f"{entry.provider}/{entry.name} "
                        f"(refusal: "
                        f"{message.get('refusal', 'none')})"
                    )

                if entry.name != self.model:

                    log.info(
                        "Used fallback %s/%s (primary: %s)",
                        entry.provider,
                        entry.name,
                        self.model,
                    )

                return text

            # ----------------------------------------------------------
            # Timeout
            # ----------------------------------------------------------

            except httpx.TimeoutException:

                if attempt < _MAX_RETRIES - 1:

                    wait = 2 ** attempt

                    log.warning(
                        "%s/%s timeout, retry in %ds",
                        entry.provider,
                        entry.name,
                        wait,
                    )

                    time.sleep(wait)
                    continue

                if not is_last:

                    log.warning(
                        "%s/%s timeout after retries, "
                        "trying next",
                        entry.provider,
                        entry.name,
                    )

                    return None

                raise

        return None

    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------

    def _try_anthropic(
        self,
        entry: ModelEntry,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        is_last: bool = False,
    ) -> str | None:
        """Try the Anthropic Messages API."""

        headers = {
            "Content-Type": "application/json",
            "x-api-key": entry.api_key,
            "anthropic-version": "2023-06-01",
        }

        # Convert OpenAI-style messages into Anthropic format.
        system_text = ""
        api_messages = []

        for message in messages:

            if message["role"] == "system":
                system_text = message["content"]

            else:
                api_messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"],
                    }
                )

        # Anthropic requires at least one user message.
        if not api_messages:
            return None

        payload: dict = {
            "model": entry.name,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }

        if system_text:
            payload["system"] = system_text

        if temperature > 0:
            payload["temperature"] = temperature

        for attempt in range(_MAX_RETRIES):

            try:

                response = self._client.post(
                    f"{entry.base_url}/v1/messages",
                    json=payload,
                    headers=headers,
                )

                # ------------------------------------------------------
                # Anthropic rate limit
                # ------------------------------------------------------

                if response.status_code == 429:

                    body = response.text.lower()

                    if (
                        "rate_limit" in body
                        or "quota" in body
                        or "rate limit" in body
                    ):

                        log.warning(
                            "anthropic/%s hit rate limit, "
                            "trying next",
                            entry.name,
                        )

                        self._exhausted[entry.name] = time.time()

                        return None

                    if attempt < _MAX_RETRIES - 1:

                        wait = 2 ** attempt + 1

                        log.warning(
                            "anthropic/%s 429, retry in %ds (%d/%d)",
                            entry.name,
                            wait,
                            attempt + 1,
                            _MAX_RETRIES,
                        )

                        time.sleep(wait)
                        continue

                    if not is_last:
                        return None

                    response.raise_for_status()

                # ------------------------------------------------------
                # Anthropic overloaded
                # ------------------------------------------------------

                if (
                    response.status_code == 529
                    and attempt < _MAX_RETRIES - 1
                ):

                    wait = 2 ** attempt + 2

                    log.warning(
                        "anthropic/%s overloaded (529), "
                        "retry in %ds",
                        entry.name,
                        wait,
                    )

                    time.sleep(wait)
                    continue

                response.raise_for_status()

                data = response.json()

                # Extract text blocks.
                text_parts = []

                for block in data.get("content", []):

                    if block.get("type") == "text":
                        text_parts.append(block["text"])

                text = "\n".join(text_parts)

                if entry.name != self.model:

                    log.info(
                        "Used fallback anthropic/%s "
                        "(primary: %s)",
                        entry.name,
                        self.model,
                    )

                return text

            except httpx.TimeoutException:

                if attempt < _MAX_RETRIES - 1:

                    wait = 2 ** attempt

                    log.warning(
                        "anthropic/%s timeout, retry in %ds",
                        entry.name,
                        wait,
                    )

                    time.sleep(wait)
                    continue

                if not is_last:
                    return None

                raise

        return None

    # ------------------------------------------------------------------
    # Convenience method
    # ------------------------------------------------------------------

    def ask(
        self,
        prompt: str,
        **kwargs,
    ) -> str:
        """Send a single user prompt."""

        return self.chat(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Close HTTP client
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._client.close()


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------

_instance: LLMClient | None = None
_quality_instance: LLMClient | None = None


# ---------------------------------------------------------------------------
# Public client getter
# ---------------------------------------------------------------------------

def get_client(
    quality: bool = False,
) -> LLMClient:
    """
    Return a module-level LLMClient singleton.

    quality=False:
        Used for normal/fast operations.

    quality=True:
        Used for higher-quality operations such as resume tailoring
        and cover letters.
    """

    global _instance
    global _quality_instance

    # ---------------------------------------------------------------
    # Quality client
    # ---------------------------------------------------------------

    if quality and os.environ.get("LLM_MODEL_QUALITY"):

        if _quality_instance is None:

            base_url, model, api_key = _detect_provider(
                quality=True,
            )

            log.info(
                "LLM quality provider: %s model: %s",
                base_url,
                model,
            )

            _quality_instance = LLMClient(
                base_url,
                model,
                api_key,
                quality=True,
            )

        return _quality_instance

    # ---------------------------------------------------------------
    # Normal client
    # ---------------------------------------------------------------

    if _instance is None:

        base_url, model, api_key = _detect_provider(
            quality=False,
        )

        log.info(
            "LLM provider: %s model: %s",
            base_url,
            model,
        )

        _instance = LLMClient(
            base_url,
            model,
            api_key,
            quality=False,
        )

    return _instance
