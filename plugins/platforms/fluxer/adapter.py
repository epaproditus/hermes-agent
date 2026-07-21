"""
Fluxer platform adapter stub.

Fluxer is an open-source Discord alternative (https://github.com/fluxer).
This adapter provides the skeleton for connecting Hermes to a Fluxer
instance via WebSocket gateway (real-time) and REST API (cron/standalone).

Environment variables:
    FLUXER_BOT_TOKEN          Bot authentication token
    FLUXER_API_URL            Base URL of the Fluxer API (e.g. https://fluxer.example.com)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Ensure Hermes core is importable when loaded as a bundled plugin
# ---------------------------------------------------------------------------
_HERMES_ROOT = Path(__file__).resolve().parents[3]
if str(_HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERMES_ROOT))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# ---- Optional dependency guards -------------------------------------------

try:
    import websockets  # noqa: F401

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None  # type: ignore[assignment]

try:
    import httpx  # noqa: F401

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

# ---- Exception classes ----------------------------------------------------


class AuthenticationError(Exception):
    """Raised when the Fluxer API returns a 401 response.

    Indicates the bot token is invalid or has been revoked. The
    reconnection layer should catch this and stop attempting to
    authenticate with the same credentials.
    """


class RateLimitError(Exception):
    """Raised when the Fluxer API rate-limits the client and the retry
    also exceeded the limit.

    Contains the parsed rate-limit headers so callers can make
    informed retry decisions.
    """

    def __init__(self, retry_after: float, headers: Dict[str, str]) -> None:
        self.retry_after = retry_after
        self.headers = headers
        super().__init__(
            f"Rate limited: retry after {retry_after}s"
        )


# ---- Fluxer REST Client ---------------------------------------------------


class FluxerRESTClient:
    """HTTP client for the Fluxer REST API.

    Wraps ``httpx.AsyncClient`` with:
    - ``Authorization: Bot <token>`` header on every request
    - Rate-limit backoff: reads ``Retry-After`` and ``X-RateLimit-*`` headers
      on 429 responses, sleeps, and retries exactly once
    - 401 detection: raises ``AuthenticationError`` for the reconnection layer
    - Lazy client creation (no I/O in ``__init__``)

    Used for out-of-process operations such as:
    - Sending messages from cron jobs (standalone sender)
    - Channel / guild metadata queries
    - File uploads
    """

    _REQUEST_TIMEOUT: float = 30.0
    _DEFAULT_RETRY_AFTER: float = 1.0
    _MAX_RETRY_AFTER: float = 60.0

    def __init__(self, base_url: str, bot_token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.bot_token = bot_token
        self._client: Optional[httpx.AsyncClient] = None

    # ---- HTTP plumbing ----------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        """Return (and lazily create) the shared ``httpx.AsyncClient``."""
        if self._client is None:
            from gateway.platforms._http_client_limits import platform_httpx_limits

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bot {self.bot_token}"},
                timeout=self._REQUEST_TIMEOUT,
                limits=platform_httpx_limits(),
            )
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute an HTTP request with auth, rate-limit backoff, and error mapping.

        Steps:
        1. Obtain the shared client
        2. Send the request
        3. On 401 → raise ``AuthenticationError``
        4. On 429 → parse ``Retry-After``, sleep, retry **once**
           (second 429 → raise ``RateLimitError``)
        5. On other errors → raise for status
        6. Return parsed JSON
        """
        client = await self._get_client()

        resp = await client.request(method, path, **kwargs)

        # 401 → auth error (non-retryable)
        if resp.status_code == 401:
            raise AuthenticationError("Invalid bot token")

        # 429 → rate-limit backoff + single retry
        if resp.status_code == 429:
            retry_after = self._parse_retry_after(resp.headers)
            await asyncio.sleep(retry_after)

            resp = await client.request(method, path, **kwargs)

            if resp.status_code == 429:
                raise RateLimitError(
                    retry_after=self._parse_retry_after(resp.headers),
                    headers=dict(resp.headers),
                )
            if resp.status_code == 401:
                raise AuthenticationError("Invalid bot token")

        # Surface remaining HTTP errors
        resp.raise_for_status()

        # 204 No Content → empty dict
        if resp.status_code == 204:
            return {}

        # Try JSON parsing — compatible with both real responses (which
        # carry application/json) and test doubles that omit the header.
        try:
            return resp.json()
        except Exception:
            return {"status_code": resp.status_code}

    @staticmethod
    def _parse_retry_after(headers: Dict[str, str]) -> float:
        """Extract ``Retry-After`` seconds from response headers.

        Falls back to ``_DEFAULT_RETRY_AFTER`` when the header is missing,
        and clamps to ``_MAX_RETRY_AFTER``.
        """
        raw = headers.get("Retry-After", "")
        if not raw:
            return FluxerRESTClient._DEFAULT_RETRY_AFTER
        try:
            return min(float(raw), FluxerRESTClient._MAX_RETRY_AFTER)
        except (ValueError, TypeError):
            return FluxerRESTClient._DEFAULT_RETRY_AFTER

    # ---- Public API methods -----------------------------------------------

    async def send_message(
        self,
        channel_id: str,
        content: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Post a message to a Fluxer channel.

        POST /channels/{channel_id}/messages

        Extra ``kwargs`` are forwarded as top-level JSON fields (e.g.
        ``embeds``, ``tts``).
        """
        body: Dict[str, Any] = {"content": content}
        body.update(kwargs)
        return await self._request(
            "POST",
            f"/channels/{channel_id}/messages",
            json=body,
        )

    async def edit_message(
        self,
        channel_id: str,
        message_id: str,
        content: str,
    ) -> Dict[str, Any]:
        """Edit an existing message.

        PATCH /channels/{channel_id}/messages/{message_id}
        """
        return await self._request(
            "PATCH",
            f"/channels/{channel_id}/messages/{message_id}",
            json={"content": content},
        )

    async def send_typing(self, channel_id: str) -> Dict[str, Any]:
        """Trigger the typing indicator in a channel.

        POST /channels/{channel_id}/typing
        """
        return await self._request(
            "POST",
            f"/channels/{channel_id}/typing",
        )

    async def create_dm(self, recipient_id: str) -> Dict[str, Any]:
        """Create a direct message channel with a user.

        POST /channels
        """
        return await self._request(
            "POST",
            "/channels",
            json={"recipient_id": recipient_id},
        )

    async def join_guild(self, guild_id: str) -> Dict[str, Any]:
        """Make the bot join a guild.

        PUT /guilds/{guild_id}/members/@me
        """
        return await self._request(
            "PUT",
            f"/guilds/{guild_id}/members/@me",
        )

    async def get_me(self) -> Dict[str, Any]:
        """Return the bot user object (liveness probe).

        GET /users/@me
        """
        return await self._request("GET", "/users/@me")

    async def get_gateway_bot(self) -> Dict[str, Any]:
        """Return gateway connection info.

        GET /gateway/bot

        Returns a dict with ``url``, ``shards``, and
        ``session_start_limit``.
        """
        return await self._request("GET", "/gateway/bot")

    async def close(self) -> None:
        """Close the underlying HTTP client, releasing connections."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ---- Fluxer Gateway Client (stub) -----------------------------------------


class FluxerGatewayClient:
    """Manages a WebSocket connection to the Fluxer gateway.

    Responsible for:
    - Establishing and maintaining the WebSocket connection
    - Receiving real-time events (messages, presence, etc.)
    - Heartbeat / keep-alive handling
    - Reconnection with exponential backoff
    """

    def __init__(self, token: str, api_url: str) -> None:
        self.token = token
        self.api_url = api_url.rstrip("/")
        self._connected = False

    async def connect(self) -> bool:
        """Open the WebSocket connection and authenticate."""
        raise NotImplementedError  # pragma: no cover

    async def disconnect(self) -> None:
        """Close the WebSocket connection gracefully."""
        raise NotImplementedError  # pragma: no cover

    @property
    def is_connected(self) -> bool:
        return self._connected


# ---- Fluxer Platform Adapter ----------------------------------------------


class FluxerAdapter(BasePlatformAdapter):
    """Platform adapter for the Fluxer messaging platform.

    Integrates Hermes with a Fluxer instance through a WebSocket gateway
    (real-time messaging) and REST API (standalone / cron delivery).
    """

    supports_code_blocks: bool = True
    supports_async_delivery: bool = True
    splits_long_messages: bool = True
    typed_command_prefix: str = "/"

    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config)
        self._token: str = os.environ.get("FLUXER_BOT_TOKEN", "")
        self._api_url: str = os.environ.get(
            "FLUXER_API_URL", "http://localhost:8090"
        )
        self._gateway: Optional[FluxerGatewayClient] = None
        self._rest: Optional[FluxerRESTClient] = None

    # ---- BasePlatformAdapter abstract methods -----------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to the Fluxer instance.

        Initialises the gateway client and opens the WebSocket connection.
        On reconnect (is_reconnect=True), applies backoff / state recovery.
        """
        self._gateway = FluxerGatewayClient(self._token, self._api_url)
        self._rest = FluxerRESTClient(
            base_url=self._api_url,
            bot_token=self._token,
        )
        connected = await self._gateway.connect()
        if connected:
            logger.info(
                "Fluxer adapter %s to %s",
                "reconnected" if is_reconnect else "connected",
                self._api_url,
            )
        return connected

    async def disconnect(self) -> None:
        """Disconnect from the Fluxer instance."""
        if self._gateway is not None:
            await self._gateway.disconnect()
            self._gateway = None
        if self._rest is not None:
            await self._rest.close()
            self._rest = None

    async def send(
        self,
        message: str,
        channel_id: str,
        message_type: MessageType = MessageType.TEXT,
        event: Optional[MessageEvent] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Send a message to a Fluxer channel.

        Attempts gateway (WebSocket) delivery first; falls back to REST
        if the gateway is not connected.
        """
        raise NotImplementedError  # pragma: no cover

    async def send_media(
        self,
        file_path: str,
        channel_id: str,
        caption: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Upload and send a media file to a Fluxer channel."""
        raise NotImplementedError  # pragma: no cover

    def is_alive(self) -> bool:
        """Return whether the adapter is currently connected."""
        return self._gateway is not None and self._gateway.is_connected

    def get_me(self) -> Optional[Dict[str, Any]]:
        """Return the bot user's identity metadata, or None if unknown."""
        return None  # pragma: no cover


# ---- Requirements check ---------------------------------------------------


def check_fluxer_requirements() -> bool:
    """Return True when all required dependencies are available.

    Must be silent (no WARNING-level logging) since it is called
    frequently during config loading.
    """
    return WEBSOCKETS_AVAILABLE and HTTPX_AVAILABLE


def _is_connected(config: PlatformConfig) -> bool:
    """Return whether the Fluxer adapter appears configured and connected."""
    token = os.environ.get("FLUXER_BOT_TOKEN", "")
    api_url = os.environ.get("FLUXER_API_URL", "")
    return bool(token) and bool(api_url)


# ---- Plugin entry point ---------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="fluxer",
        label="Fluxer",
        adapter_factory=lambda cfg: FluxerAdapter(cfg),
        check_fn=check_fluxer_requirements,
        is_connected=_is_connected,
        required_env=["FLUXER_BOT_TOKEN", "FLUXER_API_URL"],
        install_hint="pip install 'hermes-agent[fluxer]'",
        emoji="💬",
    )
