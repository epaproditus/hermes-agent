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
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

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


# ---- Gateway opcodes & close codes ----------------------------------------


class GatewayOp:
    """Fluxer-compatible Discord gateway opcodes."""
    DISPATCH = 0
    HEARTBEAT = 1
    IDENTIFY = 2
    PRESENCE_UPDATE = 3
    VOICE_STATE_UPDATE = 4
    RESUME = 6
    RECONNECT = 7
    REQUEST_GUILD_MEMBERS = 8
    INVALID_SESSION = 9
    HELLO = 10
    HEARTBEAT_ACK = 11


# ---- Fluxer Gateway Client -------------------------------------------------


class FluxerGatewayClient:
    """Manages a WebSocket connection to the Fluxer gateway.

    Implements the Discord-compatible gateway protocol with opcodes
    0-16, handling the full HELLO/IDENTIFY/READY lifecycle, periodic
    heartbeat, session resume, and dispatch routing.

    Usage::

        client = FluxerGatewayClient(token, gateway_url)
        client.on_dispatch = my_async_handler
        if await client.connect():
            # connected — events flow to on_dispatch
            ...
        await client.disconnect()
    """

    # Default heartbeat interval (ms) — used before HELLO arrives.
    _DEFAULT_HEARTBEAT_INTERVAL: int = 41250
    # Timeout for waiting on a HEARTBEAT_ACK after sending a heartbeat (ms).
    _HEARTBEAT_TIMEOUT: float = 45.0  # seconds
    # Close codes that are fatal (no reconnect possible).
    _FATAL_CLOSE_CODES: set[int] = {4004, 4010, 4012}
    # Close codes that signal a rate-limit backoff.
    _RATE_LIMIT_CLOSE_CODES: set[int] = {4008}
    # Normal disconnect code sent by the client.
    _CLOSE_CODE_NORMAL: int = 1000
    # Maximum allowed payload size for gateway messages.
    _MAX_PAYLOAD_SIZE: int = 4096

    def __init__(self, token: str, gateway_url: str) -> None:
        self.token = token
        self.gateway_url = gateway_url

        # WebSocket state
        self._ws: Any = None  # WebSocketClientProtocol
        self._connected = False
        self._close_code: Optional[int] = None

        # Session tracking
        self._session_id: Optional[str] = None
        self._last_seq: Optional[int] = None

        # Heartbeat state (wall-clock seconds)
        self._heartbeat_interval: Optional[float] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_ack_received: bool = False

        # Dispatch loop task (runs independently after handshake)
        self._dispatch_task: Optional[asyncio.Task] = None

        # Dispatch callback — set by the consumer (e.g. FluxerAdapter)
        self.on_dispatch: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None

        # Reconnect hook — set during close-code handling
        self._on_reconnect_requested: Optional[Callable[[], Awaitable[None]]] = None

        # Internal control
        self._disconnect_requested = False

    # ---- Public properties -------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Return whether the WebSocket is currently connected."""
        return self._connected

    @property
    def session_id(self) -> Optional[str]:
        """Return the session identifier for RESUME recovery."""
        return self._session_id

    @property
    def last_seq(self) -> Optional[int]:
        """Return the last sequence number for RESUME recovery."""
        return self._last_seq

    # ---- Connection lifecycle ----------------------------------------------

    async def connect(self) -> bool:
        """Open the WebSocket connection and authenticate.

        Steps:
        1. Connect to ``self.gateway_url``
        2. Wait for HELLO (op 10) → extract heartbeat interval
        3. Send IDENTIFY (op 2) with bot token
        4. Wait for READY (op 0) → extract session_id
        5. Start heartbeat loop
        6. Begin dispatch loop

        Returns ``True`` if the full handshake completed, ``False``
        otherwise.  Raises ``AuthenticationError`` on close code 4004.
        """
        try:
            async with websockets.connect(self.gateway_url) as ws:
                self._ws = ws
                self._connected = True
                self._disconnect_requested = False

                # 1. Wait for HELLO (op 10)
                hello_raw = await ws.recv()
                hello = json.loads(hello_raw)
                if hello.get("op") != GatewayOp.HELLO:
                    logger.warning(
                        "Expected HELLO (op 10), got op %s", hello.get("op")
                    )
                    await self._teardown()
                    return False

                interval_ms = hello["d"].get("heartbeat_interval", self._DEFAULT_HEARTBEAT_INTERVAL)
                self._heartbeat_interval = interval_ms / 1000.0

                # 2. Send IDENTIFY (op 2)
                identify = {
                    "op": GatewayOp.IDENTIFY,
                    "d": {
                        "token": self.token,
                        "properties": {
                            "os": "linux",
                            "browser": "hermes-agent",
                            "device": "hermes-agent",
                        },
                    },
                }
                await ws.send(json.dumps(identify))

                # 3. Wait for READY (op 0, event READY) or INVALID_SESSION
                while True:
                    ready_raw = await ws.recv()
                    ready = json.loads(ready_raw)

                    op = ready.get("op")

                    # INVALID_SESSION (op 9) → authentication failure
                    if op == GatewayOp.INVALID_SESSION:
                        logger.warning("Gateway returned INVALID_SESSION during connect")
                        await self._teardown()
                        return False

                    # DISPATCH with READY event
                    if op == GatewayOp.DISPATCH and ready.get("t") == "READY":
                        d = ready["d"]
                        self._session_id = d.get("session_id")
                        seq = ready.get("s")
                        if seq is not None:
                            self._last_seq = seq
                        logger.info(
                            "Gateway ready — session_id=%s, last_seq=%s",
                            self._session_id, self._last_seq,
                        )

                        # Fire on_dispatch for READY
                        if self.on_dispatch is not None:
                            await self.on_dispatch("READY", d)

                        break

                    # Unexpected close during handshake
                    if not ws.open:
                        await self._teardown()
                        return False

                # 4. Start heartbeat loop
                self._heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop()
                )

                # 5. Start dispatch reading loop in background
                self._dispatch_task = asyncio.create_task(
                    self._dispatch_loop()
                )

                return True

        except AuthenticationError:
            # Re-raise so the caller knows it's fatal
            await self._teardown()
            raise
        except Exception as exc:
            logger.error("WebSocket connection failed: %s", exc)
            await self._teardown()

        return self._connected

    async def disconnect(self) -> None:
        """Close the WebSocket connection gracefully.

        Sends a close frame with code 1000, stops the heartbeat loop,
        and resets connection state.  Safe to call when already
        disconnected.
        """
        if self._ws is not None:
            self._disconnect_requested = True
            try:
                await self._ws.close(code=self._CLOSE_CODE_NORMAL)
            except Exception:
                pass
        await self._teardown()

    async def resume(self) -> bool:
        """Reconnect and send RESUME (op 6) to restore the session.

        Requires a prior successful ``connect()`` that produced a
        ``session_id`` and ``last_seq``.  Returns ``True`` if the
        session was restored successfully.

        On INVALID_SESSION (op 9), returns ``False`` — the caller
        should fall back to a fresh ``connect()``.
        """
        if self._session_id is None or self._last_seq is None:
            return False

        try:
            async with websockets.connect(self.gateway_url) as ws:
                self._ws = ws
                self._connected = True
                self._disconnect_requested = False

                # Wait for HELLO
                hello_raw = await ws.recv()
                hello = json.loads(hello_raw)
                if hello.get("op") != GatewayOp.HELLO:
                    await self._teardown()
                    return False

                interval_ms = hello["d"].get("heartbeat_interval", self._DEFAULT_HEARTBEAT_INTERVAL)
                self._heartbeat_interval = interval_ms / 1000.0

                # Send RESUME (op 6)
                resume_payload = {
                    "op": GatewayOp.RESUME,
                    "d": {
                        "token": self.token,
                        "session_id": self._session_id,
                        "seq": self._last_seq,
                    },
                }
                await ws.send(json.dumps(resume_payload))

                # Wait for READY or INVALID_SESSION
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)

                    op = data.get("op")
                    if op == GatewayOp.INVALID_SESSION:
                        logger.warning("RESUME rejected — session invalid")
                        await self._teardown()
                        return False

                    if op == GatewayOp.DISPATCH and data.get("t") == "READY":
                        d = data["d"]
                        self._session_id = d.get("session_id", self._session_id)
                        seq = data.get("s")
                        if seq is not None:
                            self._last_seq = seq
                        break

                # Start heartbeat and dispatch loops
                self._heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop()
                )
                self._dispatch_task = asyncio.create_task(
                    self._dispatch_loop()
                )
                return True

        except AuthenticationError:
            await self._teardown()
            raise
        except Exception:
            logger.exception("Error during resume")
            await self._teardown()
            return False

        return True

    def can_resume(self) -> bool:
        """Return whether a RESUME is possible (session data exists)."""
        return self._session_id is not None and self._last_seq is not None

    # ---- Internal: heartbeat -----------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Send HEARTBEAT (op 1) at the interval from HELLO.

        The interval is taken from ``self._heartbeat_interval``
        (seconds).  After each heartbeat the loop waits for up to
        ``_HEARTBEAT_TIMEOUT`` seconds for a HEARTBEAT_ACK (op 11),
        which is tracked by the dispatch loop.  If the ACK does not
        arrive in time the connection is considered dead.
        """
        interval = self._heartbeat_interval or (self._DEFAULT_HEARTBEAT_INTERVAL / 1000.0)

        while self._connected and not self._disconnect_requested:
            await asyncio.sleep(interval)

            if not self._connected or self._disconnect_requested:
                break

            # Send heartbeat with the last sequence number
            seq = self._last_seq if self._last_seq is not None else None
            heartbeat = {"op": GatewayOp.HEARTBEAT, "d": seq}

            self._heartbeat_ack_received = False
            try:
                await self._ws.send(json.dumps(heartbeat))
            except Exception:
                logger.warning("Failed to send heartbeat")
                break

            # Wait for ACK (set by _dispatch_loop when it sees op 11)
            try:
                await asyncio.wait_for(
                    self._wait_for_ack(), timeout=self._HEARTBEAT_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Heartbeat ACK timeout — no response in %ss",
                    self._HEARTBEAT_TIMEOUT,
                )
                await self._ws.close(code=1000)
                break

    async def _wait_for_ack(self) -> None:
        """Wait until ``_heartbeat_ack_received`` is set by the dispatch loop."""
        while not self._heartbeat_ack_received and not self._disconnect_requested:
            await asyncio.sleep(0.05)

    # ---- Internal: dispatch loop -------------------------------------------

    async def _dispatch_loop(self) -> None:
        """Read messages from the WebSocket and route them to dispatch handling.

        Runs as a background reader: each received message that is a
        DISPATCH (op 0) is routed to ``_handle_raw_dispatch``, which
        fires ``on_dispatch``.  Other opcodes (RECONNECT op 7,
        INVALID_SESSION op 9) are handled inline.  The loop exits on
        connection close or disconnect request.
        """
        try:
            async for raw in self._ws:
                if self._disconnect_requested:
                    break

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON message from gateway")
                    continue

                op = data.get("op")

                if op == GatewayOp.DISPATCH:
                    await self._handle_raw_dispatch(data)
                elif op == GatewayOp.HEARTBEAT_ACK:
                    self._heartbeat_ack_received = True
                elif op == GatewayOp.RECONNECT:
                    logger.info("Gateway requested reconnect (op 7)")
                    if self._on_reconnect_requested is not None:
                        await self._on_reconnect_requested()
                    break
                elif op == GatewayOp.INVALID_SESSION:
                    logger.info("Gateway invalidated session (op 9)")
                    break
                # Ignore other opcodes silently
        except Exception as exc:
            # Normal on disconnect (connection dropped)
            pass

        # If the loop exits (connection dropped), clean up
        if self._connected and not self._disconnect_requested:
            # Connection dropped unexpectedly
            if self._close_code in self._RATE_LIMIT_CLOSE_CODES:
                if self._on_reconnect_requested is not None:
                    await self._on_reconnect_requested()
            elif self._close_code in self._FATAL_CLOSE_CODES:
                pass  # Caller handles fatal codes
            await self._teardown()

    async def _handle_raw_dispatch(self, data: Dict[str, Any]) -> None:
        """Process a single DISPATCH (op 0) frame.

        Updates ``_last_seq`` and calls ``on_dispatch`` if set.
        The callback receives the event type string and the event data
        dictionary.
        """
        event_type = data.get("t")
        event_data = data.get("d", {})
        seq = data.get("s")
        if seq is not None:
            self._last_seq = seq

        if event_type and self.on_dispatch is not None:
            try:
                await self.on_dispatch(event_type, event_data)
            except Exception:
                logger.exception(
                    "on_dispatch handler failed for event %s", event_type
                )

    # ---- Internal: close code handling -------------------------------------

    async def _handle_close(
        self, code: int, reason: str = ""
    ) -> None:
        """Handle a gateway close code.

        Determines whether the close code is fatal (raises
        ``AuthenticationError`` for 4004/4010/4012), rate-limit
        related (4008 — schedules a reconnect), or recoverable
        (other codes — clean teardown).
        """
        self._close_code = code
        logger.info("Gateway closed — code=%d, reason=%s", code, reason or "none")

        if code == 4004:
            raise AuthenticationError(
                f"Authentication failed (code {code}): {reason}"
            )

        if code in self._FATAL_CLOSE_CODES:
            logger.error(
                "Fatal gateway close code %d (%s) — reconnecting is not possible",
                code, reason,
            )
            return

        if code in self._RATE_LIMIT_CLOSE_CODES:
            logger.warning("Rate-limited by gateway (code %d) — will reconnect", code)
            if self._on_reconnect_requested is not None:
                await self._on_reconnect_requested()
            return

        # Other codes (4001, 4002, 4003, 4007, 4009, etc.) — recoverable
        logger.info("Gateway close code %d — recoverable", code)

    # ---- Internal: teardown ------------------------------------------------

    async def _teardown(self) -> None:
        """Reset all internal state after a disconnect.

        Stops the heartbeat task, closes the WebSocket if still open,
        and resets the connection flag.  Does NOT clear session_id or
        last_seq (they are needed for RESUME).
        """
        self._connected = False

        # Stop heartbeat
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Stop dispatch loop
        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except (asyncio.CancelledError, Exception):
                pass
            self._dispatch_task = None

        # Close WebSocket
        if self._ws is not None:
            try:
                await self._ws.close(code=self._CLOSE_CODE_NORMAL)
            except Exception:
                pass
            self._ws = None


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
        self._bot_user: Optional[Dict[str, Any]] = None

    # ---- BasePlatformAdapter abstract methods -----------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to the Fluxer instance.

        Creates the REST client first to resolve the gateway WebSocket
        URL from ``GET /gateway/bot``, then opens the WebSocket
        connection and wires ``_handle_event`` as the dispatch
        callback.

        On reconnect (is_reconnect=True), the gateway client's
        ``resume()`` method is attempted first, falling back to
        a fresh ``connect()`` if the session is invalid.
        """
        self._rest = FluxerRESTClient(
            base_url=self._api_url,
            bot_token=self._token,
        )

        # Resolve gateway URL from the REST API
        try:
            gateway_info = await self._rest.get_gateway_bot()
            gateway_url = gateway_info.get(
                "url",
                self._api_url.replace("http://", "ws://").replace("https://", "wss://")
                + "/gateway",
            )
        except Exception:
            logger.warning("Failed to resolve gateway URL, using default")
            gateway_url = self._api_url.replace("http://", "ws://").replace("https://", "wss://")

        self._gateway = FluxerGatewayClient(self._token, gateway_url)
        self._gateway.on_dispatch = self._handle_event

        if is_reconnect and self._gateway.can_resume():
            connected = await self._gateway.resume()
        else:
            connected = await self._gateway.connect()

        if connected:
            logger.info(
                "Fluxer adapter %s to %s",
                "reconnected" if is_reconnect else "connected",
                self._api_url,
            )
        return connected

    async def _handle_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Receive and route gateway dispatch events.

        Called by ``FluxerGatewayClient.on_dispatch`` for every
        DISPATCH (op 0) event received from the gateway.

        This is a basic router that will be extended by PLY-327
        (event normalization and dispatch routing).
        """
        logger.debug("Fluxer gateway event: %s", event_type)

        if event_type == "READY":
            self._bot_user = data.get("user")
        elif event_type == "MESSAGE_CREATE":
            # Basic message handling — extended by PLY-327
            pass
        elif event_type == "TYPING_START":
            pass

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
        return self._bot_user


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
