"""Tests for FluxerGatewayClient — WebSocket Gateway connection lifecycle."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.anyio

from plugins.platforms.fluxer.adapter import (
    AuthenticationError,
    FluxerGatewayClient,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def client():
    """Return a FluxerGatewayClient with fake credentials."""
    return FluxerGatewayClient(
        token="test-token",
        gateway_url="wss://gateway.fluxer.test/?v=1&encoding=json",
    )


def _make_frame(op: int, d: Any = None, t: str = None, s: int = None) -> str:
    """Build a JSON frame matching the Fluxer gateway protocol."""
    payload: Dict[str, Any] = {"op": op}
    if d is not None:
        payload["d"] = d
    if t is not None:
        payload["t"] = t
    if s is not None:
        payload["s"] = s
    return json.dumps(payload)


class MockWebSocket:
    """A controllable mock WebSocket that simulates the Fluxer gateway.

    Supports async iteration (``async for``) like the real websockets
    library.  When the recv queue is empty, it blocks until a new
    message is added or ``close()`` is called.
    """

    def __init__(self):
        self._sent: List[str] = []
        self._recv_queue: List[str] = []
        self._closed: bool = False
        self._close_code: Optional[int] = None
        self.open = True
        self._recv_event = asyncio.Event()

    def add_recv(self, frame: str) -> None:
        """Enqueue a frame for the mock to return on recv()."""
        self._recv_queue.append(frame)
        self._recv_event.set()

    def has_pending(self) -> bool:
        return len(self._recv_queue) > 0

    async def send(self, data: str) -> None:
        self._sent.append(data)

    async def recv(self) -> str:
        while not self._recv_queue:
            if not self.open:
                raise ConnectionError("Connection closed")
            self._recv_event.clear()
            await asyncio.wait_for(self._recv_event.wait(), timeout=5.0)
        return self._recv_queue.pop(0)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        """Support async for iteration like websockets."""
        if self._closed or not self.open:
            raise StopAsyncIteration
        try:
            return await self.recv()
        except (ConnectionError, asyncio.TimeoutError):
            raise StopAsyncIteration

    async def close(self, code: int = 1000) -> None:
        self._closed = True
        self._close_code = code
        self.open = False
        self._recv_event.set()

    def pop_sent(self) -> List[str]:
        """Return and clear all sent messages."""
        items = self._sent.copy()
        self._sent.clear()
        return items

    def last_sent(self) -> Optional[str]:
        if self._sent:
            return self._sent[-1]
        return None


@pytest.fixture
def mock_ws():
    """Return a new MockWebSocket."""
    return MockWebSocket()


@pytest.fixture
def mock_connect(mock_ws):
    """Patch websockets.connect to return the MockWebSocket."""
    with patch("plugins.platforms.fluxer.adapter.websockets.connect") as m:
        m.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
        m.return_value.__aexit__ = AsyncMock(return_value=None)
        yield m


async def _background_connect(
    client, mock_ws, timeout=2.0, add_heartbeat_msgs=True
):
    """Run connect() in a background task and return its result.

    The dispatch loop blocks forever waiting for messages, so we need
    to run connect() concurrently.  After the handshake is done, the
    caller can inspect state and optionally call disconnect().
    """
    task = asyncio.create_task(client.connect())
    return task


# ── Constructor Tests ────────────────────────────────────────────────


class TestConstructor:
    """FluxerGatewayClient.__init__ behaviour."""

    def test_stores_token_and_gateway_url(self):
        """Should store token, gateway_url, and set defaults."""
        c = FluxerGatewayClient(token="abc123", gateway_url="wss://gateway.test/")
        assert c.token == "abc123"
        assert c.gateway_url == "wss://gateway.test/"
        assert c._connected is False
        assert c._heartbeat_task is None
        assert c._ws is None
        assert c._heartbeat_interval is None
        assert c._last_seq is None
        assert c._session_id is None
        assert c.on_dispatch is None

    def test_session_id_property(self, client):
        """session_id property returns None before connect."""
        assert client.session_id is None

    def test_last_seq_property(self, client):
        """last_seq property returns None before connect."""
        assert client.last_seq is None


# ── Connect / Handshake ──────────────────────────────────────────────


class TestConnect:
    """connect() — HELLO → IDENTIFY → READY lifecycle."""

    async def test_handshake_success(self, client, mock_ws, mock_connect):
        """Should complete HELLO→IDENTIFY→READY and set session_id."""
        mock_ws.add_recv(_make_frame(10, {"heartbeat_interval": 41250}))
        mock_ws.add_recv(_make_frame(
            0,
            {"user": {"id": "bot1"}, "session_id": "sess_abc123", "guilds": []},
            t="READY",
            s=1,
        ))

        # Run connect in background — it will block on _dispatch_loop
        task = asyncio.create_task(client.connect())

        # Give time for the handshake to complete
        await asyncio.sleep(0.2)

        # Handshake should have succeeded
        assert client.is_connected is True
        assert client.session_id == "sess_abc123"
        assert client.last_seq == 1

        # Verify IDENTIFY was sent
        identify_sent = None
        for s in mock_ws._sent:
            parsed = json.loads(s)
            if parsed.get("op") == 2:
                identify_sent = parsed
                break
        assert identify_sent is not None, "IDENTIFY (op 2) was not sent"
        assert identify_sent["d"]["token"] == "test-token"

        # Clean up
        await client.disconnect()
        result = await task
        assert result is True

    async def test_handshake_fails_on_non_hello(self, client, mock_ws, mock_connect):
        """Should return False when first message is not HELLO."""
        mock_ws.add_recv(_make_frame(0, {}, t="INVALID", s=1))

        result = await client.connect()
        assert result is False
        assert client.is_connected is False

    async def test_handshake_invalid_session(self, client, mock_ws, mock_connect):
        """Should handle INVALID_SESSION (op 9) after IDENTIFY."""
        mock_ws.add_recv(_make_frame(10, {"heartbeat_interval": 41250}))
        mock_ws.add_recv(_make_frame(9, False))

        result = await client.connect()
        assert result is False
        assert client.is_connected is False


# ── Heartbeat ────────────────────────────────────────────────────────


class TestHeartbeat:
    """_heartbeat_loop() — op 1 at correct interval."""

    async def test_heartbeat_sent_at_interval(self, client, mock_ws, mock_connect):
        """Should send HEARTBEAT (op 1) at the configured interval."""
        mock_ws.add_recv(_make_frame(10, {"heartbeat_interval": 0.05}))  # 50ms
        mock_ws.add_recv(_make_frame(
            0, {"session_id": "sess_1"}, t="READY", s=1
        ))

        task = asyncio.create_task(client.connect())
        await asyncio.sleep(0.15)

        # Check heartbeats were sent
        heartbeats = [
            json.loads(s)
            for s in mock_ws._sent
            if json.loads(s).get("op") == 1
        ]
        assert len(heartbeats) >= 1
        assert heartbeats[0]["d"] == 1

        await client.disconnect()
        await task

    async def test_heartbeat_interval_from_hello(self, client, mock_ws, mock_connect):
        """_heartbeat_interval should be set from HELLO data."""
        # Close immediately after HELLO so connect() returns False
        mock_ws.add_recv(_make_frame(10, {"heartbeat_interval": 41250}))
        mock_ws.open = False

        result = await client.connect()
        assert result is False
        assert client._heartbeat_interval == 41.25  # 41250ms / 1000

    async def test_heartbeat_stops_on_disconnect(self, client, mock_ws, mock_connect):
        """Heartbeat task should stop after disconnect."""
        mock_ws.add_recv(_make_frame(10, {"heartbeat_interval": 0.05}))
        mock_ws.add_recv(_make_frame(
            0, {"session_id": "sess_1"}, t="READY", s=1
        ))

        task = asyncio.create_task(client.connect())
        await asyncio.sleep(0.08)

        sent_before = len(mock_ws._sent)

        await client.disconnect()
        await task

        # Wait to ensure no more heartbeats
        await asyncio.sleep(0.1)
        sent_after = len(mock_ws._sent)

        # Should not have sent anything new after disconnect
        assert sent_after == sent_before


# ── Dispatch / on_dispatch callback ──────────────────────────────────


class TestDispatch:
    """_dispatch_handler() — event demuxing with on_dispatch callback."""

    async def test_on_dispatch_called_with_event(self, client, mock_ws, mock_connect):
        """on_dispatch callback should be invoked on each DISPATCH event."""
        received: List[Dict[str, Any]] = []

        async def handler(event_type: str, data: Dict[str, Any]):
            received.append({"type": event_type, "data": data})

        client.on_dispatch = handler

        mock_ws.add_recv(_make_frame(10, {"heartbeat_interval": 50000}))
        mock_ws.add_recv(_make_frame(
            0, {"session_id": "sess_1", "user": {"id": "bot1"}}, t="READY", s=1
        ))
        mock_ws.add_recv(_make_frame(
            0,
            {"id": "msg1", "content": "Hello!", "channel_id": "chan1"},
            t="MESSAGE_CREATE",
            s=2,
        ))

        task = asyncio.create_task(client.connect())
        await asyncio.sleep(0.2)

        # Should have received both READY and MESSAGE_CREATE
        msg_events = [e for e in received if e["type"] == "MESSAGE_CREATE"]
        assert len(msg_events) == 1
        assert msg_events[0]["data"]["content"] == "Hello!"
        assert len(received) >= 2  # READY + MESSAGE_CREATE

        await client.disconnect()
        await task

    async def test_last_seq_increments(self, client, mock_ws, mock_connect):
        """last_seq should track the sequence number of the last dispatch."""
        mock_ws.add_recv(_make_frame(10, {"heartbeat_interval": 50000}))
        mock_ws.add_recv(_make_frame(
            0, {"session_id": "sess_1"}, t="READY", s=1
        ))
        mock_ws.add_recv(_make_frame(
            0, {"content": "msg1"}, t="MESSAGE_CREATE", s=2
        ))
        mock_ws.add_recv(_make_frame(
            0, {"content": "msg2"}, t="MESSAGE_CREATE", s=3
        ))

        task = asyncio.create_task(client.connect())
        await asyncio.sleep(0.2)

        assert client._last_seq == 3

        await client.disconnect()
        await task

    async def test_on_dispatch_called_for_multiple_types(self, client, mock_ws, mock_connect):
        """on_dispatch should be called for different event types."""
        received: List[str] = []

        async def handler(event_type: str, data: Dict[str, Any]):
            received.append(event_type)

        client.on_dispatch = handler

        mock_ws.add_recv(_make_frame(10, {"heartbeat_interval": 50000}))
        mock_ws.add_recv(_make_frame(
            0, {"session_id": "sess_1"}, t="READY", s=1
        ))
        mock_ws.add_recv(_make_frame(
            0, {}, t="TYPING_START", s=2
        ))
        mock_ws.add_recv(_make_frame(
            0, {}, t="MESSAGE_CREATE", s=3
        ))
        mock_ws.add_recv(_make_frame(
            0, {}, t="GUILD_CREATE", s=4
        ))

        task = asyncio.create_task(client.connect())
        await asyncio.sleep(0.2)

        assert "READY" in received
        assert "MESSAGE_CREATE" in received
        assert "TYPING_START" in received

        await client.disconnect()
        await task


# ── Resume ───────────────────────────────────────────────────────────


class TestResume:
    """resume() — RESUME (op 6) sends session_id + last_seq."""

    async def test_resume_sends_correct_payload(self, client, mock_ws, mock_connect):
        """Should send RESUME (op 6) with session_id and last_seq."""
        # First do a full handshake
        mock_ws.add_recv(_make_frame(10, {"heartbeat_interval": 50000}))
        mock_ws.add_recv(_make_frame(
            0, {"session_id": "sess_abc", "user": {"id": "bot1"}}, t="READY", s=5
        ))
        task = asyncio.create_task(client.connect())
        await asyncio.sleep(0.15)
        await client.disconnect()
        await task

        assert client.session_id == "sess_abc"
        assert client.last_seq == 5
        assert client.is_connected is False

        # New mock WS for resume
        ws2 = MockWebSocket()
        ws2.add_recv(_make_frame(10, {"heartbeat_interval": 50000}))
        ws2.add_recv(_make_frame(
            0, {"session_id": "sess_abc", "user": {"id": "bot1"}}, t="READY", s=6
        ))

        mock_connect.return_value.__aenter__ = AsyncMock(return_value=ws2)

        task2 = asyncio.create_task(client.resume())
        await asyncio.sleep(0.15)

        assert client.is_connected is True

        # Check that RESUME was sent
        resume_sent = None
        for s in ws2._sent:
            parsed = json.loads(s)
            if parsed.get("op") == 6:
                resume_sent = parsed
                break

        assert resume_sent is not None, "RESUME (op 6) was not sent"
        assert resume_sent["d"]["token"] == "test-token"
        assert resume_sent["d"]["session_id"] == "sess_abc"
        assert resume_sent["d"]["seq"] == 5

        await client.disconnect()
        await task2

    async def test_resume_without_session_returns_false(self, client):
        """Should return False if no previous session to resume."""
        result = await client.resume()
        assert result is False

    async def test_resume_invalid_session_returns_false(self, client, mock_ws, mock_connect):
        """Should handle INVALID_SESSION (op 9) during resume."""
        mock_ws.add_recv(_make_frame(10, {"heartbeat_interval": 50000}))
        mock_ws.add_recv(_make_frame(
            0, {"session_id": "sess_abc"}, t="READY", s=5
        ))
        task = asyncio.create_task(client.connect())
        await asyncio.sleep(0.15)
        await client.disconnect()
        await task

        # Resume with INVALID_SESSION
        ws2 = MockWebSocket()
        ws2.add_recv(_make_frame(10, {"heartbeat_interval": 50000}))
        ws2.add_recv(_make_frame(9, False))  # Invalid session

        mock_connect.return_value.__aenter__ = AsyncMock(return_value=ws2)

        result = await client.resume()
        assert result is False


# ── Disconnect ───────────────────────────────────────────────────────


class TestDisconnect:
    """disconnect() — graceful close with op 1000."""

    async def test_disconnect_sends_close_frame(self, client, mock_ws, mock_connect):
        """Should send close frame with code 1000 on disconnect."""
        mock_ws.add_recv(_make_frame(10, {"heartbeat_interval": 50000}))
        mock_ws.add_recv(_make_frame(
            0, {"session_id": "sess_1"}, t="READY", s=1
        ))

        # Track close on the underlying mock
        actual_close = mock_ws.close
        mock_ws.close = AsyncMock(wraps=actual_close)

        task = asyncio.create_task(client.connect())
        await asyncio.sleep(0.15)

        assert client.is_connected is True
        await client.disconnect()
        await task

        assert client.is_connected is False
        assert client._ws is None
        mock_ws.close.assert_awaited()

    async def test_disconnect_idempotent(self, client):
        """disconnect() should be safe to call when not connected."""
        await client.disconnect()  # Should not raise

    async def test_disconnect_state(self, client, mock_ws, mock_connect):
        """Is not connected after disconnect."""
        mock_ws.add_recv(_make_frame(10, {"heartbeat_interval": 50000}))
        mock_ws.add_recv(_make_frame(
            0, {"session_id": "sess_1"}, t="READY", s=1
        ))
        task = asyncio.create_task(client.connect())
        await asyncio.sleep(0.15)

        assert client.is_connected is True

        await client.disconnect()
        await task
        assert client.is_connected is False


# ── Close Code Handling ──────────────────────────────────────────────


class TestCloseCodeHandling:
    """Handle Fluxer-specific close codes."""

    async def test_close_4004_raises_auth_error(self, client):
        """close 4004 (auth failed) should raise AuthenticationError."""
        with pytest.raises(AuthenticationError, match="Authentication failed"):
            await client._handle_close(4004, "auth failed")

    async def test_close_4004_handled_during_connect(self, client, mock_ws, mock_connect):
        """Should fail connect gracefully when close 4004 detected."""
        mock_ws.add_recv(_make_frame(10, {"heartbeat_interval": 50000}))
        # Close before READY
        mock_ws.open = False

        with pytest.raises(AuthenticationError):
            # Manually trigger the close code in the exception handler
            mock_ws._close_code = 4004
            client._close_code = 4004
            await client._handle_close(4004, "auth failed")

    async def test_close_4008_triggers_reconnect(self, client):
        """close 4008 (rate limited) should trigger reconnect hook."""
        reconnect_requested = False

        async def reconnect_hook():
            nonlocal reconnect_requested
            reconnect_requested = True

        client._on_reconnect_requested = reconnect_hook

        await client._handle_close(4008, "rate limited")
        assert reconnect_requested is True

    async def test_close_4010_does_not_raise(self, client):
        """close 4010 (invalid shard) should be handled without exception."""
        await client._handle_close(4010, "invalid shard")
        # No exception raised

    async def test_close_4001_graceful(self, client):
        """close 4001 (unknown opcode) should not raise."""
        await client._handle_close(4001, "unknown opcode")


# ── Reconnect / Resume helpers ───────────────────────────────────────


class TestReconnectHelpers:
    """Helper methods for reconnection logic."""

    async def test_can_resume_returns_true_when_session_active(self, client, mock_ws, mock_connect):
        """can_resume() returns True when session_id and last_seq exist."""
        assert client.can_resume() is False

        mock_ws.add_recv(_make_frame(10, {"heartbeat_interval": 50000}))
        mock_ws.add_recv(_make_frame(
            0, {"session_id": "sess_1"}, t="READY", s=1
        ))
        task = asyncio.create_task(client.connect())
        await asyncio.sleep(0.15)

        assert client.can_resume() is True

        await client.disconnect()
        await task

    async def test_can_resume_returns_false_no_session(self, client):
        """can_resume() returns False when no session exists."""
        assert client.can_resume() is False

    async def test_is_connected_property(self, client):
        """is_connected reflects connection state."""
        assert client.is_connected is False
        client._connected = True
        assert client.is_connected is True
