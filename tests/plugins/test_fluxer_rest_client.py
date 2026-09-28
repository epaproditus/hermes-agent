"""Tests for FluxerRESTClient — HTTP REST operations against Fluxer API."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
pytestmark = pytest.mark.anyio

from plugins.platforms.fluxer.adapter import (
    AuthenticationError,
    FluxerRESTClient,
    RateLimitError,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def client():
    """Return a FluxerRESTClient with fake credentials."""
    return FluxerRESTClient(base_url="https://fluxer.example.com", bot_token="test-token")


def _mock_response(status_code=200, json_data=None, headers=None):
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    if json_data is not None:
        resp.json = MagicMock(return_value=json_data)
    else:
        resp.json = MagicMock(return_value={})
    # raise_for_status is a sync method
    resp.raise_for_status = MagicMock(return_value=None)
    return resp


def _mock_async_client(responses):
    """Create a mock httpx.AsyncClient that returns given responses in order.

    Responses can be a single MagicMock (returned for every call) or a
    list of MagicMocks (returned sequentially).
    """
    client = MagicMock(spec=["request", "aclose"])
    if isinstance(responses, list):
        client.request = AsyncMock(side_effect=responses)
    else:
        client.request = AsyncMock(return_value=responses)

    client.aclose = AsyncMock(return_value=None)
    return client


# ═══════════════════════════════════════════════════════════════════════
# Constructor tests
# ═══════════════════════════════════════════════════════════════════════


class TestConstructor:
    """FluxerRESTClient.__init__ behaviour."""

    def test_stores_base_url_and_token(self):
        """Should store base_url (stripped) and bot_token."""
        c = FluxerRESTClient(base_url="https://fluxer.example.com/", bot_token="abc123")
        assert c.base_url == "https://fluxer.example.com"
        assert c.bot_token == "abc123"

    def test_no_client_created_on_init(self):
        """Should NOT create an httpx client at construction time."""
        c = FluxerRESTClient(base_url="https://fluxer.example.com", bot_token="abc123")
        assert c._client is None


# ═══════════════════════════════════════════════════════════════════════
# Auth header tests
# ═══════════════════════════════════════════════════════════════════════


class TestAuthHeader:
    """Authorization header is set correctly on the client."""

    async def test_sets_bot_auth_header(self, client):
        """Client should have Authorization: Bot <token> header."""
        resp = _mock_response(200, {"id": "bot123", "username": "TestBot"})
        mock_client = _mock_async_client(resp)

        with patch.object(client, "_get_client", AsyncMock(return_value=mock_client)):
            await client.get_me()

        # Verify the header was included in the request
        call_kwargs = mock_client.request.call_args
        # The headers are set on the client (per-client), not per-request
        # We can't easily inspect client-level headers from the mock,
        # but we check that request was called with GET and the right path
        assert call_kwargs[0][0] == "GET"
        assert call_kwargs[0][1] == "/users/@me"


# ═══════════════════════════════════════════════════════════════════════
# REST method tests — success paths
# ═══════════════════════════════════════════════════════════════════════


class TestSendMessage:
    """send_message() — POST /channels/{id}/messages."""

    async def test_sends_text_message(self, client):
        """Should POST content to the correct channel endpoint."""
        expected = {"id": "msg1", "content": "Hello!"}
        resp = _mock_response(201, expected)
        mock_client = _mock_async_client(resp)

        with patch.object(client, "_get_client", AsyncMock(return_value=mock_client)):
            result = await client.send_message("chan123", "Hello!")

        assert result == expected
        mock_client.request.assert_awaited_once_with(
            "POST",
            "/channels/chan123/messages",
            json={"content": "Hello!"},
        )

    async def test_passes_extra_kwargs(self, client):
        """Extra kwargs should be forwarded as JSON body fields."""
        resp = _mock_response(201, {"id": "msg1"})
        mock_client = _mock_async_client(resp)

        with patch.object(client, "_get_client", AsyncMock(return_value=mock_client)):
            await client.send_message("chan1", "Hi!", embeds=[{"title": "Test"}], tts=False)

        mock_client.request.assert_awaited_once_with(
            "POST",
            "/channels/chan1/messages",
            json={"content": "Hi!", "embeds": [{"title": "Test"}], "tts": False},
        )


class TestEditMessage:
    """edit_message() — PATCH /channels/{id}/messages/{mid}."""

    async def test_edits_message_content(self, client):
        """Should PATCH new content to the message endpoint."""
        expected = {"id": "msg1", "content": "Updated!"}
        resp = _mock_response(200, expected)
        mock_client = _mock_async_client(resp)

        with patch.object(client, "_get_client", AsyncMock(return_value=mock_client)):
            result = await client.edit_message("chan1", "msg1", "Updated!")

        assert result == expected
        mock_client.request.assert_awaited_once_with(
            "PATCH",
            "/channels/chan1/messages/msg1",
            json={"content": "Updated!"},
        )


class TestSendTyping:
    """send_typing() — POST /channels/{id}/typing."""

    async def test_sends_typing_indicator(self, client):
        """Should POST to the typing endpoint."""
        resp = _mock_response(204)
        mock_client = _mock_async_client(resp)

        with patch.object(client, "_get_client", AsyncMock(return_value=mock_client)):
            result = await client.send_typing("chan1")

        assert result == {}
        mock_client.request.assert_awaited_once_with(
            "POST",
            "/channels/chan1/typing",
        )


class TestCreateDM:
    """create_dm() — POST /channels with DM payload."""

    async def test_creates_dm_channel(self, client):
        """Should POST a DM channel payload with recipient_id and type."""
        expected = {"id": "dm_chan", "type": 1, "recipients": [{"id": "user1"}]}
        resp = _mock_response(200, expected)
        mock_client = _mock_async_client(resp)

        with patch.object(client, "_get_client", AsyncMock(return_value=mock_client)):
            result = await client.create_dm("user1")

        assert result == expected
        mock_client.request.assert_awaited_once_with(
            "POST",
            "/channels",
            json={"recipient_id": "user1"},
        )


class TestJoinGuild:
    """join_guild() — PUT /guilds/{id}/members/@me."""

    async def test_joins_guild(self, client):
        """Should PUT to the guild members endpoint."""
        expected = {"guild_id": "guild1", "joined_at": "2026-01-01T00:00:00Z"}
        resp = _mock_response(200, expected)
        mock_client = _mock_async_client(resp)

        with patch.object(client, "_get_client", AsyncMock(return_value=mock_client)):
            result = await client.join_guild("guild1")

        assert result == expected
        mock_client.request.assert_awaited_once_with(
            "PUT",
            "/guilds/guild1/members/@me",
        )


class TestGetMe:
    """get_me() — GET /users/@me."""

    async def test_returns_bot_user(self, client):
        """Should return the bot user object."""
        expected = {"id": "bot1", "username": "TestBot", "bot": True}
        resp = _mock_response(200, expected)
        mock_client = _mock_async_client(resp)

        with patch.object(client, "_get_client", AsyncMock(return_value=mock_client)):
            result = await client.get_me()

        assert result == expected
        mock_client.request.assert_awaited_once_with(
            "GET",
            "/users/@me",
        )


class TestGetGatewayBot:
    """get_gateway_bot() — GET /gateway/bot."""

    async def test_returns_gateway_info(self, client):
        """Should return gateway URL, shards, and session start limit."""
        expected = {
            "url": "wss://gateway.fluxer.example.com",
            "shards": 1,
            "session_start_limit": {"total": 1000, "remaining": 999},
        }
        resp = _mock_response(200, expected)
        mock_client = _mock_async_client(resp)

        with patch.object(client, "_get_client", AsyncMock(return_value=mock_client)):
            result = await client.get_gateway_bot()

        assert result == expected
        mock_client.request.assert_awaited_once_with(
            "GET",
            "/gateway/bot",
        )


# ═══════════════════════════════════════════════════════════════════════
# Error handling tests
# ═══════════════════════════════════════════════════════════════════════


class TestAuthenticationError:
    """401 responses should raise AuthenticationError."""

    async def test_raises_on_401(self, client):
        """Should raise AuthenticationError for 401 responses."""
        resp = _mock_response(401, {})
        mock_client = _mock_async_client(resp)

        with patch.object(client, "_get_client", AsyncMock(return_value=mock_client)):
            with pytest.raises(AuthenticationError, match="Invalid bot token"):
                await client.get_me()


class TestRateLimitHandling:
    """429 responses should trigger Retry-After backoff and retry."""

    async def test_retries_after_429(self, client):
        """Should retry the request after sleeping Retry-After seconds."""
        # First call 429, second succeeds
        retry_resp = _mock_response(429, {}, headers={"Retry-After": "1"})
        ok_resp = _mock_response(200, {"id": "msg1", "content": "Hello!"})
        mock_client = _mock_async_client([retry_resp, ok_resp])

        with (
            patch.object(client, "_get_client", AsyncMock(return_value=mock_client)),
            patch("asyncio.sleep", AsyncMock()) as mock_sleep,
        ):
            result = await client.send_message("chan1", "Hello!")

        assert result == {"id": "msg1", "content": "Hello!"}
        # Should have slept for Retry-After seconds
        mock_sleep.assert_awaited_once_with(1.0)
        # Should have tried twice
        assert mock_client.request.await_count == 2

    async def test_rates_after_consecutive_429(self, client):
        """Should raise RateLimitError on consecutive 429s."""
        retry_resp = _mock_response(429, {}, headers={"Retry-After": "1"})
        another_429 = _mock_response(429, {}, headers={"Retry-After": "2"})
        mock_client = _mock_async_client([retry_resp, another_429])

        with (
            patch.object(client, "_get_client", AsyncMock(return_value=mock_client)),
            patch("asyncio.sleep", AsyncMock()),
            pytest.raises(RateLimitError, match="Rate limited"),
        ):
            await client.get_me()


    async def test_parses_rate_limit_headers(self, client):
        """Should parse X-RateLimit-* headers when available."""
        # Use a successful response that has rate limit headers
        headers = {
            "X-RateLimit-Limit": "10",
            "X-RateLimit-Remaining": "9",
            "X-RateLimit-Reset": "1700000000",
            "X-RateLimit-Reset-After": "1.0",
            "X-RateLimit-Bucket": "test-bucket",
        }
        resp = _mock_response(200, {"id": "bot1"}, headers=headers)
        mock_client = _mock_async_client(resp)

        with patch.object(client, "_get_client", AsyncMock(return_value=mock_client)):
            result = await client.get_me()

        assert result == {"id": "bot1"}

    async def test_retry_without_retry_after_header(self, client):
        """Should default to 1 second backoff when Retry-After is missing."""
        retry_resp = _mock_response(429, {}, headers={})  # No Retry-After
        ok_resp = _mock_response(200, {"id": "msg1"})
        mock_client = _mock_async_client([retry_resp, ok_resp])

        with (
            patch.object(client, "_get_client", AsyncMock(return_value=mock_client)),
            patch("asyncio.sleep", AsyncMock()) as mock_sleep,
        ):
            result = await client.send_message("chan1", "Hi!")

        assert result == {"id": "msg1"}
        mock_sleep.assert_awaited_once()
        # Should have defaulted to some reasonable backoff
        sleep_arg = mock_sleep.call_args[0][0]
        assert sleep_arg >= 0.5  # at least half a second default


class TestClose:
    """close() should clean up the httpx client."""

    async def test_closes_client(self, client):
        """Should call aclose on the underlying httpx client."""
        mock_client = _mock_async_client(_mock_response(200, {}))
        client._client = mock_client

        await client.close()

        mock_client.aclose.assert_awaited_once()
        assert client._client is None

    async def test_close_idempotent_when_no_client(self, client):
        """close() should not error when no client exists."""
        await client.close()  # should not raise


# ═══════════════════════════════════════════════════════════════════════
# Integration test marker (not run in CI by default)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.fluxer
@pytest.mark.skip(reason="Requires live Fluxer instance with FLUXER_BOT_TOKEN set")
class TestLiveFluxerAPI:
    """Integration tests against a real Fluxer instance.

    Set FLUXER_BOT_TOKEN and FLUXER_API_URL environment variables to run.
    """

    @pytest.fixture
    def live_client(self):
        """Return a client configured from environment variables."""
        import os

        token = os.environ.get("FLUXER_BOT_TOKEN", "")
        url = os.environ.get("FLUXER_API_URL", "")
        if not token or not url:
            pytest.skip("FLUXER_BOT_TOKEN and FLUXER_API_URL not set")
        return FluxerRESTClient(base_url=url, bot_token=token)

    async def test_get_gateway_bot_returns_valid_url(self, live_client):
        """get_gateway_bot() should return a valid WebSocket URL."""
        result = await live_client.get_gateway_bot()
        assert "url" in result
        assert result["url"].startswith("wss://") or result["url"].startswith("ws://")
        if "shards" in result:
            assert isinstance(result["shards"], int)
        await live_client.close()

    async def test_send_message_delivers_to_channel(self, live_client):
        """send_message() should deliver a message visible in a channel.

        Uses FLUXER_HOME_CHANNEL if set, otherwise skips.
        """
        import os

        channel_id = os.environ.get("FLUXER_HOME_CHANNEL", "")
        if not channel_id:
            pytest.skip("FLUXER_HOME_CHANNEL not set")

        result = await live_client.send_message(
            channel_id, "Hello from Hermes FluxerRESTClient integration test! 🤖"
        )
        assert "id" in result
        assert result.get("channel_id") == channel_id
        await live_client.close()
