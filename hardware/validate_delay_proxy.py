"""Offline loopback checks; every listener uses an ephemeral local port.

Run: python -B -m pytest hardware/validate_delay_proxy.py -q
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware.delay_proxy import DelayProxy  # noqa: E402


def local_port(server: asyncio.AbstractServer) -> int:
    return int(server.sockets[0].getsockname()[1])


def test_bidirectional_delay_half_close_and_metadata_only() -> None:
    async def check() -> None:
        payload = b"Authorization: sentinel-secret\r\n" + bytes(range(256)) * 600
        events = []

        async def respond(reader, writer):
            received = await reader.read()
            assert received == payload
            writer.write(received[::-1])
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        upstream = await asyncio.start_server(respond, "127.0.0.1", 0)
        proxy = DelayProxy("127.0.0.1", local_port(upstream), 60, log=events.append,
                           chunk_bytes=1024, max_buffer_chunks=8)
        server = await proxy.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", local_port(server))
            started = asyncio.get_running_loop().time()
            writer.write(payload)
            await writer.drain()
            writer.write_eof()
            received = await asyncio.wait_for(reader.read(), timeout=5)
            elapsed = asyncio.get_running_loop().time() - started
            assert received == payload[::-1]
            assert elapsed >= 0.11  # Both directions must receive their delay.
            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(0.02)
        finally:
            await proxy.close()
            upstream.close()
            await upstream.wait_closed()
        closed = [event for event in events if event["event"] == "connection_closed"]
        assert len(closed) == 1
        for direction in ("client_to_upstream", "upstream_to_client"):
            assert closed[0][direction]["received_bytes"] == len(payload)
            assert closed[0][direction]["forwarded_bytes"] == len(payload)
        assert "sentinel-secret" not in json.dumps(events)
        assert "Authorization" not in json.dumps(events)

    asyncio.run(check())


def test_streaming_chunk_deadlines_overlap_instead_of_serial_delay() -> None:
    async def check() -> None:
        delay = 0.20
        count = 8
        sent_at = []
        received_at = []
        loop = asyncio.get_running_loop()

        async def stream(reader, writer):
            assert await reader.readexactly(1) == b"?"
            for index in range(count):
                writer.write(bytes([index]))
                await writer.drain()
                sent_at.append(loop.time())
                await asyncio.sleep(0.025)
            writer.close()
            await writer.wait_closed()

        upstream = await asyncio.start_server(stream, "127.0.0.1", 0)
        proxy = DelayProxy("127.0.0.1", local_port(upstream), delay * 1000,
                           log=lambda event: None, chunk_bytes=1)
        server = await proxy.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", local_port(server))
            started = loop.time()
            writer.write(b"?")
            await writer.drain()
            writer.write_eof()
            for index in range(count):
                assert await asyncio.wait_for(reader.readexactly(1), timeout=3) == bytes([index])
                received_at.append(loop.time())
            assert await asyncio.wait_for(reader.read(), timeout=2) == b""
            writer.close()
            await writer.wait_closed()
            assert sent_at[0] - started >= delay - 0.015
            assert all(received - sent >= delay - 0.015
                       for sent, received in zip(sent_at, received_at))
            # Serial per-chunk sleeping would stretch this span to >=1.4 s.
            assert received_at[-1] - received_at[0] < 0.65
            assert received_at[-1] - sent_at[-1] < delay + 0.35
        finally:
            await proxy.close()
            upstream.close()
            await upstream.wait_closed()

    asyncio.run(check())


def test_zero_delay_concurrent_connections_are_isolated() -> None:
    async def check() -> None:
        events = []

        async def echo(reader, writer):
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
            writer.close()
            await writer.wait_closed()

        upstream = await asyncio.start_server(echo, "127.0.0.1", 0)
        proxy = DelayProxy("127.0.0.1", local_port(upstream), 0, log=events.append)
        server = await proxy.start()

        async def client(number):
            reader, writer = await asyncio.open_connection("127.0.0.1", local_port(server))
            payload = bytes([number]) * (1000 + number)
            writer.write(payload)
            await writer.drain()
            writer.write_eof()
            assert await asyncio.wait_for(reader.read(), timeout=2) == payload
            writer.close()
            await writer.wait_closed()

        try:
            await asyncio.gather(*(client(index) for index in range(4)))
        finally:
            await proxy.close()
            upstream.close()
            await upstream.wait_closed()
        closed = [event for event in events if event["event"] == "connection_closed"]
        assert len(closed) == 4
        assert len({event["connection_id"] for event in closed}) == 4
        assert sum(event["client_to_upstream"]["forwarded_bytes"] for event in closed) == 4006

    asyncio.run(check())


def test_unreachable_upstream_closes_client_and_logs_error() -> None:
    async def check() -> None:
        events = []
        placeholder = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
        unused_port = local_port(placeholder)
        placeholder.close()
        await placeholder.wait_closed()
        proxy = DelayProxy("127.0.0.1", unused_port, 0, log=events.append)
        server = await proxy.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", local_port(server))
            assert await asyncio.wait_for(reader.read(), timeout=2) == b""
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.close()
        closed = [event for event in events if event["event"] == "connection_closed"]
        assert len(closed) == 1
        assert closed[0]["status"] == "connection_error"
        assert closed[0]["error_type"] is not None

    asyncio.run(check())


def test_shutdown_cancels_pending_delays_and_retains_counters() -> None:
    async def check() -> None:
        events = []
        upstream_closed = asyncio.Event()

        async def consume(reader, writer):
            await reader.read()
            writer.close()
            await writer.wait_closed()
            upstream_closed.set()

        upstream = await asyncio.start_server(consume, "127.0.0.1", 0)
        proxy = DelayProxy("127.0.0.1", local_port(upstream), 2000, log=events.append)
        server = await proxy.start()
        reader, writer = await asyncio.open_connection("127.0.0.1", local_port(server))
        try:
            writer.write(b"pending")
            await writer.drain()
            await asyncio.sleep(0.03)
            await asyncio.wait_for(proxy.close(), timeout=0.5)
            assert await asyncio.wait_for(reader.read(), timeout=0.5) == b""
            await asyncio.wait_for(upstream_closed.wait(), timeout=0.5)
        finally:
            writer.close()
            await writer.wait_closed()
            await proxy.close()
            upstream.close()
            await upstream.wait_closed()
        closed = [event for event in events if event["event"] == "connection_closed"]
        assert len(closed) == 1
        assert closed[0]["status"] == "cancelled"
        assert closed[0]["client_to_upstream"]["received_bytes"] == 7
        assert closed[0]["client_to_upstream"]["forwarded_bytes"] == 0

    asyncio.run(check())


@pytest.mark.parametrize("delay", [-1, float("inf"), float("nan")])
def test_reject_invalid_delays(delay: float) -> None:
    with pytest.raises(ValueError):
        DelayProxy("127.0.0.1", 12345, delay)
