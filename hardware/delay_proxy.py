"""Apply symmetric software transport delay to a plain TCP connection.

Every received chunk is forwarded no earlier than its own receive time plus
the configured one-way delay. Reading continues while older chunks await
their deadlines, so streaming chunks do not each incur a serial sleep.

This is software transport emulation, not a measured WAN or packet-level
network emulator. TCP chunk boundaries depend on buffering. Congestion,
event-loop scheduling, and socket backpressure can add delay. The proxy does
not parse HTTP and never logs request or response contents.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from dataclasses import asdict, dataclass
import json
import math
import signal
import time
from typing import Callable, Dict, Optional, Set


@dataclass
class DirectionCounters:
    received_chunks: int = 0
    received_bytes: int = 0
    forwarded_chunks: int = 0
    forwarded_bytes: int = 0


def emit_json(record: Dict[str, object]) -> None:
    """Write connection metadata only; payload bytes never enter a record."""
    print(json.dumps(record, sort_keys=True), flush=True)


class DelayProxy:
    """A bounded-buffer, bidirectional TCP delay proxy with half-close support."""

    def __init__(
        self,
        upstream_host: str,
        upstream_port: int,
        one_way_delay_ms: float,
        log: Callable[[Dict[str, object]], None] = emit_json,
        chunk_bytes: int = 65536,
        max_buffer_chunks: int = 64,
        connect_timeout_s: float = 10.0,
    ) -> None:
        if not math.isfinite(one_way_delay_ms) or one_way_delay_ms < 0:
            raise ValueError("one_way_delay_ms must be finite and nonnegative")
        if not 1 <= upstream_port <= 65535:
            raise ValueError("upstream_port must lie in 1..65535")
        if chunk_bytes < 1 or max_buffer_chunks < 1:
            raise ValueError("chunk and queue capacities must be positive")
        if not math.isfinite(connect_timeout_s) or connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be finite and positive")
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.delay_s = float(one_way_delay_ms) / 1000.0
        self.chunk_bytes = chunk_bytes
        self.max_buffer_chunks = max_buffer_chunks
        self.connect_timeout_s = connect_timeout_s
        self.log = log
        self.server: Optional[asyncio.AbstractServer] = None
        self._handlers: Set[asyncio.Task] = set()
        self._connection_count = 0

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> asyncio.AbstractServer:
        if self.server is not None:
            raise RuntimeError("proxy is already started")
        self.server = await asyncio.start_server(self._handle_connection, host, port)
        self.log({
            "event": "proxy_started",
            "wall_time_ns": time.time_ns(),
            "listeners": [socket.getsockname() for socket in self.server.sockets],
            "upstream": [self.upstream_host, self.upstream_port],
            "one_way_delay_ms": self.delay_s * 1000.0,
            "emulation": "software_transport_fixed_chunk_deadline",
            "chunk_bytes": self.chunk_bytes,
            "max_buffer_chunks_per_direction": self.max_buffer_chunks,
            "payload_logging": False,
        })
        return self.server

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
        handlers = list(self._handlers)
        for task in handlers:
            task.cancel()
        if handlers:
            await asyncio.gather(*handlers, return_exceptions=True)
        # Python 3.12 waits for active connections as well as the listener.
        # Cancel handlers first so their finally blocks close the transports.
        if self.server is not None:
            await self.server.wait_closed()

    async def _forward_direction(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        counters: DirectionCounters,
    ) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_buffer_chunks)

        async def receive() -> None:
            while True:
                data = await reader.read(self.chunk_bytes)
                # Capture the deadline before waiting for queue capacity.
                deadline = loop.time() + self.delay_s
                if data:
                    counters.received_chunks += 1
                    counters.received_bytes += len(data)
                await queue.put((deadline, data))
                if not data:
                    return

        async def send() -> None:
            while True:
                deadline, data = await queue.get()
                remaining = deadline - loop.time()
                if remaining > 0:
                    await asyncio.sleep(remaining)
                if not data:
                    # Propagate FIN after the queued data while retaining the
                    # opposite direction, e.g. read-to-EOF request protocols.
                    if writer.can_write_eof():
                        writer.write_eof()
                        await writer.drain()
                    return
                writer.write(data)
                counters.forwarded_chunks += 1
                counters.forwarded_bytes += len(data)
                await writer.drain()

        tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _handle_connection(
        self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter
    ) -> None:
        handler = asyncio.current_task()
        self._handlers.add(handler)
        self._connection_count += 1
        connection_id = self._connection_count
        started = time.monotonic()
        upstream_writer: Optional[asyncio.StreamWriter] = None
        counters = {
            "client_to_upstream": DirectionCounters(),
            "upstream_to_client": DirectionCounters(),
        }
        tasks = []
        status = "completed"
        error_type = None
        self.log({
            "event": "connection_opened", "connection_id": connection_id,
            "wall_time_ns": time.time_ns(),
            "client": client_writer.get_extra_info("peername"),
        })
        try:
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(self.upstream_host, self.upstream_port),
                timeout=self.connect_timeout_s,
            )
            tasks = [
                asyncio.create_task(self._forward_direction(
                    client_reader, upstream_writer, counters["client_to_upstream"]
                )),
                asyncio.create_task(self._forward_direction(
                    upstream_reader, client_writer, counters["upstream_to_client"]
                )),
            ]
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except (OSError, asyncio.TimeoutError) as exc:
            status = "connection_error"
            # Exception messages are deliberately excluded from metadata.
            error_type = type(exc).__name__
        finally:
            async def finish() -> None:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                writers = [client_writer]
                if upstream_writer is not None:
                    writers.append(upstream_writer)
                for writer in writers:
                    writer.close()
                for writer in writers:
                    with contextlib.suppress(OSError, asyncio.TimeoutError):
                        await asyncio.wait_for(writer.wait_closed(), timeout=2.0)
                self.log({
                    "event": "connection_closed", "connection_id": connection_id,
                    "wall_time_ns": time.time_ns(), "status": status,
                    "error_type": error_type,
                    "elapsed_ms": (time.monotonic() - started) * 1000.0,
                    **{name: asdict(value) for name, value in counters.items()},
                })

            # Shutdown can arrive while this handler is already cleaning up.
            # Finish closing sockets and emitting counters even in that race.
            cleanup = asyncio.create_task(finish())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            finally:
                self._handlers.discard(handler)


async def run_proxy(args: argparse.Namespace) -> None:
    proxy = DelayProxy(args.upstream_host, args.upstream_port, args.one_way_delay_ms)
    server = await proxy.start(args.listen_host, args.listen_port)
    loop = asyncio.get_running_loop()
    stopped = asyncio.Event()
    installed_signals = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stopped.set)
        except NotImplementedError:
            continue
        installed_signals.append(signum)
    try:
        async with server:
            await stopped.wait()
    finally:
        await proxy.close()
        for signum in installed_signals:
            loop.remove_signal_handler(signum)
        emit_json({"event": "proxy_stopped", "wall_time_ns": time.time_ns()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--upstream-host", required=True)
    parser.add_argument("--upstream-port", type=int, required=True)
    parser.add_argument("--one-way-delay-ms", type=float, required=True)
    args = parser.parse_args()
    if not 0 <= args.listen_port <= 65535:
        parser.error("listen-port must lie in 0..65535")
    try:
        asyncio.run(run_proxy(args))
    except KeyboardInterrupt:
        pass
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
