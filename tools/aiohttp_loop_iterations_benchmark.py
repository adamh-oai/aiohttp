#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import statistics
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Literal, cast

from aiohttp import ClientSession, RustClientEngine, TCPConnector
from aiohttp.client_engine import AsyncioClientEngine, ClientEngine

Method = Literal["GET", "POST"]
PoolState = Literal["cold", "warm"]
POST_BODY = b"x" * 128


class BenchmarkHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        self._send_response()

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length:
            self.rfile.read(content_length)
        self._send_response()

    def _send_response(self) -> None:
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        del args


@contextmanager
def local_http_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), BenchmarkHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class LoopIterationCounter:
    def __init__(self) -> None:
        self.count = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._run_once: Callable[[], None] | None = None

    def __enter__(self) -> LoopIterationCounter:
        loop = asyncio.get_running_loop()
        run_once = cast(Callable[[], None], getattr(loop, "_run_once"))

        def counted_run_once() -> None:
            self.count += 1
            run_once()

        setattr(loop, "_run_once", counted_run_once)
        self._loop = loop
        self._run_once = run_once
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        assert self._loop is not None
        assert self._run_once is not None
        setattr(self._loop, "_run_once", self._run_once)


@dataclass(frozen=True)
class Scenario:
    engine_name: str
    method: Method
    pool_state: PoolState


def make_asyncio_engine() -> ClientEngine:
    return AsyncioClientEngine(TCPConnector())


def make_rust_engine() -> ClientEngine:
    return RustClientEngine()


def has_available_connection(session: ClientSession) -> bool:
    engine = session.client_engine
    assert engine is not None

    if isinstance(engine, AsyncioClientEngine):
        return any(engine.connector._conns.values())  # pyright: ignore[reportPrivateUsage]
    if isinstance(engine, RustClientEngine):
        return any(engine._available_connections.values())  # pyright: ignore[reportPrivateUsage]

    raise TypeError(f"Unsupported engine type: {type(engine)!r}")


async def make_request(session: ClientSession, method: Method, url: str) -> None:
    kwargs = {"data": POST_BODY} if method == "POST" else {}
    async with session.request(method, url, **kwargs) as response:
        assert response.status == 200
        assert await response.read() == b"ok"


async def measure_one(
    *,
    engine_factory: Callable[[], ClientEngine],
    method: Method,
    pool_state: PoolState,
    url: str,
) -> int:
    async with ClientSession(client_engine=engine_factory()) as session:
        if pool_state == "warm":
            await make_request(session, method, url)
            assert has_available_connection(session)
        else:
            assert not has_available_connection(session)

        with LoopIterationCounter() as counter:
            await make_request(session, method, url)

        return counter.count


async def run_scenario(
    *,
    engine_factory: Callable[[], ClientEngine],
    scenario: Scenario,
    url: str,
    warmups: int,
    repeats: int,
) -> list[int]:
    for _ in range(warmups):
        await measure_one(
            engine_factory=engine_factory,
            method=scenario.method,
            pool_state=scenario.pool_state,
            url=url,
        )

    return [
        await measure_one(
            engine_factory=engine_factory,
            method=scenario.method,
            pool_state=scenario.pool_state,
            url=url,
        )
        for _ in range(repeats)
    ]


def percentile(samples: list[int], pct: float) -> float:
    if len(samples) == 1:
        return float(samples[0])
    return statistics.quantiles(samples, n=100, method="inclusive")[int(pct) - 1]


async def run_benchmark(*, warmups: int, repeats: int) -> None:
    engine_factories = {
        "asyncio": make_asyncio_engine,
        "rust": make_rust_engine,
    }
    scenarios = [
        Scenario(engine_name, method, pool_state)
        for engine_name in engine_factories
        for method in ("GET", "POST")
        for pool_state in ("cold", "warm")
    ]

    with local_http_server() as url:
        results = {
            scenario: await run_scenario(
                engine_factory=engine_factories[scenario.engine_name],
                scenario=scenario,
                url=url,
                warmups=warmups,
                repeats=repeats,
            )
            for scenario in scenarios
        }

    print(f"warmups={warmups} repeats={repeats}")
    print()
    print("engine   method  pool  min  median  mean   p95  samples")
    print("-------  ------  ----  ---  ------  -----  ---  -------")
    for scenario in scenarios:
        samples = results[scenario]
        print(
            f"{scenario.engine_name:<7}  "
            f"{scenario.method:<6}  "
            f"{scenario.pool_state:<4}  "
            f"{min(samples):>3}  "
            f"{statistics.median(samples):>6.1f}  "
            f"{statistics.mean(samples):>5.1f}  "
            f"{percentile(samples, 95):>3.0f}  "
            f"{samples}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare aiohttp client engines by asyncio loop iterations per request."
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(run_benchmark(warmups=args.warmups, repeats=args.repeats))


if __name__ == "__main__":
    main()
