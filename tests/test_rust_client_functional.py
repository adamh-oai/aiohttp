import asyncio
import gzip
import io
import ssl
import sys
import zlib
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace

import aiohttp
import pytest
import trustme
from yarl import URL

try:
    try:
        import brotlicffi as brotli
    except ImportError:
        import brotli
except ImportError:  # pragma: no cover
    brotli = None

try:
    if sys.version_info >= (3, 14):
        import compression.zstd as zstandard  # noqa: I900
    else:
        import backports.zstd as zstandard
except ImportError:  # pragma: no cover
    zstandard = None

from aiohttp import ClientSession, FormData, RustClientEngine, SocketTimeoutError, web
from aiohttp.pytest_plugin import AiohttpServer
from aiohttp.test_utils import TestServer

try:
    from aiohttp import _rust_client as _native_client  # noqa: F401
except ImportError:
    pytestmark = pytest.mark.skip(reason="aiohttp._rust_client is not available")


async def test_rust_engine_get_content_length_body(
    aiohttp_server: AiohttpServer,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=b"native body")

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.get(server.make_url("/"))
        assert response.status == 200
        assert await response.read() == b"native body"


async def test_rust_engine_get_https_body_with_default_verification(
    aiohttp_server: AiohttpServer,
    ssl_ctx: ssl.SSLContext,
    tls_ca_certificate_pem_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=b"native tls")

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app, ssl=ssl_ctx)
    monkeypatch.setenv("SSL_CERT_FILE", tls_ca_certificate_pem_path)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.get(server.make_url("/"))
        assert response.status == 200
        assert await response.read() == b"native tls"


async def test_rust_engine_reports_https_certificate_errors(
    aiohttp_server: AiohttpServer,
    ssl_ctx: ssl.SSLContext,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=b"native tls")

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app, ssl=ssl_ctx)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        with pytest.raises(aiohttp.ClientConnectorCertificateError):
            await session.get(server.make_url("/"))


async def test_rust_engine_get_https_body_without_verification(
    aiohttp_server: AiohttpServer,
    ssl_ctx: ssl.SSLContext,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=b"native tls")

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app, ssl=ssl_ctx)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.get(server.make_url("/"), ssl=False)
        assert response.status == 200
        assert await response.read() == b"native tls"


async def test_rust_engine_close_waits_for_native_connections() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class NativeConnection:
        async def wait_closed(self) -> None:
            close_started.set()
            await allow_close.wait()

    connection = NativeConnection()
    engine = RustClientEngine()
    engine._connections.add(connection)

    close_task = asyncio.create_task(engine.close())
    await close_started.wait()
    assert not close_task.done()

    allow_close.set()
    await close_task


async def test_rust_engine_waits_for_close_after_fingerprint_mismatch() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class NativeConnection:
        closed = False

        def peer_certificate_der(self) -> bytes:
            return b"certificate"

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            close_started.set()
            await allow_close.wait()

    connection = NativeConnection()
    engine = RustClientEngine()
    engine._connections.add(connection)
    request = SimpleNamespace(url=URL("https://example.com"))

    mismatch_task = asyncio.create_task(
        engine._check_fingerprint(request, connection, b"\x00" * 32)
    )
    await close_started.wait()
    assert not mismatch_task.done()

    allow_close.set()
    with pytest.raises(aiohttp.ServerFingerprintMismatch):
        await mismatch_task
    assert connection not in engine._connections


async def test_rust_engine_uses_server_hostname_for_tls_identity(
    aiohttp_server: AiohttpServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ca = trustme.CA()
    leaf = ca.issue_cert("localhost")
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    leaf.configure_cert(server_context)

    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=b"native tls")

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app, ssl=server_context)

    with ca.cert_pem.tempfile() as ca_cert_pem:
        monkeypatch.setenv("SSL_CERT_FILE", ca_cert_pem)
        async with ClientSession(client_engine=RustClientEngine()) as session:
            response = await session.get(
                server.make_url("/"), server_hostname="localhost"
            )
            assert response.status == 200
            assert await response.read() == b"native tls"


async def test_rust_engine_get_https_body_with_fingerprint(
    aiohttp_server: AiohttpServer,
    ssl_ctx: ssl.SSLContext,
    tls_certificate_fingerprint_sha256: bytes,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=b"native tls")

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app, ssl=ssl_ctx)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.get(
            server.make_url("/"),
            ssl=aiohttp.Fingerprint(tls_certificate_fingerprint_sha256),
        )
        assert response.status == 200
        assert await response.read() == b"native tls"


async def test_rust_engine_reports_fingerprint_mismatch(
    aiohttp_server: AiohttpServer,
    ssl_ctx: ssl.SSLContext,
    tls_certificate_fingerprint_sha256: bytes,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=b"native tls")

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app, ssl=ssl_ctx)
    bad_fingerprint = b"\x00" * len(tls_certificate_fingerprint_sha256)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        with pytest.raises(aiohttp.ServerFingerprintMismatch) as exc_info:
            await session.get(
                server.make_url("/"),
                ssl=aiohttp.Fingerprint(bad_fingerprint),
            )

    assert exc_info.value.expected == bad_fingerprint
    assert exc_info.value.got == tls_certificate_fingerprint_sha256


async def test_rust_engine_waits_for_100_continue_before_uploading(
    aiohttp_server: AiohttpServer,
) -> None:
    body_started = asyncio.Event()
    expect_seen = asyncio.Event()

    async def body() -> AsyncIterator[bytes]:
        body_started.set()
        yield b"native payload"

    async def handler(request: web.Request) -> web.Response:
        assert await request.read() == b"native payload"
        return web.Response(body=b"ok")

    async def expect_handler(request: web.Request) -> None:
        assert request.headers["Expect"].lower() == "100-continue"
        assert not body_started.is_set()
        expect_seen.set()
        assert request.transport is not None
        request.transport.write(b"HTTP/1.1 100 Continue\r\n\r\n")

    app = web.Application()
    app.router.add_post("/", handler, expect_handler=expect_handler)
    server = await aiohttp_server(app)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.post(
            server.make_url("/"), data=body(), expect100=True
        )
        assert response.status == 200
        assert await response.read() == b"ok"

    assert expect_seen.is_set()
    assert body_started.is_set()


async def test_rust_engine_skips_upload_when_expect_gets_final_response(
    aiohttp_server: AiohttpServer,
) -> None:
    body_started = asyncio.Event()

    async def body() -> AsyncIterator[bytes]:
        body_started.set()
        yield b"native payload"

    async def handler(request: web.Request) -> web.Response:
        raise AssertionError("handler should not run after rejected expectation")

    async def expect_handler(request: web.Request) -> None:
        raise web.HTTPForbidden()

    app = web.Application()
    app.router.add_post("/", handler, expect_handler=expect_handler)
    server = await aiohttp_server(app)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.post(
            server.make_url("/"), data=body(), expect100=True
        )
        assert response.status == 403

    assert not body_started.is_set()


async def test_rust_engine_reuses_connection_after_body_eof(
    aiohttp_server: AiohttpServer,
) -> None:
    transports: list[object] = []

    async def handler(request: web.Request) -> web.Response:
        transports.append(request.transport)
        return web.Response(body=b"native body")

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        first = await session.get(server.make_url("/"))
        assert await first.read() == b"native body"
        second = await session.get(server.make_url("/"))
        assert await second.read() == b"native body"

    assert len(transports) == 2
    assert transports[0] is transports[1]


async def test_rust_engine_get_chunked_body(aiohttp_server: AiohttpServer) -> None:
    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"native ")
        await response.write(b"chunks")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.get(server.make_url("/"))
        assert response.status == 200
        assert await response.read() == b"native chunks"


async def test_rust_engine_rejects_chunk_boundary_reads(
    aiohttp_server: AiohttpServer,
) -> None:
    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"native ")
        await response.write(b"chunks")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.get(server.make_url("/"))
        with pytest.raises(NotImplementedError, match="chunk-boundary reads"):
            await response.content.readchunk()


async def test_rust_engine_streams_response_body_before_eof(
    aiohttp_server: AiohttpServer,
) -> None:
    release_tail = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"first")
        await release_tail.wait()
        await response.write(b"second")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await asyncio.wait_for(session.get(server.make_url("/")), timeout=1)
        assert await response.content.readany() == b"first"
        release_tail.set()
        assert await response.read() == b"second"


async def test_rust_engine_times_out_waiting_for_response_headers(
    aiohttp_server: AiohttpServer,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        await asyncio.sleep(0.1)
        return web.Response(body=b"late")

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    timeout = aiohttp.ClientTimeout(total=None, sock_read=0.01)
    async with ClientSession(client_engine=RustClientEngine()) as session:
        with pytest.raises(SocketTimeoutError):
            await session.get(server.make_url("/"), timeout=timeout)


async def test_rust_engine_times_out_between_response_chunks(
    aiohttp_server: AiohttpServer,
) -> None:
    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"first")
        await asyncio.sleep(0.1)
        await response.write(b"second")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    timeout = aiohttp.ClientTimeout(total=None, sock_read=0.01)
    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.get(server.make_url("/"), timeout=timeout)
        assert await response.content.readany() == b"first"
        with pytest.raises(SocketTimeoutError):
            await response.content.readany()


async def test_rust_engine_total_timeout_covers_body_reads(
    aiohttp_server: AiohttpServer,
) -> None:
    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        await asyncio.sleep(0.1)
        await response.write(b"late")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    timeout = aiohttp.ClientTimeout(total=0.01, sock_read=None)
    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.get(server.make_url("/"), timeout=timeout)
        with pytest.raises(asyncio.TimeoutError):
            await response.read()


async def test_rust_engine_does_not_start_read_timeout_during_upload(
    aiohttp_server: AiohttpServer,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=await request.read())

    async def body() -> AsyncIterator[bytes]:
        await asyncio.sleep(0.05)
        yield b"native upload"

    app = web.Application()
    app.router.add_post("/", handler)
    server = await aiohttp_server(app)

    timeout = aiohttp.ClientTimeout(total=None, sock_read=0.01)
    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.post(server.make_url("/"), data=body(), timeout=timeout)
        assert await response.read() == b"native upload"


async def test_rust_engine_post_buffered_body(aiohttp_server: AiohttpServer) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=await request.read())

    app = web.Application()
    app.router.add_post("/", handler)
    server = await aiohttp_server(app)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.post(server.make_url("/"), data=b"native upload")
        assert response.status == 200
        assert await response.read() == b"native upload"


async def test_rust_engine_post_file_body(aiohttp_server: AiohttpServer) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=await request.read())

    app = web.Application()
    app.router.add_post("/", handler)
    server = await aiohttp_server(app)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.post(
            server.make_url("/"),
            data=io.BytesIO(b"native file"),
        )
        assert response.status == 200
        assert await response.read() == b"native file"


async def test_rust_engine_post_multipart_body(aiohttp_server: AiohttpServer) -> None:
    async def handler(request: web.Request) -> web.Response:
        post = await request.post()
        uploaded = post["file"]
        assert hasattr(uploaded, "file")
        uploaded_body = await asyncio.to_thread(uploaded.file.read)
        return web.Response(
            body=f"{post['field']}:{uploaded_body.decode()}".encode()
        )

    app = web.Application()
    app.router.add_post("/", handler)
    server = await aiohttp_server(app)

    form = FormData()
    form.add_field("field", "value")
    form.add_field("file", io.BytesIO(b"native multipart"), filename="data.txt")

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.post(server.make_url("/"), data=form)
        assert response.status == 200
        assert await response.read() == b"value:native multipart"


async def test_rust_engine_post_async_iterable_body(
    aiohttp_server: AiohttpServer,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=await request.read())

    async def body() -> AsyncIterator[bytes]:
        yield b"native "
        yield b"stream"

    app = web.Application()
    app.router.add_post("/", handler)
    server = await aiohttp_server(app)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.post(server.make_url("/"), data=body())
        assert response.status == 200
        assert await response.read() == b"native stream"


@pytest.mark.parametrize("compression", ("deflate", "gzip"))
async def test_rust_engine_compresses_request_body(
    aiohttp_server: AiohttpServer,
    compression: str,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        assert request.headers["Content-Encoding"] == compression
        assert request.headers["Transfer-Encoding"].lower() == "chunked"
        return web.Response(body=await request.read())

    app = web.Application()
    app.router.add_post("/", handler)
    server = await aiohttp_server(app)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.post(
            server.make_url("/"),
            data=b"native compressed upload",
            compress=compression,
        )
        assert response.status == 200
        assert await response.read() == b"native compressed upload"


def _compress_raw_deflate(body: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return compressor.compress(body) + compressor.flush()


@pytest.mark.parametrize(
    ("encoding", "compress"),
    [
        ("gzip", gzip.compress),
        ("deflate", zlib.compress),
        ("raw-deflate", _compress_raw_deflate),
        pytest.param(
            "br",
            lambda body: brotli.compress(body),  # type: ignore[union-attr]
            marks=pytest.mark.skipif(brotli is None, reason="brotli is not installed"),
        ),
        pytest.param(
            "zstd",
            lambda body: zstandard.compress(body),  # type: ignore[union-attr]
            marks=pytest.mark.skipif(zstandard is None, reason="zstd is not installed"),
        ),
    ],
)
async def test_rust_engine_decompresses_response(
    aiohttp_server: AiohttpServer,
    encoding: str,
    compress: Callable[[bytes], bytes],
) -> None:
    compressed = compress(b"compressed body")
    header_encoding = "deflate" if encoding == "raw-deflate" else encoding

    async def handler(request: web.Request) -> web.Response:
        return web.Response(
            body=compressed,
            headers={"Content-Encoding": header_encoding},
        )

    app = web.Application()
    app.router.add_get("/", handler)
    server: TestServer = await aiohttp_server(app)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.get(server.make_url("/"))
        assert await response.read() == b"compressed body"
        assert response.content.total_raw_bytes == len(compressed)


async def test_rust_engine_preserves_compressed_response_when_disabled(
    aiohttp_server: AiohttpServer,
) -> None:
    compressed = gzip.compress(b"compressed body")

    async def handler(request: web.Request) -> web.Response:
        return web.Response(
            body=compressed,
            headers={"Content-Encoding": "gzip"},
        )

    app = web.Application()
    app.router.add_get("/", handler)
    server: TestServer = await aiohttp_server(app)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.get(server.make_url("/"), auto_decompress=False)
        assert await response.read() == compressed
        assert response.content.total_raw_bytes == len(compressed)


@pytest.mark.skipif(zstandard is None, reason="zstd is not installed")
async def test_rust_engine_decompresses_multiframe_zstd_response(
    aiohttp_server: AiohttpServer,
) -> None:
    assert zstandard is not None
    compressed = zstandard.compress(b"left ") + zstandard.compress(b"right")

    async def handler(request: web.Request) -> web.Response:
        return web.Response(
            body=compressed,
            headers={"Content-Encoding": "zstd"},
        )

    app = web.Application()
    app.router.add_get("/", handler)
    server: TestServer = await aiohttp_server(app)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.get(server.make_url("/"))
        assert await response.read() == b"left right"
        assert response.content.total_raw_bytes == len(compressed)


async def test_rust_engine_reports_bad_compressed_payload(
    aiohttp_server: AiohttpServer,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(
            body=b"not gzip",
            headers={"Content-Encoding": "gzip"},
        )

    app = web.Application()
    app.router.add_get("/", handler)
    server: TestServer = await aiohttp_server(app)

    async with ClientSession(client_engine=RustClientEngine()) as session:
        response = await session.get(server.make_url("/"))
        with pytest.raises(aiohttp.ClientPayloadError):
            await response.read()
