"""Behavioral conformance tests shared by the client engine implementations."""

import asyncio
import gzip
import io
import ssl
from collections.abc import AsyncIterator
from typing import Literal

import pytest

import aiohttp
from aiohttp import (
    BasicAuth,
    ClientHandlerType,
    ClientRequest,
    ClientResponse,
    ClientSession,
    FormData,
    hdrs,
    web,
)
from aiohttp.pytest_plugin import AiohttpServer

try:
    from aiohttp_rs import RustClientEngine
    from aiohttp_rs import _rust_client as _native_client  # noqa: F401
except ImportError:
    RustClientEngine = None  # type: ignore[assignment, misc]
    _RUST_AVAILABLE = False
else:
    _RUST_AVAILABLE = True


ClientEngineName = Literal["asyncio", "rust"]


@pytest.fixture(params=("asyncio", "rust"))
def client_engine_name(request: pytest.FixtureRequest) -> ClientEngineName:
    if request.param == "rust" and not _RUST_AVAILABLE:
        pytest.skip("aiohttp_rs._rust_client is not available")
    return request.param


def make_session(
    client_engine_name: ClientEngineName, **kwargs: object
) -> ClientSession:
    if client_engine_name == "rust":
        assert RustClientEngine is not None
        return ClientSession(client_engine=RustClientEngine(), **kwargs)
    return ClientSession(**kwargs)


async def test_redirect_history_and_cookie_jar(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
    async def redirect_handler(request: web.Request) -> web.Response:
        response = web.Response(status=302, headers={hdrs.LOCATION: "/final"})
        response.set_cookie("seen", "1")
        return response

    async def final_handler(request: web.Request) -> web.Response:
        assert request.cookies["seen"] == "1"
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/start", redirect_handler)
    app.router.add_get("/final", final_handler)
    server = await aiohttp_server(app)

    async with make_session(
        client_engine_name,
        cookie_jar=aiohttp.CookieJar(unsafe=True),
    ) as session:
        response = await session.get(server.make_url("/start"))
        assert response.status == 200
        assert [item.status for item in response.history] == [302]
        assert response.url.path == "/final"
        assert await response.text() == "ok"


async def test_session_auth_and_skip_auto_headers(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        assert request.headers[hdrs.AUTHORIZATION] == "Basic bG9naW46cGFzcw=="
        assert hdrs.USER_AGENT not in request.headers
        return web.Response()

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    async with make_session(
        client_engine_name,
        auth=BasicAuth("login", "pass"),
    ) as session:
        response = await session.get(
            server.make_url("/"), skip_auto_headers={hdrs.USER_AGENT}
        )
        assert response.status == 200


async def test_client_middleware_mutates_request(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(text=request.headers["X-From-Middleware"])

    async def middleware(
        request: ClientRequest, handler: ClientHandlerType
    ) -> ClientResponse:
        request.headers["X-From-Middleware"] = "present"
        return await handler(request)

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    async with make_session(
        client_engine_name,
        middlewares=(middleware,),
    ) as session:
        response = await session.get(server.make_url("/"))
        assert await response.text() == "present"


async def test_request_middlewares_override_session_middlewares(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(text=request.headers.get(hdrs.AUTHORIZATION, "none"))

    async def session_middleware(
        request: ClientRequest, handler: ClientHandlerType
    ) -> ClientResponse:
        request.headers[hdrs.AUTHORIZATION] = "Bearer session"
        return await handler(request)

    async def request_middleware(
        request: ClientRequest, handler: ClientHandlerType
    ) -> ClientResponse:
        request.headers[hdrs.AUTHORIZATION] = "Bearer request"
        return await handler(request)

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    async with make_session(
        client_engine_name,
        middlewares=(session_middleware,),
    ) as session:
        session_response = await session.get(server.make_url("/"))
        assert await session_response.text() == "Bearer session"

        request_response = await session.get(
            server.make_url("/"), middlewares=(request_middleware,)
        )
        assert await request_response.text() == "Bearer request"

        disabled_response = await session.get(server.make_url("/"), middlewares=())
        assert await disabled_response.text() == "none"


async def test_raise_for_status(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        raise web.HTTPNotFound()

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    async with make_session(
        client_engine_name,
        raise_for_status=True,
    ) as session:
        with pytest.raises(aiohttp.ClientResponseError):
            await session.get(server.make_url("/"))


async def test_request_raise_for_status_can_disable_session_policy(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        raise web.HTTPBadRequest()

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    async with make_session(
        client_engine_name,
        raise_for_status=True,
    ) as session:
        response = await session.get(
            server.make_url("/"), raise_for_status=False
        )
        assert response.status == 400


async def test_redirect_to_other_origin_drops_request_auth(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
    async def final_handler(request: web.Request) -> web.Response:
        assert hdrs.AUTHORIZATION not in request.headers
        return web.Response()

    final_app = web.Application()
    final_app.router.add_get("/final", final_handler)
    final_server = await aiohttp_server(final_app)

    async def redirect_handler(request: web.Request) -> web.Response:
        assert request.headers[hdrs.AUTHORIZATION] == "Basic dXNlcjpwYXNz"
        raise web.HTTPFound(final_server.make_url("/final"))

    redirect_app = web.Application()
    redirect_app.router.add_get("/start", redirect_handler)
    redirect_server = await aiohttp_server(redirect_app)

    async with make_session(client_engine_name) as session:
        response = await session.get(
            redirect_server.make_url("/start"),
            auth=BasicAuth("user", "pass"),
        )
        assert response.status == 200


async def test_redirect_policy_and_post_method_rewrite(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
    async def final_handler(request: web.Request) -> web.Response:
        return web.Response(text=request.method)

    async def redirect_handler(request: web.Request) -> web.Response:
        raise web.HTTPFound("/final")

    app = web.Application()
    app.router.add_get("/final", final_handler)
    app.router.add_post("/start", redirect_handler)
    server = await aiohttp_server(app)

    async with make_session(client_engine_name) as session:
        redirected = await session.post(server.make_url("/start"), data=b"body")
        assert redirected.status == 200
        assert [item.status for item in redirected.history] == [302]
        assert await redirected.text() == "GET"

        not_redirected = await session.post(
            server.make_url("/start"),
            data=b"body",
            allow_redirects=False,
        )
        assert not_redirected.status == 302
        assert not_redirected.history == ()


async def test_trace_lifecycle_and_response_chunks(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
    events: list[str] = []
    trace_config = aiohttp.TraceConfig()

    async def on_request_start(
        session: ClientSession, context: object, params: object
    ) -> None:
        events.append("start")

    async def on_request_headers_sent(
        session: ClientSession, context: object, params: object
    ) -> None:
        events.append("headers")

    async def on_request_end(
        session: ClientSession, context: object, params: object
    ) -> None:
        events.append("end")

    async def on_response_chunk_received(
        session: ClientSession, context: object, params: object
    ) -> None:
        events.append("chunk")

    trace_config.on_request_start.append(on_request_start)
    trace_config.on_request_headers_sent.append(on_request_headers_sent)
    trace_config.on_request_end.append(on_request_end)
    trace_config.on_response_chunk_received.append(on_response_chunk_received)

    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=b"trace body")

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    async with make_session(
        client_engine_name,
        trace_configs=(trace_config,),
    ) as session:
        response = await session.get(server.make_url("/"))
        assert await response.read() == b"trace body"

    assert events == ["start", "headers", "end", "chunk"]


async def test_trace_request_exception_on_timeout(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
    events: list[str] = []
    trace_config = aiohttp.TraceConfig()

    async def on_request_end(
        session: ClientSession, context: object, params: object
    ) -> None:
        events.append("end")

    async def on_request_exception(
        session: ClientSession, context: object, params: object
    ) -> None:
        events.append("exception")

    trace_config.on_request_end.append(on_request_end)
    trace_config.on_request_exception.append(on_request_exception)

    async def handler(request: web.Request) -> web.Response:
        await asyncio.sleep(0.1)
        return web.Response()

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    timeout = aiohttp.ClientTimeout(total=None, sock_read=0.01)
    async with make_session(
        client_engine_name,
        trace_configs=(trace_config,),
    ) as session:
        with pytest.raises(aiohttp.SocketTimeoutError):
            await session.get(server.make_url("/"), timeout=timeout)

    assert events == ["exception"]


async def test_sock_read_timeout_waiting_for_headers(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        await asyncio.sleep(0.1)
        return web.Response()

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    timeout = aiohttp.ClientTimeout(total=None, sock_read=0.01)
    async with make_session(client_engine_name) as session:
        with pytest.raises(aiohttp.SocketTimeoutError):
            await session.get(server.make_url("/"), timeout=timeout)


async def test_streaming_response_can_be_consumed_before_eof(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
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

    async with make_session(client_engine_name) as session:
        response = await asyncio.wait_for(session.get(server.make_url("/")), timeout=1)
        assert await response.content.readany() == b"first"
        release_tail.set()
        assert await response.read() == b"second"


async def test_stream_reader_compatible_public_methods(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=b"line one\nline two\n")

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    async with make_session(client_engine_name) as session:
        response = await session.get(server.make_url("/"))
        assert await response.content.readline() == b"line one\n"
        assert await response.content.readexactly(4) == b"line"
        assert [chunk async for chunk in response.content.iter_chunked(3)] == [
            b" tw",
            b"o\n",
        ]


async def test_total_timeout_covers_body_reads(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
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
    async with make_session(client_engine_name) as session:
        response = await session.get(server.make_url("/"), timeout=timeout)
        with pytest.raises(asyncio.TimeoutError):
            await response.read()


async def test_redirect_replays_buffered_upload(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
    seen: list[tuple[str, bytes]] = []

    async def redirect_handler(request: web.Request) -> web.Response:
        seen.append(("redirect", await request.read()))
        raise web.HTTPTemporaryRedirect("/final")

    async def final_handler(request: web.Request) -> web.Response:
        seen.append(("final", await request.read()))
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_post("/start", redirect_handler)
    app.router.add_post("/final", final_handler)
    server = await aiohttp_server(app)

    async with make_session(client_engine_name) as session:
        response = await session.post(server.make_url("/start"), data=b"payload")
        assert response.status == 200
        assert [item.status for item in response.history] == [307]
        assert await response.text() == "ok"

    assert seen == [("redirect", b"payload"), ("final", b"payload")]


async def test_async_iterable_upload(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
    async def body() -> AsyncIterator[bytes]:
        yield b"streamed "
        yield b"body"

    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=await request.read())

    app = web.Application()
    app.router.add_post("/", handler)
    server = await aiohttp_server(app)

    async with make_session(client_engine_name) as session:
        response = await session.post(server.make_url("/"), data=body())
        assert await response.read() == b"streamed body"


async def test_multipart_upload(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
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
    form.add_field("file", io.BytesIO(b"upload"), filename="data.txt")

    async with make_session(client_engine_name) as session:
        response = await session.post(server.make_url("/"), data=form)
        assert await response.read() == b"value:upload"


@pytest.mark.parametrize("compression", ("gzip", "deflate"))
async def test_request_compression(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
    compression: str,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        assert request.headers[hdrs.CONTENT_ENCODING] == compression
        return web.Response(body=await request.read())

    app = web.Application()
    app.router.add_post("/", handler)
    server = await aiohttp_server(app)

    async with make_session(client_engine_name) as session:
        response = await session.post(
            server.make_url("/"),
            data=b"compressed upload",
            compress=compression,
        )
        assert await response.read() == b"compressed upload"


async def test_response_auto_decompression_can_be_disabled(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
    compressed = gzip.compress(b"compressed response")

    async def handler(request: web.Request) -> web.Response:
        return web.Response(
            body=compressed,
            headers={hdrs.CONTENT_ENCODING: "gzip"},
        )

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    async with make_session(client_engine_name) as session:
        decoded = await session.get(server.make_url("/"))
        assert await decoded.read() == b"compressed response"

        encoded = await session.get(server.make_url("/"), auto_decompress=False)
        assert await encoded.read() == compressed


async def test_max_headers_can_be_raised_per_request(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(headers={f"Custom-{i}": "x" for i in range(130)})

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)

    async with make_session(client_engine_name) as session:
        with pytest.raises(aiohttp.ClientResponseError):
            await session.get(server.make_url("/"))

        response = await session.get(server.make_url("/"), max_headers=140)
        assert response.headers["Custom-129"] == "x"


async def test_https_can_disable_verification(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
    ssl_ctx: ssl.SSLContext,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=b"tls body")

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app, ssl=ssl_ctx)

    async with make_session(client_engine_name) as session:
        response = await session.get(server.make_url("/"), ssl=False)
        assert await response.read() == b"tls body"


async def test_https_fingerprint(
    aiohttp_server: AiohttpServer,
    client_engine_name: ClientEngineName,
    ssl_ctx: ssl.SSLContext,
    tls_certificate_fingerprint_sha256: bytes,
) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=b"tls body")

    app = web.Application()
    app.router.add_get("/", handler)
    server = await aiohttp_server(app, ssl=ssl_ctx)

    async with make_session(client_engine_name) as session:
        response = await session.get(
            server.make_url("/"),
            ssl=aiohttp.Fingerprint(tls_certificate_fingerprint_sha256),
        )
        assert await response.read() == b"tls body"
