from __future__ import annotations

import asyncio
import contextlib
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import TYPE_CHECKING, NoReturn, Protocol

from multidict import CIMultiDict, CIMultiDictProxy

from . import hdrs
from .client_exceptions import (
    ClientConnectorCertificateError,
    ClientConnectorSSLError,
    ClientPayloadError,
    ClientResponseError,
    ConnectionTimeoutError,
    ServerFingerprintMismatch,
    SocketTimeoutError,
    cert_errors,
    ssl_errors,
)
from .compression_utils import (
    HAS_BROTLI,
    HAS_ZSTD,
    BrotliDecompressor,
    ZLibDecompressor,
    ZSTDDecompressor,
)
from .helpers import EMPTY_BODY_METHODS, BaseTimerContext, TimerNoop
from .http import HttpVersion
from .http_exceptions import ContentEncodingError, HttpProcessingError
from .http_parser import RawResponseMessage

if TYPE_CHECKING:
    from .client import ClientTimeout
    from .client_reqrep import ClientRequest, ClientResponse, PreparedClientRequest
    from .connector import BaseConnector, Connection
    from .payload import Payload
    from .tracing import Trace

__all__ = (
    "AsyncioClientEngine",
    "AsyncioClientExchange",
    "RustClientEngine",
    "ClientBodyStream",
    "ClientConnection",
    "ClientEngine",
    "ClientEngineCapabilities",
    "ClientExchange",
    "PayloadUploadSource",
    "UploadSource",
    "UploadKind",
    "UploadPlan",
    "UploadReplayability",
)

DEFAULT_NATIVE_READ_BUFSIZE = 2**16


class UploadKind(Enum):
    EMPTY = "empty"
    BUFFERED = "buffered"
    FILE = "file"
    ASYNC_ITERABLE = "async_iterable"
    MULTIPART = "multipart"
    GENERIC_PAYLOAD = "generic_payload"


class UploadReplayability(Enum):
    REPLAYABLE = "replayable"
    ONE_SHOT = "one_shot"
    CONSUMED = "consumed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class UploadPlan:
    kind: UploadKind
    size: int | None
    replayability: UploadReplayability
    autoclose: bool

    @property
    def replayable(self) -> bool:
        return self.replayability is UploadReplayability.REPLAYABLE


class UploadSource(Protocol):
    async def iter_chunks(self) -> AsyncIterator[bytes]: ...


class _ChunkQueueWriter:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=1)

    async def write(self, chunk: bytes) -> None:
        if chunk:
            await self._queue.put(bytes(chunk))

    async def write_eof(self, chunk: bytes = b"") -> None:
        if chunk:
            await self.write(chunk)

    async def drain(self) -> None:
        pass

    def enable_compression(
        self, encoding: str = "deflate", strategy: int | None = None
    ) -> None:
        raise NotImplementedError("upload source writer does not support compression")

    def enable_chunking(self) -> None:
        raise NotImplementedError("upload source writer does not support chunking")

    def send_headers(self) -> None:
        pass

    async def finish(self) -> None:
        await self._queue.put(None)

    async def iter_chunks(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            yield chunk


class PayloadUploadSource:
    def __init__(self, body: "Payload", content_length: int | None) -> None:
        self._body = body
        self._content_length = content_length

    async def iter_chunks(self) -> AsyncIterator[bytes]:
        writer = _ChunkQueueWriter()
        producer = asyncio.create_task(
            self._produce(writer),
            name="aiohttp-payload-upload-source",
        )
        try:
            async for chunk in writer.iter_chunks():
                yield chunk
            await producer
        finally:
            if not producer.done():
                producer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await producer

    async def _produce(self, writer: _ChunkQueueWriter) -> None:
        try:
            await self._body.write_with_length(writer, self._content_length)
        finally:
            await writer.finish()


class _UploadCursor:
    def __init__(
        self,
        source: UploadSource,
        *,
        on_chunk: Callable[[bytes], Awaitable[None]] | None = None,
    ) -> None:
        self._iterator = source.iter_chunks().__aiter__()
        self._on_chunk = on_chunk

    async def next_chunk(self) -> bytes | None:
        try:
            chunk = await anext(self._iterator)
        except StopAsyncIteration:
            return None
        if self._on_chunk is not None:
            await self._on_chunk(chunk)
        return chunk


@dataclass(frozen=True, slots=True)
class ClientEngineCapabilities:
    supports_connectors: bool = False
    supports_custom_request_classes: bool = False
    supports_custom_response_classes: bool = False
    supports_custom_ws_response_classes: bool = False
    supports_websockets: bool = False
    supported_upload_kinds: frozenset[UploadKind] = frozenset(UploadKind)


@dataclass(frozen=True, slots=True)
class AttemptOptions:
    timeout: "ClientTimeout"
    timer: BaseTimerContext
    read_until_eof: bool
    auto_decompress: bool
    read_bufsize: int
    max_line_size: int
    max_field_size: int
    max_headers: int


class ClientEngine(Protocol):
    capabilities: ClientEngineCapabilities
    allowed_protocol_schema_set: frozenset[str]

    async def send_one_attempt(
        self,
        request: "ClientRequest",
        traces: list["Trace"],
        options: AttemptOptions,
    ) -> "ClientResponse": ...

    async def close(self) -> None: ...

    @property
    def closed(self) -> bool: ...


class ClientConnection(Protocol):
    @property
    def closed(self) -> bool: ...

    def add_callback(self, callback: Callable[[], None]) -> None: ...

    def release(self) -> None: ...

    def close(self) -> None: ...


class ClientBodyStream(Protocol):
    def __aiter__(self) -> AsyncIterator[bytes]: ...

    def iter_chunked(self, n: int) -> AsyncIterator[bytes]: ...

    def iter_any(self) -> AsyncIterator[bytes]: ...

    def iter_chunks(self) -> AsyncIterator[tuple[bytes, bool]]: ...

    def get_read_buffer_limits(self) -> tuple[int, int]: ...

    async def read(self, n: int = -1) -> bytes: ...

    async def readany(self) -> bytes: ...

    async def readexactly(self, n: int) -> bytes: ...

    async def readline(self) -> bytes: ...

    async def readuntil(self, separator: bytes = b"\n") -> bytes: ...

    async def readchunk(self) -> tuple[bytes, bool]: ...

    def exception(self) -> type[BaseException] | BaseException | None: ...

    def set_exception(
        self, exc: type[BaseException] | BaseException, exc_cause: BaseException = ...
    ) -> None: ...

    def on_eof(self, callback: Callable[[], None]) -> None: ...

    async def wait_eof(self) -> None: ...

    def is_eof(self) -> bool: ...

    def at_eof(self) -> bool: ...

    def read_nowait(self, n: int = -1) -> bytes: ...

    def unread_data(self, data: bytes) -> None: ...

    def close(self) -> None: ...

    @property
    def total_raw_bytes(self) -> int: ...


class ClientExchange(Protocol):
    __aiohttp_exchange__: bool

    @property
    def connection(self) -> "ClientConnection | None": ...

    @property
    def upgraded(self) -> bool: ...

    async def read(self) -> tuple["RawResponseMessage", ClientBodyStream]: ...

    def release(self) -> None: ...

    def close(self) -> None: ...


class AsyncioClientExchange:
    __aiohttp_exchange__ = True

    def __init__(self, connection: "Connection") -> None:
        self._connection = connection

    @property
    def connection(self) -> "Connection":
        return self._connection

    @property
    def upgraded(self) -> bool:
        protocol = self._connection.protocol
        return protocol is not None and protocol.upgraded

    async def read(self) -> tuple["RawResponseMessage", ClientBodyStream]:
        protocol = self._connection.protocol
        assert protocol is not None
        return await protocol.read()

    def release(self) -> None:
        self._connection.release()

    def close(self) -> None:
        self._connection.close()


class AsyncioClientEngine:
    capabilities = ClientEngineCapabilities(
        supports_connectors=True,
        supports_custom_request_classes=True,
        supports_custom_response_classes=True,
        supports_custom_ws_response_classes=True,
        supports_websockets=True,
    )

    def __init__(self, connector: "BaseConnector") -> None:
        self._connector = connector

    @property
    def allowed_protocol_schema_set(self) -> frozenset[str]:
        return self._connector.allowed_protocol_schema_set

    @property
    def connector(self) -> "BaseConnector":
        return self._connector

    @property
    def closed(self) -> bool:
        return self._connector.closed

    async def close(self) -> None:
        await self._connector.close()

    async def send_one_attempt(
        self,
        request: "ClientRequest",
        traces: list["Trace"],
        options: AttemptOptions,
    ) -> "ClientResponse":
        try:
            conn = await self._connector.connect(
                request, traces=traces, timeout=options.timeout
            )
        except asyncio.TimeoutError as exc:
            raise ConnectionTimeoutError(
                f"Connection timeout to host {request.url}"
            ) from exc

        assert conn.protocol is not None
        conn.protocol.set_response_params(
            timer=options.timer,
            skip_payload=request.method in EMPTY_BODY_METHODS,
            read_until_eof=options.read_until_eof,
            auto_decompress=options.auto_decompress,
            read_timeout=options.timeout.sock_read,
            read_bufsize=options.read_bufsize,
            timeout_ceil_threshold=self._connector._timeout_ceil_threshold,
            max_line_size=options.max_line_size,
            max_field_size=options.max_field_size,
            max_headers=options.max_headers,
        )
        try:
            resp = await request.send(conn)
            try:
                await resp.start(conn)
            except BaseException:
                resp.close()
                raise
        except BaseException:
            conn.close()
            raise
        return resp


class _BufferedClientBodyStream:
    def __init__(self, body: bytes) -> None:
        self._buffer = body
        self._offset = 0
        self._exception: type[BaseException] | BaseException | None = None
        self._on_eof: list[Callable[[], None]] = []
        self._eof_notified = False
        self._total_raw_bytes = len(body)

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self.iter_any()

    async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:
        while chunk := await self.read(n):
            yield chunk

    async def iter_any(self) -> AsyncIterator[bytes]:
        while chunk := await self.readany():
            yield chunk

    async def iter_chunks(self) -> AsyncIterator[tuple[bytes, bool]]:
        if chunk := await self.read():
            yield chunk, False

    def get_read_buffer_limits(self) -> tuple[int, int]:
        return (0, 0)

    async def read(self, n: int = -1) -> bytes:
        return self.read_nowait(n)

    async def readany(self) -> bytes:
        return self.read_nowait()

    async def readexactly(self, n: int) -> bytes:
        chunk = self.read_nowait(n)
        if len(chunk) != n:
            raise asyncio.IncompleteReadError(chunk, n)
        return chunk

    async def readline(self) -> bytes:
        self._raise_exception()
        remaining = self._buffer[self._offset :]
        newline = remaining.find(b"\n")
        if newline < 0:
            return self.read_nowait()
        return self.read_nowait(newline + 1)

    async def readuntil(self, separator: bytes = b"\n") -> bytes:
        if not separator:
            raise ValueError("Separator should be at least one-byte string")
        self._raise_exception()
        remaining = self._buffer[self._offset :]
        index = remaining.find(separator)
        if index < 0:
            partial = self.read_nowait()
            raise asyncio.IncompleteReadError(partial, None)
        return self.read_nowait(index + len(separator))

    async def readchunk(self) -> tuple[bytes, bool]:
        return await self.read(), False

    def exception(self) -> type[BaseException] | BaseException | None:
        return self._exception

    def set_exception(
        self, exc: type[BaseException] | BaseException, exc_cause: BaseException = ...
    ) -> None:
        self._exception = exc

    def on_eof(self, callback: Callable[[], None]) -> None:
        if self.at_eof():
            callback()
            return
        self._on_eof.append(callback)

    async def wait_eof(self) -> None:
        self._notify_eof_if_needed()

    def is_eof(self) -> bool:
        return self._offset >= len(self._buffer)

    def at_eof(self) -> bool:
        return self.is_eof()

    def read_nowait(self, n: int = -1) -> bytes:
        self._raise_exception()
        if n == 0:
            return b""
        if self.at_eof():
            self._notify_eof_if_needed()
            return b""
        if n < 0:
            chunk = self._buffer[self._offset :]
            self._offset = len(self._buffer)
        else:
            end = min(len(self._buffer), self._offset + n)
            chunk = self._buffer[self._offset : end]
            self._offset = end
        self._notify_eof_if_needed()
        return chunk

    def unread_data(self, data: bytes) -> None:
        if not data:
            return
        self._buffer = data + self._buffer[self._offset :]
        self._offset = 0
        self._eof_notified = False

    def close(self) -> None:
        self._offset = len(self._buffer)
        self._notify_eof_if_needed()

    @property
    def total_raw_bytes(self) -> int:
        return self._total_raw_bytes

    def _raise_exception(self) -> None:
        if self._exception is None:
            return
        if isinstance(self._exception, type):
            raise self._exception()
        raise self._exception

    def _notify_eof_if_needed(self) -> None:
        if not self.at_eof() or self._eof_notified:
            return
        self._eof_notified = True
        callbacks = self._on_eof
        self._on_eof = []
        for callback in callbacks:
            callback()


class _RustClientBodyStream:
    def __init__(
        self,
        body: object,
        timer: BaseTimerContext,
        compression: str | None = None,
    ) -> None:
        self._body = body
        self._timer = timer
        self._compression = compression
        self._buffer = bytearray()
        self._eof = False
        self._exception: type[BaseException] | BaseException | None = None
        self._on_eof: list[Callable[[], None]] = []
        self._eof_notified = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self.iter_any()

    @property
    def _protocol(self) -> NoReturn:
        raise NotImplementedError(
            "RustClientEngine response streams do not expose StreamReader internals"
        )

    async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:
        while chunk := await self.read(n):
            yield chunk

    async def iter_any(self) -> AsyncIterator[bytes]:
        while chunk := await self.readany():
            yield chunk

    async def iter_chunks(self) -> AsyncIterator[tuple[bytes, bool]]:
        raise NotImplementedError(
            "RustClientEngine does not support HTTP chunk-boundary reads yet"
        )
        yield b"", False

    def get_read_buffer_limits(self) -> tuple[int, int]:
        return (0, 0)

    async def read(self, n: int = -1) -> bytes:
        self._raise_exception()
        if n == 0:
            return b""
        if n < 0:
            await self._fill_until_eof()
            return self.read_nowait()
        await self._fill_to(n)
        return self.read_nowait(n)

    async def readany(self) -> bytes:
        self._raise_exception()
        if not self._buffer and not self._eof:
            await self._read_next_chunk()
        return self.read_nowait()

    async def readexactly(self, n: int) -> bytes:
        if n < 0:
            raise ValueError("readexactly size can not be less than zero")
        await self._fill_to(n)
        chunk = self.read_nowait(n)
        if len(chunk) != n:
            raise asyncio.IncompleteReadError(chunk, n)
        return chunk

    async def readline(self) -> bytes:
        return await self.readuntil(b"\n")

    async def readuntil(self, separator: bytes = b"\n") -> bytes:
        if not separator:
            raise ValueError("Separator should be at least one-byte string")
        self._raise_exception()
        while True:
            index = self._buffer.find(separator)
            if index >= 0:
                return self.read_nowait(index + len(separator))
            if self._eof:
                return self.read_nowait()
            await self._read_next_chunk()

    async def readchunk(self) -> tuple[bytes, bool]:
        raise NotImplementedError(
            "RustClientEngine does not support HTTP chunk-boundary reads yet"
        )

    def exception(self) -> type[BaseException] | BaseException | None:
        return self._exception

    def set_exception(
        self, exc: type[BaseException] | BaseException, exc_cause: BaseException = ...
    ) -> None:
        self._exception = exc

    def on_eof(self, callback: Callable[[], None]) -> None:
        if self._eof:
            callback()
            return
        self._on_eof.append(callback)

    async def wait_eof(self) -> None:
        await self._fill_until_eof()

    def is_eof(self) -> bool:
        return self._eof

    def at_eof(self) -> bool:
        return self._eof and not self._buffer

    def read_nowait(self, n: int = -1) -> bytes:
        self._raise_exception()
        if n == 0 or not self._buffer:
            return b""
        if n < 0 or n >= len(self._buffer):
            chunk = bytes(self._buffer)
            self._buffer.clear()
            return chunk
        chunk = bytes(self._buffer[:n])
        del self._buffer[:n]
        return chunk

    def unread_data(self, data: bytes) -> None:
        if data:
            self._buffer[:0] = data

    def close(self) -> None:
        self._body.close()  # type: ignore[attr-defined]
        self._eof = True
        self._notify_eof()

    @property
    def total_raw_bytes(self) -> int:
        return self._body.total_raw_bytes  # type: ignore[attr-defined]

    async def _fill_to(self, n: int) -> None:
        while len(self._buffer) < n and not self._eof:
            await self._read_next_chunk()

    async def _fill_until_eof(self) -> None:
        while not self._eof:
            await self._read_next_chunk()

    async def _read_next_chunk(self) -> None:
        self._raise_exception()
        try:
            with self._timer:
                chunk = await self._body.next_chunk()  # type: ignore[attr-defined]
        except asyncio.TimeoutError as exc:
            error = SocketTimeoutError("Timeout on reading data from socket")
            self._exception = error
            raise error from exc
        except ValueError as exc:
            if self._compression is None:
                raise
            self._raise_payload_error(
                ContentEncodingError(
                    f"Can not decode content-encoding: {self._compression}"
                )
            )
        if chunk is None:
            self._eof = True
            self._notify_eof()
            return
        self._buffer.extend(chunk)

    def _raise_payload_error(self, exc: ContentEncodingError) -> NoReturn:
        error = ClientPayloadError(f"Response payload is not completed: {exc!r}")
        self._exception = error
        raise error from exc

    def _raise_exception(self) -> None:
        if self._exception is None:
            return
        if isinstance(self._exception, type):
            raise self._exception()
        raise self._exception

    def _notify_eof(self) -> None:
        if self._eof_notified:
            return
        self._eof_notified = True
        callbacks = self._on_eof
        self._on_eof = []
        for callback in callbacks:
            callback()


class _DecompressingClientBodyStream:
    def __init__(self, body: ClientBodyStream, encoding: str) -> None:
        self._body = body
        self._encoding = encoding
        self._decompressor = self._make_decompressor(encoding)
        self._started_decoding = False
        self._buffer = bytearray()
        self._eof = False
        self._exception: type[BaseException] | BaseException | None = None
        self._on_eof: list[Callable[[], None]] = []
        self._eof_notified = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self.iter_any()

    @property
    def _protocol(self) -> NoReturn:
        raise NotImplementedError(
            "RustClientEngine response streams do not expose StreamReader internals"
        )

    async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:
        while chunk := await self.read(n):
            yield chunk

    async def iter_any(self) -> AsyncIterator[bytes]:
        while chunk := await self.readany():
            yield chunk

    async def iter_chunks(self) -> AsyncIterator[tuple[bytes, bool]]:
        raise NotImplementedError(
            "RustClientEngine does not support HTTP chunk-boundary reads yet"
        )
        yield b"", False

    def get_read_buffer_limits(self) -> tuple[int, int]:
        return self._body.get_read_buffer_limits()

    async def read(self, n: int = -1) -> bytes:
        self._raise_exception()
        if n == 0:
            return b""
        if n < 0:
            await self._fill_until_eof()
            return self.read_nowait()
        await self._fill_to(n)
        return self.read_nowait(n)

    async def readany(self) -> bytes:
        self._raise_exception()
        while not self._buffer and not self._eof:
            await self._read_next_chunk()
        return self.read_nowait()

    async def readexactly(self, n: int) -> bytes:
        if n < 0:
            raise ValueError("readexactly size can not be less than zero")
        await self._fill_to(n)
        chunk = self.read_nowait(n)
        if len(chunk) != n:
            raise asyncio.IncompleteReadError(chunk, n)
        return chunk

    async def readline(self) -> bytes:
        return await self.readuntil(b"\n")

    async def readuntil(self, separator: bytes = b"\n") -> bytes:
        if not separator:
            raise ValueError("Separator should be at least one-byte string")
        self._raise_exception()
        while True:
            index = self._buffer.find(separator)
            if index >= 0:
                return self.read_nowait(index + len(separator))
            if self._eof:
                return self.read_nowait()
            await self._read_next_chunk()

    async def readchunk(self) -> tuple[bytes, bool]:
        raise NotImplementedError(
            "RustClientEngine does not support HTTP chunk-boundary reads yet"
        )

    def exception(self) -> type[BaseException] | BaseException | None:
        return self._exception

    def set_exception(
        self, exc: type[BaseException] | BaseException, exc_cause: BaseException = ...
    ) -> None:
        self._exception = exc

    def on_eof(self, callback: Callable[[], None]) -> None:
        if self._eof:
            callback()
            return
        self._on_eof.append(callback)

    async def wait_eof(self) -> None:
        await self._fill_until_eof()

    def is_eof(self) -> bool:
        return self._eof

    def at_eof(self) -> bool:
        return self._eof and not self._buffer

    def read_nowait(self, n: int = -1) -> bytes:
        self._raise_exception()
        if n == 0 or not self._buffer:
            return b""
        if n < 0 or n >= len(self._buffer):
            chunk = bytes(self._buffer)
            self._buffer.clear()
            return chunk
        chunk = bytes(self._buffer[:n])
        del self._buffer[:n]
        return chunk

    def unread_data(self, data: bytes) -> None:
        if data:
            self._buffer[:0] = data

    def close(self) -> None:
        self._body.close()  # type: ignore[attr-defined]
        self._eof = True
        self._notify_eof()

    @property
    def total_raw_bytes(self) -> int:
        return self._body.total_raw_bytes

    async def _fill_to(self, n: int) -> None:
        while len(self._buffer) < n and not self._eof:
            await self._read_next_chunk()

    async def _fill_until_eof(self) -> None:
        while not self._eof:
            await self._read_next_chunk()

    async def _read_next_chunk(self) -> None:
        self._raise_exception()
        raw_chunk = await self._body.readany()
        if not raw_chunk:
            self._finish()
            return
        try:
            self._buffer.extend(self._decompress(raw_chunk))
        except ContentEncodingError as exc:
            self._raise_payload_error(exc)

    def _decompress(self, chunk: bytes) -> bytes:
        if (
            not self._started_decoding
            and self._encoding == "deflate"
            and chunk[0] & 0xF != 8
        ):
            self._decompressor = ZLibDecompressor(
                encoding=self._encoding,
                suppress_deflate_header=True,
            )

        chunks = []
        try:
            chunks.append(self._decompressor.decompress_sync(chunk))
            while self._decompressor.data_available:
                chunks.append(self._decompressor.decompress_sync(b""))
        except Exception as exc:
            raise ContentEncodingError(
                f"Can not decode content-encoding: {self._encoding}"
            ) from exc
        self._started_decoding = True
        return b"".join(chunks)

    def _finish(self) -> None:
        try:
            flushed = self._decompressor.flush()
            if flushed:
                self._buffer.extend(flushed)
            if (
                self._body.total_raw_bytes > 0
                and self._encoding == "deflate"
                and not self._decompressor.eof  # type: ignore[union-attr]
            ):
                raise ContentEncodingError("deflate")
        except ContentEncodingError as exc:
            self._raise_payload_error(exc)
        except Exception:
            self._raise_payload_error(
                ContentEncodingError(
                    f"Can not decode content-encoding: {self._encoding}"
                )
            )
        self._eof = True
        self._notify_eof()

    def _raise_payload_error(self, exc: ContentEncodingError) -> NoReturn:
        error = ClientPayloadError(f"Response payload is not completed: {exc!r}")
        self._exception = error
        raise error from exc

    def _raise_exception(self) -> None:
        if self._exception is None:
            return
        if isinstance(self._exception, type):
            raise self._exception()
        raise self._exception

    def _notify_eof(self) -> None:
        if self._eof_notified:
            return
        self._eof_notified = True
        callbacks = self._on_eof
        self._on_eof = []
        for callback in callbacks:
            callback()

    @staticmethod
    def _make_decompressor(
        encoding: str,
    ) -> BrotliDecompressor | ZLibDecompressor | ZSTDDecompressor:
        if encoding == "br":
            if not HAS_BROTLI:
                raise ContentEncodingError(
                    "Can not decode content-encoding: brotli (br). "
                    "Please install `Brotli`"
                )
            return BrotliDecompressor()
        if encoding == "zstd":
            if not HAS_ZSTD:
                raise ContentEncodingError(
                    "Can not decode content-encoding: zstandard (zstd). "
                    "Please install `backports.zstd`"
                )
            return ZSTDDecompressor()
        return ZLibDecompressor(encoding=encoding)


class _RustClientExchange:
    __aiohttp_exchange__ = True
    __aiohttp_native_exchange__ = True

    def __init__(
        self,
        message: RawResponseMessage,
        body: bytes | object,
        *,
        engine: "RustClientEngine | None" = None,
        connection_key: tuple[str, str, int, int] | None = None,
        native_connection: object | None = None,
        force_close: bool = False,
        timer: BaseTimerContext | None = None,
        auto_decompress: bool = False,
        compression: str | None = None,
    ) -> None:
        self._message = message
        body_stream: ClientBodyStream = (
            _BufferedClientBodyStream(body)
            if isinstance(body, bytes)
            else _RustClientBodyStream(
                body,
                timer or TimerNoop(),
                compression if auto_decompress else None,
            )
        )
        self._body = (
            _DecompressingClientBodyStream(body_stream, compression)
            if isinstance(body, bytes) and auto_decompress and compression is not None
            else body_stream
        )
        self._engine = engine
        self._connection_key = connection_key
        self._native_connection = native_connection
        self._force_close = force_close
        self._released = False
        self._callbacks: list[Callable[[], None]] = []

    @property
    def connection(self) -> "_RustClientExchange":
        return self

    @property
    def host(self) -> str | None:
        if self._connection_key is None:
            return None
        return self._connection_key[1]

    @property
    def port(self) -> int | None:
        if self._connection_key is None:
            return None
        return self._connection_key[2]

    @property
    def closed(self) -> bool:
        if self._released:
            return True
        native_connection = self._native_connection
        return native_connection is not None and native_connection.closed  # type: ignore[attr-defined]

    def add_callback(self, callback: Callable[[], None]) -> None:
        if self.closed:
            callback()
            return
        self._callbacks.append(callback)

    @property
    def upgraded(self) -> bool:
        return False

    @property
    def protocol(self) -> NoReturn:
        raise NotImplementedError(
            "RustClientEngine connections do not expose asyncio protocol internals"
        )

    @property
    def _protocol(self) -> NoReturn:
        raise NotImplementedError(
            "RustClientEngine connections do not expose asyncio protocol internals"
        )

    @property
    def transport(self) -> NoReturn:
        raise NotImplementedError(
            "RustClientEngine connections do not expose asyncio transport internals"
        )

    async def read(self) -> tuple[RawResponseMessage, ClientBodyStream]:
        return self._message, self._body

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._notify_release()
        engine = self._engine
        native_connection = self._native_connection
        if (
            engine is not None
            and native_connection is not None
            and self._connection_key is not None
            and self._body.is_eof()
            and not self._force_close
            and not self._message.should_close
        ):
            engine._release_connection(self._connection_key, native_connection)
            return
        self._body.close()
        if engine is not None and native_connection is not None:
            engine._close_connection(native_connection)

    def close(self) -> None:
        if self._released:
            return
        self._released = True
        self._notify_release()
        self._body.close()
        engine = self._engine
        native_connection = self._native_connection
        if engine is not None and native_connection is not None:
            engine._close_connection(native_connection)

    def _notify_release(self) -> None:
        callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            callback()


class RustClientEngine:
    capabilities = ClientEngineCapabilities()
    allowed_protocol_schema_set = frozenset({"http", "https"})

    def __init__(self) -> None:
        self._closed = False
        self._available_connections: dict[
            tuple[str, str, int, int, bool, str | None, bytes | None], list[object]
        ] = {}
        self._connections: set[object] = set()

    @property
    def closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        self._closed = True
        for connection in self._connections:
            connection.close()  # type: ignore[attr-defined]
        self._available_connections.clear()
        self._connections.clear()

    async def send_one_attempt(
        self,
        request: "ClientRequest",
        traces: list["Trace"],
        options: AttemptOptions,
    ) -> "ClientResponse":
        self._validate_request(request)
        self._validate_options(options)
        prepared = request.prepare_for_send(force_close=False)
        self._validate_prepared(prepared)
        if hdrs.ACCEPT_ENCODING in prepared.auto_headers:
            prepared.headers[hdrs.ACCEPT_ENCODING] = "gzip, deflate, br, zstd"
        host = request.url.raw_host or ""
        tls_server_name = (
            (request.server_hostname or host).rstrip(".")
            if request.url.scheme == "https"
            else None
        )
        tls_fingerprint = self._tls_fingerprint(request)
        connection_key = (
            request.url.scheme,
            host,
            request.url.port or (443 if request.url.scheme == "https" else 80),
            options.max_headers,
            request.url.scheme != "https" or request.ssl is True,
            tls_server_name,
            tls_fingerprint,
        )
        native_connection = await self._acquire_connection(
            connection_key,
            request=request,
            traces=traces,
            sock_connect=options.timeout.sock_connect,
        )
        self._check_fingerprint(request, native_connection, tls_fingerprint)
        try:
            if traces:
                await request._on_headers_request_sent(
                    prepared.method,
                    request.url,
                    prepared.headers,
                )
            raw_response = await self._send_http1_request(
                native_connection,
                prepared.method,
                prepared.target,
                prepared.version.major,
                prepared.version.minor,
                list(prepared.headers.items()),
                _UploadCursor(
                    prepared.upload_source,
                    on_chunk=(
                        (
                            lambda chunk: request._on_chunk_request_sent(
                                prepared.method,
                                request.url,
                                chunk,
                            )
                        )
                        if traces
                        else None
                    ),
                ),
                content_length=prepared.content_length,
                compression=prepared.compression,
                expect_continue=prepared.expect_continue,
                read_until_eof=options.read_until_eof,
                auto_decompress=options.auto_decompress,
                skip_payload=request.method in EMPTY_BODY_METHODS,
                max_headers=options.max_headers,
                sock_read=options.timeout.sock_read,
            )
        except asyncio.TimeoutError as exc:
            self._close_connection(native_connection)
            raise SocketTimeoutError("Timeout on reading data from socket") from exc
        except HttpProcessingError as exc:
            self._close_connection(native_connection)
            response = request._create_response(None)
            raise ClientResponseError(
                response.request_info,
                response.history,
                status=exc.code,
                message=exc.message,
                headers=exc.headers,
            ) from exc
        (
            version_minor,
            code,
            reason,
            raw_headers,
            response_body,
            should_close,
            compression,
            upgrade,
            chunked,
        ) = raw_response

        headers = CIMultiDict[str]()
        for raw_name, raw_value in raw_headers:
            headers.add(
                raw_name.decode("utf-8", "surrogateescape"),
                raw_value.decode("utf-8", "surrogateescape"),
            )
        message = RawResponseMessage(
            HttpVersion(1, version_minor),
            code,
            reason,
            CIMultiDictProxy(headers),
            tuple(raw_headers),
            should_close,
            compression,
            upgrade,
            chunked,
        )
        response = request._create_response(None)
        await response.start(
            _RustClientExchange(
                message,
                response_body,
                engine=self,
                connection_key=connection_key,
                native_connection=native_connection,
                force_close=self._request_forces_close(prepared),
                timer=options.timer,
                auto_decompress=options.auto_decompress,
                compression=compression,
            )
        )
        return response

    def _validate_request(self, request: "ClientRequest") -> None:
        if isinstance(request.ssl, ssl.SSLContext):
            raise NotImplementedError(
                "RustClientEngine does not support SSLContext ssl= values"
            )
        if request.proxy is not None:
            raise NotImplementedError("RustClientEngine does not support proxies")

    def _validate_prepared(self, prepared: "PreparedClientRequest") -> None:
        pass

    def _validate_options(self, options: AttemptOptions) -> None:
        if options.read_bufsize != DEFAULT_NATIVE_READ_BUFSIZE:
            raise NotImplementedError(
                "RustClientEngine does not support custom read_bufsize values"
            )
        if options.max_line_size != 8190:
            raise NotImplementedError(
                "RustClientEngine does not support custom max_line_size values"
            )
        if options.max_field_size != 8190:
            raise NotImplementedError(
                "RustClientEngine does not support custom max_field_size values"
            )

    async def _acquire_connection(
        self,
        key: tuple[str, str, int, int, bool, str | None, bytes | None],
        *,
        request: "ClientRequest",
        traces: list["Trace"],
        sock_connect: float | None,
    ) -> object:
        available = self._available_connections.get(key)
        while available:
            connection = available.pop()
            if not connection.closed:  # type: ignore[attr-defined]
                for trace in traces:
                    await trace.send_connection_reuseconn()
                return connection
            self._connections.discard(connection)
        if available == []:
            self._available_connections.pop(key, None)

        try:
            for trace in traces:
                await trace.send_connection_create_start()
            connection = await self._open_http1_connection(
                *key,
                sock_connect=sock_connect,
            )
            for trace in traces:
                await trace.send_connection_create_end()
        except cert_errors as exc:
            raise ClientConnectorCertificateError(request.connection_key, exc) from exc
        except ssl_errors as exc:
            raise ClientConnectorSSLError(request.connection_key, exc) from exc
        except asyncio.TimeoutError as exc:
            raise ConnectionTimeoutError(
                f"Connection timeout to host {key[0]}://{key[1]}:{key[2]}"
            ) from exc
        self._connections.add(connection)
        return connection

    def _release_connection(
        self,
        key: tuple[str, str, int, int, bool, str | None, bytes | None],
        connection: object,
    ) -> None:
        if self._closed or connection.closed:  # type: ignore[attr-defined]
            self._close_connection(connection)
            return
        self._available_connections.setdefault(key, []).append(connection)

    def _close_connection(self, connection: object) -> None:
        connection.close()  # type: ignore[attr-defined]
        self._connections.discard(connection)

    def _request_forces_close(self, prepared: "PreparedClientRequest") -> bool:
        connection = prepared.headers.get("Connection")
        if connection is None:
            return False
        return any(
            token.strip().lower() == "close" for token in connection.split(",")
        )

    def _tls_fingerprint(self, request: "ClientRequest") -> bytes | None:
        if type(request.ssl) is bool:
            return None
        return request.ssl.fingerprint  # type: ignore[union-attr]

    def _check_fingerprint(
        self,
        request: "ClientRequest",
        connection: object,
        expected: bytes | None,
    ) -> None:
        if expected is None or request.url.scheme != "https":
            return
        peer_certificate_der = connection.peer_certificate_der()  # type: ignore[attr-defined]
        if peer_certificate_der is None:
            self._close_connection(connection)
            raise RuntimeError("RustClientEngine TLS connection is missing peer certificate")
        got = sha256(peer_certificate_der).digest()
        if got != expected:
            self._close_connection(connection)
            raise ServerFingerprintMismatch(
                expected,
                got,
                request.url.raw_host or "",
                request.url.port or 443,
            )

    async def _open_http1_connection(
        self,
        scheme: str,
        host: str,
        port: int,
        max_headers: int,
        verify_tls: bool,
        tls_server_name: str | None,
        tls_fingerprint: bytes | None,
        *,
        sock_connect: float | None,
    ) -> object:
        try:
            from . import _rust_client
        except ImportError as exc:
            raise RuntimeError(
                "RustClientEngine requires the aiohttp._rust_client extension"
            ) from exc

        return await _rust_client.open_http1_connection(
            host,
            port,
            max_headers,
            sock_connect,
            scheme == "https",
            verify_tls,
            tls_server_name,
        )

    async def _send_http1_request(
        self,
        connection: object,
        method: str,
        target: str,
        version_major: int,
        version_minor: int,
        headers: list[tuple[str, str]],
        upload_cursor: _UploadCursor,
        *,
        content_length: int | None,
        compression: str | None,
        expect_continue: bool,
        read_until_eof: bool,
        auto_decompress: bool,
        skip_payload: bool,
        max_headers: int,
        sock_read: float | None,
    ) -> tuple[
        int,
        int,
        str,
        list[tuple[bytes, bytes]],
        object,
        bool,
        str | None,
        bool,
        bool,
    ]:
        try:
            from . import _rust_client
        except ImportError as exc:
            raise RuntimeError(
                "RustClientEngine requires the aiohttp._rust_client extension"
            ) from exc

        return await _rust_client.send_http1_request(
            connection,
            method,
            target,
            version_major,
            version_minor,
            headers,
            upload_cursor,
            content_length,
            compression,
            expect_continue,
            read_until_eof,
            auto_decompress,
            skip_payload,
            max_headers,
            sock_read,
        )
