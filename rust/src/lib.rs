use std::io::{self, Write};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use brotli::DecompressorWriter;
use bytes::Bytes;
use flate2::write::{DeflateDecoder, MultiGzDecoder, ZlibDecoder};
use flate2::write::{GzEncoder, ZlibEncoder};
use flate2::Compression;
use http_body::{Body, Frame, SizeHint};
use http_body_util::BodyExt;
use hyper::client::conn::http1;
use hyper::header::{HeaderName, HeaderValue, CONNECTION, CONTENT_ENCODING};
use hyper::{Method, Request, Uri, Version};
use hyper_util::rt::TokioIo;
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyType;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{verify_tls12_signature, verify_tls13_signature, WebPkiSupportedAlgorithms};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{
    ClientConfig, DigitallySignedStruct, Error as RustlsError, RootCertStore, SignatureScheme,
};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, oneshot, Mutex, Notify};
use tokio::task::AbortHandle;
use tokio::time::timeout;
use tokio_rustls::TlsConnector;
use zstd::stream::write::Decoder as ZstdDecoder;

type RawHeaders = Vec<(Vec<u8>, Vec<u8>)>;
type NativeResponse = (
    u8,
    u16,
    String,
    RawHeaders,
    Py<ResponseBody>,
    bool,
    Option<String>,
    bool,
    bool,
);

struct ChannelBody {
    receiver: mpsc::Receiver<Result<Bytes, io::Error>>,
    size_hint: SizeHint,
}

enum RequestCompressorKind {
    Gzip(GzEncoder<Vec<u8>>),
    Deflate(ZlibEncoder<Vec<u8>>),
}

struct RequestCompressor {
    kind: RequestCompressorKind,
    emitted: usize,
}

impl RequestCompressor {
    fn new(compression: Option<&str>) -> PyResult<Option<Self>> {
        let Some(compression) = compression else {
            return Ok(None);
        };
        let kind = match compression {
            "gzip" => {
                RequestCompressorKind::Gzip(GzEncoder::new(Vec::new(), Compression::default()))
            }
            "deflate" => {
                RequestCompressorKind::Deflate(ZlibEncoder::new(Vec::new(), Compression::default()))
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported request compression {other:?}"
                )));
            }
        };
        Ok(Some(Self { kind, emitted: 0 }))
    }

    fn compress_chunk(&mut self, chunk: &[u8]) -> io::Result<Vec<u8>> {
        match &mut self.kind {
            RequestCompressorKind::Gzip(encoder) => {
                encoder.write_all(chunk)?;
                Ok(take_new_bytes(encoder.get_ref(), &mut self.emitted))
            }
            RequestCompressorKind::Deflate(encoder) => {
                encoder.write_all(chunk)?;
                Ok(take_new_bytes(encoder.get_ref(), &mut self.emitted))
            }
        }
    }

    fn finish(self) -> io::Result<Vec<u8>> {
        match self.kind {
            RequestCompressorKind::Gzip(encoder) => {
                let emitted = self.emitted;
                let bytes = encoder.finish()?;
                Ok(bytes[emitted..].to_vec())
            }
            RequestCompressorKind::Deflate(encoder) => {
                let emitted = self.emitted;
                let bytes = encoder.finish()?;
                Ok(bytes[emitted..].to_vec())
            }
        }
    }
}

fn take_new_bytes(bytes: &[u8], emitted: &mut usize) -> Vec<u8> {
    let chunk = bytes[*emitted..].to_vec();
    *emitted = bytes.len();
    chunk
}

enum ResponseDecoderKind {
    DeferredDeflate,
    Deflate(ZlibDecoder<Vec<u8>>),
    RawDeflate(DeflateDecoder<Vec<u8>>),
    Gzip(MultiGzDecoder<Vec<u8>>),
    Brotli(DecompressorWriter<Vec<u8>>),
    Zstd(ZstdDecoder<'static, Vec<u8>>),
}

struct ResponseDecoder {
    encoding: String,
    kind: ResponseDecoderKind,
    emitted: usize,
}

impl ResponseDecoder {
    fn new(encoding: &str) -> PyResult<Self> {
        let kind = match encoding {
            "deflate" => ResponseDecoderKind::DeferredDeflate,
            "gzip" => ResponseDecoderKind::Gzip(MultiGzDecoder::new(Vec::new())),
            "br" => ResponseDecoderKind::Brotli(DecompressorWriter::new(Vec::new(), 4096)),
            "zstd" => ResponseDecoderKind::Zstd(ZstdDecoder::new(Vec::new()).map_err(|error| {
                PyRuntimeError::new_err(format!(
                    "failed to initialize zstd response decoder: {error}"
                ))
            })?),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported response compression {other:?}"
                )));
            }
        };
        Ok(Self {
            encoding: encoding.to_owned(),
            kind,
            emitted: 0,
        })
    }

    fn decode_chunk(&mut self, chunk: &[u8]) -> io::Result<Vec<u8>> {
        if matches!(self.kind, ResponseDecoderKind::DeferredDeflate) {
            self.kind = if chunk.first().is_some_and(|byte| byte & 0xF == 8) {
                ResponseDecoderKind::Deflate(ZlibDecoder::new(Vec::new()))
            } else {
                ResponseDecoderKind::RawDeflate(DeflateDecoder::new(Vec::new()))
            };
        }

        match &mut self.kind {
            ResponseDecoderKind::DeferredDeflate => Ok(Vec::new()),
            ResponseDecoderKind::Deflate(decoder) => {
                decoder.write_all(chunk)?;
                decoder.flush()?;
                Ok(take_new_bytes(decoder.get_ref(), &mut self.emitted))
            }
            ResponseDecoderKind::RawDeflate(decoder) => {
                decoder.write_all(chunk)?;
                decoder.flush()?;
                Ok(take_new_bytes(decoder.get_ref(), &mut self.emitted))
            }
            ResponseDecoderKind::Gzip(decoder) => {
                decoder.write_all(chunk)?;
                decoder.flush()?;
                Ok(take_new_bytes(decoder.get_ref(), &mut self.emitted))
            }
            ResponseDecoderKind::Brotli(decoder) => {
                decoder.write_all(chunk)?;
                decoder.flush()?;
                Ok(take_new_bytes(decoder.get_ref(), &mut self.emitted))
            }
            ResponseDecoderKind::Zstd(decoder) => {
                decoder.write_all(chunk)?;
                decoder.flush()?;
                Ok(take_new_bytes(decoder.get_ref(), &mut self.emitted))
            }
        }
    }

    fn finish(self) -> io::Result<Vec<u8>> {
        match self.kind {
            ResponseDecoderKind::DeferredDeflate => Ok(Vec::new()),
            ResponseDecoderKind::Deflate(decoder) => {
                let bytes = decoder.finish()?;
                Ok(bytes[self.emitted..].to_vec())
            }
            ResponseDecoderKind::RawDeflate(decoder) => {
                let bytes = decoder.finish()?;
                Ok(bytes[self.emitted..].to_vec())
            }
            ResponseDecoderKind::Gzip(decoder) => {
                let bytes = decoder.finish()?;
                Ok(bytes[self.emitted..].to_vec())
            }
            ResponseDecoderKind::Brotli(mut decoder) => {
                decoder.close()?;
                let bytes = decoder.into_inner().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "brotli decoder close failed")
                })?;
                Ok(bytes[self.emitted..].to_vec())
            }
            ResponseDecoderKind::Zstd(mut decoder) => {
                decoder.flush()?;
                let bytes = decoder.into_inner();
                Ok(bytes[self.emitted..].to_vec())
            }
        }
    }
}

const BODY_WAITING: u8 = 0;
const BODY_CONTINUE: u8 = 1;
const BODY_SKIP: u8 = 2;

struct RequestBodyGate {
    state: AtomicU8,
    notify: Notify,
}

impl RequestBodyGate {
    fn new(expect_continue: bool) -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU8::new(if expect_continue {
                BODY_WAITING
            } else {
                BODY_CONTINUE
            }),
            notify: Notify::new(),
        })
    }

    fn continue_upload(&self) {
        self.set_state(BODY_CONTINUE);
    }

    fn skip_upload(&self) {
        self.set_state(BODY_SKIP);
    }

    fn set_state(&self, next: u8) {
        if self
            .state
            .compare_exchange(BODY_WAITING, next, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.notify.notify_waiters();
        }
    }

    async fn should_send_body(&self) -> bool {
        loop {
            let notified = self.notify.notified();
            match self.state.load(Ordering::Acquire) {
                BODY_CONTINUE => return true,
                BODY_SKIP => return false,
                _ => notified.await,
            }
        }
    }
}

#[pyclass]
struct NativeConnection {
    sender: Arc<Mutex<http1::SendRequest<ChannelBody>>>,
    closed: Arc<AtomicBool>,
    abort_handle: AbortHandle,
    peer_certificate_der: Option<Vec<u8>>,
}

#[pymethods]
impl NativeConnection {
    #[getter]
    fn closed(&self) -> bool {
        self.closed.load(Ordering::Relaxed)
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Relaxed);
        self.abort_handle.abort();
    }

    fn peer_certificate_der<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.peer_certificate_der
            .as_deref()
            .map(|cert| PyBytes::new(py, cert))
    }
}

#[pyclass]
struct ResponseBody {
    body: Option<hyper::body::Incoming>,
    locals: pyo3_async_runtimes::TaskLocals,
    read_timeout: Option<Duration>,
    decoder: Option<ResponseDecoder>,
    total_raw_bytes: u64,
}

#[pymethods]
impl ResponseBody {
    #[getter]
    fn total_raw_bytes(&self) -> u64 {
        self.total_raw_bytes
    }

    fn next_chunk<'py>(slf: Py<Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let locals = slf.borrow(py).locals.clone();
        pyo3_async_runtimes::tokio::future_into_py_with_locals(
            py,
            locals.clone(),
            pyo3_async_runtimes::tokio::scope(locals, async move {
                let (body, read_timeout, mut decoder) = Python::attach(|py| {
                    let mut body = slf.borrow_mut(py);
                    (body.body.take(), body.read_timeout, body.decoder.take())
                });
                let Some(mut body) = body else {
                    return Ok(None);
                };
                loop {
                    let frame = if let Some(read_timeout) = read_timeout {
                        timeout(read_timeout, body.frame())
                            .await
                            .map_err(|_| read_timeout_error())?
                    } else {
                        body.frame().await
                    };
                    let Some(frame) = frame else {
                        let chunk = match decoder.take() {
                            Some(decoder) => {
                                let encoding = decoder.encoding.clone();
                                decoder
                                    .finish()
                                    .map_err(|error| response_decode_error(&encoding, error))?
                            }
                            None => Vec::new(),
                        };
                        if chunk.is_empty() {
                            return Ok(None);
                        }
                        return Ok(Some(chunk));
                    };
                    let frame = frame.map_err(hyper_error)?;
                    if let Ok(data) = frame.into_data() {
                        let raw_chunk = data.to_vec();
                        Python::attach(|py| {
                            slf.borrow_mut(py).total_raw_bytes += raw_chunk.len() as u64
                        });
                        let chunk = match decoder.as_mut() {
                            Some(decoder) => decoder
                                .decode_chunk(&raw_chunk)
                                .map_err(|error| response_decode_error(&decoder.encoding, error))?,
                            None => raw_chunk,
                        };
                        if !chunk.is_empty() {
                            Python::attach(|py| {
                                let mut response_body = slf.borrow_mut(py);
                                response_body.body = Some(body);
                                response_body.decoder = decoder;
                            });
                            return Ok(Some(chunk));
                        }
                    }
                }
            }),
        )
    }

    fn close(&mut self) {
        self.body = None;
        self.decoder = None;
    }
}

impl Body for ChannelBody {
    type Data = Bytes;
    type Error = io::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        match self.receiver.poll_recv(cx) {
            Poll::Ready(Some(Ok(chunk))) => Poll::Ready(Some(Ok(Frame::data(chunk)))),
            Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(error))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.receiver.is_closed() && self.receiver.is_empty()
    }

    fn size_hint(&self) -> SizeHint {
        self.size_hint.clone()
    }
}

#[pyfunction]
fn is_available() -> bool {
    true
}

#[pyfunction]
#[pyo3(signature = (
    host,
    port,
    max_headers,
    sock_connect,
    use_tls,
    verify_tls,
    tls_server_name,
))]
fn open_http1_connection<'py>(
    py: Python<'py>,
    host: String,
    port: u16,
    max_headers: usize,
    sock_connect: Option<f64>,
    use_tls: bool,
    verify_tls: bool,
    tls_server_name: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
    pyo3_async_runtimes::tokio::future_into_py_with_locals(
        py,
        locals.clone(),
        pyo3_async_runtimes::tokio::scope(locals, async move {
            open_http1_connection_impl(
                &host,
                port,
                max_headers,
                sock_connect,
                use_tls,
                verify_tls,
                tls_server_name.as_deref(),
            )
            .await
        }),
    )
}

async fn open_http1_connection_impl(
    host: &str,
    port: u16,
    max_headers: usize,
    sock_connect: Option<f64>,
    use_tls: bool,
    verify_tls: bool,
    tls_server_name: Option<&str>,
) -> PyResult<Py<NativeConnection>> {
    if max_headers == 0 {
        return Err(PyValueError::new_err("max_headers must be positive"));
    }

    let connect = TcpStream::connect((host, port));
    let stream = if let Some(sock_connect) = timeout_duration(sock_connect)? {
        timeout(sock_connect, connect)
            .await
            .map_err(|_| connect_timeout_error())?
            .map_err(io_error)?
    } else {
        connect.await.map_err(io_error)?
    };

    if use_tls {
        let config = tls_client_config(verify_tls)?;
        let connector = TlsConnector::from(config);
        let tls_server_name = tls_server_name.unwrap_or(host);
        let server_name = ServerName::try_from(tls_server_name.to_owned()).map_err(|_| {
            PyValueError::new_err(format!("invalid TLS server name {tls_server_name:?}"))
        })?;
        let stream = connector
            .connect(server_name, stream)
            .await
            .map_err(tls_error)?;
        let peer_certificate_der = stream
            .get_ref()
            .1
            .peer_certificates()
            .and_then(|certs| certs.first())
            .map(|cert| cert.as_ref().to_vec());
        return open_http1_connection_over_io(stream, max_headers, peer_certificate_der).await;
    }

    open_http1_connection_over_io(stream, max_headers, None).await
}

async fn open_http1_connection_over_io<I>(
    stream: I,
    max_headers: usize,
    peer_certificate_der: Option<Vec<u8>>,
) -> PyResult<Py<NativeConnection>>
where
    I: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let (sender, connection) = http1::Builder::new()
        .max_headers(max_headers)
        .handshake(TokioIo::new(stream))
        .await
        .map_err(hyper_error)?;
    let closed = Arc::new(AtomicBool::new(false));
    let connection_closed = closed.clone();
    let connection_task = tokio::spawn(async move {
        if let Err(error) = connection.await {
            eprintln!("aiohttp Rust client HTTP/1 connection error: {error}");
        }
        connection_closed.store(true, Ordering::Relaxed);
    });
    let abort_handle = connection_task.abort_handle();

    Python::attach(|py| {
        Py::new(
            py,
            NativeConnection {
                sender: Arc::new(Mutex::new(sender)),
                closed,
                abort_handle,
                peer_certificate_der,
            },
        )
    })
}

#[derive(Debug)]
struct NoCertificateVerification {
    supported_algs: WebPkiSupportedAlgorithms,
}

impl ServerCertVerifier for NoCertificateVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, RustlsError> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, RustlsError> {
        verify_tls12_signature(message, cert, dss, &self.supported_algs)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, RustlsError> {
        verify_tls13_signature(message, cert, dss, &self.supported_algs)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.supported_algs.supported_schemes()
    }
}

fn tls_client_config(verify_tls: bool) -> PyResult<Arc<ClientConfig>> {
    let mut config = if verify_tls {
        let mut roots = RootCertStore::empty();
        let certs = rustls_native_certs::load_native_certs();
        if certs.certs.is_empty() && !certs.errors.is_empty() {
            return Err(PyOSError::new_err(format!(
                "failed to load native root certificates: {:?}",
                certs.errors[0]
            )));
        }
        let (added, _ignored) = roots.add_parsable_certificates(certs.certs);
        if added == 0 {
            return Err(PyOSError::new_err("no native root certificates found"));
        }
        ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth()
    } else {
        let supported_algs = rustls::crypto::CryptoProvider::get_default()
            .map(|provider| provider.signature_verification_algorithms)
            .unwrap_or_else(|| {
                rustls::crypto::aws_lc_rs::default_provider().signature_verification_algorithms
            });
        ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(NoCertificateVerification {
                supported_algs,
            }))
            .with_no_client_auth()
    };
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(Arc::new(config))
}

#[pyfunction]
#[pyo3(signature = (
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
    sock_read
))]
#[allow(clippy::too_many_arguments)]
fn send_http1_request<'py>(
    py: Python<'py>,
    connection: Py<NativeConnection>,
    method: String,
    target: String,
    version_major: u8,
    version_minor: u8,
    headers: Vec<(String, String)>,
    upload_cursor: Py<PyAny>,
    content_length: Option<u64>,
    compression: Option<String>,
    expect_continue: bool,
    read_until_eof: bool,
    auto_decompress: bool,
    skip_payload: bool,
    max_headers: usize,
    sock_read: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
    pyo3_async_runtimes::tokio::future_into_py_with_locals(
        py,
        locals.clone(),
        pyo3_async_runtimes::tokio::scope(locals.clone(), async move {
            send_http1_request_impl(
                connection,
                &method,
                &target,
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
                locals,
            )
            .await
        }),
    )
}

#[allow(clippy::too_many_arguments)]
async fn send_http1_request_impl(
    connection: Py<NativeConnection>,
    method: &str,
    target: &str,
    version_major: u8,
    version_minor: u8,
    headers: Vec<(String, String)>,
    upload_cursor: Py<PyAny>,
    content_length: Option<u64>,
    compression: Option<String>,
    expect_continue: bool,
    read_until_eof: bool,
    auto_decompress: bool,
    skip_payload: bool,
    max_headers: usize,
    sock_read: Option<f64>,
    locals: pyo3_async_runtimes::TaskLocals,
) -> PyResult<NativeResponse> {
    if max_headers == 0 {
        return Err(PyValueError::new_err("max_headers must be positive"));
    }
    let (sender, closed) = Python::attach(|py| {
        let connection = connection.borrow(py);
        (connection.sender.clone(), connection.closed.clone())
    });
    if closed.load(Ordering::Relaxed) {
        return Err(PyOSError::new_err("connection is closed"));
    }

    let read_timeout = timeout_duration(sock_read)?;
    let compressor = RequestCompressor::new(compression.as_deref())?;
    let body_gate = RequestBodyGate::new(expect_continue);
    let (body, body_driver, body_done) = make_request_body(
        upload_cursor,
        content_length,
        compressor,
        locals.clone(),
        body_gate.clone(),
    );
    tokio::spawn(body_driver);
    let mut request = build_request(method, target, version_major, version_minor, headers, body)?;
    if expect_continue {
        let informational_gate = body_gate.clone();
        hyper::ext::on_informational(&mut request, move |response| {
            if response.status().as_u16() != 101 {
                informational_gate.continue_upload();
            }
        });
    }
    let mut sender = sender.lock().await;
    let response = wait_for_response_headers(
        sender.send_request(request),
        body_done,
        read_timeout,
        closed,
    )
    .await;
    body_gate.skip_upload();
    let response = response?;
    drop(sender);
    let (parts, body) = response.into_parts();
    let version_minor = match parts.version {
        Version::HTTP_10 => 0,
        Version::HTTP_11 => 1,
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "RustClientEngine received unsupported response version {other:?}"
            )));
        }
    };
    let raw_headers = parts
        .headers
        .iter()
        .map(|(name, value)| (name.as_str().as_bytes().to_vec(), value.as_bytes().to_vec()))
        .collect::<RawHeaders>();
    let chunked = is_chunked(&raw_headers);
    let has_content_length = parts.headers.contains_key("content-length");
    let should_close = should_close(
        version_minor,
        parts.status.as_u16(),
        &parts.headers,
        has_content_length,
        chunked,
    );
    let compression = response_compression(&parts.headers);
    let upgrade = is_upgrade(&parts.headers);
    let body = if skip_payload
        || response_has_no_body(parts.status.as_u16())
        || (!read_until_eof && !has_content_length && !chunked)
    {
        None
    } else {
        Some(body)
    };
    let decoder = if auto_decompress {
        compression
            .as_deref()
            .map(ResponseDecoder::new)
            .transpose()?
    } else {
        None
    };
    let body = Python::attach(|py| {
        Py::new(
            py,
            ResponseBody {
                body,
                locals: locals.clone(),
                read_timeout,
                decoder,
                total_raw_bytes: 0,
            },
        )
    })?;

    Ok((
        version_minor,
        parts.status.as_u16(),
        parts
            .status
            .canonical_reason()
            .unwrap_or_default()
            .to_owned(),
        raw_headers,
        body,
        should_close,
        compression,
        upgrade,
        chunked,
    ))
}

fn build_request(
    method: &str,
    target: &str,
    version_major: u8,
    version_minor: u8,
    headers: Vec<(String, String)>,
    body: ChannelBody,
) -> PyResult<Request<ChannelBody>> {
    let method = Method::from_bytes(method.as_bytes())
        .map_err(|error| PyValueError::new_err(format!("invalid request method: {error}")))?;
    let uri = target
        .parse::<Uri>()
        .map_err(|error| PyValueError::new_err(format!("invalid request target: {error}")))?;
    let version = match (version_major, version_minor) {
        (1, 0) => Version::HTTP_10,
        (1, 1) => Version::HTTP_11,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported request HTTP version {version_major}.{version_minor}"
            )));
        }
    };

    let mut request = Request::builder()
        .method(method)
        .uri(uri)
        .version(version)
        .body(body)
        .map_err(|error| PyValueError::new_err(format!("invalid request: {error}")))?;
    for (name, value) in headers {
        let name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|error| PyValueError::new_err(format!("invalid header name: {error}")))?;
        let value = HeaderValue::from_str(&value)
            .map_err(|error| PyValueError::new_err(format!("invalid header value: {error}")))?;
        request.headers_mut().append(name, value);
    }
    Ok(request)
}

fn make_request_body(
    upload_cursor: Py<PyAny>,
    content_length: Option<u64>,
    mut compressor: Option<RequestCompressor>,
    locals: pyo3_async_runtimes::TaskLocals,
    body_gate: Arc<RequestBodyGate>,
) -> (
    ChannelBody,
    impl std::future::Future<Output = ()> + Send + 'static,
    oneshot::Receiver<()>,
) {
    let (sender, receiver) = mpsc::channel(1);
    let mut size_hint = SizeHint::new();
    if let Some(length) = content_length {
        size_hint.set_exact(length);
    }
    let body = ChannelBody {
        receiver,
        size_hint,
    };
    let (done_sender, done_receiver) = oneshot::channel();
    let driver = async move {
        if !body_gate.should_send_body().await {
            let _ = done_sender.send(());
            return;
        }
        loop {
            let next_chunk = Python::attach(|py| {
                pyo3_async_runtimes::into_future_with_locals(
                    &locals,
                    upload_cursor.bind(py).call_method0("next_chunk")?,
                )
            });
            let next_chunk = match next_chunk {
                Ok(next_chunk) => next_chunk,
                Err(error) => {
                    let _ = sender.send(Err(py_error(error))).await;
                    break;
                }
            };
            let chunk = match next_chunk.await {
                Ok(chunk) => Python::attach(|py| chunk.extract::<Option<Vec<u8>>>(py)),
                Err(error) => Err(error),
            };
            let chunk = match chunk {
                Ok(Some(chunk)) => chunk,
                Ok(None) => break,
                Err(error) => {
                    let _ = sender.send(Err(py_error(error))).await;
                    break;
                }
            };
            let chunk = if let Some(compressor) = compressor.as_mut() {
                match compressor.compress_chunk(&chunk) {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        let _ = sender.send(Err(error)).await;
                        break;
                    }
                }
            } else {
                chunk
            };
            if !chunk.is_empty() && sender.send(Ok(Bytes::from(chunk))).await.is_err() {
                break;
            }
        }
        if let Some(compressor) = compressor {
            match compressor.finish() {
                Ok(chunk) => {
                    if !chunk.is_empty() {
                        let _ = sender.send(Ok(Bytes::from(chunk))).await;
                    }
                }
                Err(error) => {
                    let _ = sender.send(Err(error)).await;
                }
            }
        }
        let _ = done_sender.send(());
    };
    (body, driver, done_receiver)
}

async fn wait_for_response_headers<F>(
    response: F,
    mut body_done: oneshot::Receiver<()>,
    read_timeout: Option<Duration>,
    closed: Arc<AtomicBool>,
) -> PyResult<hyper::Response<hyper::body::Incoming>>
where
    F: std::future::Future<Output = Result<hyper::Response<hyper::body::Incoming>, hyper::Error>>,
{
    tokio::pin!(response);

    let response = if let Some(read_timeout) = read_timeout {
        tokio::select! {
            response = &mut response => response,
            _ = &mut body_done => {
                timeout(read_timeout, &mut response)
                    .await
                    .map_err(|_| read_timeout_error())?
            }
        }
    } else {
        response.await
    };

    response.map_err(|error| {
        closed.store(true, Ordering::Relaxed);
        hyper_error(error)
    })
}

fn timeout_duration(seconds: Option<f64>) -> PyResult<Option<Duration>> {
    let Some(seconds) = seconds else {
        return Ok(None);
    };
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(PyValueError::new_err(
            "timeout must be a finite non-negative value",
        ));
    }
    Ok(Some(Duration::from_secs_f64(seconds)))
}

fn response_has_no_body(code: u16) -> bool {
    (100..200).contains(&code) || code == 204 || code == 304
}

fn response_compression(headers: &hyper::HeaderMap) -> Option<String> {
    let value = headers.get(CONTENT_ENCODING)?.to_str().ok()?;
    let lower = value.to_ascii_lowercase();
    matches!(lower.as_str(), "gzip" | "deflate" | "br" | "zstd").then_some(value.to_owned())
}

fn is_chunked(headers: &RawHeaders) -> bool {
    header_values(headers, b"transfer-encoding").any(|value| {
        std::str::from_utf8(value)
            .ok()
            .and_then(|value| value.rsplit(',').next())
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("chunked"))
    })
}

fn is_upgrade(headers: &hyper::HeaderMap) -> bool {
    header_has_token(headers, CONNECTION, "upgrade") && headers.contains_key("upgrade")
}

fn should_close(
    version_minor: u8,
    code: u16,
    headers: &hyper::HeaderMap,
    has_content_length: bool,
    chunked: bool,
) -> bool {
    if header_has_token(headers, CONNECTION, "close") {
        return true;
    }
    if header_has_token(headers, CONNECTION, "keep-alive") {
        return false;
    }
    if version_minor == 0 {
        return true;
    }
    if response_has_no_body(code) {
        return false;
    }
    !(has_content_length || chunked)
}

fn header_has_token(headers: &hyper::HeaderMap, name: HeaderName, token: &str) -> bool {
    headers.get_all(name).iter().any(|value| {
        value.to_str().ok().is_some_and(|value| {
            value
                .split(',')
                .map(str::trim)
                .any(|part| part.eq_ignore_ascii_case(token))
        })
    })
}

fn header_values<'a>(
    headers: &'a RawHeaders,
    name: &'a [u8],
) -> impl Iterator<Item = &'a [u8]> + 'a {
    headers.iter().filter_map(move |(header_name, value)| {
        header_name
            .eq_ignore_ascii_case(name)
            .then_some(value.as_slice())
    })
}

fn io_error(error: io::Error) -> PyErr {
    PyOSError::new_err(error.to_string())
}

fn tls_error(error: io::Error) -> PyErr {
    if error
        .get_ref()
        .and_then(|error| error.downcast_ref::<rustls::Error>())
        .is_some_and(|error| matches!(error, rustls::Error::InvalidCertificate(_)))
    {
        return python_ssl_error("SSLCertVerificationError", error.to_string());
    }
    python_ssl_error("SSLError", error.to_string())
}

fn python_ssl_error(class_name: &str, message: String) -> PyErr {
    Python::attach(|py| {
        let Ok(ssl) = py.import("ssl") else {
            return PyOSError::new_err(message);
        };
        let Ok(exception_type) = ssl
            .getattr(class_name)
            .and_then(|exception_type| Ok(exception_type.cast_into::<PyType>()?))
        else {
            return PyOSError::new_err(message);
        };
        PyErr::from_type(exception_type, (message,))
    })
}

fn connect_timeout_error() -> PyErr {
    PyTimeoutError::new_err("Connection timeout")
}

fn read_timeout_error() -> PyErr {
    PyTimeoutError::new_err("Timeout on reading data from socket")
}

fn py_error(error: PyErr) -> io::Error {
    io::Error::other(error.to_string())
}

fn response_decode_error(encoding: &str, error: io::Error) -> PyErr {
    PyValueError::new_err(format!(
        "Can not decode content-encoding: {encoding}: {error}"
    ))
}

fn hyper_error(error: hyper::Error) -> PyErr {
    if error.is_parse() {
        return response_parse_error(error);
    }
    PyOSError::new_err(error.to_string())
}

fn response_parse_error(error: hyper::Error) -> PyErr {
    let raw_message = error.to_string();
    let message = if raw_message == "message head is too large" {
        "Too many headers received".to_owned()
    } else {
        raw_message
    };
    Python::attach(|py| {
        let Ok(http_exceptions) = py.import("aiohttp.http_exceptions") else {
            return PyOSError::new_err(message);
        };
        let Ok(exception_type) = http_exceptions
            .getattr("BadHttpMessage")
            .and_then(|exception_type| Ok(exception_type.cast_into::<PyType>()?))
        else {
            return PyOSError::new_err(message);
        };
        PyErr::from_type(exception_type, (message,))
    })
}

#[pymodule]
fn _rust_client(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeConnection>()?;
    module.add_class::<ResponseBody>()?;
    module.add_function(wrap_pyfunction!(is_available, module)?)?;
    module.add_function(wrap_pyfunction!(open_http1_connection, module)?)?;
    module.add_function(wrap_pyfunction!(send_http1_request, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::header::{HeaderMap, HeaderValue};

    #[test]
    fn detects_response_metadata() {
        let raw_headers = vec![
            (b"Transfer-Encoding".to_vec(), b"gzip, chunked".to_vec()),
            (b"Content-Encoding".to_vec(), b"gzip".to_vec()),
        ];
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        headers.insert(CONNECTION, HeaderValue::from_static("keep-alive"));

        assert!(is_chunked(&raw_headers));
        assert_eq!(response_compression(&headers), Some("gzip".to_owned()));
        assert!(!should_close(1, 200, &headers, false, true));
    }
}
