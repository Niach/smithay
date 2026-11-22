//! A time limit on the window manager's round trips to its X server.
//!
//! The window manager asks its X server on the compositor's thread and
//! waits for the answers (`start_wm`, the geometry and properties of every
//! new window, ...). An X server that answers slowly (a client floods it,
//! holds a server grab, the server is short of memory) makes the
//! compositor wait. A window manager started with
//! [`X11Wm::start_wm_with_round_trip_limit`](super::X11Wm::start_wm_with_round_trip_limit)
//! gives its server `limit` per call: a call that has not returned by then
//! shuts the connection down. The call (and every later one) fails at once,
//! the event thread ends, and the handler hears
//! [`XwmHandler::disconnected`](super::XwmHandler::disconnected): the slow
//! server costs its own clients, not the compositor's frames.
//!
//! Only calls that can block are timed: sending (the server may not read),
//! flushing, waiting for a reply or an error, the extension and request
//! size queries. The event thread's wait for the next event is not: an
//! idle server sends nothing.

use std::{
    fmt,
    io::IoSlice,
    os::unix::net::UnixStream,
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

use x11rb::{
    connection::{
        BufWithFds, Connection, DiscardMode, RawEventAndSeqNumber, ReplyOrError, RequestConnection,
        RequestKind, SequenceNumber,
    },
    cookie::{Cookie, CookieWithFds, VoidCookie},
    errors::{ConnectionError, ParseError, ReplyOrIdError},
    protocol::{xproto::Setup, Event},
    rust_connection::{DefaultStream, RustConnection},
    utils::RawFdContainer,
    x11_utils::{ExtensionInformation, TryParse, TryParseFd, X11Error},
};

/// The window manager's connection: x11rb's, with an optional time limit
/// on every call that can block.
pub struct XwmConnection {
    conn: RustConnection,
    watch: Option<Arc<Watch>>,
}

impl fmt::Debug for XwmConnection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XwmConnection")
            .field("limit", &self.watch.as_ref().map(|watch| watch.limit))
            .finish_non_exhaustive()
    }
}

impl XwmConnection {
    /// Connects over `socket` (screen 0, as Xwayland has one), with a
    /// limit of `limit` per call when there is one: the handshake has it
    /// too.
    pub(super) fn connect(socket: UnixStream, limit: Option<Duration>) -> Result<Self, Box<dyn std::error::Error>> {
        let watch = match limit {
            Some(limit) => Some(Watch::start(socket.try_clone()?, limit)?),
            None => None,
        };
        let stream = DefaultStream::from_unix_stream(socket)?.0;
        let handshake = || RustConnection::connect_to_stream(stream, 0);
        let conn = match &watch {
            Some(watch) => watch.timed(handshake),
            None => handshake(),
        };
        match conn {
            Ok(conn) => Ok(Self { conn, watch }),
            Err(e) => {
                if let Some(watch) = &watch {
                    watch.close();
                }
                Err(e.into())
            }
        }
    }

    /// A call took longer than the limit and the connection was shut down.
    pub fn timed_out(&self) -> bool {
        self.watch.as_ref().is_some_and(|watch| watch.timed_out())
    }

    fn timed<T>(&self, call: impl FnOnce() -> T) -> T {
        match &self.watch {
            Some(watch) => watch.timed(call),
            None => call(),
        }
    }
}

impl Drop for XwmConnection {
    fn drop(&mut self) {
        if let Some(watch) = &self.watch {
            watch.close();
        }
    }
}

/// A cookie of x11rb's connection, its sequence number now ours: the
/// cookie must not discard the reply on its drop (it holds a reference and
/// a number, nothing to free).
fn adopt<C>(cookie: C, sequence: impl FnOnce(&C) -> SequenceNumber) -> SequenceNumber {
    let number = sequence(&cookie);
    std::mem::forget(cookie);
    number
}

impl RequestConnection for XwmConnection {
    type Buf = Vec<u8>;

    fn send_request_with_reply<R>(
        &self,
        bufs: &[IoSlice<'_>],
        fds: Vec<RawFdContainer>,
    ) -> Result<Cookie<'_, Self, R>, ConnectionError>
    where
        R: TryParse,
    {
        let sent = self.timed(|| self.conn.send_request_with_reply::<R>(bufs, fds))?;
        Ok(Cookie::new(self, adopt(sent, Cookie::sequence_number)))
    }

    fn send_request_with_reply_with_fds<R>(
        &self,
        bufs: &[IoSlice<'_>],
        fds: Vec<RawFdContainer>,
    ) -> Result<CookieWithFds<'_, Self, R>, ConnectionError>
    where
        R: TryParseFd,
    {
        let sent = self.timed(|| self.conn.send_request_with_reply_with_fds::<R>(bufs, fds))?;
        Ok(CookieWithFds::new(self, adopt(sent, CookieWithFds::sequence_number)))
    }

    fn send_request_without_reply(
        &self,
        bufs: &[IoSlice<'_>],
        fds: Vec<RawFdContainer>,
    ) -> Result<VoidCookie<'_, Self>, ConnectionError> {
        let sent = self.timed(|| self.conn.send_request_without_reply(bufs, fds))?;
        Ok(VoidCookie::new(self, adopt(sent, VoidCookie::sequence_number)))
    }

    fn discard_reply(&self, sequence: SequenceNumber, kind: RequestKind, mode: DiscardMode) {
        self.conn.discard_reply(sequence, kind, mode)
    }

    fn prefetch_extension_information(&self, extension_name: &'static str) -> Result<(), ConnectionError> {
        self.timed(|| self.conn.prefetch_extension_information(extension_name))
    }

    fn extension_information(
        &self,
        extension_name: &'static str,
    ) -> Result<Option<ExtensionInformation>, ConnectionError> {
        self.timed(|| self.conn.extension_information(extension_name))
    }

    fn wait_for_reply_or_raw_error(
        &self,
        sequence: SequenceNumber,
    ) -> Result<ReplyOrError<Self::Buf>, ConnectionError> {
        self.timed(|| self.conn.wait_for_reply_or_raw_error(sequence))
    }

    fn wait_for_reply(&self, sequence: SequenceNumber) -> Result<Option<Self::Buf>, ConnectionError> {
        self.timed(|| self.conn.wait_for_reply(sequence))
    }

    fn wait_for_reply_with_fds_raw(
        &self,
        sequence: SequenceNumber,
    ) -> Result<ReplyOrError<BufWithFds<Self::Buf>, Self::Buf>, ConnectionError> {
        self.timed(|| self.conn.wait_for_reply_with_fds_raw(sequence))
    }

    fn check_for_raw_error(&self, sequence: SequenceNumber) -> Result<Option<Self::Buf>, ConnectionError> {
        self.timed(|| self.conn.check_for_raw_error(sequence))
    }

    fn prefetch_maximum_request_bytes(&self) {
        self.timed(|| self.conn.prefetch_maximum_request_bytes())
    }

    fn maximum_request_bytes(&self) -> usize {
        self.timed(|| self.conn.maximum_request_bytes())
    }

    fn parse_error(&self, error: &[u8]) -> Result<X11Error, ParseError> {
        self.conn.parse_error(error)
    }

    fn parse_event(&self, event: &[u8]) -> Result<Event, ParseError> {
        self.conn.parse_event(event)
    }
}

impl Connection for XwmConnection {
    fn wait_for_raw_event_with_sequence(&self) -> Result<RawEventAndSeqNumber<Self::Buf>, ConnectionError> {
        // The event thread's: an idle server sends nothing, for as long as it likes.
        self.conn.wait_for_raw_event_with_sequence()
    }

    fn poll_for_raw_event_with_sequence(
        &self,
    ) -> Result<Option<RawEventAndSeqNumber<Self::Buf>>, ConnectionError> {
        self.conn.poll_for_raw_event_with_sequence()
    }

    fn flush(&self) -> Result<(), ConnectionError> {
        self.timed(|| self.conn.flush())
    }

    fn setup(&self) -> &Setup {
        self.conn.setup()
    }

    fn generate_id(&self) -> Result<u32, ReplyOrIdError> {
        // Asks the server once the ids run out (XC-MISC).
        self.timed(|| self.conn.generate_id())
    }
}

/// The watch over one connection's calls: a thread that sleeps while no
/// call runs and shuts the socket down when one outlasts the limit.
struct Watch {
    limit: Duration,
    socket: UnixStream,
    state: Mutex<State>,
    wake: Condvar,
}

#[derive(Default)]
struct State {
    /// Calls running now (the compositor's thread, and an event source's
    /// drop on another).
    running: u32,
    /// When the oldest of them began; a new number per stretch of calls.
    since: Option<(Instant, u64)>,
    stretch: u64,
    timed_out: bool,
    closed: bool,
}

impl Watch {
    fn start(socket: UnixStream, limit: Duration) -> std::io::Result<Arc<Self>> {
        let watch = Arc::new(Self {
            limit,
            socket,
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
        });
        let watcher = Arc::clone(&watch);
        std::thread::Builder::new()
            .name("xwm-round-trips".into())
            .spawn(move || watcher.run())?;
        Ok(watch)
    }

    fn timed<T>(&self, call: impl FnOnce() -> T) -> T {
        {
            let mut state = self.state.lock().unwrap();
            state.running += 1;
            if state.running == 1 {
                state.stretch += 1;
                state.since = Some((Instant::now(), state.stretch));
                self.wake.notify_one();
            }
        }
        let result = call();
        let mut state = self.state.lock().unwrap();
        state.running -= 1;
        if state.running == 0 {
            state.since = None;
        }
        result
    }

    fn timed_out(&self) -> bool {
        self.state.lock().unwrap().timed_out
    }

    fn close(&self) {
        self.state.lock().unwrap().closed = true;
        self.wake.notify_one();
    }

    fn run(&self) {
        let mut state = self.state.lock().unwrap();
        loop {
            if state.closed || state.timed_out {
                return;
            }
            let Some((began, stretch)) = state.since else {
                state = self.wake.wait(state).unwrap();
                continue;
            };
            let deadline = began + self.limit;
            let now = Instant::now();
            if now < deadline {
                state = self.wake.wait_timeout(state, deadline - now).unwrap().0;
                continue;
            }
            if state.since.is_some_and(|(_, current)| current == stretch) {
                state.timed_out = true;
                drop(state);
                tracing::warn!(limit = ?self.limit, "X server did not answer in time: shutting its window manager's connection down");
                // Every poll on the socket returns now, every read ends:
                // the waiting call fails, so does the event thread.
                let _ = self.socket.shutdown(std::net::Shutdown::Both);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn a_call_over_the_limit_shuts_the_socket_down() {
        let (ours, mut theirs) = UnixStream::pair().unwrap();
        let watch = Watch::start(ours, Duration::from_millis(50)).unwrap();
        watch.timed(|| std::thread::sleep(Duration::from_millis(10)));
        assert!(!watch.timed_out(), "a quick call is not cut");
        watch.timed(|| std::thread::sleep(Duration::from_millis(300)));
        assert!(watch.timed_out());
        // The peer reads the end of the stream.
        let mut buf = [0u8; 1];
        assert_eq!(theirs.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn quick_calls_in_a_row_are_never_cut() {
        let (ours, _theirs) = UnixStream::pair().unwrap();
        let watch = Watch::start(ours, Duration::from_millis(50)).unwrap();
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(300) {
            watch.timed(|| std::thread::sleep(Duration::from_millis(5)));
        }
        assert!(!watch.timed_out());
        watch.close();
    }

    #[test]
    fn a_blocked_read_returns_once_the_limit_passes() {
        let (ours, _theirs) = UnixStream::pair().unwrap();
        let reader = ours.try_clone().unwrap();
        let watch = Watch::start(ours, Duration::from_millis(50)).unwrap();
        let start = Instant::now();
        // As a reply that never comes: the read waits until the watch cuts it.
        let read = watch.timed(|| (&reader).read(&mut [0u8; 1]));
        assert!(matches!(read, Ok(0)));
        assert!(start.elapsed() < Duration::from_secs(2));
        assert!(watch.timed_out());
    }
}
