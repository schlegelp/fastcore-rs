//! Sizing the rayon pools everything else in this crate runs on.
//!
//! Two levers, for two different situations:
//!
//! - [`with_pool`] caps a *single call*. This is what the per-function `threads` /
//!   `n_cores` arguments are wired to. It builds a fresh pool for the duration of
//!   the call, so it costs `n` thread spawns every time — fine for one big
//!   all-by-all NBLAST, wasteful if the call is small and in a loop.
//! - [`set_num_threads`] sizes the *process-wide* pool, once, and costs nothing
//!   afterwards. This is the one to reach for when the caller is itself running
//!   this library across several processes.
//!
//! That second case is not hypothetical, and it is why this module exists. The
//! default pool takes every core the process can see, which is the right answer
//! for a single interactive call and the wrong one under a process pool: navis'
//! `heal_skeleton(nl, parallel=True, n_cores=20)` on a 224-core node used to run
//! 20 workers x 224 threads = 4480 threads over 224 cores, and measured *slower*
//! than the same work on one worker (6.71 s vs 5.10 s) while burning 2.3x the CPU.
//! A library cannot detect that situation on its own — nothing tells a worker
//! process that it is one of twenty — so the caller needs a way to say so.

use std::fmt;

/// Why sizing the global pool failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadPoolError {
    /// `set_num_threads(0)`. Rayon reads a zero as "pick a default", which is the
    /// opposite of what a caller passing an explicit cap wants, so it is rejected
    /// rather than quietly honoured as "all cores".
    ZeroThreads,
    /// The global pool already exists at a different size. It is built at most
    /// once per process, and whatever ran first won: an earlier
    /// `set_num_threads`, `RAYON_NUM_THREADS`, a call to [`num_threads`], or
    /// simply the first parallel call in the process.
    AlreadyInitialised {
        /// Threads the existing pool has.
        current: usize,
        /// Threads the caller asked for.
        requested: usize,
    },
}

impl fmt::Display for ThreadPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroThreads => {
                write!(f, "thread count must be >= 1")
            }
            Self::AlreadyInitialised { current, requested } => write!(
                f,
                "the global thread pool is already running with {current} thread(s), \
                 cannot resize it to {requested}; call `set_num_threads` before the \
                 first parallel call in this process"
            ),
        }
    }
}

impl std::error::Error for ThreadPoolError {}

/// Size the process-wide rayon pool. Call before any other work in this crate.
///
/// Succeeds at most once per process — but is idempotent, so calling it repeatedly
/// with the same `n` is fine (worth knowing for callers that run this from a
/// worker-init hook that fires more than once).
///
/// # Errors
///
/// [`ThreadPoolError::ZeroThreads`] if `n == 0`, and
/// [`ThreadPoolError::AlreadyInitialised`] if the pool was already built at a
/// different size — see that variant for what builds it.
pub fn set_num_threads(n: usize) -> Result<(), ThreadPoolError> {
    if n == 0 {
        return Err(ThreadPoolError::ZeroThreads);
    }

    // Emscripten (Pyodide/WebAssembly) cannot spawn threads at all, so there is no
    // pool to size — `build_global` fails with `ENOSYS`. Everything there already
    // runs on the calling thread, which is what the caller asked for if they asked
    // for one; treat any other request as satisfied too rather than failing a call
    // that is purely advisory. Mirrors `with_pool`.
    #[cfg(target_os = "emscripten")]
    {
        return Ok(());
    }

    #[cfg(not(target_os = "emscripten"))]
    {
        if rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .is_ok()
        {
            return Ok(());
        }

        // `build_global` reports only *that* it lost the race, not to whom, so ask
        // the pool itself how big it is. Deliberately only on this path:
        // `current_num_threads` builds the global pool as a side effect, and
        // calling it first would be the thing that makes `build_global` fail.
        let current = rayon::current_num_threads();
        if current == n {
            Ok(())
        } else {
            Err(ThreadPoolError::AlreadyInitialised {
                current,
                requested: n,
            })
        }
    }
}

/// Threads in the pool the *current* context runs on: the per-call pool inside
/// [`with_pool`], otherwise the global one.
///
/// Note that asking is not free of consequence — outside a [`with_pool`] scope
/// this builds the global pool if it does not exist yet, which is exactly what
/// makes a later [`set_num_threads`] fail.
pub fn num_threads() -> usize {
    rayon::current_num_threads()
}

/// Run `f` on a rayon pool capped to `threads` workers, or on the default global
/// pool when `threads` is `None`/`Some(0)`. A fresh scoped pool is built per call
/// only when a cap is requested, so the common (uncapped) path is zero-overhead.
pub(crate) fn with_pool<R, F>(threads: Option<usize>, f: F) -> R
where
    R: Send,
    F: FnOnce() -> R + Send,
{
    // Emscripten (Pyodide/WebAssembly) cannot spawn threads at all, so there is no
    // pool to build — `ThreadPoolBuilder::build` fails with `ENOSYS`. Honour the
    // call by running serially rather than panicking on a thread count we cannot
    // satisfy; rayon's implicit pool already degrades to the calling thread there.
    #[cfg(target_os = "emscripten")]
    {
        let _ = threads;
        return f();
    }

    #[cfg(not(target_os = "emscripten"))]
    match threads {
        Some(n) if n >= 1 => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("failed to build rayon thread pool")
            .install(f),
        _ => f(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_rejected() {
        assert_eq!(set_num_threads(0), Err(ThreadPoolError::ZeroThreads));
    }

    #[test]
    fn with_pool_caps_the_call() {
        assert_eq!(with_pool(Some(3), num_threads), 3);
        assert_eq!(with_pool(Some(1), num_threads), 1);
    }

    /// `None` and `Some(0)` both mean "leave it alone", so they must agree with
    /// whatever the ambient pool is rather than with each other in isolation.
    #[test]
    fn with_pool_passes_through() {
        let ambient = with_pool(Some(5), || {
            (
                with_pool(None, num_threads),
                with_pool(Some(0), num_threads),
            )
        });
        assert_eq!(ambient, (5, 5));
    }
}
