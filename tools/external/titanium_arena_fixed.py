#!/usr/bin/env python3
"""Run titanium_arena with robust I/O and Titanium's native session protocol.

Zquoridor's adapter speaks conventional UCI. Titanium's `uci` executable mode
is intentionally only UCI-style: its documented protocol is
`reset / position [MOVES] / go TIME_SEC / quit`, replying `ready` and
`bestmove`, not `uciok`/`readyok`. This shim keeps the generic arena/referee but
uses the correct protocol for each engine.

Pondering is disabled for Titanium so a fixed-movetime match gives neither
engine compute on the opponent's clock. Both stdout and stderr are drained on
background threads: leaving stderr as an unread PIPE can deadlock an engine
before it emits bestmove. Timeout errors include the exact move history so any
remaining engine-side stall is directly reproducible.
"""
from __future__ import annotations

import collections
import os
import queue
import selectors
import subprocess
import threading
import time
from typing import Deque, List, Sequence, Tuple

import titanium_arena as arena

# Fair fixed-movetime benchmark: do not let Titanium search on Zquoridor's clock.
os.environ["TITANIUM_PONDERING"] = "0"


def _stdout_reader_loop(stream, q: "queue.Queue[object]") -> None:
    try:
        while True:
            line = stream.readline()
            if line == "":
                q.put(None)
                return
            q.put(line.rstrip("\r\n"))
    except BaseException as exc:
        q.put(exc)


def _stderr_reader_loop(stream, tail: Deque[str]) -> None:
    try:
        while True:
            line = stream.readline()
            if line == "":
                return
            tail.append(line.rstrip("\r\n"))
    except BaseException as exc:
        tail.append(f"<stderr reader failed: {exc!r}>")


def _start_readers(self) -> None:
    if getattr(self, "_line_queue", None) is None:
        self._line_queue = queue.Queue()
        t = threading.Thread(
            target=_stdout_reader_loop,
            args=(self.p.stdout, self._line_queue),
            name=f"{self.name}-stdout-reader",
            daemon=True,
        )
        self._reader_thread = t
        t.start()

    if getattr(self, "_stderr_tail", None) is None:
        self._stderr_tail = collections.deque(maxlen=4000)
        if self.p.stderr is not None:
            t = threading.Thread(
                target=_stderr_reader_loop,
                args=(self.p.stderr, self._stderr_tail),
                name=f"{self.name}-stderr-reader",
                daemon=True,
            )
            self._stderr_reader_thread = t
            t.start()


def _stderr_tail_text(self, max_chars: int = 12000) -> str:
    tail = "\n".join(getattr(self, "_stderr_tail", ()))
    return tail[-max_chars:]


def _readline_timeout(self, timeout: float) -> str:
    _start_readers(self)
    try:
        item = self._line_queue.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError(
            f"{self.name}: timeout waiting for output; rc={self.p.poll()}; "
            f"stderr_tail={_stderr_tail_text(self)!r}"
        )

    if item is None:
        raise RuntimeError(
            f"{self.name}: process exited rc={self.p.poll()} "
            f"stderr_tail={_stderr_tail_text(self)!r}"
        )
    if isinstance(item, BaseException):
        raise RuntimeError(f"{self.name}: stdout reader failed: {item!r}")
    return str(item).strip()


_original_init = arena.UCIEngine.__init__
_original_bestmove = arena.UCIEngine.bestmove


def _init(self, argv: Sequence[str], name: str) -> None:
    if name != "titanium":
        _original_init(self, argv, name)
        # Replace selector-based reads with queue readers before subsequent I/O,
        # and drain stderr continuously for the entire match.
        _start_readers(self)
        return

    self.argv = list(argv)
    self.name = name
    self.p = subprocess.Popen(
        self.argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    assert self.p.stdin and self.p.stdout
    # close() in the original class expects a selector object.
    self.sel = selectors.DefaultSelector()
    _start_readers(self)
    self._send("reset")
    self._wait_for("ready", 20.0)


def _bestmove(self, history: Sequence[str], movetime_ms: int) -> Tuple[str, float, List[str]]:
    if self.name != "titanium":
        return _original_bestmove(self, history, movetime_ms)

    # Titanium native session syntax: no `startpos moves` keywords.
    pos = "position"
    if history:
        pos += " " + " ".join(history)
    self._send(pos)
    try:
        self._wait_for("ready", 20.0)
    except Exception as exc:
        raise RuntimeError(
            f"titanium failed while setting history ({len(history)} plies): "
            f"{' '.join(history)}; {exc}"
        ) from exc

    t0 = time.monotonic()
    # Titanium's fixed-time argument is in seconds.
    self._send(f"go {movetime_ms / 1000.0:.6f}")
    info: List[str] = []
    timeout = max(10.0, movetime_ms / 1000.0 * 8.0 + 3.0)
    deadline = time.monotonic() + timeout
    try:
        while True:
            line = self._readline_timeout(max(0.01, deadline - time.monotonic()))
            if line.startswith("info "):
                info.append(line)
                if "error" in line.lower():
                    raise RuntimeError(f"{self.name}: {line}")
            elif line.startswith("error "):
                raise RuntimeError(f"{self.name}: {line}")
            elif line.startswith("bestmove "):
                return line.split()[1], time.monotonic() - t0, info
    except Exception as exc:
        raise RuntimeError(
            f"titanium search failed at history ({len(history)} plies): "
            f"{' '.join(history)}; movetime_ms={movetime_ms}; "
            f"stderr_tail={_stderr_tail_text(self)!r}; {exc}"
        ) from exc


arena.UCIEngine._readline_timeout = _readline_timeout
arena.UCIEngine.__init__ = _init
arena.UCIEngine.bestmove = _bestmove

if __name__ == "__main__":
    raise SystemExit(arena.main())
