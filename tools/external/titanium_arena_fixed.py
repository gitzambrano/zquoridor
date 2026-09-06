#!/usr/bin/env python3
"""Run titanium_arena with a robust stdout reader.

The original harness mixed selectors with TextIOWrapper.readline(). Python may
read several UCI lines into the TextIOWrapper buffer after a single fd-ready
event; the next selector wait then blocks even though `uciok`/`readyok` is
already buffered in user space. Patch UCIEngine before main() instantiates it
and drain stdout continuously on a background thread instead.
"""
from __future__ import annotations

import queue
import threading

import titanium_arena as arena


def _reader_loop(stream, q: "queue.Queue[object]") -> None:
    try:
        while True:
            line = stream.readline()
            if line == "":
                q.put(None)
                return
            q.put(line.rstrip("\r\n"))
    except BaseException as exc:  # surface reader failures to the caller
        q.put(exc)


def _readline_timeout(self, timeout: float) -> str:
    q = getattr(self, "_line_queue", None)
    if q is None:
        q = queue.Queue()
        self._line_queue = q
        t = threading.Thread(
            target=_reader_loop,
            args=(self.p.stdout, q),
            name=f"{self.name}-uci-reader",
            daemon=True,
        )
        self._reader_thread = t
        t.start()

    try:
        item = q.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError(f"{self.name}: timeout waiting for output")

    if item is None:
        err = ""
        if self.p.stderr:
            try:
                err = self.p.stderr.read()
            except Exception:
                pass
        raise RuntimeError(
            f"{self.name}: process exited rc={self.p.poll()} stderr={err[-2000:]}"
        )
    if isinstance(item, BaseException):
        raise RuntimeError(f"{self.name}: stdout reader failed: {item!r}")
    return str(item).strip()


arena.UCIEngine._readline_timeout = _readline_timeout

if __name__ == "__main__":
    raise SystemExit(arena.main())
