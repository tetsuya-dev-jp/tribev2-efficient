import threading

import pytest

from benchmarks.benchmark_video_prefetch import SerialFrameReader


def test_serial_control_runs_only_when_consumed():
    calls = []
    caller = threading.get_ident()

    def read(value):
        calls.append((threading.get_ident(), value))
        return value + 1

    with SerialFrameReader(max_workers=1) as reader:
        pending = reader.submit(read, 4)
        assert calls == []
        assert pending.result() == 5
        assert calls == [(caller, 4)]
        reader.submit(read, 9)
    assert calls == [(caller, 4)]


def test_serial_control_propagates_read_failure():
    def fail():
        raise OSError("read failed")

    with SerialFrameReader(max_workers=1) as reader:
        pending = reader.submit(fail)
        with pytest.raises(OSError, match="read failed"):
            pending.result()
