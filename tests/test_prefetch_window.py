"""PrefetchWindow: what is loading ahead, wrap-round, stale drops and the free-RAM floor."""
import threading
import unittest
from unittest import mock

from mull import prefetch
from mull.prefetch import PrefetchWindow


class RecordingLoader:
    def __init__(self):
        self.calls, self.lock = [], threading.Lock()

    def __call__(self, key):
        with self.lock:
            self.calls.append(key)
        return f"weights-{key}"


class TestPrefetchWindow(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(prefetch, "has_room_to_prefetch", return_value=True)
        self.room = patcher.start()
        self.addCleanup(patcher.stop)

    def test_returns_the_layer_and_loads_the_next_ones(self):
        load = RecordingLoader()
        w = PrefetchWindow([0, 1, 2, 3, 4], load, size=2)
        self.assertEqual(w.take(0), "weights-0")
        self.assertEqual(sorted(w.inflight), [1, 2])
        self.assertEqual(w.take(1), "weights-1")
        self.assertEqual(sorted(w.inflight), [2, 3])

    def test_each_layer_is_loaded_once_per_pass(self):
        load = RecordingLoader()
        w = PrefetchWindow(list(range(6)), load, size=2)
        for _ in range(3):
            for k in range(6):
                self.assertEqual(w.take(k), f"weights-{k}")
        for future in list(w._inflight.values()):   # let the last read-aheads finish
            future.result()
        # One synchronous load of layer 0, every other load from the window, +2 loaded ahead.
        self.assertEqual(len(load.calls), 3 * 6 + 2)

    def test_wraps_round_into_the_next_pass(self):
        w = PrefetchWindow([10, 11, 12, 13], RecordingLoader(), size=2)
        w.take(12)
        self.assertEqual(sorted(w.inflight), [10, 13])

    def test_stale_loads_are_dropped(self):
        w = PrefetchWindow(list(range(8)), RecordingLoader(), size=2)
        w.take(0)
        w.take(5)   # jumped ahead (e.g. a new generate() call after stopping early)
        self.assertEqual(sorted(w.inflight), [6, 7])

    def test_size_zero_loads_on_demand(self):
        load = RecordingLoader()
        w = PrefetchWindow([0, 1, 2], load, size=0)
        self.assertEqual(w.take(1), "weights-1")
        self.assertEqual(w.inflight, [])
        self.assertEqual(load.calls, [1])

    def test_load_now_is_used_for_misses(self):
        now, ahead = RecordingLoader(), RecordingLoader()
        w = PrefetchWindow([0, 1, 2], ahead, size=1, load_now=now)
        w.take(0)
        self.assertEqual(now.calls, [0])

    def test_low_free_ram_stops_read_ahead(self):
        self.room.return_value = False
        load = RecordingLoader()
        with mock.patch("builtins.print") as printed:
            w = PrefetchWindow([0, 1, 2, 3], load, size=2, size_of=lambda k: 1 << 30)
            self.assertEqual(w.take(0), "weights-0")
            self.assertEqual(w.take(1), "weights-1")
        self.assertEqual(w.inflight, [])
        self.assertEqual(load.calls, [0, 1])
        self.assertEqual(printed.call_count, 1)   # warned once, not per layer

    def test_room_check_uses_the_layer_size(self):
        w = PrefetchWindow([0, 1, 2], RecordingLoader(), size=1, size_of=lambda k: 100 + k)
        w.take(0)
        self.room.assert_called_with(101)

    def test_single_layer_order_never_prefetches(self):
        w = PrefetchWindow(["only"], RecordingLoader(), size=3)
        self.assertEqual(w.size, 0)
        self.assertEqual(w.take("only"), "weights-only")


if __name__ == "__main__":
    unittest.main()
