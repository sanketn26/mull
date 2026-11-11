"""Read-ahead must never push free RAM below the safety floor."""
import os
import unittest
from unittest import mock

from mull import memory as utils

GB = 1024 ** 3


class TestFreeRamFloor(unittest.TestCase):
    def setUp(self):
        os.environ.pop("MULL_MIN_FREE_RAM_GB", None)

    def test_floor_is_ten_percent_of_ram(self):
        with mock.patch.object(utils, "system_ram_bytes", return_value=(64 * GB, 30 * GB)):
            self.assertEqual(utils.min_free_ram_bytes(), int(6.4 * GB))

    def test_floor_is_at_least_one_gb(self):
        with mock.patch.object(utils, "system_ram_bytes", return_value=(4 * GB, 2 * GB)):
            self.assertEqual(utils.min_free_ram_bytes(), GB)

    def test_env_override(self):
        with mock.patch.dict(os.environ, {"MULL_MIN_FREE_RAM_GB": "3"}):
            self.assertEqual(utils.min_free_ram_bytes(), 3 * GB)

    def test_prefetch_allowed_only_when_it_leaves_the_floor(self):
        # 24GB machine: floor 2.4GB.
        with mock.patch.object(utils, "system_ram_bytes", return_value=(24 * GB, 4 * GB)):
            self.assertTrue(utils.has_room_to_prefetch(1 * GB))      # leaves 3GB
            self.assertFalse(utils.has_room_to_prefetch(2 * GB))     # would leave 2GB
        with mock.patch.object(utils, "system_ram_bytes", return_value=(24 * GB, 2 * GB)):
            self.assertFalse(utils.has_room_to_prefetch(0))          # already below the floor

    def test_system_ram_is_sane(self):
        total, available = utils.system_ram_bytes()
        self.assertGreater(total, 0)
        self.assertTrue(0 <= available <= total)


if __name__ == "__main__":
    unittest.main()
