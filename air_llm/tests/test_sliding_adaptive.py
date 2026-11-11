"""Sliding adaptive processing: budget planning and memory_percent validation."""
import unittest

from airllm.utils import DEFAULT_MEMORY_PERCENT, plan_resident_layers, plan_sliding, validate_memory_percent

GB = 1024 ** 3


class TestPlanResidentLayers(unittest.TestCase):
    def test_nothing_fits_falls_back_to_streaming(self):
        # Budget smaller than the reserve: keep nothing, stream one layer at a time.
        self.assertEqual(plan_resident_layers({0: GB, 1: GB}, budget_bytes=GB, reserve_bytes=2 * GB), [])

    def test_everything_fits(self):
        sizes = {i: GB for i in range(4)}
        self.assertEqual(plan_resident_layers(sizes, budget_bytes=10 * GB, reserve_bytes=GB), [0, 1, 2, 3])

    def test_fills_budget_without_exceeding_it(self):
        sizes = {i: GB for i in range(10)}
        chosen = plan_resident_layers(sizes, budget_bytes=5 * GB, reserve_bytes=2 * GB)
        self.assertEqual(len(chosen), 3)
        self.assertLessEqual(sum(sizes[i] for i in chosen), 3 * GB)

    def test_smallest_modules_first(self):
        # A tiny norm and a decoder layer fit; the big lm_head does not.
        sizes = {0: 4 * GB, 1: 2 * GB, 2: 1}
        self.assertEqual(plan_resident_layers(sizes, budget_bytes=3 * GB, reserve_bytes=0), [1, 2])

    def test_result_is_in_layer_order(self):
        sizes = {5: 3, 1: 2, 3: 1}
        self.assertEqual(plan_resident_layers(sizes, budget_bytes=100, reserve_bytes=0), [1, 3, 5])


class TestPlanSliding(unittest.TestCase):
    """The strategy follows from layer sizes, free memory and the percentage."""

    LAYERS = {i: GB for i in range(80)}   # a 70B-class model: 80 layers of ~1GB

    def test_everything_fits_so_nothing_streams(self):
        plan = plan_sliding({i: GB for i in range(4)}, budget_bytes=20 * GB)
        self.assertEqual(plan.strategy, 'all-resident')
        self.assertEqual(plan.streamed_bytes, 0)

    def test_roomy_budget_keeps_layers_with_a_two_layer_window(self):
        # One large read nearly saturates the SSD, so a window beyond 2 only costs memory that is
        # better spent on resident layers.
        plan = plan_sliding(self.LAYERS, budget_bytes=30 * GB)
        self.assertEqual(plan.strategy, 'resident+window')
        self.assertEqual(plan.window, 2)
        self.assertEqual(len(plan.resident), 25)
        self.assertLessEqual(plan.resident_bytes + plan.reserve_bytes, plan.budget_bytes)

    def test_floor_is_one_layer_at_a_time(self):
        for budget in (int(2.5 * GB), GB // 2):
            plan = plan_sliding(self.LAYERS, budget_bytes=budget)
            self.assertEqual(plan.strategy, 'single-layer')
            self.assertEqual(plan.window, 0)
            self.assertEqual(plan.resident, [])

    def test_prefetching_off_means_no_window(self):
        self.assertEqual(plan_sliding(self.LAYERS, budget_bytes=30 * GB, max_window=0).window, 0)

    def test_discrete_gpu_window_is_bounded_by_host_ram(self):
        # VRAM only holds the running layer; the window waits in host RAM.
        plan = plan_sliding(self.LAYERS, budget_bytes=6 * GB, unified_memory=False,
                            host_budget_bytes=int(3.5 * GB))
        self.assertEqual(plan.window, 2)
        self.assertEqual(len(plan.resident), 4)

    def test_kv_cache_is_reserved(self):
        without = plan_sliding(self.LAYERS, budget_bytes=30 * GB)
        with_kv = plan_sliding(self.LAYERS, budget_bytes=30 * GB, kv_cache_bytes=5 * GB)
        self.assertLess(len(with_kv.resident), len(without.resident))

    def test_model_that_fits_in_free_ram_reads_from_the_page_cache(self):
        model = {i: GB for i in range(13)}
        self.assertTrue(plan_sliding(model, budget_bytes=int(3.5 * GB), free_ram_bytes=20 * GB).os_cache)

    def test_model_too_big_for_free_ram_is_read_from_the_ssd(self):
        self.assertFalse(plan_sliding(self.LAYERS, budget_bytes=6 * GB, free_ram_bytes=20 * GB).os_cache)

    def test_window_is_kept_small_enough_to_stay_in_the_page_cache(self):
        # A 4-layer window would push the streamed layers out of the page cache.
        model = {i: GB for i in range(20)}
        plan = plan_sliding(model, budget_bytes=10 * GB, free_ram_bytes=27 * GB)
        self.assertTrue(plan.os_cache)
        self.assertLessEqual(plan.window, 2)


class TestValidateMemoryPercent(unittest.TestCase):
    def test_default_is_25(self):
        self.assertEqual(DEFAULT_MEMORY_PERCENT, 25)

    def test_zero_and_none_turn_it_off(self):
        self.assertIsNone(validate_memory_percent(0))
        self.assertIsNone(validate_memory_percent(None))

    def test_valid_values(self):
        self.assertEqual(validate_memory_percent(25), 25.0)
        self.assertEqual(validate_memory_percent(100), 100.0)
        self.assertEqual(validate_memory_percent(12.5), 12.5)

    def test_rejects_out_of_range_and_non_numbers(self):
        for bad in (-1, 101, '25', True):
            with self.assertRaises(ValueError):
                validate_memory_percent(bad)


if __name__ == '__main__':
    unittest.main()
