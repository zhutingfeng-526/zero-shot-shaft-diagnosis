"""Environment inspection must not change numerical backend settings."""
import unittest

import torch

from shaft_diagnosis.runtime import runtime_report


def flags():
    return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
            torch.are_deterministic_algorithms_enabled(), torch.get_num_threads(),
            torch.get_num_interop_threads(), torch.is_autocast_enabled())


class RuntimeTests(unittest.TestCase):
    def test_report_is_serializable_and_read_only(self):
        import json
        before = flags()
        report = runtime_report()
        self.assertEqual(before, flags())
        self.assertEqual(report["packages"]["torch"], torch.__version__)
        self.assertEqual(report["backend_flags"]["matmul_allow_tf32"], before[0])
        self.assertNotIn("driver_query", report)
        json.dumps(report, allow_nan=False)

    def test_nondefault_flags_are_reported_not_reset(self):
        before = flags()
        try:
            torch.backends.cuda.matmul.allow_tf32 = not before[0]
            report = runtime_report()
            self.assertEqual(report["backend_flags"]["matmul_allow_tf32"], not before[0])
            self.assertEqual(flags()[0], not before[0])
        finally:
            torch.backends.cuda.matmul.allow_tf32 = before[0]
        self.assertEqual(before, flags())
