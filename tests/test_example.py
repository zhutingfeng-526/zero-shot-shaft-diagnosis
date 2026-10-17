import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).absolute().parents[1] / "scripts/make_example.py"
SPEC = importlib.util.spec_from_file_location("make_example", SCRIPT)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


class ExampleTests(unittest.TestCase):
    def test_numeric_history_contract_and_determinism(self):
        first = EXAMPLE.example_data()
        second = EXAMPLE.example_data()
        self.assertEqual(set(first), {"raw", "rpm"})
        self.assertEqual(first["raw"].shape, (10, 48000))
        self.assertEqual(first["raw"].dtype, np.float32)
        self.assertEqual(first["rpm"].shape, ())
        self.assertEqual(float(first["rpm"]), 1200.0)
        self.assertTrue(np.isfinite(first["raw"]).all())
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_write_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data/history.npz"
            EXAMPLE.write_example(path)
            original = path.read_bytes()
            with np.load(path, allow_pickle=False) as source:
                self.assertEqual(set(source.files), {"raw", "rpm"})
                np.testing.assert_array_equal(source["raw"], EXAMPLE.example_data()["raw"])
            with self.assertRaises(FileExistsError):
                EXAMPLE.write_example(path)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
