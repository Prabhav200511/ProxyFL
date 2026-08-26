import unittest

import torch

from models import filter_trusted_weights


class TrustFilterTests(unittest.TestCase):
    def test_deviation_uses_supplied_global_reference(self):
        reference = {"w": torch.tensor([0.0])}
        entries = [
            ("honest", {"w": torch.tensor([0.1])}),
            ("outlier", {"w": torch.tensor([10.0])}),
        ]
        trusted, log = filter_trusted_weights(
            entries, reference_weights=reference, threshold=1.0)
        outcomes = {name: accepted for name, _, accepted in log}
        self.assertEqual(outcomes, {"honest": True, "outlier": False})
        self.assertEqual(len(trusted), 1)


if __name__ == "__main__":
    unittest.main()
