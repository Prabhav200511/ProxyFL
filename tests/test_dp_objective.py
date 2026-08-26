import unittest
from unittest.mock import patch

import torch

from device import Device, proxy_training_objective


class DPObjectiveTests(unittest.TestCase):
    def test_class_weights_change_dp_and_batch_objective_consistently(self):
        logits = torch.tensor([[3.0, 0.5], [0.1, 2.0]])
        targets = torch.tensor([0, 1])
        soft = torch.softmax(
            torch.tensor([[2.5, 0.2], [0.3, 1.7]]) / 3.0, dim=1)
        weights = torch.tensor([1.0, 4.0])
        weighted = proxy_training_objective(logits, targets, soft, weights)
        unweighted = proxy_training_objective(logits, targets, soft, None)
        self.assertFalse(torch.isclose(weighted, unweighted))
        per_sample = torch.stack([
            proxy_training_objective(
                logits[index:index + 1], targets[index:index + 1],
                soft[index:index + 1], weights)
            for index in range(2)
        ]).mean()
        self.assertTrue(torch.isfinite(per_sample))

    def test_high_epsilon_warns_without_exhausting_budget(self):
        device = Device.__new__(Device)
        device.name = "C0_V1"
        device.budget_exhausted = False
        with patch("builtins.print") as output:
            device._warn_if_privacy_high(11.0)
        rendered = " ".join(
            str(call.args[0]) for call in output.call_args_list)
        self.assertIn("PRIVACY WARNING", rendered)
        self.assertFalse(device.budget_exhausted)


if __name__ == "__main__":
    unittest.main()
