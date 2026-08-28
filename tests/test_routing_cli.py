import unittest
from unittest.mock import patch

import main
from config import RSU_LAYOUT


class RoutingCliTests(unittest.TestCase):
    def test_default_layout_unchanged_and_expanded_grid_has_unique_positions(self):
        self.assertEqual(main.cluster_layout(None), list(RSU_LAYOUT))
        expanded = main.cluster_layout(10)
        self.assertEqual(expanded[:5], list(RSU_LAYOUT))
        self.assertEqual(len({(x, y) for _, _, x, y in expanded}), 10)
        with self.assertRaises(ValueError):
            main.cluster_layout(21)  # existing per-cluster port allocation overlaps server

    def test_both_dataset_children_receive_routing_and_scenario_flags(self):
        with patch("sys.argv", ["main.py", "--dataset", "both", "--routing", "aodv", "--clusters", "2", "--vehicles", "3"]):
            with patch("main.subprocess.run") as run:
                main.main()
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            args = call.args[0]
            self.assertEqual(args[args.index("--routing") + 1], "aodv")
            self.assertEqual(args[args.index("--clusters") + 1], "2")
            self.assertEqual(args[args.index("--vehicles") + 1], "3")


if __name__ == "__main__":
    unittest.main()
