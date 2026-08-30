import io
import threading
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import main
from config import RSU_LAYOUT, TOTAL_ROUNDS
from round_coordinator import RoundCoordinator


class RoutingCliTests(unittest.TestCase):
    def test_main_wait_surfaces_worker_abort_without_waiting_for_server(self):
        waiter = getattr(main, "wait_for_server_completion", None)
        self.assertIsNotNone(waiter, "main must observe coordinator aborts")
        coordinator = RoundCoordinator(["A"], ["R"])
        coordinator.abort("A worker failed")

        with self.assertRaisesRegex(RuntimeError, "A worker failed"):
            waiter(threading.Event(), coordinator, poll_interval=0.001)

    def test_worker_join_has_one_bounded_deadline(self):
        joiner = getattr(main, "join_device_workers", None)
        self.assertIsNotNone(joiner, "device joins must have a bounded deadline")
        release = threading.Event()
        worker = threading.Thread(
            target=release.wait, kwargs={"timeout": 1}, daemon=True)
        worker.start()
        device = type("DeviceStub", (), {
            "name": "A",
            "training_thread": worker,
        })()
        coordinator = RoundCoordinator(["A"], ["R"])

        try:
            with self.assertRaisesRegex(RuntimeError, "A"):
                joiner([device], coordinator, timeout=0.01)
            with self.assertRaisesRegex(RuntimeError, "did not finish"):
                coordinator.raise_if_aborted()
        finally:
            release.set()
            worker.join(timeout=1)

    def test_round_audit_rejects_missing_server_or_vehicle_rounds(self):
        audit = getattr(main, "validate_round_outputs", None)
        self.assertIsNotNone(audit, "a final configured-round audit is required")
        training_logger = type("Log", (), {
            "global_proxy_acc": {1: 70.0, 2: 71.0},
            "vehicle_train_loss": {
                "A": {1: 0.5, 2: 0.4},
                "B": {1: 0.6},
            },
        })()

        with self.assertRaisesRegex(RuntimeError, "B.*rounds.*2"):
            audit(training_logger, 2, ["A", "B"])

    def test_round_audit_accepts_exact_configured_sequence(self):
        audit = getattr(main, "validate_round_outputs", None)
        self.assertIsNotNone(audit, "a final configured-round audit is required")
        training_logger = type("Log", (), {
            "global_proxy_acc": {1: 70.0, 2: 71.0},
            "vehicle_train_loss": {
                "A": {1: 0.5, 2: 0.4},
                "B": {1: 0.6, 2: 0.5},
            },
        })()

        audit(training_logger, 2, ["A", "B"])

    def test_default_run_uses_configured_total_rounds(self):
        captured = []
        with patch("sys.argv", ["main.py", "--dataset", "vanet"]), \
                patch("main.run_single_simulation",
                      side_effect=lambda **kwargs: captured.append(kwargs)):
            main.main()

        self.assertEqual(captured[0]["total_rounds"], TOTAL_ROUNDS)

    def test_standard_runs_use_dataset_appropriate_routing_defaults(self):
        captured = []

        def record_run(**kwargs):
            captured.append(kwargs)

        with patch("sys.argv", ["main.py", "--dataset", "vanet"]), \
                patch("main.run_single_simulation", side_effect=record_run):
            main.main()
        with patch("sys.argv", ["main.py", "--dataset", "mnist"]), \
                patch("main.run_single_simulation", side_effect=record_run):
            main.main()
        with patch("sys.argv", [
                "main.py", "--dataset", "vanet", "--routing", "direct"
        ]), patch("main.run_single_simulation", side_effect=record_run):
            main.main()

        self.assertEqual(
            [run["routing"] for run in captured],
            ["aodv", "direct", "direct"],
        )

    def test_both_dataset_default_uses_aodv_only_for_vanet(self):
        with patch("sys.argv", ["main.py", "--dataset", "both"]):
            with patch("main.subprocess.run") as run:
                main.main()

        commands = [call.args[0] for call in run.call_args_list]
        routing_by_dataset = {
            command[command.index("--dataset") + 1]:
            command[command.index("--routing") + 1]
            for command in commands
        }
        self.assertEqual(routing_by_dataset, {
            "mnist": "direct",
            "vanet": "aodv",
        })

    def test_both_dataset_summary_lists_aodv_plot_artifacts(self):
        output = io.StringIO()
        with patch("sys.argv", ["main.py", "--dataset", "both"]), \
                patch("main.subprocess.run"), redirect_stdout(output):
            main.main()

        rendered = output.getvalue()
        for filename in (
                "vanet_aodv_routing_overhead_vs_rounds.png",
                "vanet_communication_volume_vs_rounds.png",
                "vanet_normalized_routing_load_vs_rounds.png",
                "vanet_aodv_network_latency_vs_rounds.png",
        ):
            self.assertIn(f"plots/{filename}", rendered)

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
