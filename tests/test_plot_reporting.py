"""Regression tests for plot inputs, output, and non-overlapping legends."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd

import plot_metrics
from logger import TrainingLogger
from routing_sim import RoutingSimulator, TopologySnapshot


class PlotReportingTests(unittest.TestCase):
    def test_throughput_plot_uses_round_wireless_goodput_mbps(self) -> None:
        rows = [
            {"node": "C0_V1", "round": 1, "training_ms": 1,
             "vanet_wireless_bits": 8_000_000, "vanet_airtime_s": 1.0},
            {"node": "RSU_0", "round": 1,
             "vanet_wireless_bits": 8_000_000, "vanet_airtime_s": 1.0},
            {"node": "Server", "round": 1,
             "throughput_bytes_per_sec": 999_000_000},
            {"node": "C0_V1", "round": 2, "training_ms": 1,
             "vanet_wireless_bits": 6_000_000, "vanet_airtime_s": 0.5},
            {"node": "RSU_0", "round": 2,
             "vanet_wireless_bits": 3_000_000, "vanet_airtime_s": 1.0},
            {"node": "Server", "round": 2,
             "throughput_bytes_per_sec": 999_000_000},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "metrics.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            with patch.object(plot_metrics, "_save_fig"), patch.object(
                    plt, "plot", wraps=plt.plot) as plotted:
                plot_metrics.plot_metrics_from_csv(path)
            goodput_call = next(
                call for call in plotted.call_args_list
                if call.kwargs.get("label") == "Modeled VANET Goodput"
            )
        self.assertEqual(list(goodput_call.args[0]), [1, 2])
        self.assertEqual(list(goodput_call.args[1]), [8.0, 6.0])
        plt.close("all")

    def test_constant_coverage_explanation_does_not_claim_mobility(self):
        frame = pd.DataFrame({
            "node": ["Server", "Server"],
            "round": [1, 2],
            "throughput_bytes_per_sec": [1.0, 1.0],
        })
        vehicle_group = pd.DataFrame(index=[1, 2])
        coverage = pd.Series([3.0, 3.0], index=[1, 2])
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plot_metrics, "PLOT_OUTPUT_DIR", directory):
                plot_metrics._write_vanet_plot_explanations(
                    frame, vehicle_group, coverage)
            path = os.path.join(directory, "vanet_plot_explanations.md")
            with open(path, encoding="utf-8") as source:
                explanation = source.read()
        self.assertIn("remained constant", explanation)
        self.assertNotIn("moving vehicles enter", explanation)

    def test_legacy_device_ids_are_rendered_as_vehicle_ids(self) -> None:
        log_text = """VEHICLE TRAINING UPDATES
| 1 | C1_D2 | 0.5 | 80% |
GLOBAL PROXY MODEL EVALUATION
| 1 | 75% |
PRIVATE MODEL TEST ACCURACY
| 1 | C1_D2 | 82% |
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "training_logs.txt")
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write(log_text)
            _, private_accuracy, training_loss = plot_metrics.parse_logs(log_path)

        self.assertEqual(list(private_accuracy), ["C1_V2"])
        self.assertEqual(list(training_loss), ["C1_V2"])

    def test_training_logger_routes_accuracy_and_loss_to_plots_folder(self) -> None:
        training_logger = TrainingLogger()
        training_logger.log_vehicle(1, "C0_V1", 0.8, 0.7)
        training_logger.log_private_accuracy(1, "C0_V1", 0.72)
        training_logger.log_global(1, 0.68)

        with tempfile.TemporaryDirectory() as temp_dir:
            original_directory = os.getcwd()
            os.chdir(temp_dir)
            try:
                training_logger.save_logs("vanet_training_logs.txt")
                with patch.object(plot_metrics, "PLOT_OUTPUT_DIR", "plots"):
                    training_logger.generate_plots(prefix="vanet")
                self.assertTrue(os.path.exists(
                    os.path.join("plots", "vanet_accuracy_vs_rounds.png")))
                self.assertTrue(os.path.exists(
                    os.path.join("plots", "vanet_loss_vs_rounds.png")))
                self.assertFalse(os.path.exists("vanet_accuracy_vs_rounds.png"))
                self.assertFalse(os.path.exists("vanet_loss_vs_rounds.png"))
            finally:
                os.chdir(original_directory)

    def test_training_logger_includes_aodv_plots_and_explanations(self) -> None:
        training_logger = TrainingLogger()
        training_logger.log_vehicle(1, "C0_V1", 0.8, 0.7)
        training_logger.log_private_accuracy(1, "C0_V1", 0.72)
        training_logger.log_global(1, 0.68)
        metric_rows = [
            {
                "node": "C0_V1", "round": 1, "training_ms": 1.0,
                "train_loss": 0.8, "private_test_accuracy_pct": 72.0,
                "vanet_wireless_bits": 8_000.0,
                "vanet_airtime_s": 0.01,
            },
            {
                "node": "Server", "round": 1,
                "global_proxy_accuracy_pct": 68.0,
                "vehicles_in_range_total": 1.0,
                "vehicles_assigned_total": 1.0,
            },
        ]
        simulator = RoutingSimulator(capacity=lambda distance: 1_000_000.0)
        simulator.submit(
            "C0_V1", "RSU_0", 1250, 1200, 1,
            TopologySnapshot.from_edges(
                ["C0_V1", "relay", "RSU_0"],
                [("C0_V1", "relay", 100), ("relay", "RSU_0", 100)],
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            original_directory = os.getcwd()
            os.chdir(temp_dir)
            try:
                training_logger.save_logs("vanet_training_logs.txt")
                pd.DataFrame(metric_rows).to_csv(
                    "vanet_metrics.csv", index=False)
                simulator.ledger.export(
                    "vanet", simulator.metadata(
                        traffic="Federated Learning model envelopes"))
                with patch.object(plot_metrics, "PLOT_OUTPUT_DIR", "plots"):
                    training_logger.generate_plots(prefix="vanet")
                    from routing_plots import plot_routing_metrics
                    plot_routing_metrics(
                        "vanet_routing_rounds.csv", prefix="vanet")

                expected_aodv_files = [
                    "vanet_aodv_routing_overhead_vs_rounds.png",
                    "vanet_communication_volume_vs_rounds.png",
                    "vanet_normalized_routing_load_vs_rounds.png",
                    "vanet_aodv_network_latency_vs_rounds.png",
                ]
                for filename in expected_aodv_files:
                    self.assertTrue(os.path.exists(
                        os.path.join("plots", filename)), filename)

                report_path = os.path.join(
                    "plots", "vanet_plot_explanations.md")
                with open(report_path, encoding="utf-8") as report_file:
                    report = report_file.read()
                for filename in [
                        "vanet_accuracy_vs_rounds.png",
                        *expected_aodv_files,
                ]:
                    heading = f"## `{filename}`"
                    image = f"![{filename}]({filename})"
                    self.assertIn(heading, report)
                    section = report.split(heading, 1)[1].split("## `", 1)[0]
                    self.assertIn(image, section)
                    self.assertLess(
                        section.index(image),
                        section.index("Why the line rises and falls"),
                    )
            finally:
                os.chdir(original_directory)

    def test_vehicle_frame_excludes_rsu_rows_with_new_vehicle_ids(self) -> None:
        frame = pd.DataFrame({
            "node": ["C0_V1", "C1_V2", "RSU_0_Central", "Server"],
            "round": [1, 1, 1, 1],
            "training_ms": [10.0, 20.0, 1000.0, 2000.0],
        })

        vehicles = plot_metrics._vehicle_frame(frame)

        self.assertEqual(vehicles["node"].tolist(), ["C0_V1", "C1_V2"])

    def test_legend_is_above_title_and_axes(self) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for index in range(23):
            ax.plot([1, 2], [index, index + 1], label=f"Series {index}")
        ax.set_title("Coverage")

        plot_metrics._legend_above()
        plot_metrics._apply_plot_layout()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_box = ax.get_legend().get_window_extent(renderer)
        axes_box = ax.get_window_extent(renderer)
        title_box = ax.title.get_window_extent(renderer)

        self.assertGreaterEqual(legend_box.y0, axes_box.y1)
        self.assertGreaterEqual(legend_box.y0, title_box.y1)
        plt.close(fig)

    def test_round_ticks_remain_readable_for_long_runs(self) -> None:
        fig, ax = plt.subplots()
        tick_setter = getattr(plot_metrics, "_set_round_ticks", None)
        self.assertIsNotNone(tick_setter, "_set_round_ticks is required")

        tick_setter(range(1, 101), max_ticks=12)

        ticks = [int(value) for value in ax.get_xticks()]
        self.assertLessEqual(len(ticks), 12)
        self.assertEqual(ticks[0], 1)
        self.assertEqual(ticks[-1], 100)
        plt.close(fig)

    def test_vanet_plotting_creates_coverage_graph_and_explanations(self) -> None:
        rows = []
        for round_num, central, north, total in [(1, 2, 1, 3), (2, 1, 1, 2)]:
            rows.extend([
                {
                    "node": "C0_V1", "round": round_num,
                    "training_ms": 100 + round_num,
                    "train_loss": 1.0 / round_num,
                    "private_test_accuracy_pct": 70 + round_num,
                    "action_to_response_ms": 1000 + round_num * 10,
                },
                {
                    "node": "RSU_0_Central", "round": round_num,
                    "vehicles_in_range": central, "vehicles_assigned": 2,
                },
                {
                    "node": "RSU_1_North", "round": round_num,
                    "vehicles_in_range": north, "vehicles_assigned": 1,
                },
                {
                    "node": "Server", "round": round_num,
                    "vehicles_in_range_total": total,
                    "vehicles_assigned_total": 3,
                    "throughput_bytes_per_sec": 5000 / round_num,
                    "global_proxy_accuracy_pct": 60 + round_num,
                },
            ])

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "vanet_metrics.csv")
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            with patch.object(plot_metrics, "PLOT_OUTPUT_DIR", temp_dir):
                plot_metrics.plot_metrics_from_csv(csv_path, prefix="vanet")

            coverage_path = os.path.join(
                temp_dir, "vanet_vehicles_in_range_vs_rounds.png")
            explanations_path = os.path.join(
                temp_dir, "vanet_plot_explanations.md")
            self.assertTrue(os.path.exists(coverage_path))
            self.assertGreater(os.path.getsize(coverage_path), 0)
            self.assertTrue(os.path.exists(explanations_path))
            with open(explanations_path, encoding="utf-8") as explanation_file:
                explanations = explanation_file.read()
            self.assertIn("vanet_vehicles_in_range_vs_rounds.png", explanations)
            self.assertIn("vanet_throughput_vs_rounds.png", explanations)
            self.assertIn("Why the line rises and falls", explanations)
            image = "![vanet_vehicles_in_range_vs_rounds.png]"
            self.assertIn(image, explanations)
            coverage_section = explanations.split(
                "## `vanet_vehicles_in_range_vs_rounds.png`", 1)[1]
            self.assertLess(
                coverage_section.index(image),
                coverage_section.index("Why the line rises and falls"),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
