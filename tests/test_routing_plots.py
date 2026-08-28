import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from routing_sim import RoutingSimulator, TopologySnapshot
from routing_plots import build_routing_figures, plot_routing_metrics


class RoutingPlotTests(unittest.TestCase):
    def test_actual_series_use_ip_routing_bytes_and_undefined_nrl_gap(self):
        sim = RoutingSimulator(capacity=lambda d: 1e6)
        sim.submit("A", "C", 1250, 1250, 1,
                   TopologySnapshot.from_edges(["A", "B", "C"], [("A", "B", 100), ("B", "C", 100)]))
        sim.submit("A", "C", 1250, 1250, 2,
                   TopologySnapshot.from_edges(["A", "B", "C"], []))
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "synthetic"
            sim.ledger.export(prefix, sim.metadata(traffic="synthetic test"))
            csv_path = str(prefix) + "_routing_rounds.csv"
            figures = build_routing_figures(csv_path, prefix="synthetic")
            try:
                route = figures["aodv_routing_overhead_vs_rounds"].axes[0]
                self.assertEqual(list(route.lines[0].get_ydata()), [104 / 1024, 156 / 1024])
                self.assertEqual(list(route.lines[3].get_ydata()), [200 / 1024, 156 / 1024])
                nrl = figures["normalized_routing_load_vs_rounds"].axes[0]
                self.assertEqual(nrl.lines[0].get_ydata()[0], 2)
                self.assertNotEqual(nrl.lines[0].get_ydata()[1], nrl.lines[0].get_ydata()[1])
                self.assertTrue(any("N/A" in text.get_text() for text in nrl.texts))
                self.assertGreaterEqual(nrl.get_xlim()[1], 2)
                self.assertIn("synthetic", route.get_title().lower())
            finally:
                for figure in figures.values():
                    plt.close(figure)
            paths = plot_routing_metrics(csv_path, "synthetic", output_dir=Path(temporary) / "plots")
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(Path(path).stat().st_size > 10000 for path in paths))
            Path(str(prefix) + "_routing_metadata.json").write_text(json.dumps({"routing_mode": "direct"}))
            self.assertEqual(plot_routing_metrics(csv_path, "direct"), [])

    def test_legacy_input_without_routing_metadata_does_not_fabricate_graph(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "legacy.csv"
            csv_path.write_text("round,bytes_tx\n1,100\n")
            self.assertEqual(build_routing_figures(csv_path), {})


if __name__ == "__main__":
    unittest.main()
