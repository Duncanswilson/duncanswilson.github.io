"""Portable report serialization and source-label safety tests."""

import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from flyreward.report import write_report
from flyreward.types import SimulationResult


class ReportTests(unittest.TestCase):
    def make_result(self, preset="baseline"):
        return SimulationResult(
            config={"preset": preset, "duration": 0.2},
            time=np.array([0.0, 0.1, 0.2]),
            traces={"dopamine_concentration": np.array([0.0, 1.0, 2.0]), "custom_signal": np.array([3.0, 4.0, 5.0])},
            metrics={"dopamine_exposure_auc": np.float64(0.2), "mean_rate_hz": 12.5},
            metadata={"biological_validation": "not_validated"},
        )

    def read_payload(self, document):
        match = re.search(r'<script id="experiment-data" type="application/json">(.*?)</script>', document, re.DOTALL)
        self.assertIsNotNone(match)
        return json.loads(match.group(1))

    def test_complete_standalone_report_preserves_data(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "report.html"
            written = write_report([self.make_result()], {"synthetic": True, "n_neurons": 5}, target, optimization={"objective": "dopamine_exposure_auc", "score": np.float64(0.2)})
            self.assertEqual(written, target.resolve())
            document = written.read_text()
        payload = self.read_payload(document)
        self.assertEqual(payload["schema"], "flyreward-report-v1")
        self.assertEqual(payload["runs"][0]["traces"]["custom_signal"], [3.0, 4.0, 5.0])
        self.assertEqual(payload["runs"][0]["time"], [0.0, 0.1, 0.2])
        self.assertEqual(payload["optimization"]["score"], 0.2)
        self.assertEqual(payload["graph_metadata"]["n_neurons"], 5)
        self.assertIn("MaleCNS reward simulation", document)
        self.assertIn("abstract global proxy", document)
        self.assertIn("baseline</th>", document)
        self.assertIn("12.5</td>", document)
        self.assertNotRegex(document, r'<script[^>]+src=')
        self.assertNotRegex(document, r'<link[^>]+href=')
        self.assertNotIn("__PAYLOAD__", document)
        self.assertNotIn("__METRIC_TABLE__", document)

    def test_source_labels_cannot_escape_json_or_table(self):
        dangerous = '</script><script>alert("source")</script>&\u2028\u2029'
        result = self.make_result(dangerous)
        result.metrics['<img src=x onerror="alert(1)">'] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            target = write_report([result], {"source": dangerous}, Path(directory) / "report.html")
            document = target.read_text()
        payload = self.read_payload(document)
        self.assertEqual(payload["runs"][0]["config"]["preset"], dangerous)
        self.assertEqual(payload["graph_metadata"]["source"], dangerous)
        self.assertNotIn(dangerous, document)
        self.assertNotIn('<img src=x', document)
        self.assertIn("\\u003c/script\\u003e", document)
        self.assertIn("&lt;/script&gt;", document)
        self.assertEqual(document.count("</script>"), 2)

    def test_nonfinite_values_are_explicit_gaps(self):
        result = self.make_result()
        result.traces["dopamine_concentration"] = np.array([np.nan, np.inf, -np.inf])
        result.metrics["mean_rate_hz"] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            target = write_report([result], {}, Path(directory) / "report.html")
            document = target.read_text()
        payload = self.read_payload(document)
        self.assertEqual(payload["runs"][0]["traces"]["dopamine_concentration"], [None, None, None])
        self.assertIsNone(payload["runs"][0]["metrics"]["mean_rate_hz"])
        self.assertIn("Not recorded</td>", document)

    def test_template_tokens_in_source_are_preserved_as_data(self):
        result = self.make_result("__PAYLOAD__ __METRIC_TABLE__")
        with tempfile.TemporaryDirectory() as directory:
            target = write_report([result], {"source": "__METRIC_TABLE__"}, Path(directory) / "report.html")
            document = target.read_text()
        payload = self.read_payload(document)
        self.assertEqual(payload["runs"][0]["config"]["preset"], "__PAYLOAD__ __METRIC_TABLE__")
        self.assertEqual(payload["graph_metadata"]["source"], "__METRIC_TABLE__")

    def test_empty_results_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            write_report([], {}, Path("unused.html"))


if __name__ == "__main__":
    unittest.main()
