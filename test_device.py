import unittest
from types import SimpleNamespace
import tempfile
import json

import numpy as np

from device import resolve_device
from dataset.postgres_plan_dataset import PostgresPlanDataSet, template_family
from model_arch import QPPNet


class DeviceTest(unittest.TestCase):
    def test_explicit_cpu(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")

    def test_auto_returns_supported_device(self):
        self.assertIn(resolve_device().type, {"cpu", "cuda", "mps"})

    def test_forward_uses_selected_device(self):
        with tempfile.TemporaryDirectory() as save_dir:
            model = QPPNet(SimpleNamespace(
                device="auto", save_dir=save_dir, test_time=False,
                batch_size=1, dataset="PSQLTPCH", SGD=False, lr=1e-3,
                scheduler=False, start_epoch=0,
            ))
            width = model.dim_dict["Seq Scan"]
            sample = {
                "node_type": "Seq Scan",
                "real_node_type": "Seq Scan",
                "subbatch_size": 1,
                "feat_vec": np.zeros((1, width), dtype=np.float32),
                "children_plan": [],
                "total_time": np.ones(1, dtype=np.float32),
            }
            _, prediction = model._forward_oneQ_batch(sample)
            self.assertEqual(prediction.device.type, model.device.type)
            model.evaluate([sample])
            self.assertGreaterEqual(model.last_pred_err, 0)
            self.assertEqual(model.last_truths.tolist(), [100.0])

    def test_operator_dimensions_can_come_from_a_workload(self):
        with tempfile.TemporaryDirectory() as save_dir:
            dimensions = {"Custom Scan": 5}
            model = QPPNet(SimpleNamespace(
                device="cpu", save_dir=save_dir, test_time=False,
                batch_size=1, dataset="Postgres", dim_dict=dimensions,
                SGD=False, lr=1e-3, scheduler=False, start_epoch=0,
            ))
        self.assertEqual(model.dim_dict, dimensions)
        self.assertEqual(model.units["Custom Scan"].dense_block[0].in_features, 5)

    def test_postgres_plans_split_whole_template_families(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as output:
            output.write(json.dumps({"kind": "header"}) + "\n")
            for family in range(5):
                for sample in range(2):
                    output.write(json.dumps({"kind": "plan", "record": {
                        "key": f"dsb:query{family:03d}_s{sample}",
                        "label_duration_ms": 100 + family,
                        "plan": {"Plan": {
                            "Node Type": "Seq Scan", "Plan Width": 8,
                            "Plan Rows": 10, "Startup Cost": 0,
                            "Total Cost": 1, "Actual Total Time": 1,
                            "Relation Name": f"table_{family}",
                        }},
                    }}) + "\n")
            output.flush()
            dataset = PostgresPlanDataSet(SimpleNamespace(
                data_dir=output.name, batch_size=2, split_seed=2027,
            ))
        train = {template_family(record["key"])
                 for record in dataset.train_records}
        test = {template_family(record["key"])
                for record in dataset.test_records}
        self.assertFalse(train & test)
        self.assertEqual(len(train | test), 5)
        self.assertEqual(dataset.datasize, 8)
        self.assertLess(max(np.abs(group["feat_vec"]).max()
                            for group in dataset.test_dataset), 100)


if __name__ == "__main__":
    unittest.main()
