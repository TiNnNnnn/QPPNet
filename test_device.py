import unittest
from types import SimpleNamespace
import tempfile

import numpy as np

from device import resolve_device
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


if __name__ == "__main__":
    unittest.main()
