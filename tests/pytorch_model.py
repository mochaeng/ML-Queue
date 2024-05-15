import unittest
import torch.jit
import torch.nn as nn
import json
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class PytorchModel(unittest.TestCase):
    def test_model_load(self):
        path = "data/centralized_0_scripted.pt"
        model = torch.jit.load(path)

        self.assertIsInstance(model, nn.Module)

    def test_scaler(self):
        scaler_path = "data/centralized_scaler.pkl"
        scaler = joblib.load(scaler_path)

        self.assertIsInstance(scaler, MinMaxScaler)

    def test_scaler_transform(self):
        data_path = "data/data.json"
        with open(data_path, "r+") as f:
            data = json.load(f)

        line = [value for feature, value in data.items()]
        values = np.array([line])

        scaler_path = "data/centralized_scaler.pkl"
        scaler: MinMaxScaler = joblib.load(scaler_path)
        scaled_values = scaler.transform(values)

        self.assertIsInstance(scaled_values, np.ndarray)

    def test_model_infer(self):
        path = "data/centralized_0_scripted.pt"
        model: nn.Module = torch.jit.load(path)

        data_path = "data/data.json"
        with open(data_path, "r+") as f:
            data = json.load(f)

        line = [value for feature, value in data.items()]
        values = np.array([line])
        scaler_path = "data/centralized_scaler.pkl"
        scaler: MinMaxScaler = joblib.load(scaler_path)
        scaled_values = scaler.transform(values)

        tensors = torch.tensor(scaled_values, dtype=torch.float32).to("cuda")

        output = model(tensors)
        pred = torch.round(torch.sigmoid(output))

        self.assertEqual(pred.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
