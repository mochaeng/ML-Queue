import pika
import json
import pika.channel
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np


class PytorchClassifier:
    def __init__(self, model_path: str, scaler_path: str) -> None:
        self.model: torch.nn.Module = torch.jit.load(model_path)
        self.scaler: MinMaxScaler = joblib.load(scaler_path)

    def get_prediction(self, data: dict):
        values_from_features = [value for _, value in data.items()]
        values = np.array([values_from_features])
        scaled_values = self.scaler.transform(values)
        tensor_values = torch.tensor(scaled_values, dtype=torch.float32).to("cuda")
        self.model.eval()
        output = self.model(tensor_values)
        prediction = torch.round(torch.sigmoid(output))
        return prediction.item()


def on_request(ch, method, props, body):
    json_data = json.loads(body)
    prediction = classifier.get_prediction(json_data)

    if prediction == 1:
        msg = {"classification": True}
    else:
        msg = {"classification": False}

    response = json.dumps(msg).encode()

    ch.basic_publish(
        exchange="",
        routing_key=props.reply_to,
        properties=pika.BasicProperties(correlation_id=props.correlation_id),
        body=response,
    )

    ch.basic_ack(delivery_tag=method.delivery_tag)


if __name__ == "__main__":
    model_path = "data/centralized_0_scripted.pt"
    scaler_path = "data/centralized_scaler.pkl"

    classifier = PytorchClassifier(model_path=model_path, scaler_path=scaler_path)

    queue_name = "rpc_queue"
    connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))

    channel = connection.channel()
    channel.queue_declare(queue=queue_name)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=queue_name, on_message_callback=on_request)

    print(" [x] Awaiting RPC requests")
    channel.start_consuming()
