import pika
from uuid import uuid4
import json
import time


class DetectionRpcClient:
    def __init__(self, hostname: str, routing_key: str) -> None:
        self.routing_key = routing_key
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(hostname))
        self.channel = self.connection.channel()
        result = self.channel.queue_declare(queue="", exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True,
        )
        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, data: bytes):
        self.response = None
        self.corr_id = str(uuid4())

        self.channel.basic_publish(
            exchange="",
            routing_key=self.routing_key,
            properties=pika.BasicProperties(
                content_type="application/json",
                correlation_id=self.corr_id,
                reply_to=self.callback_queue,
            ),
            body=request,
        )

        while self.response is None:
            self.connection.process_data_events(time_limit=None)

        json_response = json.loads(self.response)

        return json_response


if __name__ == "__main__":
    data_path = "data/data.json"
    with open(data_path, "r+") as f:
        data = json.load(f)

    request = json.dumps(data).encode()

    detectionRPC = DetectionRpcClient(hostname="localhost", routing_key="rpc_queue")
    t1 = time.time()
    response = detectionRPC.call(request)
    t2 = time.time()

    print(response)
    print(f"Time: {t2 - t1}")
