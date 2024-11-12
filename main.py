import base64
import http.server
import io
import json
import logging
import time
from typing import Dict, List, Tuple, Type, TypedDict

import PIL.Image
import torch.cuda
import transformers

HOST: str = "localhost"
PORT: int = 24680
LOGGING_LEVEL: int = logging.INFO
MODEL: str = "facebook/detr-resnet-50"


class RequestModel(TypedDict):
    image: str  # base64 encoded jpg image


class ResponseResultModel(TypedDict):
    label: str
    score: float
    box: List[Dict[str, int]]


class ResponseModel(TypedDict):
    results: List[ResponseResultModel]


class DetectionRequestHandler(http.server.BaseHTTPRequestHandler):
    server: "DetectionServer"

    def __init__(self, *args, **kwargs) -> None:
        self.protocol_version = "HTTP/1.1"
        super().__init__(*args, **kwargs)

    def do_POST(self) -> None:
        pipeline = self.server.pipeline

        logging.debug("Loading request...")
        content_length = int(self.headers.get("Content-Length", 0))
        request_json = self.rfile.read(content_length).decode("utf-8")
        request = json.loads(request_json)

        # Decode base64 image
        image_bytes = base64.b64decode(request["image"])
        image = PIL.Image.open(io.BytesIO(image_bytes))

        logging.debug("Performing inference...")
        start_time = time.perf_counter()
        response = pipeline(image)
        end_time = time.perf_counter()
        logging.info("Inference done in %s seconds", end_time - start_time)

        logging.debug("Sending response...")
        response_json = json.dumps(response)
        response_bytes = response_json.encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

        logging.debug("Response sent")


class DetectionServer(http.server.HTTPServer):
    pipeline: transformers.pipelines.ObjectDetectionPipeline

    def __init__(
        self,
        server_address: Tuple[str, int],
        RequestHandlerClass: Type[DetectionRequestHandler],
    ) -> None:
        logging.info("Loading model %s...", MODEL)

        device = 0 if torch.cuda.is_available() else -1
        pipeline = transformers.pipeline(
            task="object-detection",
            model=MODEL,
            device=device,
        )
        assert isinstance(pipeline, transformers.pipelines.ObjectDetectionPipeline)
        self.pipeline = pipeline

        logging.info("Model loaded")

        super().__init__(server_address, RequestHandlerClass)


def main() -> None:
    logging.basicConfig(level=LOGGING_LEVEL)

    server_address = (HOST, PORT)
    httpd = DetectionServer(server_address, DetectionRequestHandler)

    logging.info("Starting server on %s:%s...", HOST, PORT)

    httpd.serve_forever()


if __name__ == "__main__":
    main()
