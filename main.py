import http.server
import logging
import pickle
from typing import Dict, List, Tuple, Type, TypedDict

import PIL.Image
import transformers

HOST: str = "localhost"
PORT: int = 24680
LOGGING_LEVEL: int = logging.DEBUG


class RequestModel(TypedDict):
    image: PIL.Image.Image


class ResponseResultModel(TypedDict):
    label: str
    score: float
    box: List[Dict[str, int]]


class ResponseModel(TypedDict):
    results: List[ResponseResultModel]


class DetectionRequestHandler(http.server.BaseHTTPRequestHandler):
    server: "DetectionServer"

    def do_POST(self) -> None:
        pipeline = self.server.pipeline

        content_length = int(self.headers.get("Content-Length", 0))
        request = pickle.loads(self.rfile.read(content_length))

        image = request["image"]

        response = pipeline(image)

        self.send_response(200)
        self.end_headers()

        self.wfile.write(pickle.dumps(response))


class DetectionServer(http.server.HTTPServer):
    pipeline: transformers.pipelines.ObjectDetectionPipeline

    def __init__(
        self,
        server_address: Tuple[str, int],
        RequestHandlerClass: Type[DetectionRequestHandler],
    ) -> None:
        pipeline = transformers.pipeline(task="object-detection")
        assert isinstance(pipeline, transformers.pipelines.ObjectDetectionPipeline)
        self.pipeline = pipeline

        super().__init__(server_address, RequestHandlerClass)


def main() -> None:
    logging.basicConfig(level=LOGGING_LEVEL)

    server_address = (HOST, PORT)
    # Use DetectionServer instead of HTTPServer
    httpd = DetectionServer(server_address, DetectionRequestHandler)

    logging.info("Starting server on %s:%s", HOST, PORT)

    httpd.serve_forever()


if __name__ == "__main__":
    main()
