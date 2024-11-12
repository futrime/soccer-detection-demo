import pickle
import time

import PIL.Image
import requests

from main import HOST, PORT, RequestModel, ResponseModel


def main() -> None:
    session = requests.Session()

    image = PIL.Image.open("test.png")

    request: RequestModel = {
        "image": image,
    }

    data = pickle.dumps(request, protocol=2)

    while True:
        start_time = time.perf_counter()
        response = session.post(
            f"http://{HOST}:{PORT}",
            data=data,
            timeout=10,
        )
        end_time = time.perf_counter()

        # Parse response
        results: ResponseModel = pickle.loads(response.content)

        print(f"Response received in {end_time - start_time} seconds:")
        print(results)
        print()


if __name__ == "__main__":
    main()
