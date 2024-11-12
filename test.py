import pickle

import PIL.Image
import requests

from main import HOST, PORT, RequestModel, ResponseModel


def main() -> None:
    image = PIL.Image.open("test.png")

    request: RequestModel = {
        "image": image,
    }

    data = pickle.dumps(request)
    response = requests.post(
        f"http://{HOST}:{PORT}",
        data=data,
        timeout=1000,
    )

    # Parse response
    results: ResponseModel = pickle.loads(response.content)

    print(results)


if __name__ == "__main__":
    main()
