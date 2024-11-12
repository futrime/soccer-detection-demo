import base64
import io
import time

import PIL.Image
import requests

from main import HOST, PORT, RequestModel, ResponseModel


def main() -> None:
    session = requests.Session()

    # Load and convert image to base64
    image = PIL.Image.open("test.png")

    # Convert image to RGB format
    image = image.convert("RGB")

    # Convert image to JPEG format in memory
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()

    # Encode as base64
    img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")

    request: RequestModel = {
        "image": img_base64,
    }

    while True:
        start_time = time.perf_counter()
        response = session.post(
            f"http://{HOST}:{PORT}",
            json=request,  # requests will automatically convert dict to JSON
            timeout=10,
        )
        end_time = time.perf_counter()

        # Parse JSON response
        results: ResponseModel = response.json()

        print(f"Response received in {end_time - start_time} seconds:")
        print(results)
        print()


if __name__ == "__main__":
    main()
