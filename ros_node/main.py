#!/usr/bin/env python

import base64
import io
import pickle

import cv2
import PIL.Image
import requests
import rosnode
import rospy
import rosservice
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from webots_ros.srv import set_int

DETECTION_SERVER_URL = "http://localhost:24680"

node_name = None
session = requests.Session()


def detect_node_name():
    """Detects and sets the name of the robot node."""

    global node_name

    node_names = filter(
        lambda x: x.startswith("/red_player_1"), rosnode.get_node_names()
    )
    if len(list(node_names)) == 0:
        raise RuntimeError("No robot node found")

    node_name = list(node_names)[0]

    print("Node name %s detected" % node_name)


def enable_camera():
    """Enables the camera of the robot."""

    global node_name

    service_name = "%s/Camera/enable" % node_name

    print("Waiting for service %s" % service_name)

    rospy.wait_for_service(service_name)
    enable_camera_service = rospy.ServiceProxy(service_name, set_int)
    enable_camera_service(128)

    print("Camera enabled")


def detect_ball(image):
    """Detects the ball in the image with the detection server.

    Args:
        image (PIL.Image.Image): The image to detect the ball in.

    Returns:
        List[Any]: A list of detection results.
    """
    # Convert image to RGB format
    image = image.convert("RGB")

    # Convert image to JPEG format in memory
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()

    # Encode as base64
    img_base64 = base64.b64encode(img_byte_arr)  # Python 2.7 compatible

    data = {
        "image": img_base64,
    }

    response = session.post(
        DETECTION_SERVER_URL,
        json=data,  # Use json parameter instead of raw pickle
        timeout=10,
    )

    results = response.json()  # Parse JSON response directly

    ball_results = []
    if "results" in results:
        for result in results["results"]:
            if result["label"] != "sports ball":
                continue
            ball_results.append(result)

    return ball_results


def camera_image_callback(image_msg):
    """Callback function for the camera image subscriber.

    Args:
        image_msg (sensor_msgs.msg.Image): The image message from the camera.
    """

    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(image_msg, "bgr8")

    # Convert to PIL image
    image = PIL.Image.fromarray(image)

    results = detect_ball(image)

    print(results)


def main():
    global node_name

    rospy.init_node("my_bot")

    detect_node_name()

    enable_camera()

    camera_image_sub = rospy.Subscriber(
        "%s/Camera/image" % node_name, Image, camera_image_callback
    )

    rospy.spin()


if __name__ == "__main__":
    main()
