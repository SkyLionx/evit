from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval
import os
import cv2 as cv


def extract_ros_bag_csv(file_path: str):
    """
    Extracts topics from a ROS bag file.
    A new folder will be created in the dataset directory and the topic messages will be extracted there.
    """
    from bagpy import bagreader

    br = bagreader(file_path)

    print("Extracting topics to csv files. This may take a while...")
    for topic in br.topics:
        print("Extracting topic", topic + "... ", end="")
        data = br.message_by_topic(topic)
        print("Saved into", data)


def bytes_to_image(
    bytes_string: str, width: int, height: int, channels: int = 3
) -> np.ndarray:
    """
    Convert a binary string with a b prefix (e.g b'\x00\x45') to a uint8 image
    """
    img = np.frombuffer(bytes_string, dtype=np.uint8)
    img = img.reshape((height, width, channels))
    return img


def save_raw_images_from_csv(csv_path, save_path):
    """
    Load the csv file specified in `csv_path` and extract black and white png
    images to the `save_path` folder
    """
    raw_images_data = pd.read_csv(csv_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for index, row in raw_images_data.iterrows():
        width, height = row["width"], row["height"]
        image_data = literal_eval(row["data"])
        image = bytes_to_image(image_data, width, height, channels=1)
        image = image.reshape(height, width)

        img_path = os.path.join(save_path, str(index) + ".png")
        plt.imsave(img_path, image, cmap="gray")


def save_color_images_from_csv(csv_path, save_path):
    raw_images_data = pd.read_csv(csv_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for index, row in raw_images_data.iterrows():
        width, height = row["width"], row["height"]
        image_data = literal_eval(row["data"])
        image = bytes_to_image(image_data, width, height, channels=3)
        image = image.reshape(height, width, 3)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        img_path = os.path.join(save_path, str(index) + ".png")
        plt.imsave(img_path, image)


def extract_rosbag(ros_path, extract_csv=True, extract_images=True):
    """
    Extract a .bag file containing events.
    If `extract_csv` is True, it's going to create a csv file for each topic
    in a folder called as the bag file in the same directory.
    If `extract_images` is True, it's going to extract raw and colored images
    in a folder called as the bag file in the same directory.
    """
    if extract_csv:
        extract_ros_bag_csv(ros_path)

    if extract_images:
        extract_folder = os.path.join(
            os.path.dirname(dataset_path),
            os.path.basename(dataset_path).replace(".bag", ""),
        )

        dest_folder = os.path.join(extract_folder, "raw_images")
        print("Extracting raw images into", dest_folder)
        raw_images_path = os.path.join(extract_folder, "dvs-image_raw.csv")
        save_raw_images_from_csv(raw_images_path, dest_folder)

        dest_folder = os.path.join(extract_folder, "color_images")
        print("Extracting color images into", dest_folder)
        color_images_path = os.path.join(extract_folder, "dvs-image_color.csv")
        save_color_images_from_csv(color_images_path, dest_folder)


if __name__ == "__main__":
    dataset_path = r"..\03 - Dataset\CED_simple\simple_color_keyboard_2.bag"
    extract_rosbag(dataset_path)
