import os
from ast import literal_eval
from typing import Generator, List, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from media_utils import bgr_to_rgb, image_from_buffer
from utils import is_using_colab

if is_using_colab():
    os.system("pip install -q bagpy")

import rosbag


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


def save_raw_images_from_csv(csv_path: str, save_path: str):
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


def save_color_images_from_csv(csv_path: str, save_path: str):
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


def extract_rosbag(
    ros_path: str, extract_csv: bool = True, extract_images: bool = True
):
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


def inspect_dataset(dataset_path: str):
    bag = rosbag.Bag(dataset_path)

    for topic, msg, t in bag.read_messages(topics=["/dvs/image_raw"]):
        image_raw_msg = msg
        break

    for topic, msg, t in bag.read_messages(topics=["/dvs/image_color"]):
        image_color_msg = msg
        break

    for topic, msg, t in bag.read_messages(topics=["/dvs/events"]):
        events_msg = msg
        break

    bag.close()

    print("Raw Image Event")
    print(str(image_raw_msg)[:200] + "...")
    print()

    print("Color Image Event")
    print(str(image_color_msg)[:200] + "...")
    print()

    print("Events Event")
    print(str(events_msg)[:300] + "...")
    print()


def inspect_message_timestamps(dataset_path: str):
    i = 0
    print("{:5}{:20}{}".format("", "Topic Name", "Event Timestamp"))
    with rosbag.Bag(dataset_path) as b:
        for topic, msg, _ in b.read_messages(
            topics=["/dvs/events", "/dvs/image_color"]
        ):
            print("{:3}) {:20}{}".format(i, topic, msg.header.stamp.to_sec()))

            if topic == "/dvs/events":
                print(
                    "{:25}First event timestamp {} Last event timestamp {}".format(
                        "", msg.events[0].ts.to_sec(), msg.events[-1].ts.to_sec()
                    )
                )

            i += 1
            if i == 15:
                break


# Load dataset as Pandas dataframes
def load_bag_as_dataframes(dataset_path: str, image_type: str):
    """
    Load the dataset from a rosbag file and return two pandas dataframe with events and images.
    `image_type` can be either `raw` or `color`.
    """
    if image_type not in ["raw", "color"]:
        raise Exception(
            "image_type argument can be either 'raw' or 'color', not " + image_type
        )

    image_topic = "/dvs/image_" + image_type

    bag = rosbag.Bag(dataset_path)

    events_list = []
    images_list = []

    for topic, msg, t in tqdm(bag.read_messages(), total=bag.get_message_count()):
        if topic == image_topic:
            seq = msg.header.seq
            secs = msg.header.stamp.to_sec()
            width = msg.width
            height = msg.height
            is_bigendian = msg.is_bigendian
            encoding = msg.encoding
            step = msg.step
            data = np.frombuffer(msg.data, dtype=np.uint8)
            image_event = [seq, secs, width, height, is_bigendian, encoding, step, data]
            images_list.append(image_event)
        elif topic == "/dvs/events":
            seq = msg.header.seq
            events = msg.events

            for event in events:
                polarity = event.polarity
                x = event.x
                y = event.y
                secs = event.ts.to_sec()

                event = [seq, x, y, secs, polarity]
                events_list.append(event)

    bag.close()

    events_df = pd.DataFrame(events_list, columns=["seq", "x", "y", "secs", "polarity"])
    images_df = pd.DataFrame(
        images_list,
        columns=[
            "seq",
            "secs",
            "width",
            "height",
            "is_bigendian",
            "encoding",
            "step",
            "data",
        ],
    )

    return events_df, images_df


def plot_number_of_events_per_frame(events_df: pd.DataFrame, images_df: pd.DataFrame):
    n_events_per_image = []

    prev_timestamp = -1
    for i in range(1, len(images_df)):
        image_timestamp = images_df.iloc[i].secs
        n_events = len(
            events_df[
                (prev_timestamp <= events_df.secs) & (events_df.secs <= image_timestamp)
            ]
        )
        n_events_per_image.append(n_events)
        prev_timestamp = image_timestamp

    plt.suptitle("Number of events per image frame")
    plt.xlabel("Frame number")
    plt.ylabel("Number of events")
    plt.plot(n_events_per_image)


def plot_image_frames_frequency(images_df: pd.DataFrame):
    deltas = images_df.secs.diff()
    mean_secs = np.mean(deltas)
    plt.ylim(mean_secs - 0.001, mean_secs + 0.001)
    plt.ylabel("Delta frame (s)")
    plt.xlabel("Frame number")
    plt.plot(deltas[1:])
    print(
        "Average milliseconds: {:.4f}, Average fps: {:.2f}".format(
            mean_secs * 1000, 1 / mean_secs
        )
    )


class Event:
    def __init__(self, x: int, y: int, timestamp: float, polarity: int):
        """
        Event captured by the camera.\n
        `x`, `y` are the coordinate of the sensors,\n
        `polarity` can be True if positive, False if negative.
        """
        self.x = x
        self.y = y
        self.timestamp = timestamp
        self.polarity = polarity


def create_event_grid(
    events: List[Event], w: int, h: int, n_temp_bins: int
) -> np.array:
    """
    Create an event grid with shape `(n_temp_bins, h, w)`.
    Each event polarity is assigned proportionally to the two closest temporal bins (bilinear interpolation).
    Events should be sorted by timestamp.
    """
    event_grid = np.zeros(shape=(n_temp_bins, h, w))
    smallest_timestamp = events[0].timestamp
    highest_timestamp = events[-1].timestamp

    for event in events:
        temp_bin = (
            (event.timestamp - smallest_timestamp)
            / (highest_timestamp - smallest_timestamp + 1e-6)
        ) * (n_temp_bins - 1)

        int_bin = int(temp_bin)
        decimal_part = temp_bin - int_bin

        event_grid[int_bin, event.y, event.x] += event.polarity * (1 - decimal_part)
        if decimal_part != 0:
            event_grid[int_bin + 1, event.y, event.x] += event.polarity * decimal_part
    return event_grid


DatasetBatch = Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]


def dataset_generator_from_bag(
    bag_path: str, image_type: str, n_temp_bins: int = 10
) -> Generator[DatasetBatch, None, None]:
    if image_type not in ["color", "raw"]:
        raise Exception(
            "image_type must be either 'color' or 'raw'. "
            + image_type
            + " is not a valid option."
        )

    # Preload image frames to get timestamps
    images = []
    w, h = 0, 0
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, _ in bag.read_messages(topics=["/dvs/image_" + image_type]):
            w, h = msg.width, msg.height
            channels = 3 if image_type == "color" else 1
            image = bgr_to_rgb(image_from_buffer(msg.data, w, h, channels))
            images.append((msg.header.stamp.to_sec(), image))
    print("Images loaded")

    current_img_idx = 1
    events_batch: List[Event] = []

    with rosbag.Bag(bag_path) as bag:
        msg_count = bag.get_message_count("/dvs/events")

        # Group events before image frame
        for _, msg, _ in tqdm(bag.read_messages("/dvs/events"), total=msg_count):
            for event in msg.events:
                cur_image_ts = images[current_img_idx][0]
                event_obj = Event(event.x, event.y, event.ts.to_sec(), event.polarity)

                if event_obj.timestamp <= cur_image_ts:
                    events_batch.append(event_obj)
                else:
                    current_img_idx += 1
                    event_grid = create_event_grid(
                        events_batch, w, h, n_temp_bins=n_temp_bins
                    )
                    events_batch = []
                    yield (
                        (images[current_img_idx - 2][1], event_grid),
                        images[current_img_idx - 1][1],
                    )
                    if current_img_idx == len(images):
                        return


def save_batches_to_disk(ds: Generator[DatasetBatch, None, None], dst_folder: str):
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)
    for i, batch in enumerate(ds):
        torch.save(batch, os.path.join(dst_folder, "batch_{:04}.pt".format(i)))


def dataset_generator_from_batches(path: str) -> DatasetBatch:
    for batch_file in os.listdir(path):
        if batch_file.endswith(".pt"):
            yield torch.load(os.path.join(path, batch_file))


if __name__ == "__main__":
    dataset_path = r"..\03 - Dataset\CED_simple\simple_color_keyboard_2.bag"
    extract_rosbag(dataset_path)

    events_df, images_df = load_bag_as_dataframes(dataset_path, image_type="color")

    ds_gen = dataset_generator_from_bag(dataset_path, "color", n_temp_bins=10)
    dst_folder = os.path.join(
        os.path.dirname(dataset_path),
        os.path.basename(dataset_path).replace(".bag", ""),
        "batches",
    )
    print("Saving batches to", dst_folder)
    save_batches_to_disk(ds_gen, dst_folder)
