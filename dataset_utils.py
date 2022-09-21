import os
from ast import literal_eval
from typing import Generator, Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from media_utils import bgr_to_rgb, rgb_to_bgr, denorm_img, image_from_buffer, Model
from utils import is_using_colab

if is_using_colab():
    os.system("pip install -q bagpy")

import rosbag
from bagpy import bagreader


def extract_ros_bag_csv(file_path: str):
    """
    Extracts topics from a ROS bag file.
    A new folder will be created in the dataset directory and the topic messages will be extracted there.

    Args:
        file_path (str): path of the rosbag file.
    """
    br = bagreader(file_path)

    print("Extracting topics to csv files. This may take a while...")
    for topic in br.topics:
        print("Extracting topic", topic + "... ", end="")
        data = br.message_by_topic(topic)
        print("Saved into", data)


def save_images_from_csv(csv_path: str, save_path: str):
    """
    Load the specified image csv file and save png files.

    Args:
        csv_path (str): path of the source csv.
        save_path (str): folder where the images will be saved.
    """
    raw_images_data = pd.read_csv(csv_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for index, row in raw_images_data.iterrows():
        width, height = row["width"], row["height"]
        image_data = literal_eval(row["data"])
        channels = 1 if "mono" in row["encoding"] else 3
        image = image_from_buffer(image_data, width, height, channels)
        if channels == 3:
            cmap = None
            image = bgr_to_rgb(image)
        else:
            cmap = "gray"

        img_path = os.path.join(save_path, str(index) + ".png")
        plt.imsave(img_path, image, cmap=cmap)


def extract_rosbag(
    ros_path: str, extract_csv: bool = True, extract_images: bool = True
):
    """
    Extract a .bag file containing events and images.

    Args:
        ros_path (str): rosbag file path.
        extract_csv (bool, optional): if True it's going to create a csv file
        for each topic in a folder called as the bag file in the same directory. Defaults to True.
        extract_images (bool, optional): if True it's going to extract raw and colored images
        in a folder called as the bag file in the same directory. Defaults to True.
    """
    if extract_csv:
        extract_ros_bag_csv(ros_path)

    if extract_images:
        extract_folder = os.path.join(
            os.path.dirname(ros_path),
            os.path.basename(ros_path).replace(".bag", ""),
        )

        dest_folder = os.path.join(extract_folder, "raw_images")
        print("Extracting raw images into", dest_folder)
        raw_images_path = os.path.join(extract_folder, "dvs-image_raw.csv")
        save_images_from_csv(raw_images_path, dest_folder)

        dest_folder = os.path.join(extract_folder, "color_images")
        print("Extracting color images into", dest_folder)
        color_images_path = os.path.join(extract_folder, "dvs-image_color.csv")
        save_images_from_csv(color_images_path, dest_folder)


def inspect_bag(rosbag_path: str):
    """
    Print a (truncated) message for each topic contained in the rosbag file.

    Args:
        dataset_path (str): path of the rosbag file.
    """
    bag = rosbag.Bag(rosbag_path)

    topics_types, topics_info = bag.get_type_and_topic_info()

    for topic_name in topics_info.keys():
        messages = iter(bag.read_messages(topics=[topic_name]))
        _topic, msg, _t = next(messages)

        print(topic_name)
        print(str(msg)[:200] + "...")
        print()

    bag.close()


def inspect_message_timestamps(
    rosbag_path: str, events_topic="/dvs/events", images_topic="/dvs/image_color"
):
    """
    Print events and images timestamps to check their sequenciality.

    Args:
        rosbag_path (str): path of the rosbag file.
        events_topic (str, optional): topic of the events messages. Defaults to "/dvs/events".
        images_topic (str, optional): topic of the images messages. Defaults to "/dvs/image_color".
    """
    i = 0
    print("{:5}{:20}{}".format("", "Topic Name", "Event Timestamp"))
    with rosbag.Bag(rosbag_path) as b:
        for topic, msg, _ in b.read_messages(topics=[events_topic, images_topic]):
            print("{:3}) {:20}{}".format(i, topic, msg.header.stamp.to_sec()))

            if topic == events_topic:
                print(
                    "{:25}First event timestamp {} Last event timestamp {}".format(
                        "", msg.events[0].ts.to_sec(), msg.events[-1].ts.to_sec()
                    )
                )

            i += 1
            if i == 15:
                break


def load_bag_as_dataframes(
    rosbag_path: str,
    events_topic="/dvs/events",
    images_topic="/dvs/image_color",
    max_events: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset from a rosbag file and return two pandas dataframe with events and images.

    Args:
        rosbag_path (str): path of the rosbag file.
        events_topic (str, optional): topic of the events messages. Defaults to "/dvs/events".
        images_topic (str, optional): topic of the images messages. Defaults to "/dvs/image_color".
        max_events (int, optional): maximum number of events to process.
        In case it is None, all the events will be loaded. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: events and images dataframes.
    """
    with rosbag.Bag(rosbag_path) as bag:

        events_list = []
        images_list = []

        n_events = 0
        for topic, msg, _ in tqdm(bag.read_messages(), total=bag.get_message_count()):
            if topic == images_topic:
                seq = msg.header.seq
                secs = msg.header.stamp.to_sec()
                width = msg.width
                height = msg.height
                is_bigendian = msg.is_bigendian
                encoding = msg.encoding
                step = msg.step
                data = np.frombuffer(msg.data, dtype=np.uint8)

                image_event = [
                    seq,
                    secs,
                    width,
                    height,
                    is_bigendian,
                    encoding,
                    step,
                    data,
                ]
                images_list.append(image_event)
            elif topic == events_topic:
                seq = msg.header.seq
                events = msg.events

                for event in events:
                    polarity = event.polarity
                    x = event.x
                    y = event.y
                    secs = event.ts.to_sec()

                    event = [seq, x, y, secs, polarity]
                    events_list.append(event)

                    n_events += 1
                    if n_events == max_events:
                        break

            if n_events == max_events:
                break

    events_columns = ["seq", "x", "y", "secs", "polarity"]
    images_columns = [
        "seq",
        "secs",
        "width",
        "height",
        "is_bigendian",
        "encoding",
        "step",
        "data",
    ]
    events_df = pd.DataFrame(events_list, columns=events_columns)
    images_df = pd.DataFrame(images_list, columns=images_columns)

    return events_df, images_df


def plot_number_of_events_per_frame(events_df: pd.DataFrame, images_df: pd.DataFrame):
    """
    Plot the number of events between each frame.

    Args:
        events_df (pd.DataFrame): dataframe of the events.
        images_df (pd.DataFrame): dataframe of the images.
    """
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
    """
    Plot the number of produced images per each second.

    Args:
        images_df (pd.DataFrame): dataframe of the images.
    """
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
    """Event captured by the camera."""

    def __init__(self, x: int, y: int, timestamp: float, polarity: bool):
        """
        Construct an event originated from an Event Camera.

        Args:
            x (int): x coordinate of the sensor.
            y (int): y coordinate of the sensor.
            timestamp (float): timestamp of the event.
            polarity (bool): polarity of the event.
            False means negative and True means positive.
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

    Args:
        events (List[Event]): list of events to create the grid.
        w (int): width of the resulting grid.
        h (int): height of the resulting grid.
        n_temp_bins (int): number of temporal bins for the grid.

    Returns:
        np.array: event grid with shape (`n_temps_bins`, `h`, `w`).
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


DatasetBatch = Tuple[np.ndarray, np.ndarray]


def get_sensor_size(bag_path: str, image_topic: str) -> Tuple[int, int]:
    """
    Get the sensor size used to generate images in a rosbag file.

    Args:
        bag_path (str): path of the rosbag file.
        image_topic (str): topic of the images messages.

    Returns:
        Tuple[int, int]: width and height of the sensor.
    """
    with rosbag.Bag(bag_path) as bag:
        _, msg, _ = next(iter(bag.read_messages(topics=[image_topic])))
        w, h = msg.width, msg.height
    return w, h


def rosbag_images_generator(
    bag: rosbag.Bag, image_topic: str, crop_size: Tuple[int, int] = None
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Yield images contained in an already opened rosbag file.

    Args:
        bag (rosbag.Bag): already opened rosbag handle.
        image_topic (str): topic of the images.
        crop_size (Tuple[int, int], optional): width and height to crop the image.
        Only the top left rectangle will be cropped if specified. Defaults to None.

    Yields:
        Generator[int, np.ndarray]: timestamp and image array.
    """
    _, msg, _ = next(iter(bag.read_messages(topics=[image_topic])))
    w, h = msg.width, msg.height
    channels = 1 if "mono" in msg.encoding else 3

    for _, msg, _ in bag.read_messages(topics=[image_topic]):
        cur_image_ts = msg.header.stamp.to_sec()
        image = bgr_to_rgb(image_from_buffer(msg.data, w, h, channels))
        if crop_size:
            image = image[: crop_size[1], : crop_size[0]]
        yield cur_image_ts, image


def dataset_generator_from_bag(
    bag_path: str,
    events_topic="/dvs/events",
    image_topic="/dvs/image_color",
    n_temp_bins: int = 10,
    crop_size: Tuple[int, int] = None,
    min_n_events: int = None,
) -> Generator[DatasetBatch, None, None]:
    """
    Yield training pairs formed by an event grid and the resulting output image.

    Args:
        bag_path (str): rosbag file path.
        events_topic (str, optional): topic of the events. Defaults to "/dvs/events".
        image_topic (str, optional): topic of the images. Defaults to "/dvs/image_color".
        n_temp_bins (int, optional): number of themporal bins. Defaults to 10.
        crop_size(Tuple[int, int], optional): accept events only in the top-left sector defined
        by this coordinates assuming origin at (0, 0). Defaults to None.
        min_n_events: batch at least this number of events before
        yielding the next event grid. Defaults to None.

    Yields:
        Generator[DatasetBatch, None, None]: training pair (event grid, output image).
    """
    if crop_size:
        w, h = crop_size
    else:
        w, h = get_sensor_size(bag_path, image_topic)

    events_batch: List[Event] = []

    with rosbag.Bag(bag_path) as bag:
        images_gen = iter(rosbag_images_generator(bag, image_topic, crop_size))
        cur_image_ts, cur_image = next(images_gen)

        msg_count = bag.get_message_count(events_topic)

        # Group events before image frame
        for _, msg, _ in tqdm(bag.read_messages(events_topic), total=msg_count):
            for event in msg.events:
                if crop_size and (event.x >= w or event.y >= h):
                    continue

                event_obj = Event(event.x, event.y, event.ts.to_sec(), event.polarity)

                if event_obj.timestamp <= cur_image_ts:
                    events_batch.append(event_obj)
                else:
                    if not min_n_events and len(events_batch) == 0:
                        print(
                            "Warning, there are no events between two frames, skipping one."
                        )
                        continue

                    if not min_n_events or len(events_batch) >= min_n_events:
                        event_grid = create_event_grid(
                            events_batch, w, h, n_temp_bins=n_temp_bins
                        )

                        yield (event_grid, cur_image)

                    try:
                        cur_image_ts, cur_image = next(images_gen)
                    except StopIteration:
                        return

                    events_batch = [event_obj]


def dataset_generator_from_batches(path: str) -> Generator[DatasetBatch, None, None]:
    """
    Yield samples from batches stored on the disk.

    Args:
        path (str): folder path containin batches files.

    Yields:
        Generator[DatasetBatch]: loaded batch.
    """
    for batch_file in os.listdir(path):
        if batch_file.endswith(".pt"):
            yield torch.load(os.path.join(path, batch_file))


def save_samples_to_disk(dataset: Iterable, dst_folder: str):
    """
    Save samples to the disk using the torch framework.

    Args:
        dataset (Iterable): iterable or generator containing the samples to save.
        dst_folder (str): destination folder where samples are going to be saved.
    """
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    for i, batch in enumerate(dataset):
        torch.save(batch, os.path.join(dst_folder, "batch_{:04}.pt".format(i)))


def save_events_frames_view(
    video_path: str,
    iterable: Iterable,
    model: Model = None,
    fps: int = 30,
    denorm: bool = False,
):
    """
    Save a visualization that contains the events and the ground truth frames.
    It can also optionally show the predicted images of a model.

    Args:
        video_path (str): path of the video that will be saved.
        iterable (Iterable): dataloader to get the input data.
        model (Model, optional): model used for inference. Defaults to None.
        fps (int, optional): fps of the resulting video. Defaults to 30.
        denorm (bool, optional): if True the frame will be denormalized. Defaults to False.
    """
    input_image = False
    try:
        (_, events), img_out = next(iter(iterable))
        input_image = True
    except:
        events, img_out = next(iter(iterable))

    h, w, _ = img_out.shape
    columns = 2 if model is None else 3

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter(video_path, fourcc, fps, (w * columns, h))

    try:
        for data in tqdm(iterable):
            if input_image:
                (_, events), gt_img = data
            else:
                events, gt_img = data

            if model is not None:
                pred = model(events[None, ...].to(model.device)).detach().cpu()

            for bin_ in events:
                event_frame = np.repeat(bin_.reshape(h, w, 1), 3, axis=2)
                event_frame = denorm_img(event_frame)

                images = [event_frame]
                if model is not None:
                    images.append(np.einsum("chw -> hwc", pred[0]))
                images.append(gt_img)

                frame = np.hstack(images)
                if denorm:
                    frame = denorm_img(frame)
                frame = rgb_to_bgr(frame).astype(np.uint8)

                out.write(frame)
    finally:
        out.release()


def save_events_frames_view_from_bag(
    video_path: str,
    rosbag_path: str,
    events_topic: str = "/dvs/events",
    image_topic: str = "/dvs/image_color",
    model: Model = None,
    n_bins: int = 10,
    fps: int = 30,
):
    gen = dataset_generator_from_bag(rosbag_path, events_topic, image_topic, n_bins)
    save_events_frames_view(video_path, gen, model, fps, denorm=False)


def save_events_frames_view_from_batches(
    video_path: str,
    batches_path: str,
    model: Model = None,
    fps: int = 30,
):
    gen = dataset_generator_from_batches(batches_path)
    save_events_frames_view(video_path, gen, model, fps, denorm=False)


if __name__ == "__main__":
    bags = r"G:\VM\Shared Folder\bags\DIV2K_0.5"
    failed_bags = []
    for bag_name in os.listdir(bags):
        try:
            bag_path = os.path.join(bags, bag_name)
            ds_gen = dataset_generator_from_bag(
                bag_path, "/cam0/events", "/cam0/image_raw", n_temp_bins=10
            )
            folder_path = os.path.join(
                r"G:\VM\Shared Folder\preprocess_800_0.5", bag_name.replace(".bag", "")
            )
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            dst_folder = os.path.join(folder_path, "batches")
            print("Saving batches to", dst_folder)
            save_samples_to_disk(ds_gen, dst_folder)
        except:
            failed_bags.append(bag_name)
    if failed_bags:
        print("The following bags have failed:")
        print("\n".join(failed_bags))
