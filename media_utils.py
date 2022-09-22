from typing import Iterable, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd
import torch
import pytorch_lightning as pl
from tqdm import tqdm

Model = Union[torch.nn.Module, pl.LightningModule]


def norm_img(img: np.ndarray) -> np.ndarray:
    """
    Normalize the image values from the range [0, 255] to [0.0, 1.0].

    Args:
        img (np.ndarray): image to normalize.

    Returns:
        np.ndarray: image normalized in [0.0, 1.0].
    """
    return img.astype(np.float32) / 255.0


def denorm_img(img: np.ndarray) -> np.ndarray:
    """
    Denormalize the image values from the range [0.0, 1.0] to [0, 255].

    Args:
        img (np.ndarray): image to denormalize.

    Returns:
        np.ndarray: image normalized in [0, 255].
    """
    return (img * 255).astype(np.uint8)


def plot_img(
    img: np.ndarray,
    grid: bool = False,
    real_ticks: bool = False,
    show: bool = True,
    cmap: str = "gray",
):
    """
    Plot an image array using matplotlib.

    Args:
        img (np.ndarray): image array that needs to be plotted.
        grid (bool, optional): show a grid over the image.
        real_ticks (bool, optional): shift ticks in order to have 0 at the origin of the image. Defaults to False. Defaults to False.
        show (bool, optional): show the image produced (don't set it if you're using the function just to create the plot). Defaults to True.
        cmap (str, optional): colormap string compatible with matplotlib. Defaults to "gray".
    """
    if real_ticks:
        plt.xticks(np.arange(img.shape[1] + 1) - 0.5, np.arange(img.shape[1] + 1))
        plt.yticks(np.arange(img.shape[0] + 1) - 0.5, np.arange(img.shape[0] + 1))
    plt.grid(grid)
    plt.imshow(img, cmap=cmap)
    if show:
        plt.show()


def plot_square(
    imgs: Iterable[np.ndarray], titles: Iterable[str] = None, size: int = 4
):
    """
    Plot a square figure showing the `imgs`.

    Args:
        imgs (Iterable[np.ndarray]): iterable of images that needs to be plotted.
        titles (Iterable[str], optional): titles that will be shown on the images. Defaults to None.
        size (int, optional): size factor of a single image. Defaults to 4.
    """
    n = len(imgs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.floor(np.sqrt(n)))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * size, rows * size))
    for i, ax in enumerate(axs.flatten()):

        if i >= len(imgs):
            ax.remove()
            continue

        ax.imshow(imgs[i])
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()


def image_from_buffer(
    data: bytes, width: int, height: int, channels: int
) -> np.ndarray:
    """
    Build a numpy array representing an image from a byte buffer.

    Args:
        data (bytes): bytes representing the image data.
        width (int): width of the image.
        height (int): height of the image.
        channels (int): number of channels of the image.

    Returns:
        np.ndarray: built uint8 numpy array representing the image.
    """
    return (
        np.frombuffer(data, dtype=np.uint8).reshape(height, width, channels).squeeze()
    )


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert an image from BGR format to the RGB format.

    Args:
        img (np.ndarray): image in BGR to convert.

    Returns:
        np.ndarray: converted RGB image.
    """
    return img[..., ::-1]


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """
    Convert an image from RGB format to the BGR format.

    Args:
        img (np.ndarray): image in RGB to convert.

    Returns:
        np.ndarray: converted BGR image.
    """
    return img[..., ::-1]


def save_video(path: str, frames: Iterable[np.ndarray], fps: int):
    """
    Save a video on the disk with the provided frames.

    Args:
        path (str): path of the resulting video file.
        frames (Iterable[np.ndarray]): frames to be saved.
        fps (int): fps of the resulting video.
    """

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()


def save_video_tensors(path: str, frames: Iterable[torch.tensor], fps: int):
    """
    Save a video on the disk with the provided tensors

    Args:
        path (_type_): path of the resulting video file.
        frames (_type_): frames with values in [0.0, 1.0] to be saved.
        fps (_type_): fps of the resulting video.
    """
    frames = torch.stack(frames)
    assert (
        frames.min() >= 0.0 and frames.max() <= 1.0
    ), "frames range should be in [0, 1]"
    frames = [(frame * 255).numpy().astype(np.uint8) for frame in frames]
    save_video(path, frames, fps)


def save_predicted_video(
    model: Model,
    dataloader: torch.utils.data.DataLoader,
    video_path: str,
    fps: int = 30,
):
    """
    Save the output of the model when given as input a video.

    Args:
        model (Model): model used in order to process the events.
        dataloader (torch.utils.data.DataLoader): dataloader to get the input data.
        video_path (str): path of the resulting video.
        fps (int, optional): fps of the resulting video. Defaults to 30.
    """
    frames = []
    for events, img_out in tqdm(dataloader):
        with torch.no_grad():
            output = model(events.to(model.device))
        image_output = torch.einsum("bchw -> bhwc", output)
        for image in image_output:
            frames.append(image.detach().cpu())
    save_video_tensors(video_path, frames, fps)


def predict_n_images(
    dataset: torch.utils.data.Dataset, n_imgs: int, model: Model
) -> Iterable[np.ndarray]:
    """
    Uniformly select `n_images` samples from the `dataset` and predict their results.

    Args:
        dataset (torch.utils.data.Dataset): dataset to get the events from.
        n_imgs (int): number of images to produce.
        model (Model): model used for inference.
    Returns:
        Iterable[np.ndarray]: list of resulting images.
    """
    eval_events = []

    indexes = np.linspace(0, len(dataset) - 1, n_imgs).astype(np.int32)
    for idx in indexes:
        events, out_img = dataset[idx]
        eval_events.append(torch.tensor(events))
    eval_events = torch.stack(eval_events).to(model.device)

    model.eval()
    with torch.no_grad():
        results = model(eval_events)
        results = np.einsum("bchw -> bhwc", results.detach().cpu())
    return results


def get_empty_bayer_images(
    w: int, h: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return four grey empty images of size `w` x `h`, one for each Bayer channel (RGGB).

    Args:
        w (int): width of the images.
        h (int): height of the images.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: four images for each Bayer channel (RGGB).
    """
    red_img = np.ones(shape=(h, w, 3), dtype=np.uint8) * 127
    green1_img = red_img.copy()
    green2_img = red_img.copy()
    blue_img = red_img.copy()
    return red_img, green1_img, green2_img, blue_img


def save_visual_bayer_events(
    events_df: pd.DataFrame,
    w: int,
    h: int,
    output_dir: str,
    batch_n_events: int = 10000,
):
    """
    Save batches of events in a image form.
    Each image is going to have four sectors, one for each RGGB filter,
    showing the respective events polarity accumulated.

    Args:
        events_df (pd.DataFrame): dataframe containg the events.
        w (int): width of the sensor.
        h (int): height of the sensor.
        output_dir (str): folder where to save the images.
        batch_n_events (int): number of events to batch. Defaults to 10000.
    """

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    red_img, green1_img, green2_img, blue_img = get_empty_bayer_images(w, h)

    for i, (_, event) in enumerate(tqdm(events_df.iterrows(), total=len(events_df))):
        cur_timestamp = int(event.secs)
        mod = (event.x % 2, event.y % 2)

        if mod == (0, 0):
            red_img[event.y][event.x] = [255, 0, 0] if event.polarity else [0, 0, 0]
        elif mod == (1, 0):
            green1_img[event.y][event.x] = [0, 255, 0] if event.polarity else [0, 0, 0]
        elif mod == (0, 1):
            green2_img[event.y][event.x] = [0, 255, 0] if event.polarity else [0, 0, 0]
        elif mod == (1, 1):
            blue_img[event.y][event.x] = [0, 0, 255] if event.polarity else [0, 0, 0]

        if i % batch_n_events == 0:
            final_img = np.vstack(
                (np.hstack((red_img, green1_img)), np.hstack((green2_img, blue_img)))
            )
            plt.suptitle(cur_timestamp)
            plt.imsave(os.path.join(output_dir, f"{i}.png"), final_img)
            red_img, green1_img, green2_img, blue_img = get_empty_bayer_images(w, h)


def save_visual_accumulated_events(
    events_df: pd.DataFrame,
    w: int,
    h: int,
    output_dir: str,
    batch_n_events: int = 10000,
    pol_factor: int = 20,
):
    """
    Save accumulated events images.

    Args:
        events_df (pd.DataFrame): dataframe containg the events.
        w (int): width of the sensor.
        h (int): height of the sensor.
        output_dir (str): folder where to save the images.
        batch_n_events (int): number of events to batch. Defaults to 10000.
        pol_factor (int, optional): factor to which each polarity is going to be multiplied by. Defaults to 20.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    reconstructed_image = np.zeros(shape=(h, w), dtype=np.uint8)

    for i, (_, event) in enumerate(tqdm(events_df.iterrows(), total=len(events_df))):
        cur_timestamp = int(event.secs)
        reconstructed_image[event.y][event.x] += event.polarity * pol_factor

        if i % batch_n_events == 0:
            plt.imsave(
                os.path.join(output_dir, f"{i}.png"),
                reconstructed_image.astype(np.uint8),
                vmin=0,
                vmax=255,
            )
            reconstructed_image = np.zeros(shape=(h, w), dtype=np.uint8)
