from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd
import torch
from tqdm import tqdm


def plot_img(img: np.ndarray, grid=False, real_ticks=False, show=True, cmap="gray"):
    if real_ticks:
        plt.xticks(np.arange(img.shape[1] + 1) - 0.5, np.arange(img.shape[1] + 1))
        plt.yticks(np.arange(img.shape[0] + 1) - 0.5, np.arange(img.shape[0] + 1))
    plt.grid(grid)
    plt.imshow(img, cmap=cmap)
    if show:
        plt.show()


def image_from_buffer(
    data: bytes, width: int, height: int, channels: int
) -> np.ndarray:
    return (
        np.frombuffer(data, dtype=np.uint8).reshape(height, width, channels).squeeze()
    )


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return img[..., ::-1]


def save_video(path, frames, fps):
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()


def save_video_tensors(path, frames, fps):
    frames = torch.stack(frames)
    assert (
       frames.min() >= 0.0 and frames.max() <= 1.0
    ), "frames range should be in [0, 1]"
    height, width = frames[0].shape[:2]
    frames = [(frame * 255).numpy().astype(np.uint8) for frame in frames]
    save_video(path, frames, fps)

def save_predicted_video(model, device, dataloader, output_name, fps=30):
  frames = []
  for (img_in, events), img_out in tqdm(dataloader):
      with torch.no_grad():
          output = model(events.to(device))
      image_output = torch.einsum("bchw -> bhwc", output).squeeze()
      for image in image_output:
          frames.append(image.detach().cpu())
  save_video_tensors(output_name, frames, fps)


def get_empty_images(
    w: int, h: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return four grey empty images of size `w` x `h`, one for each Bayer channel (RGGB).
    """
    red_img = np.ones(shape=(h, w, 3), dtype=np.uint8) * 127
    green1_img = red_img.copy()
    green2_img = red_img.copy()
    blue_img = red_img.copy()
    return red_img, green1_img, green2_img, blue_img


def save_visual_events(
    events_df: pd.DataFrame, w: int, h: int, output_dir, batch_n_events=10000
):
    """
    Save visual events by batching them.
    The output is going to have four sectors, one for each RGGB filter.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    reconstructed_image = np.zeros(shape=(h, w), dtype=np.uint8)
    red_img, green1_img, green2_img, blue_img = get_empty_images(w, h)

    if not os.path.exists(os.path.join(output_dir, "visual_events")):
        os.mkdir(os.path.join(output_dir, "visual_events"))
    if not os.path.exists(os.path.join(output_dir, "visual_images")):
        os.mkdir(os.path.join(output_dir, "visual_images"))

    for i, (index, event) in enumerate(
        tqdm(events_df.iterrows(), total=len(events_df))
    ):
        cur_timestamp = int(event.secs)
        mod = (event.x % 2, event.y % 2)

        reconstructed_image[event.y][event.x] += event.polarity * 20

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
            plt.imsave(os.path.join(output_dir, f"visual_events/{i}.png"), final_img)
            plt.imsave(
                os.path.join(output_dir, f"visual_images/{i}.png"),
                reconstructed_image.astype(np.uint8),
                vmin=0,
                vmax=255,
            )
            reconstructed_image = np.zeros(shape=(h, w), dtype=np.uint8)
            red_img, green1_img, green2_img, blue_img = get_empty_images(w, h)
