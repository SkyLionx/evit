from typing import Iterable, Tuple
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd
import torch
from tqdm import tqdm


def norm_img(img: np.ndarray):
    return img.astype(np.float32) / 255.0


def denorm_img(img: np.ndarray):
    return (img * 255).astype(np.uint8)


def plot_img(img: np.ndarray, grid=False, real_ticks=False, show=True, cmap="gray"):
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
    return (
        np.frombuffer(data, dtype=np.uint8).reshape(height, width, channels).squeeze()
    )


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return img[..., ::-1]


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
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
    for events, img_out in tqdm(dataloader):
        with torch.no_grad():
            output = model(events.to(device))
        image_output = torch.einsum("bchw -> bhwc", output)
        for image in image_output:
            frames.append(image.detach().cpu())
    save_video_tensors(output_name, frames, fps)


def save_events_frames_visualization(
    sensor_size, filename, dataloader, model=None, fps=30
):
    w, h = sensor_size
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    columns = 2 if model is None else 3
    out = cv2.VideoWriter(filename, fourcc, fps, (w * columns, h))

    for events, img_out in tqdm(dataloader):
        if model is not None:
            pred = model(events.to(model.device)).detach().cpu()

        for i, batch in enumerate(events):
            gt_img = img_out[i]
            for bin_ in batch:
                event_frame = torch.repeat_interleave(bin_.reshape(h, w, 1), 3, dim=2)

                images = [event_frame]
                if model is not None:
                    images.append(torch.einsum("chw -> hwc", pred[i]))
                images.append(gt_img)

                frame = np.hstack(images)
                frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                out.write(frame)

    out.release()


def predict_n_images(dataset, n_imgs, model):
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
