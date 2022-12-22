import os
import numpy as np
import cv2
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing.event_accumulator import ImageEvent
import matplotlib.pyplot as plt
from media_utils import bgr_to_rgb
from typing import List, Tuple, Dict


def tb_decode_image_from_event(event: ImageEvent) -> np.ndarray:
    """
    Create an image from a TensorBoard ImageEvent

    Args:
        event (ImageEvent): event of the image.

    Returns:
        np.ndarray: decoded image.
    """
    s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
    image = cv2.imdecode(s, cv2.IMREAD_COLOR)
    return image


def tb_images_to_video(
    event_acc: event_accumulator.EventAccumulator, tag: str, out_path: str, fps: int
):
    """
    # Save one image tag as a progression video.

    Args:
        event_acc (event_accumulator.EventAccumulator): event accumulator of the experiment.
        tag (str): tag of the images to save.
        out_path (str): output folder where the video will be saved.
        fps (int): fps of the video.
    """

    events = event_acc.Images(tag)

    first_event = next(iter(events))
    img = tb_decode_image_from_event(first_event)
    h, w = img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for index, event in enumerate(events):
        s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
        image = cv2.imdecode(s, cv2.IMREAD_COLOR)
        text = f"{tag} - Step: {event.step}"
        font = cv2.FONT_HERSHEY_DUPLEX
        text_w, text_h = cv2.getTextSize(text, font, 1, 1)[0]
        text_x = (image.shape[1] - text_w) // 2
        text_y = 25
        cv2.putText(image, text, (text_x, text_y), font, 1, (0, 0, 0))
        out.write(image)


def tb_images_to_videos(experiment_path: str, out_path: str, fps: int = 15):
    """
    Save each image tag as a progression video.

    Args:
        experiment_path (str): path of the experiment.
        out_path (str): path where the video will be saved.
        fps (int, optional): fps of the video. Defaults to 15.
    """
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    event_acc = event_accumulator.EventAccumulator(
        experiment_path, size_guidance={"images": 0}
    )
    event_acc.Reload()

    for tag in event_acc.Tags()["images"]:
        path = os.path.join(out_path, tag + ".mp4")
        tb_images_to_video(event_acc, tag, path, fps=fps)


def tb_retrieve_best_metric(
    experiment_path: str,
    target_metric: str,
    mode: str,
    additional_metrics: List[str] = [],
) -> Tuple:
    """
    Get the best metric value(s) and epoch from a run.

    Args:
        experiment_path (str): path of the TensorBoard experiment.
        target_metric (str): metric to monitor.
        mode (str): how the best metric is choesen. It can be "min" or "max".
        additional_metrics (List[str], optional): list of additional metrics that will be returned
        taken from the same epoch of the best metric. Defaults to [].

    Returns:
        Tuple: (best_epoch, best_metric_value) and if additional metrics are specified,
        ((best_epoch, best_metric_value), (best_add_metric_value1, best_add_metric_value2, ...))
    """
    assert mode in ["min", "max"], "mode must be 'min' or 'max', it cannot be " + mode
    event_acc = event_accumulator.EventAccumulator(experiment_path)
    event_acc.Reload()

    best_additional_values = []

    events = event_acc.Scalars(target_metric)
    values = [event.value for event in events]
    if mode == "max":
        best_idx = np.argmax(values)
    else:
        best_idx = np.argmin(values)
    best_value = values[best_idx]

    output = (best_idx, best_value)

    for metric in additional_metrics:
        events = event_acc.Scalars(metric)
        values = [event.value for event in events]
        best_value = values[best_idx]
        best_additional_values.append(best_value)

    if additional_metrics:
        output = (output, best_additional_values)

    return output


class TB_Report:
    def __init__(self, runs_paths: str, tags_to_show: Dict[str, List[str]]):
        """
        TensorBoard report with images and scalars plotted as graphs from multiple runs.

        Args:
            runs_paths (str): base path where the runs are stored.
            tags_to_show (Dict[str, List[str]]): dictionary of tags that will be showed.
            The keys can be "scalars" or "images" and the values must be lists of tags.
        """
        self.runs_paths = runs_paths
        self.tags_to_show = tags_to_show

    def _compute_rows(self) -> Tuple[int, int]:
        """
        Compute how many rows the report should have.

        Raises:
            Exception: raised when a tag category is not supported.

        Returns:
            Tuple[int, int]: (number of scalars rows, number of images rows)
        """
        scalars = 0
        images = 0
        for category, tags in self.tags_to_show.items():
            if category == "scalars":
                scalars += len(tags)
            elif category == "images":
                images += len(self.runs_paths) * len(tags)
            else:
                raise Exception("Category {} is not supported.".format(category))
        return scalars, images

    def generate(
        self,
        output_path: str,
        width: int = 20,
        scalars_height: int = 2,
        images_height: int = 6,
    ):
        """
        Generate the report.

        Args:
            output_path (str): path of the figure that will be saved.
            width (int, optional): width size of the figure. Defaults to 20.
            scalars_height (int, optional): height of the scalars graphs. Defaults to 2.
            images_height (int, optional): height of the images. Defaults to 6.
        """
        scalars, images = self._compute_rows()
        rows = scalars + images
        height = scalars * scalars_height + images * images_height

        fig, axs = plt.subplots(rows, 1, figsize=(width, height))
        for run_idx, run_path in enumerate(self.runs_paths):
            run_name = os.path.basename(run_path)
            event_acc = event_accumulator.EventAccumulator(run_path)
            event_acc.Reload()

            ax_idx = 0
            for category, tags in self.tags_to_show.items():
                for tag in tags:
                    if category == "scalars":
                        events = event_acc.Scalars(tag)
                        steps = [event.step for event in events]
                        values = [event.value for event in events]
                        axs[ax_idx].plot(steps, values, label=run_name)
                        axs[ax_idx].set_title(tag)
                        axs[ax_idx].legend()
                        ax_idx += 1
                    elif category == "images":
                        ax_idx += run_idx
                        image_event = list(event_acc.Images(tag))[-1]
                        img = bgr_to_rgb(tb_decode_image_from_event(image_event))
                        axs[ax_idx].imshow(img)
                        axs[ax_idx].set_title(tag + " | " + run_name)
                        ax_idx += len(self.runs_paths) - run_idx
        plt.savefig(output_path)


if __name__ == "__main__":
    # Transform image progression into video
    # experiment_path = r"E:\Cartelle Personali\Fabrizio\Universita\Magistrale\Tesi\05 - Experiments\lightning_logs\"
    # run_name = "Large - 32-64 conv, 1 il, 1e-2 fl, batchnorm relu"
    # tb_images_to_videos(os.path.join(experiment_path, run_name), run_name)

    # experiment_path = r"E:\Cartelle Personali\Fabrizio\Universita\Magistrale\Tesi\Materiale da mostrare\12-05\lightning_logs\Large - 1 il, 1e-2 fl, bn relu, maxpool, polarity fix"
    experiment_path = r"E:\Cartelle Personali\Fabrizio\Universita\Magistrale\Tesi\05 - Experiments\lightning_logs\Large - VisionTransformerConv final"

    # Generate TensorBoard report
    # base_path = r"E:\Cartelle Personali\Fabrizio\Universita\Magistrale\Tesi\05 - Experiments\lightning_logs"
    # paths = [
    #     os.path.join(base_path, "Large - 32-64 conv, 1 il, 1e-2 fl"),
    #     os.path.join(base_path, "Large, Long - 1 il, 1e-2 fl, bn relu, maxpool"),
    #     # os.path.join(
    #     #     base_path, "Large - 32-64 conv, 1 il, 1e-2 fl, batchnorm relu, strided conv"
    #     # ),
    #     # os.path.join(base_path, "Large - 32-64 conv, 0.8 il, 0.2 fl"),
    # ]
    # tags_to_show = {
    #     "scalars": ["train_loss", "val_MSE", "val_SSIM", "val_LPIPS"],
    #     "images": ["train", "valid"],
    # }
    # TB_Report(paths, tags_to_show).generate("Report.pdf")

    # Compute best metric and epoch
    (best_idx, best_lpips), (best_mse, best_ssim) = tb_retrieve_best_metric(
        experiment_path, "val_LPIPS", "min", additional_metrics=["val_MSE", "val_SSIM"]
    )

    print(
        "Best Epoch: {}, MSE: {:.4f}, SSIM: {:.4f}, LPIPS: {:.4f}".format(
            best_idx, best_mse, best_ssim, best_lpips
        )
    )

    # (best_idx, best_val_loss), (
    #     best_mse,
    #     best_ssim,
    #     best_lpips,
    # ) = tb_retrieve_best_metric(
    #     experiment_path,
    #     "val_loss",
    #     "min",
    #     additional_metrics=["val_MSE", "val_SSIM", "val_LPIPS"],
    # )

    # print(
    #     "Best Epoch: {}, MSE: {:.4f}, SSIM: {:.4f}, LPIPS: {:.4f}".format(
    #         best_idx, best_mse, best_ssim, best_lpips
    #     )
    # )
