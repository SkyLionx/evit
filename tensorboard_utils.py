import os
import numpy as np
import cv2
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from media_utils import bgr_to_rgb


def tb_decode_image_from_event(event):
    s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
    image = cv2.imdecode(s, cv2.IMREAD_COLOR)
    return image


def tb_images_to_video(event_acc, tag, out_path, fps):
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


def tb_images_to_videos(experiment_path, out_path, fps=15):
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    event_acc = event_accumulator.EventAccumulator(
        experiment_path, size_guidance={"images": 0}
    )
    event_acc.Reload()

    for tag in event_acc.Tags()["images"]:
        path = os.path.join(out_path, tag + ".mp4")
        tb_images_to_video(event_acc, tag, path, fps=fps)


class TB_Report:
    def __init__(self, runs_paths, tags_to_show) -> None:
        self.runs_paths = runs_paths
        self.tags_to_show = tags_to_show

    def _compute_rows(self):
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

    def generate(self, width=20, scalars_height=2, images_height=6):
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
        plt.savefig("Report.pdf")

    # for tag in ["images"]:
    #     path = os.path.join(out_path, tag + ".mp4")
    #     tb_images_to_video(event_acc, tag, path, fps=fps)


if __name__ == "__main__":
    # experiment_path = r"E:\Cartelle Personali\Fabrizio\Universita\Magistrale\Tesi\05 - Experiments\lightning_logs"
    # run_name = "Large - 32-64 conv, 1 il, 1e-2 fl, batchnorm relu"
    # tb_images_to_videos(os.path.join(experiment_path, run_name), run_name)

    base_path = r"E:\Cartelle Personali\Fabrizio\Universita\Magistrale\Tesi\05 - Experiments\lightning_logs"
    paths = [
        os.path.join(base_path, "Large - 32-64 conv, 1 il, 1e-2 fl"),
        os.path.join(base_path, "Large, Long - 1 il, 1e-2 fl, bn relu, maxpool"),
        # os.path.join(
        #     base_path, "Large - 32-64 conv, 1 il, 1e-2 fl, batchnorm relu, strided conv"
        # ),
        # os.path.join(base_path, "Large - 32-64 conv, 0.8 il, 0.2 fl"),
    ]
    tags_to_show = {
        "scalars": ["train_loss", "val_MSE", "val_SSIM", "val_LPIPS"],
        "images": ["train", "valid"],
    }
    TB_Report(paths, tags_to_show).generate()
