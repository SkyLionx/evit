import os
import numpy as np
import cv2
from tensorboard.backend.event_processing import event_accumulator


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
    event_acc = event_accumulator.EventAccumulator(
        experiment_path, size_guidance={"images": 0}
    )
    event_acc.Reload()

    for tag in event_acc.Tags()["images"]:
        path = os.path.join(out_path, tag + ".mp4")
        tb_images_to_video(event_acc, tag, path, fps=fps)


if __name__ == "__main__":
    experiment_path = r"E:\Cartelle Personali\Fabrizio\Universita\Magistrale\Tesi\05 - Experiments\lightning_logs\Large - 1e-4 lr, 1e-2 fl, fixed + more conv"
    tb_images_to_videos(experiment_path, "Large - 1e-4 lr, 1e-2 fl, fixed + more conv")
