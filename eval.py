import torch
import os
import json

import matplotlib.pyplot as plt

from torchmetrics import (
    MeanSquaredError as MSE,
    StructuralSimilarityIndexMeasure as SSIM,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from tqdm import tqdm

from dataset_utils import get_dataset
from models import get_model


def eval_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    output_folder: str,
    disable_progress=False,
):

    # Predict and save images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mse = MSE()
    ssim = SSIM(data_range=1)
    lpips = LPIPS(net_type="vgg", normalize=True)

    i = 0
    for batch in tqdm(dataloader, disable=disable_progress):
        events, imgs = batch

        # Compute results
        model.eval()
        with torch.no_grad():
            out = model.predict_images(batch)

        # Compute metrics
        imgs = imgs.permute(0, 3, 1, 2)
        greyscale_imgs = imgs.shape[1] == 1
        mse(out, imgs)
        ssim(out, imgs)
        if greyscale_imgs:  # greyscale images
            out_rgb = torch.repeat_interleave(out, 3, 1)
            imgs_rgb = torch.repeat_interleave(imgs, 3, 1)
            lpips(out_rgb, imgs_rgb)
        else:
            lpips(out, imgs)

        # Save images and ground truths
        for j in range(len(out)):
            pred_img = out[j].permute(1, 2, 0).squeeze().detach().cpu().numpy()
            gt_img = imgs[j].permute(1, 2, 0).squeeze().detach().cpu().numpy()
            cmap = "gray" if greyscale_imgs else None
            filename = f"{i:04}_pred.png"
            plt.imsave(os.path.join(output_folder, filename), pred_img, cmap=cmap)
            filename = f"{i:04}_gt.png"
            plt.imsave(os.path.join(output_folder, filename), gt_img, cmap=cmap)
            i += 1

    # Save metrics
    metrics = {
        "MSE": mse.compute().item(),
        "SSIM": ssim.compute().item(),
        "LPIPS": lpips.compute().item(),
    }
    with open(os.path.join(output_folder, "metrics.json"), "w", encoding="utf8") as fp:
        json.dump(metrics, fp)


def eval_VisionTransformerConv(
    results_path, checkpoint_path, dataloader, reconstruct_colors=False
):
    from models.transformer import VisionTransformerConv, predict_color_images

    model = VisionTransformerConv.load_from_checkpoint(
        checkpoint_path,
        feature_loss_weight=None,
        image_loss_weight=None,
        map_location="cuda",
    )

    path_split = checkpoint_path.split(os.sep)
    RUN_NAME = path_split[path_split.index("checkpoints") - 1]
    suffix = " - " + ("last" if "last.ckpt" in checkpoint_path else "best")

    if reconstruct_colors:
        from types import MethodType

        model.predict_images = MethodType(predict_color_images, model)
        suffix = " color " + suffix

    output_folder = os.path.join(results_path, RUN_NAME + suffix)
    eval_model(model, dataloader, output_folder)


def eval_StudentK(results_path, checkpoint_path, dataloader):
    from models.teacher_student import StudentK

    teacher = get_model(
        {
            "class_name": "Teacher",
            "teacher_path": "teacher-epoch=287-step=18144.ckpt",
            "MODEL_PARAMS": {"lr": None},
        }
    )
    model = StudentK.load_from_checkpoint(
        checkpoint_path, teacher=teacher, map_location="cuda"
    )

    path_split = checkpoint_path.split(os.sep)
    RUN_NAME = path_split[path_split.index("checkpoints") - 1]

    output_folder = os.path.join(results_path, RUN_NAME)
    eval_model(model, dataloader, output_folder)


def prepare_dataset(
    base_dataset_path,
    dataset_name="DIV2K_5_FIX",
    split="valid",
    batch_size=16,
    crop_size=(128, 128),
):
    dataset_params = {
        "limit": None,
        "preload_to_RAM": True,
        "crop_size": crop_size,
        "events_normalization": None,
        "convert_to_bw": False,
    }

    if "BW" in dataset_name.upper():
        dataset_params["convert_to_bw"] = True

    dataloader_params = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": True,
    }

    dataset = get_dataset(
        base_dataset_path, dataset_name, dataset_params, splits=[split]
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, **dataloader_params
    )

    print("Loaded samples: {} \t batches: {:<10}".format(len(dataset), len(dataloader)))

    return dataset, dataloader


def eval_EventsVisualization(results_path, dataset, show_images=False):
    from media_utils import gen_visual_bayer_events

    i = 0
    for events, imgs in dataset:
        img = gen_visual_bayer_events(events)
        res_folder = os.path.join(results_path, "EventsVisualization")
        if not os.path.exists(res_folder):
            os.mkdir(res_folder)
        out_path = os.path.join(res_folder, f"{i:04}.png")
        plt.imsave(out_path, img)
        if show_images:
            plt.imshow(img)
            plt.show()
        i += 1


def eval_E2VID(results_path, weights_path, dataloader):
    # Append their repo to path to fix import issues
    import sys

    sys.path.append("rpg_e2vid")

    from rpg_e2vid.model.model import E2VIDRecurrent


if __name__ == "__main__":
    results_path = r"..\06 - Results"

    base_dataset_path = r"C:\datasets"

    valid_dataset, valid_dataloader = prepare_dataset(
        base_dataset_path,
        dataset_name="DIV2K_5_FIX_SMALL",
        split="valid",
        batch_size=16,
    )

    # Color VisionTransformerConv
    paths = [
        r"E:\Cartelle Personali\Fabrizio\Universita\Magistrale\Tesi\Materiale da mostrare\12-05\lightning_logs\Large - 1 il, 1e-2 fl, bn relu, maxpool, polarity fix\checkpoints\epoch=175-step=11088.ckpt",
        r"E:\Cartelle Personali\Fabrizio\Universita\Magistrale\Tesi\05 - Experiments\lightning_logs\Large - VisionTransformerConv final\checkpoints\epoch=499-last.ckpt",
    ]
    for checkpoint_path in paths:
        eval_VisionTransformerConv(results_path, checkpoint_path, valid_dataloader)
    checkpoint_path = r"E:\Cartelle Personali\Fabrizio\Universita\Magistrale\Tesi\05 - Experiments\lightning_logs_00\Large - ViTConv black and white\checkpoints\epoch=273-step=17262.ckpt"

    # Color black and white VisionTransformerConv
    eval_VisionTransformerConv(
        results_path, checkpoint_path, valid_dataloader, reconstruct_colors=True
    )

    # StudentK
    checkpoint_path = r"E:\Cartelle Personali\Fabrizio\Universita\Magistrale\Tesi\05 - Experiments\lightning_logs\Large - StudentK LeakyReLU PlateauLR\checkpoints\epoch=221-last.ckpt"
    eval_StudentK(results_path, checkpoint_path, valid_dataloader)

    # Events visualization
    eval_EventsVisualization(results_path, valid_dataset)

    # Black and white VisionTransformerConv
    # It's at the end of the eval because there is the need to load the bw dataset
    valid_dataset, valid_dataloader = prepare_dataset(
        +base_dataset_path,
        dataset_name="DIV2K_5_BW_FIX",
        split="valid",
        batch_size=16,
    )
    eval_VisionTransformerConv(results_path, checkpoint_path, valid_dataloader)