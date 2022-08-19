import os
from typing import Callable, Dict
import torch
import time
import numpy as np
import json
import matplotlib.pyplot as plt

from utils import format_current_date

def train_generic(
    model: torch.nn.Module,
    device: str,
    train_ds: torch.utils.data.DataLoader,
    params: Dict,
    log_path: str = None,
    input_process_fn: Callable = None,
    output_process_fn: Callable = None,
    valid_ds: torch.utils.data.DataLoader = None,
    save_best_model: bool = False,
    save_imgs_after_n_epochs: int = None,
    save_times: bool = False,
):

    # Log initialization
    if log_path:
        now = format_current_date()
        
        experiment_dir = os.path.join(log_path, now)
        os.mkdir(experiment_dir)

        if save_imgs_after_n_epochs:
            os.mkdir(os.path.join(experiment_dir, "imgs"))

        with open(os.path.join(experiment_dir, "metadata.json"), "w", encoding="utf8") as log_file:
            log = {
                "model_class": model.__class__.__name__, 
                "params": params,
                "creation_date": now,
                "device": device
            }
            log_epochs = []
            json.dump(log, log_file, indent=2)

    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    # This is going to be train loss if no valid data has been provided
    best_loss = -1
    
    for epoch in range(params["n_epochs"]):
        save_epochs = save_imgs_after_n_epochs
        save_imgs = save_epochs and (epoch + 1) % save_epochs == 0

        model.train()
        start_epoch_time = time.time()
        print(
            "Epoch {}/{} Step {}/{} Loss: {:.5f}".format(
                epoch + 1, params["n_epochs"], 1, len(train_ds), 0
            ),
            end="",
        )

        times = {
            "dataset_time": 0,
            "input_process_time": 0,
            "to_device_time": 0,
            "inference_time": 0,
            "save_train_imgs_time": 0,
            "output_process_time": 0,
            "compute_loss_time": 0,
            "backward_pass_time": 0,
            "append_losses_time": 0,
            "print_step_time": 0,
            "batch_time": 0,
            "eval_time": 0,
            "save_best_model_time": 0,
            "save_imgs_time": 0,
            "total_epoch_time": 0,
        }

        dataset_time_var = time.time()
        for step, batch in enumerate(train_ds):
            batch_losses = []
            times["dataset_time"] += time.time() - dataset_time_var          

            if input_process_fn:
                t = time.time()
                X, y = input_process_fn(batch)
                times["input_process_time"] += time.time() - t
            else:
                X, y = batch
            
            t = time.time()
            X = X.to(device)
            y = y.to(device)
            times["to_device_time"] += time.time() - t
            
            t = time.time()
            model_output = model(X)
            times["inference_time"] += time.time() - t

            # Save some training images to check progress
            train_imgs_out = []
            if save_imgs and len(train_imgs_out) < 5:
                t = time.time()
                outs = model_output[:5].detach()
                ys = y[:5].detach()
                train_imgs_out = list(zip(outs, ys))
                times["save_train_imgs_time"] += time.time() - t

            if output_process_fn:
                t = time.time()
                model_output = output_process_fn(model_output)
                times["output_process_time"] += time.time() - t


            t = time.time()
            loss: torch.Tensor = criterion(model_output, y)
            times["compute_loss_time"] += time.time() - t

            t = time.time()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            times["backward_pass_time"] += time.time() - t

            t = time.time()
            detached_loss = loss.detach()
            batch_losses.append(detached_loss)
            times["append_losses_time"] += time.time() - t


            t = time.time()
            print(
                "\rEpoch {}/{} Step {}/{} Loss: {:.5f}".format(
                    epoch + 1, params["n_epochs"], step + 1, len(train_ds), detached_loss
                ),
                end="",
            )
            times["print_step_time"] += time.time() - t

            times["batch_time"] += time.time() - dataset_time_var
            dataset_time_var = time.time()

        # Epoch has ended
        elapsed_time = time.time() - start_epoch_time
        train_loss = torch.mean(torch.Tensor(batch_losses)).detach()
        del batch_losses

        print(
            "\rEpoch {}/{} Step {}/{} Mean Loss: {:.5f} Elapsed Seconds: {}s".format(
                epoch + 1,
                params["n_epochs"],
                step + 1,
                len(train_ds),
                train_loss,
                int(elapsed_time),
            ), end=""
        )

        # Validation step
        if valid_ds:
            t = time.time()
            model.eval()
            batch_losses = []
            with torch.no_grad():
                for step, batch in enumerate(valid_ds):
                    if input_process_fn:
                        X, y = input_process_fn(batch)
                    else:
                        X, y = batch
                    
                    X = X.to(device)
                    y = y.to(device)
                    
                    model_output = model(X)

                    val_imgs_out = []
                    if save_imgs:
                        outs = model_output
                        ys = y[:5]
                        val_imgs_out = list(zip(outs, ys))

                    if output_process_fn:
                        model_output = output_process_fn(model_output)

                    loss: torch.Tensor = criterion(model_output, y)
                    batch_losses.append(loss.detach())
            valid_loss = torch.mean(torch.Tensor(batch_losses)).detach()
            del batch_losses

            times["eval_time"] += time.time() - t
            print(" - Valid Loss: {:.5f}".format(valid_loss), end="")

        print()

        # Save best model
        if save_best_model:
            t = time.time()
            current_loss = loss if not valid_ds else valid_loss
            loss_key = "train_loss" if not valid_ds else "valid_loss"

            if best_loss == -1 or current_loss < best_loss:
                model_filename = f"model.pt"
                model_path = os.path.join(experiment_dir, model_filename)
                torch.save(model.state_dict(), model_path)
               
                best_loss = current_loss
                
                log.update({"best_model": {
                    "epoch": epoch,
                    loss_key: float(current_loss)
                }})
            times["save_best_model_time"] += time.time() - t

        # Optionally save images
        if save_imgs:
            t = time.time()
            for i, (out, y) in enumerate(train_imgs_out + val_imgs_out):
                ds = "train" if i < len(train_imgs_out) else "valid"
                filename = f"{ds}_{epoch:04}_{i:04}.png"
                out = torch.permute(out, (1, 2, 0)).detach().cpu().numpy()
                y = torch.permute(y, (1, 2, 0)).detach().cpu().numpy()
                plt.subplot(1, 2, 1)
                plt.title("Model Output")
                plt.axis("off")
                plt.imshow(out)
                plt.subplot(1, 2, 2)
                plt.title("Ground Truth")
                plt.axis("off")
                plt.imshow(y)
                plt.savefig(os.path.join(experiment_dir, "imgs", filename))
            times["save_imgs_time"] += time.time() - t

        times["total_epoch_time"] = time.time() - start_epoch_time

        # Save log
        if log_path:
            with open(os.path.join(experiment_dir, "metadata.json"), "w", encoding="utf8") as log_file:
                epoch_info = {
                    "train_loss": float(train_loss),
                    "elapsed_seconds": int(elapsed_time),
                }

                if valid_ds:
                    epoch_info.update({"valid_loss": float(valid_loss)})

                log_epochs.append(epoch_info)
                log.update({"epochs": log_epochs})

                if save_times:
                    log.update({"times": times})

                json.dump(log, log_file, indent=2)

        # Optionally print times
        # for name, value in times.items():
        #     print(name.replace("_", " ").title(), value)

def train_unet(
    model: torch.nn.Module,
    device: str,
    train_ds: torch.utils.data.DataLoader,
    params: Dict,
    log_path: str = None,
    valid_ds: torch.utils.data.DataLoader = None,
    save_best_model: bool = False,
    save_times: bool = False,
):
    def in_fn(batch):
        (input_images, events_tensors), ground_truth_images = batch
        # events_tensors: (batch, bins, height, width)
        input_images = torch.einsum("bhwc -> bchw", input_images)
        ground_truth_images = torch.einsum("bhwc -> bchw", ground_truth_images)

        input_tensors = torch.hstack((input_images, events_tensors))
        return input_tensors, ground_truth_images

    train_generic(model, device, train_ds, params, log_path=log_path, input_process_fn=in_fn, valid_ds=valid_ds, save_best_model=save_best_model, save_times=save_times)


def train_transformer(
    model: torch.nn.Module,
    device: str,
    train_ds: torch.utils.data.DataLoader,
    params: Dict,
    log_path: str = None,
    valid_ds: torch.utils.data.DataLoader = None,
    save_best_model: bool = False,
    save_times: bool = False,
):

    def in_fn(batch):
        (input_images, events_tensors), ground_truth_images = batch
        ground_truth_images = torch.einsum("bhwc -> bchw", ground_truth_images)
        # events_tensors: (batch, bins, height, width)
        return events_tensors, ground_truth_images

    train_generic(model, device, train_ds, params, log_path=log_path, input_process_fn=in_fn, valid_ds=valid_ds, save_best_model=save_best_model, save_times=save_times)

def train_autoencoder(
    model: torch.nn.Module,
    device: str,
    train_ds: torch.utils.data.Dataset,
    params: Dict,
    log_path: str = None,
    valid_ds: torch.utils.data.DataLoader = None,
    save_best_model: bool = False,
    save_times: bool = False,
):

    def in_fn(sample):
        (_, batch), _ = sample
        batch = torch.unsqueeze(torch.from_numpy(batch), 1)
        return batch, batch

    train_generic(model, device, train_ds, params, log_path=log_path, input_process_fn=in_fn, valid_ds=valid_ds, save_best_model=save_best_model, save_times=save_times)        

def train_vit(
    model: torch.nn.Module,
    device: str,
    train_ds: torch.utils.data.DataLoader,
    params: Dict,
    log_path: str = None,
    valid_ds: torch.utils.data.DataLoader = None,
    save_best_model: bool = False,
    save_imgs_after_n_epochs: int = None,
    save_times: bool = False,
):

    def in_fn(batch):
        (input_images, events_tensors), ground_truth_images = batch
        events_tensors = events_tensors[:, :, :256, :256]
        ground_truth_images = torch.einsum("bhwc -> bchw", ground_truth_images)[:, :, :256, :256]
        return events_tensors, ground_truth_images

    train_generic(model, device, train_ds, params, log_path=log_path, 
    input_process_fn=in_fn, valid_ds=valid_ds, save_best_model=save_best_model, 
    save_imgs_after_n_epochs=save_imgs_after_n_epochs, save_times=save_times)