import os
from typing import Callable, Dict
import torch
import time
import numpy as np
import json

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
    save_best_model: bool = False
):

    if log_path:
        now = format_current_date()
        
        experiment_dir = os.path.join(log_path, now)
        os.mkdir(experiment_dir)

        with open(os.path.join(experiment_dir, "metadata.json"), "w", encoding="utf8")as log_file:
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

    best_valid_loss = -1

    for epoch in range(params["n_epochs"]):
        model.train()
        start_epoch_time = time.time()
        batch_losses = []
        print(
            "Epoch {}/{} Step {}/{} Loss: {:.5f}".format(
                epoch + 1, params["n_epochs"], 1, len(train_ds), 0
            ),
            end="",
        )

        for step, batch in enumerate(train_ds):

            if input_process_fn:
                X, y = input_process_fn(batch)
            else:
                X, y = batch
            
            X = X.to(device)
            y = y.to(device)
            
            model_output = model(X)

            if output_process_fn:
                model_output = output_process_fn(model_output)

            loss: torch.Tensor = criterion(model_output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_losses.append(loss.cpu().detach())

            print(
                "\rEpoch {}/{} Step {}/{} Loss: {:.5f}".format(
                    epoch + 1, params["n_epochs"], step + 1, len(train_ds), loss
                ),
                end="",
            )

        # Epoch has ended
        elapsed_time = time.time() - start_epoch_time
        loss = np.mean(batch_losses)

        print(
            "\rEpoch {}/{} Step {}/{} Mean Loss: {:.5f} Elapsed Seconds: {}s".format(
                epoch + 1,
                params["n_epochs"],
                step + 1,
                len(train_ds),
                loss,
                int(elapsed_time),
            ), end=""
        )

        if valid_ds:
            model.eval()
            with torch.no_grad():
                for step, batch in enumerate(valid_ds):
                    if input_process_fn:
                        X, y = input_process_fn(batch)
                    else:
                        X, y = batch
                    
                    X = X.to(device)
                    y = y.to(device)
                    
                    model_output = model(X)

                    if output_process_fn:
                        model_output = output_process_fn(model_output)

                    loss: torch.Tensor = criterion(model_output, y)
                    batch_losses.append(loss.cpu().detach())
            valid_loss = np.mean(batch_losses)

            print(" - Valid Loss: {:.5f}".format(valid_loss), end="")

        print()

        if save_best_model:
            if best_valid_loss == -1 or valid_loss < best_valid_loss:
                model_filename = f"model.pt"
                model_path = os.path.join(experiment_dir, model_filename)
                torch.save(model.state_dict(), model_path)
            best_valid_loss = valid_loss
            log.update({"best_model": {
                "epoch": epoch,
                "valid_loss": float(valid_loss)
            }})

        if log_path:
            with open(os.path.join(experiment_dir, "metadata.json"), "w", encoding="utf8")as log_file:
                epoch_info = {
                    "train_loss": float(loss),
                    "elapsed_seconds": int(elapsed_time),
                }

                if valid_ds:
                    epoch_info.update({"valid_loss": float(valid_loss)})

                log_epochs.append(epoch_info)
                log.update({"epochs": log_epochs})

                json.dump(log, log_file, indent=2)


def train_unet(
    model: torch.nn.Module,
    device: str,
    train_ds: torch.utils.data.DataLoader,
    params: Dict,
    log_path: str = None,
    valid_ds: torch.utils.data.DataLoader = None,
    save_best_model: bool = False
):
    def in_fn(batch):
        (input_images, events_tensors), ground_truth_images = batch
        # events_tensors: (batch, bins, height, width)
        input_images = torch.einsum("bhwc -> bchw", input_images)
        ground_truth_images = torch.einsum("bhwc -> bchw", ground_truth_images)

        input_tensors = torch.hstack((input_images, events_tensors))
        return input_tensors, ground_truth_images

    train_generic(model, device, train_ds, params, log_path=log_path, input_process_fn=in_fn, valid_ds=valid_ds, save_best_model=save_best_model)


def train_transformer(
    model: torch.nn.Module,
    device: str,
    train_ds: torch.utils.data.DataLoader,
    params: Dict,
    log_path: str = None,
    valid_ds: torch.utils.data.DataLoader = None,
    save_best_model: bool = False
):

    def in_fn(batch):
        (input_images, events_tensors), ground_truth_images = batch
        ground_truth_images = torch.einsum("bhwc -> bchw", ground_truth_images)
        # events_tensors: (batch, bins, height, width)
        return events_tensors, ground_truth_images

    train_generic(model, device, train_ds, params, log_path=log_path, input_process_fn=in_fn, valid_ds=valid_ds, save_best_model=save_best_model)

def train_autoencoder(
    model: torch.nn.Module,
    device: str,
    train_ds: torch.utils.data.Dataset,
    params: Dict,
    log_path: str = None,
    valid_ds: torch.utils.data.DataLoader = None,
    save_best_model: bool = False
):

    def in_fn(sample):
        (_, batch), _ = sample
        batch = torch.unsqueeze(torch.from_numpy(batch), 1)
        return batch, batch

    train_generic(model, device, train_ds, params, log_path=log_path, input_process_fn=in_fn, valid_ds=valid_ds, save_best_model=save_best_model)        
