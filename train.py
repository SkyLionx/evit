from typing import Dict
import torch
import time
import numpy as np


def train_unet(
    model: torch.nn.Module,
    device: str,
    train_ds: torch.utils.data.DataLoader,
    n_epochs: int,
):
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    for epoch in range(n_epochs):
        start_epoch_time = time.time()
        batch_losses = []
        print(
            "Epoch {}/{} Step {}/{} Loss: {:.5f}".format(
                epoch + 1, n_epochs, 1, len(train_ds), 0
            ),
            end="",
        )
        for step, batch in enumerate(train_ds):
            (input_images, events_tensors), ground_truth_images = batch
            # events_tensors: (batch, bins, height, width)
            input_images = torch.einsum("bhwc -> bchw", input_images)

            input_tensors = torch.hstack((input_images, events_tensors))
            input_tensors = input_tensors.to(device)

            generated_images = model(input_tensors)

            ground_truth_images = torch.einsum("bhwc -> bchw", ground_truth_images).to(
                device
            )
            loss: torch.Tensor = criterion(generated_images, ground_truth_images)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_losses.append(loss.cpu().detach())

            print(
            "\rEpoch {}/{} Step {}/{} Mean Loss: {:.5f} Elapsed Seconds: {}s".format(
                epoch + 1,
                n_epochs,
                step + 1,
                len(train_ds),
                np.mean(batch_losses),
                int(elapsed_time),
            )
        )
        elapsed_time = time.time() - start_epoch_time
        print(
            "Epoch {} Step {}/{} Mean Loss: {:.3f} Elapsed Seconds: {}s".format(
                0, 0, len(train_ds), np.mean(batch_losses), int(elapsed_time)
            )
        )


def train_transformer(
    model: torch.nn.Module,
    device: str,
    train_ds: torch.utils.data.DataLoader,
    params: Dict,
):

    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    model.train()
    for epoch in range(params["n_epochs"]):
        start_epoch_time = time.time()
        batch_losses = []
        print(
            "Epoch {}/{} Step {}/{} Loss: {:.5f}".format(
                epoch + 1, params["n_epochs"], 1, len(train_ds), 0
            ),
            end="",
        )
        for step, batch in enumerate(train_ds):
            (input_images, events_tensors), ground_truth_images = batch
            # events_tensors: (batch, bins, height, width)
            events_tensors = events_tensors.to(device)

            generated_images = model(events_tensors)

            ground_truth_images = torch.einsum("bhwc -> bchw", ground_truth_images).to(
                device
            )
            loss: torch.Tensor = criterion(generated_images, ground_truth_images)
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
        elapsed_time = time.time() - start_epoch_time
        print(
            "\rEpoch {}/{} Step {}/{} Mean Loss: {:.5f} Elapsed Seconds: {}s".format(
                epoch + 1,
                params["n_epochs"],
                step + 1,
                len(train_ds),
                np.mean(batch_losses),
                int(elapsed_time),
            )
        )
