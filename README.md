# EViT - Learning to Reconstruct Color Images from Event-based Neuromorphic Cameras using a Transformer Architecture

> Thesis project for the Computer Science Master Degree at Sapienza University A.Y. 2021/2022
>
> The thesis complete with and in-depth description of the model is available in the file [`thesis.pdf`](https://github.com/SkyLionx/evit/blob/master/thesis.py)

In this project a **CNN-Transformer based** architecture called **EViT** is used to transform a **color event stream** produced by Neuromophic Cameras into **RGB images**.
In order to achieve this task, a **synthetic dataset** has been generated using the [ESIM simulator](https://github.com/uzh-rpg/rpg_esim).

The repo also includes the [rpg_e2vid repo](https://github.com/uzh-rpg/rpg_e2vid) which has been used in order to compare the model with E2VID.

## Contributors

<a href="https://github.com/SkyLionx" target="_blank">
  <img src="https://img.shields.io/badge/Profile-Fabrizio%20Rossi-green?style=for-the-badge&logo=github&labelColor=blue&color=white">
</a>

## Architecture
The **architecture** of the model is composed of three main modules and is summarized in the following picture:
<div align="center">
    <img src="https://user-images.githubusercontent.com/23276420/228008022-3d20094b-e0d2-4f12-91a6-3538a4056d45.png"/>
</div>

The **Convolutional Encoder** is used to encode temporal and spatial information extracted locally from the input events. The **feature maps** obtained are divided into patches of fixed size in a similar fashion to the **ViT model**, and they are unrolled across the filters dimension. 
The patches are then flattened and summed with positional encoding vectors in order to form a sequence that is given as input to the **Transformer Encoder**, which can correlate global dependencies among the local features.
After recomposing back the feature maps, a **Convolutional Decoder** is finally employed in order to transform the enriched global features into an RGB image depicting the scene captured by the events processed.

## Results

The model has been evaluated on the **synthetic test set** and on the [**CED dataset**](https://rpg.ifi.uzh.ch/CED.html) which contains data coming from the real DAVIS346 event camera.

### Synthetic Test Set
<div align="center">
    <img width="750" src="https://user-images.githubusercontent.com/23276420/228009736-d3278289-1024-49f5-8cf7-9d5608955f99.png"/>
</div>

As can be seen, on the synthetic test set, the model is able to reconstruct the majority of the edges with good fidelity, hence the 60% SSIM, while colors tend to not match perfectly the ground truth image, respecting however the brightness changes. Moreover, the model is also capable of allucinating some parts of the images where there aren’t too many events available, although details are not very defined.

### CED Dataset
<div align="center">
    <img width="750" src="https://user-images.githubusercontent.com/23276420/228009781-c575a6ba-2284-4386-9e47-e9fb3074d6ed.png"/>
</div>

On the CED dataset instead, it can be seen that the model proposed is again able to identify and reconstruct edges but it has problems to reconstruct colors.
In particular, in the last row is reported an interesting scenario that shows how the two models react when no events are available for parts of the image, since in this case the camera is fixed and there are only people dancing. By looking at the results, it seems that the model proposed can’t handle very well the parts in which events are too sparse, probably due to the fact that the sequences contained in the synthetic dataset on which it has been trained are much denser of events.

## Folder structure

In this section is described a brief overview on how the project is organized in folders:

- **/models**: contains the code for all the models
    - **autoencoder.py**: autoencoder to encode events grids
    - **cnn.py**: basic CNN encoder-decoder architecture to transform event grids into RGB images
    - **convit.py**: [ConViT model](https://github.com/facebookresearch/convit) which has been tested to perform color image reconstruction. It is a transformer architecture that initializes attention weights in order to perform convolutions and then it can learn to improve them.
    - **modules.py**: it contains some useful models like PatchExtrator or PositionalEncoding
    - **teacher_student.py**: numerous Teacher-Student architectures
    - **tests.py**: simple tests of architectures to check if they work
    - **transformer.py**: different Transformers architecture. The best performing one and used in the thesis is VisionTransformerConv.
    - **unet.py**: initial architecture to perform the task taking in input images and events using a UNet architecture
- **/dataset_utils.py**: many utils to load and preprocess datasets, inspect events and save visualizations
- **/eval.py**: code to produce the qualitative results shown in the thesis
- **/exploration.ipynb**: CED dataset analysis, some tests and generation of images for different events representations
- **image_config_generator.py**: script used to generate the synthetic dataset using the ESIM simulator
- **/media_utils.py**: utils to process images and produce videos
- **/my_utils.py**: PyTorch Lightning callbacks and misc functions
- **/old_train.py**: training methods which have been used before switching to the PyTorch Lightning framework
- **/tensorboard_utils.py**: images and metrics utilities for TensorBoard, used also to generate reports
- **/train.py**: training procedure with PyTorch Lightning framework and callbacks
- **/training.ipynb**: main notebook used to launch model trainings

## How to train a model

In the [training.ipynb](training.ipynb) file, it is possible to configure an experiment in order to load a dataset and train a model. In order to do that, it is necessary to run the first 4 cells.
In the second cell, the parameters of the experiment can be set up using a dictionary structured in the following way:
```python
PARAMS = {
    # Choose which device to use for the training
    "DEVICE": "gpu" | "cpu",
    # Relative or absolute path where the lightning_logs folder will be stored
    "EXPERIMENTS_DIR" : "/path/to/experiments",
    # Base bath where each dataset will be found
    "BASE_DATASETS_PATH" : "/path/to/datasets",
    # Name of the dataset to load
    # Additional datasets can be added by changing the get_dataset function in dataset_utils.py
    "DATASET_NAME": "DIV2K_5_FIX" | "DIV2K_5_FIX_SMALL" | "DIV2K_5_BW_FIX",

    "DATASET_PARAMS" : {
        # This dict is directly passed to the dataset instatiation so any positional argument can be used, for example:
        "limit": None,
        "preload_to_RAM": True,
        "crop_size": (128, 128),
        "events_normalization": "z_score_non_zero",
        "convert_to_bw": False
    },

    "DATALOADER_PARAMS" : {
        # This dict is directly passed to the PyTorch dataloader instantiations so any positional argument can be used, for example:
        "batch_size": 16,
        "num_workers": 0,
        "pin_memory": True,
    },

    "MODEL": {
        # Name of the class which will be instantiated. It must be supported by the get_model function defined in models/__init__.py
        "class_name": "VisionTransformerConv",

        # If training a student-teacher model, the following can also be defined to instantiate the Teacher:
        # "teacher": "TeacherTest",
        # "teacher_path": "teachertest-epoch=40-step=10250.ckpt",
        "MODEL_PARAMS" : {
            # This dict is directly passed to the model constructor so any positional argument can be used, for example:
            "input_shape": (10, 128, 128),
            "patch_size": (32, 32),
            "heads": 4,
            "layers_number": 1,
            "learning_rate": 1e-4,
            "image_loss_weight": 1,
            "feature_loss_weight": 1e-2,
            "color_output": True,
        },
    },

    "TRAINING_PARAMS": {
        "n_epochs": 500,
        # This is going to be the name of the run reported on TensorBoard (and the name of the experiment folder)
        "comment": "Experiment description"
    }
}
```

## Dataset notes
In order to generate the synthetic dataset, the [_feature/color_](https://github.com/uzh-rpg/rpg_esim/tree/feature/color) branch of the ESIM simulator was used, enabling the generation of color events. I was not able to make it run under Windows, so I used a VM running Lubuntu.
To generate the dataset, DIV2K images were used by performing automatic simulations using the [image_config_generator.py](image_config_generator.py) script.
The configuration used is the default one, but some parameters are overwritten in the script.

## Technologies
For this project, the following technologies have been used:
- PyTorch and PyTorch Lightning to build and train the models
- TensorBoard for experiments logging and comparison
- Matplotlib for plotting
- OpenCV for demosaicing algorithms and video writing
