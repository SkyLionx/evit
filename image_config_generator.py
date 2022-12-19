import subprocess
from threading import Thread
import time
import os
from random import randint

# Setup constants
ENV_COMMAND = "source /home/fabrizio/setupeventsim_color.sh && "
CONFIG_PATH = "/home/fabrizio/sim_ws_color/src/rpg_esim/event_camera_simulator/esim_ros/cfg/example_color.conf"
COCO_DATASET_PATH = "/media/sf_Shared_Folder/val2017"
DIV2K_TRAIN_DATASET_PATH = "/media/sf_Shared_Folder/DIV2K_train_HR"
DIV2K_VALID_DATASET_PATH = "/media/sf_Shared_Folder/DIV2K_valid_HR"

def parse_args_from_file(config_path):
    with open(config_path) as file:
        args = {}
        for line in file:
            line = line.strip()
            
            # If not empty line or comment
            if len(line) == 0 or line.startswith("#"):
                continue
            
            equals = line.index("=")
            name = line[:equals]
            value = line[equals + 1:]
            args[name] = value
        return args
        
def args_dict_to_string(args):
    string = ""
    for key, value in args.items():
        string += key + "=" + value
        string += " "
    return string

def launch_process(command, stdout=False, stderr=True):
    stdout = None if stdout else subprocess.DEVNULL
    stderr = None if stderr else subprocess.DEVNULL
    proc = subprocess.Popen(ENV_COMMAND + command, shell=True, executable="/bin/bash", stdout=stdout, stderr=stderr)
    return proc

def launch_ros_core():
    core = launch_process("roscore")
    return core

def wait_for_master():
    master_up = False
    while not master_up:
        try:
            launch_process("rostopic list", stderr=False)
            master_up = True
        except Exception as e:
            print("Master not ready")
            print(e)
            time.sleep(.5)
    
def launch_experiment(image_path, file_path):
    args = parse_args_from_file(CONFIG_PATH)
    args["--renderer_texture"] = image_path
    args["--simulate_color_events"] = "false"

    ext = file_path.split(".")[-1]
    if ext == "bag":
        args["--path_to_output_bag"] = file_path
    elif ext == "bin":
        args["--path_to_output_bin"] = file_path
    else:
        print("File extension {} not supported.".format(ext))

    args["--random_seed"] = str(randint(1, int(1e5)))
    # args["--renderer_preprocess_gaussian_blur"] = "0.1"
    # args["--renderer_preprocess_median_blur"] = "0.1"
    command = "rosrun esim_ros esim_node " + args_dict_to_string(args)
    print(args_dict_to_string(args))
    return launch_process(command)

def get_file_ext(path):
    return path.split(".")[-1]
    
def get_images_path(dataset_folder, n=1):
    IMG_EXTS = set(["jpg", "png"])
    
    paths = []
    i = 0
    for image in os.listdir(dataset_folder):
        if get_file_ext(image) in IMG_EXTS:
            paths.append(os.path.join(dataset_folder, image))
        i += 1
        if n and i == n:
            break
    return paths            


if __name__ == "__main__":
    #images_paths = [
        #"/home/fabrizio/sim_ws_color/src/rpg_esim/event_camera_simulator/imp/imp_planar_renderer/textures/rocks.jpg",
        #"/home/fabrizio/sim_ws_color/src/rpg_esim/event_camera_simulator/imp/imp_planar_renderer/textures/forest.jpg"
    #]
    
    #images_paths = [
        #"/media/sf_Shared_Folder/DIV2K_train_HR/0010.png",
        #"/media/sf_Shared_Folder/DIV2K_train_HR/0011.png",
        #"/media/sf_Shared_Folder/DIV2K_train_HR/0029.png",
        #"/media/sf_Shared_Folder/DIV2K_train_HR/0030.png",
        #"/media/sf_Shared_Folder/DIV2K_train_HR/0038.png",
        #"/media/sf_Shared_Folder/DIV2K_train_HR/0059.png",
        #"/media/sf_Shared_Folder/DIV2K_train_HR/0062.png",
        #"/media/sf_Shared_Folder/DIV2K_train_HR/0072.png",
        #"/media/sf_Shared_Folder/DIV2K_train_HR/0082.png",
        #"/media/sf_Shared_Folder/DIV2K_train_HR/0088.png",
    #]

    images_paths = [
        # "/media/sf_Shared_Folder/test_image_black.png",
        # "/media/sf_Shared_Folder/DIV2K_train_HR/0001.png",
        # "/media/sf_Shared_Folder/DIV2K_train_HR/0002.png",
        # "/media/sf_Shared_Folder/DIV2K_train_HR/0003.png",
        # "/media/sf_Shared_Folder/blue_circle.png",
    ]

    OUTPUT_DIR = "/media/sf_Shared_Folder/bags/DIV2K_0.5_bw"
    
    images_paths = get_images_path(DIV2K_VALID_DATASET_PATH, n=None)
    # images_paths = get_images_path(COCO_DATASET_PATH, n=None)
    
    print("Launching ROSCore...")
    core = launch_ros_core()

    wait_for_master()
    print("Master ready")
    
    # Create output foulder if needed
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    for image_path in images_paths:
        ext = get_file_ext(image_path)
        file_name = os.path.basename(image_path).replace(ext, "bag")
        file_path = os.path.join(OUTPUT_DIR, file_name)

        print("Launching simulation ({})...".format(file_name), end=" ", flush=True)
        launch_experiment(image_path, file_path).wait()
        print("Done")

    core.kill()

