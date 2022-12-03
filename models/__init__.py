from typing import Dict, Any
import importlib
import os

from utils import is_using_colab


def get_model(model_params: Dict[str, Any]):
    # TODO: implement older models

    model_name = model_params["class_name"]
    module_name = ""

    if "Teacher" in model_name or "Student" in model_name:
        module_name = "models.teacher_student"
    elif "Transformer" in model_name:
        module_name = "models.transformer"
    elif "Test" in model_name:
        module_name = "models.tests"
    elif "ConViT" in model_name:
        module_name = "models.convit"
    elif model_name == "Events2Image":
        module_name = "models.2dretr"
    else:
        raise Exception("Model " + model_name + " not supported.")

    module = importlib.import_module(module_name)

    if model_name.startswith("Student"):

        # Load Teacher
        teacher_name = model_params["teacher"]
        teacher_path = model_params["teacher_path"]

        from models.teacher_student import Teacher, TeacherTanh

        if teacher_name == "Teacher":
            if is_using_colab():
                os.system("gdown 1Z1R7C1G28mQQZZYyQvRZxJXRoyLG-fT0")
            teacher = Teacher.load_from_checkpoint(teacher_path, lr=1e-3)
        elif teacher_name == "TeacherTanh":
            if is_using_colab():
                os.system("gdown 1hXs6J68UZ1aWgkY-taRyNe4hlR3XiyX7")
            teacher = TeacherTanh.load_from_checkpoint(teacher_path, lr=1e-3)

        model_params["MODEL_PARAMS"]["teacher"] = teacher

    ModelClass = getattr(module, model_name)
    model = ModelClass(**model_params["MODEL_PARAMS"])

    if model_name.startswith("Student"):
        # Remove the teacher instance in order to save the parameters in a JSON file
        del model_params["MODEL_PARAMS"]["teacher"]

    return model
