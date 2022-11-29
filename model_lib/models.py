from enum import Enum


ModelType = Enum(
    "ModelType",
    ["CNN-2", "CNN-5", "ResNet-18", "ResNet-50", "ViT", "AudioRNN", "rt-cnn"],
)


def load_model_setting(model_type: ModelType):
    if model_type == "CNN-2":
        from model_lib.vision_models import SimpleCNN as Model
    elif model_type == "CNN-5":
        from model_lib.vision_models import MediumCNN as Model
    elif model_type == "ResNet-18":
        from torchvision.models import resnet18 as Model
    elif model_type == "ResNet-50":
        from torchvision.models import resnet50 as Model
    elif model_type == "AudioRNN":
        from model_lib.audio_rnn_model import Model
    elif model_type == "rt-cnn":
        from model_lib.rtNLP_cnn_model import Model
    else:
        raise NotImplementedError("Unknown model type %s" % model_type)

    return Model
