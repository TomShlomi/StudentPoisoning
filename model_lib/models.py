from enum import Enum
import torch.nn as nn

from utils_basic import DatasetConfig


model_types = ["CNN-2", "CNN-5", "ResNet-18", "ResNet-50", "ViT", "AudioRNN", "rt-cnn"]
ModelType = Enum("ModelType", model_types)


def load_model_setting(model_type: ModelType, config: DatasetConfig) -> nn.Module:
    if model_type in ("CNN-2", "CNN-5"):
        if model_type == "CNN-2":
            from model_lib.vision_models import SimpleCNN as Model
        elif model_type == "CNN-5":
            from model_lib.vision_models import MediumCNN as Model

        c_in, h_in, w_in = config.input_size
        return Model(c_in=c_in, h_in=h_in, w_in=w_in, num_classes=config.num_classes)

    elif model_type in ("ResNet-18", "ResNet-50"):
        if model_type == "ResNet-18":
            from model_lib.vision_models import ResNet18 as Model
        elif model_type == "ResNet-50":
            from model_lib.vision_models import ResNet50 as Model

        return Model(num_classes=config.num_classes)

    elif model_type == "AudioRNN":
        from model_lib.audio_rnn_model import Model

    elif model_type == "rt-cnn":
        from model_lib.rtNLP_cnn_model import Model

    else:
        raise NotImplementedError("Unknown model type %s" % model_type)
