# Meta Neural Trojan Detection

This repo provides an implementation of detecting Trojans in machine learning models as introduced [here](https://arxiv.org/abs/1910.03137).

## Installation

The code successfully runs on Python 3.6 and PyTorch 1.6.0. The PyTorch package need to be manually installed as shown [here](https://pytorch.org/) for different platforms and CUDA drivers. Other required packages can be installed by:
```bash
pip install -r requirements.txt
```

The MNIST and CIFAR-10 datasets will be downloaded at running time. To run the audio task, one need to download the [SpeechCommand v0.02 dataset](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz), extract it under `raw_data/speech_command` folder and run `python audio_preprocess.py`. To run the NLP task, one need to download the [pretrained GoogleNews word embedding](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit), extract it under `raw_data/rt_polarity` folder and run `python rtNLP_preprocess.py`; the movie review data is already in the folder. The Irish Smart Meter Electricity data is private and we do not include them here.

## Training Shadow Models and Target Models

The original repository used three parts for training the shadow (training) models and the target (evaluation) models: training the benign models (`train_basic_benign.py`), training the shadow models with jumbo learning (`train_basic_jumbo.py`) and training the target models with certain Trojans (`train_basic_trojaned.py`).

This repository merges them into one file under a single executable that can be called as follows:

```bash
python train_models.py --task cifar10 --model ResNet-18 basic
python train_models.py --task cifar10 --model CNN-5 student_poison \
    --teacher ResNet-18 \
    --teacher_weights "./shadow_model_ckpt/cifar10/models/target_basic_resnet-18_0.model" \
    --attack_type patch \
    --inject_p 0.5
```

## Training and Evaluating the Meta-Classifier

`run_meta.py` trains and evaluates the meta-classifier using jumbo learning and `run_meta_oneclass.py` trains and evaluates the meta-classifier using one-class learning. An example of training the meta-classifier with jumbo learning on the MNIST task and evaluating on modification attack:

```bash
python run_meta.py --task mnist --troj_type M
```

## Tensorboard

Can upload experiments using [TensorBoard.dev](https://tensorboard.dev/experiments/).
