import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WordEmb:
    """
    Word embeddings
    """

    # Not an nn.Module so that it will not be saved and trained
    def __init__(self, emb_path):
        w2v_value = np.load(emb_path)
        self.embed = nn.Embedding(*w2v_value.shape)
        self.embed.weight.data = torch.FloatTensor(w2v_value)

    def calc_emb(self, x):
        return self.embed(x)


class Model(nn.Module):
    def __init__(self, emb_path="./raw_data/rt_polarity/saved_emb.npy"):
        super(Model, self).__init__()

        self.embed_static = WordEmb(emb_path)
        self.conv1_3 = nn.Conv2d(1, 100, (3, 300))
        self.conv1_4 = nn.Conv2d(1, 100, (4, 300))
        self.conv1_5 = nn.Conv2d(1, 100, (5, 300))
        self.output = nn.Linear(3 * 100, 1)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed_static.calc_emb(x).unsqueeze(1)
        score = self.emb_forward(x)
        return score

    def emb_forward(self, x):
        x_3 = self.conv_and_pool(x, self.conv1_3)
        x_4 = self.conv_and_pool(x, self.conv1_4)
        x_5 = self.conv_and_pool(x, self.conv1_5)
        x = torch.cat((x_3, x_4, x_5), dim=1)
        x = F.dropout(x, 0.5, training=self.training)
        score = self.output(x).squeeze(1)
        return score

    def loss(self, pred, label):
        return F.binary_cross_entropy_with_logits(pred, label.float())

    def emb_info(self):
        emb_matrix = self.embed_static.embed.weight.data
        emb_mean = emb_matrix.mean(0)
        emb_std = emb_matrix.std(0, unbiased=True)
        return emb_mean, emb_std
