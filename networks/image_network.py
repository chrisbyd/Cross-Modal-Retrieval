import torchvision.models as models
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

class ImageNet(nn.Module):
    def __init__(self, hash_length):
        super(ImageNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, hash_length)
        self.tanh=torch.nn.Tanh()

    def forward(self, x):
        resnet_feature=self.resnet(x)
        image_feature=self.tanh(resnet_feature)
        return image_feature

