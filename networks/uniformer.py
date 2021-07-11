import math
import json
from typing import NamedTuple
import numpy as np
import torch
from scipy import ndimage
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from .text_transformer import Transformer as TextFormer
from .New_vision_transformer import Transformer as VisionFormer
import torch.nn as nn
from .New_vision_transformer import Block as VBlock
import utils.checkpoint as checkpoint
from utils import np2th
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

class Uniformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        modelConfig = BertConfig.from_pretrained('./pretrained_dir/bert_pretrain/bert-base-uncased-config.json')
        self.textformer = BertModel.from_pretrained(
            './pretrained_dir/bert_pretrain/bert-base-uncased-pytorch_model.bin', config=modelConfig)
       # self.textformer = TextFormer(config)
        self.visionformer = VisionFormer(config,config["crop_size"], False)
        self.c_blocks = nn.ModuleList([VBlock(config, False)  for _ in range(self.config['n_common_layers'])])
        self.text_hash_layer = Linear(self.config["dim"], self.config["hash_length"])
        self.image_hash_layer = Linear(self.config["dim"], self.config["hash_length"])
        self.tanh = nn.Tanh()
        self.initialize_common_params()

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)

        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def initialize_common_params(self):
        for block in self.c_blocks:
            block.apply(self.weights_init_kaiming)



    def forward_text(self, x, segs, mask):
        h, _ = self.textformer(x, segs, mask)
        for block in self.c_blocks:
            h,_ = block(h)
        h = self.text_hash_layer(h[:, 0])
        text_feature = self.tanh(h)
        return text_feature


    def forward_image(self, x):
        h,_ = self.visionformer(x)
        for block in self.c_blocks:
            h,_ = block(h)
        h = self.image_hash_layer(h[:,0])
        image_feature = self.tanh(h)
        return  image_feature

    def load_vision_from(self,pretrained_file):
        weights = np.load(pretrained_file)
        with torch.no_grad():
            self.visionformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.visionformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.visionformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.visionformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.visionformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.visionformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.visionformer.embeddings.position_embeddings.copy_(posemb)
            else:
                raise NotImplementedError("the number of patches do not support")

            for bname, block in self.visionformer.encoder.named_children():
                print(f"The bname is {bname}, ")

                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

    def load_model(self, pretrained_file = None):
        text_weight_path = pretrained_file['text']
        img_weight_path  = pretrained_file['vision']
        print("Starting loading pretrained text transformer ")
        # if text_weight_path.endswith('.ckpt'):  # checkpoint file in tensorflow
        #     print("Start loading ckpt")
        #     checkpoint.load_model(self.textformer, text_weight_path)

        print("Starting loading pretrained image transformer")
        self.load_vision_from(img_weight_path)
        weights = np.load(img_weight_path)
        #for uname in range(self.config['num_v_layers'], 12):
        n_block = int(self.config['num_v_layers'])
        for uname, unit in self.c_blocks.named_children():
            unit.load_from(weights, n_block= n_block)
            n_block += 1


















    
    