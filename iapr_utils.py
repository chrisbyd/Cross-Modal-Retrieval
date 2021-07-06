import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as torchdata
import os
import re


        






# root = '/home/disk1/zhaoyuying/dataset/iapr-tc12_255labels'
# train_file = os.path.join(root, 'iapr_train')
# test_file = os.path.join(root, 'iapr_test')
# retrieval_file = os.path.join(root, 'iapr_retrieval')
#
# mean = (0.4914, 0.4822, 0.4465)
# std = (0.2023, 0.1994, 0.2010)
# transform_train = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=mean, std=std)
# ])
# train_set = MyDataset(txt=train_file, transform=transform_train)
# train_loader = torchdata.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
#
# iii = 0
# for batch_idx, (images, texts, labels) in enumerate(train_loader):
#     # print('batch_idx = ', batch_idx)
#     iii += 1


# # words =  ['32', '32289']
# # words =  ['30', '30583']
# # words =  ['22', '22116']
# # words =  ['12', '12660']
# words =  ['03', '3580']

# text_path = os.path.join('../dataset/iapr-tc12_255labels/annotations', words[0], words[1]+'.txt')

# tokenizer = BertTokenizer.from_pretrained('../models/tokenization_bert/bert-base-uncased-vocab.txt')

# # text
# tokenized_text = None
# for line in open(text_path):
#     text = "[CLS]" + line + "[SEP]"
#     tokenized_text = tokenizer.tokenize(text)

# print('len(tokenized_text) = ', len(tokenized_text))