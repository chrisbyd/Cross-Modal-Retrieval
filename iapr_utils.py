import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as torchdata
import os
import re


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(torchdata.Dataset):
    def __init__(self, args, data_list_path ,transform=None, loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.data_list_path = data_list_path
        self.data_root =  args.data_root
        self.raw_to_new_label = self.construct_label_dic()
        self.data = self.get_data()
    
    def get_data(self):
        data = []
        with open(self.data_list_path, 'r') as f:
            data_lines = f.readlines()
            for data_line in data_lines:
                img_path = os.path.join(self.data_root,data_line.split(";")[0])
                new_label = [self.raw_to_new_label[int(label)] for label in data_line.split(";")[1].split()]
                binary_label = self.new_label_to_binary(new_label)
                data.append((img_path, binary_label, data_line.split(";")[2]))
        return data
    
    def construct_label_dic(self):
        all_label_path = os.path.join(self.data_root,'imid_anno_label.txt')
        label_set = set()
        with open(all_label_path, 'r') as f:
            data_lines = f.readlines()
            for data_line in data_lines:
                labels = data_line.split(";")[2].split()
                for label in labels:
                    label_set.add(int(label))
        label_to_new = {}
        for index, label in enumerate(label_set):
            label_to_new[label] = index
        return label_to_new
        
    def new_label_to_binary(self, new_label_list):
        a = np.zeros(255,np.float32)
        a[new_label_list] = 1
        return a      


    def __getitem__(self, index):
     
        img_path, label, text = self.data[index]
        img_path = img_path.strip()
        # img
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        # text
        text = '[CLS]' + " "  + text + " "+ '[SEP]'
        return img, text, label
        

    def __len__(self):
        return len(self.data)
    


def IAPR_dataloader(args):

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


    root = args.data_root
    train_file = os.path.join(root, 'train.txt')
    test_file = os.path.join(root, 'test.txt')
    retrieval_file = os.path.join(root, 'database.txt')

    train_set = MyDataset(args, data_list_path=train_file, transform=transform_train)
    train_loader = torchdata.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_set = MyDataset(args,data_list_path=test_file, transform=transform_test)
    test_loader = torchdata.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    db_set = MyDataset(args,data_list_path=retrieval_file, transform=transform_test)
    db_loader = torchdata.DataLoader(db_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, db_loader



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