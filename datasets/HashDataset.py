import numpy as np
from PIL import Image
import torch.utils.data as torchdata
import os




def default_loader(path):
    return Image.open(path).convert('RGB')


class HashDataset(torchdata.Dataset):
    def __init__(self, args, data_list_paths, transform=None, loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.data_list_paths = data_list_paths
        self.data_root = args["data_root"]
        self.raw_to_new_label = self.construct_label_dic()
        self.data = self.get_data()

    def get_data(self):
        data = []
        for path in self.data_list_paths:
            with open(path, 'r') as f:
                data_lines = f.readlines()
                for data_line in data_lines:
                    img_path = os.path.join(self.data_root, data_line.split(";")[0])
                    new_label = [self.raw_to_new_label[int(label)] for label in data_line.split(";")[1].split()]
                    binary_label = self.new_label_to_binary(new_label)
                    data.append((img_path, binary_label, data_line.split(";")[2]))
        return data

    def construct_label_dic(self):
        all_label_path = os.path.join(self.data_root, 'imid_anno_label.txt')
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
        a = np.zeros(255, np.float32)
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
        text = '[CLS]' + ' ' + text + " " + '[SEP]'

        return img, text, label

    def __len__(self):
        return len(self.data)