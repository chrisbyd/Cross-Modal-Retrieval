import torchvision.models as models
import torch
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig


class ImageNet(nn.Module):
    def __init__(self, hash_length, num_classes):
        super(ImageNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        #self.fc = nn.Linear(1000,512)
        self.classifier = nn.Linear(hash_length,num_classes)
        self.hash = nn.Linear(1000,hash_length)
        self.tanh=torch.nn.Tanh() 

    def forward(self, x):
        raw_feature =self.resnet(x)
        
        hash_feature = self.hash(raw_feature)
        hash_feature_tanh=self.tanh(hash_feature)
        if self.training:
            class_feature = self.classifier(hash_feature)
            return class_feature, hash_feature_tanh
        return hash_feature_tanh


class TextNet(nn.Module):
    def __init__(self,  code_length, num_classes):
        super(TextNet, self).__init__()

        modelConfig = BertConfig.from_pretrained('./bert_pretrain/bert-base-uncased-config.json')
        self.textExtractor = BertModel.from_pretrained('./bert_pretrain/bert-base-uncased-pytorch_model.bin', config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size
        self.classifier = nn.Linear(code_length, num_classes)
        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output=self.textExtractor(tokens, token_type_ids=segments, attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]  #output[0](batch size, sequence length, model hidden dimension)
         
        hash_features = self.fc(text_embeddings)
        hash_features_tanh=self.tanh(hash_features)
        if self.training:
            class_features = self.classifier(hash_features)
            return class_features, hash_features_tanh
        return hash_features_tanh