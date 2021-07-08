import torchvision.models as models
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig




class TextNet(nn.Module):
    def __init__(self,  hash_length):
        super(TextNet, self).__init__()

        modelConfig = BertConfig.from_pretrained('./pretrained_dir/bert_pretrain/bert-base-uncased-config.json')
        self.textExtractor = BertModel.from_pretrained('./pretrained_dir/bert_pretrain/bert-base-uncased-pytorch_model.bin', config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size

        self.fc = nn.Linear(embedding_dim, hash_length)
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output=self.textExtractor(tokens, token_type_ids=segments, attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]  #output[0](batch size, sequence length, model hidden dimension)



        hash_features = self.fc(text_embeddings)
        hash_features=self.tanh(hash_features)
        return hash_features