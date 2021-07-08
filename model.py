import  pytorch_lightning as pl
import torch.optim as optim
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from  networks import TextNet
from  networks import ImageNet
from  utils.utils import CrossModel_triplet_loss, Variable, get_tokens, compute_result_CrossModel, compute_mAP_MultiLabels
from networks import VisionTransformerHash
from networks import TextTransformerHash
import torch

class CrossRetrievalModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_net = TextTransformerHash(self.config)
        self.image_net = VisionTransformerHash(config= self.config)
        self.tokenizer = BertTokenizer.from_pretrained('./pretrained_dir/bert_pretrain/bert-base-uncased-vocab.txt')
        self.previous_epoch = -1


    def configure_optimizers(self):
        optimier = optim.Adam(list(self.image_net.parameters())+list(self.text_net.parameters()),
                              lr = self.config['lr'], weight_decay=self.config['weight_decay'])
        return optimier

    def forward_image(self, image_input):
        image_feature = self.image_net(image_input)
        return  image_feature

    def forward_text(self, tokens, segments, input_masks):
        text_feature = self.text_net(tokens, segments, input_masks)
        return text_feature

    def forward_all(self, image_input, text_input):
        images = Variable(image_input)
        image_feature = self.forward_image(images)
        # text
        tokens, segments, input_masks = get_tokens(text_input, self.tokenizer)
        text_feature = self.forward_text(tokens, segments, input_masks)

        return image_feature, text_feature

    def training_step(self, batch, batch_index):
        images, texts, labels = batch
        image_hash_feature, text_hash_feature = self.forward_all(images, texts)
        image_triplet_loss, text_triplet_loss, \
        image_text_triplet_loss, text_image_triplet_loss, \
        len_triplets = CrossModel_triplet_loss(image_hash_feature, text_hash_feature, labels, self.config['margin'])

        loss = image_triplet_loss + text_triplet_loss + image_text_triplet_loss + text_image_triplet_loss
        with torch.no_grad():
            if self.current_epoch % self.config['eval_interval'] == 0 and self.previous_epoch != self.current_epoch:
                self.validation()
                self.previous_epoch = self.current_epoch
        self.log('Training bi-directional triplet loss', loss)
        return  loss

    def validation(self):
        query_loader = self.trainer.datamodule.query_loader()
        gallery_loader = self.trainer.datamodule.gallery_loader()
        print("Start computing the hash codes for images and texts")
        tst_image_binary, tst_text_binary, tst_label, tst_time = compute_result_CrossModel(query_loader, self.image_net,
                                                                                           self.text_net, self.tokenizer)
        db_image_binary, db_text_binary, db_label, db_time = compute_result_CrossModel(gallery_loader, self.image_net, self.text_net,
                                                                                       self.tokenizer)
        # print('test_codes_time = %.6f, db_codes_time = %.6f'%(tst_time ,db_time))
        print("Start computing mAP")
        it_mAP = compute_mAP_MultiLabels(db_text_binary, tst_image_binary, db_label, tst_label)
        ti_mAP = compute_mAP_MultiLabels(db_image_binary, tst_text_binary, db_label, tst_label)
        print(f" the i-to-I mAP is {it_mAP}, the T-to-i mAP is {ti_mAP}")
        self.log("retrieval image to text mAP" , it_mAP)
        self.log("retrieval text to image mAP" , ti_mAP)

    def load_model(self, checkpoint_file, pretrained_file):
        if checkpoint_file is not None:
            self.load_state_dict()
            pass
        elif pretrained_file is not None:
            self.text_net.load_from(pretrained_file['text'])
            self.image_net.load_from(pretrained_file['vision'])
        else:
            raise NotImplementedError("U need either supply a checkpoint or a pretrained file")











