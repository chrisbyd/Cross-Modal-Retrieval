import os
import copy
import pytorch_lightning as pl
from pytorch_lightning import trainer
from configs.config import ex
from  model import CrossRetrievalModel
from datasets import HashDataModule

@ex.automain
def main(_config):
    config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    logger = pl.loggers.TensorBoardLogger(
        config["log_dir"],
        name=f'{config["exp_name"]}_seed{config["seed"]}',
    )
    trainer = pl.Trainer(
        gpus = config['num_gpus'],
        precision = config['precision'],
        max_epochs = config['max_epochs'],
        logger = logger,
        log_every_n_steps= config['log_interval'],
    )
    dm = HashDataModule(config= config)
    model = CrossRetrievalModel(config= config)
    pretrained_file = {}
    pretrained_file['vision'] = './pretrained_dir/ViT-B_32.npz'
    pretrained_file['text'] = './pretrained_dir/bert_pretrain/bert_model.ckpt'
    model.load_model(None,pretrained_file)

    if not config["test_only"]:
        trainer.fit(model, datamodule= dm)
    else:
        trainer.validate(model, datamodule= dm)
