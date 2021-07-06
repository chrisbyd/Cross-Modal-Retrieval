from sacred import Experiment
ex =  Experiment("UniFormerHash")


@ex.config
def config():
    exp_name = 'uniformerHash'
    seed = 0
    dataset_name = 'iaprtc12'
    batch_size = 32
    max_epochs = 300
    hash_length = 32

    #optimizer config
    lr =  1e-5
    image_lr = 1e-4
    text_lr = 1e-5
    momentum = 0.9
    weight_decay = 0.0005
    margin = 12

    #gpu training
    num_gpus = 1
    precision = 16

    #log config
    log_dir = './results/'
    log_interval = 10
    val_check_interval =  1000

    #from checkpoint
    load_path = ''

    #directories
    data_root = 'data/iaprtc12'

    test_only = False
