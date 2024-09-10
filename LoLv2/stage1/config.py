from easydict import EasyDict as edict


class Configs:
    model_version = 'lolv2_acca'

    img_path = "/dataset/kunzhou/project/low_light_noisy/lol_dataset/LOL-v2/Real_captured/Train/"
    img_val_path = "/dataset/kunzhou/project/low_light_noisy/lol_dataset/LOL-v2/Real_captured/Test/"

    normalize = True 
    batch_size = 1

    val_batch_size = 1

    num_epochs = 1500 

    lr = 5e-4
    weight_decay = 0.0005

config = Configs()
