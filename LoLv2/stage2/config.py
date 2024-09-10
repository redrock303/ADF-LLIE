from easydict import EasyDict as edict


class Configs:
    model_version = 'unet_lap'
    # dataset
    DATASET = edict()
    DATASET.DATASETS = []
    DATASET.SCALE = 4
    DATASET.PATCH_WIDTH = 64
    DATASET.PATCH_HEIGHT = 64
    DATASET.REPEAT = 1
    DATASET.AUG_MODE = True
    DATASET.VALUE_RANGE = 255.0
    DATASET.SEED = 0
    # dataloader
    DATALOADER = edict()
    DATALOADER.IMG_PER_GPU = 32
    DATALOADER.NUM_WORKERS = 8

    # model
    MODEL = edict()
   
    MODEL.DEVICE = 'cuda'

    # solver
    SOLVER = edict()
    SOLVER.OPTIMIZER = 'Adamax'
    SOLVER.BASE_LR = 3e-4
    SOLVER.WARM_UP_FACTOR = 0.1
    SOLVER.WARM_UP_ITER = 2000
    SOLVER.MAX_ITER = 600000
    SOLVER.WEIGHT_DECAY = 0
    SOLVER.MOMENTUM = 0
    SOLVER.BIAS_WEIGHT = 0.0

    # initialization
    CONTINUE_ITER = None
    INIT_MODEL = None


    # log and save
    LOG_PERIOD = 20
    SAVE_PERIOD = 2000

    # validation
    VAL = edict()
    VAL.PERIOD = 2000
    VAL.DATASET = ''
    VAL.IMG_PER_GPU = 1
    VAL.NUM_WORKERS = 1
    VAL.MAX_NUM = 100
    VAL.SAVE_IMG = True
    VAL.TO_Y = True





config = Configs()