from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.I2T_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# OCTAR training-only modules: RTG, R-MTD, PTA, and APR.
_C.MODEL.OCTAR = CN()
_C.MODEL.OCTAR.ENABLED = False
_C.MODEL.OCTAR.RMTD_WEIGHT = 0.03
_C.MODEL.OCTAR.APR_WEIGHT = 0.05
_C.MODEL.OCTAR.WARMUP_EPOCHS = 10
_C.MODEL.OCTAR.RTG_TOKEN_RATIO = 0.6
_C.MODEL.OCTAR.RTG_BATCH_TOPK_RATIO = 0.5
_C.MODEL.OCTAR.RTG_SCORE_THRESHOLD = -999.0
_C.MODEL.OCTAR.RTG_ENSURE_ONE = True
_C.MODEL.OCTAR.CANDIDATE_RATIO = 0.6
_C.MODEL.OCTAR.CANDIDATE_SCORE_THRESHOLD = -999.0
_C.MODEL.OCTAR.REGION_BINS = [0.0, 0.35, 0.70, 1.0]
_C.MODEL.OCTAR.REGION_MIN_TOKENS = 1
_C.MODEL.OCTAR.REGION_CONF_THRESHOLD = 0.0
_C.MODEL.OCTAR.REGION_CONF_TOPK_RATIO = 0.5
_C.MODEL.OCTAR.REGION_ADAPTIVE_QUOTA = True
_C.MODEL.OCTAR.REGION_QUOTA_TEMPERATURE = 0.05
_C.MODEL.OCTAR.MASK_OVERLAP_THRESHOLD = 0.5
_C.MODEL.OCTAR.PTA_ALPHA = 0.3
_C.MODEL.OCTAR.PTA_TEMPERATURE = 0.07
_C.MODEL.OCTAR.APR_MOMENTUM = 0.1
_C.MODEL.OCTAR.NORMALIZE_FEATURES = True
_C.MODEL.OCTAR.EPS = 1e-6
_C.MODEL.OCTAR.RTG_POSITIVE_PROMPTS = [
    "a photo of a complete person.",
    "a photo of a visible body part of a pedestrian.",
    "a photo of pedestrian clothing.",
    "a photo of the upper body of a pedestrian.",
    "a photo of the lower body of a pedestrian.",
    "a photo of pants of a pedestrian.",
    "a photo of shoes of a pedestrian.",
]
_C.MODEL.OCTAR.RTG_NEGATIVE_PROMPTS = [
    "a photo of an occluding object.",
    "a photo of a bag blocking a person.",
    "a photo of a backpack blocking a person.",
    "a photo of background.",
    "a photo of road.",
    "a photo of floor.",
    "a photo of another pedestrian occluding the person.",
]

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# Semantic Occlusion Composer (SOC).
_C.INPUT.SOC = CN()
_C.INPUT.SOC.ENABLED = False
_C.INPUT.SOC.PROB = 0.5
_C.INPUT.SOC.AREA_RANGE = [0.15, 0.35]
_C.INPUT.SOC.ASPECT_RANGE = [0.3, 3.3]
_C.INPUT.SOC.BLEND_ALPHA = 1.0
_C.INPUT.SOC.OCC_REID_WEIGHT = 0.5
_C.INPUT.SOC.AVOID_SAME_ID = True
_C.INPUT.SOC.EDGE_RATIO = 0.2
_C.INPUT.SOC.BODY_X_RANGE = [0.2, 0.8]
_C.INPUT.SOC.BODY_Y_RANGE = [0.15, 0.85]
_C.INPUT.SOC.PARTS = [
    [0.15, 0.35],
    [0.35, 0.65],
    [0.65, 0.85],
]
_C.INPUT.SOC.NUM_SOURCE_CANDIDATES = 8
_C.INPUT.SOC.NUM_TARGET_CANDIDATES = 8
_C.INPUT.SOC.TOPK = 3
_C.INPUT.SOC.SAMPLE_MODE = "softmax"
_C.INPUT.SOC.TEMPERATURE = 0.2
_C.INPUT.SOC.SRC_WEIGHT = 1.0
_C.INPUT.SOC.TAR_WEIGHT = 1.0
_C.INPUT.SOC.HARD_WEIGHT = 0.5
_C.INPUT.SOC.UNREAL_WEIGHT = 0.2
_C.INPUT.SOC.TARGET_ID_WEIGHT = 0.5
_C.INPUT.SOC.TARGET_NEG_WEIGHT = 0.0
_C.INPUT.SOC.MIN_VISIBLE_RATIO = 0.55
_C.INPUT.SOC.SOURCE_POSITIVE_PROMPTS = [
    "a photo of an occluding object.",
    "a photo of a bag.",
    "a photo of a backpack.",
    "a photo of an umbrella.",
    "a photo of clothes of another person.",
    "a photo of a body part of another pedestrian.",
]
_C.INPUT.SOC.SOURCE_NEGATIVE_PROMPTS = [
    "a photo of empty background.",
    "a photo of road.",
    "a photo of floor.",
    "a photo of blurred texture.",
]
_C.INPUT.SOC.TARGET_PROMPTS = [
    "a photo of a visible body part of a pedestrian.",
    "a photo of pedestrian clothing.",
    "a photo of the upper body of a pedestrian.",
    "a photo of the lower body of a pedestrian.",
    "a photo of pants of a pedestrian.",
    "a photo of shoes of a pedestrian.",
]
_C.INPUT.SOC.TARGET_NEGATIVE_PROMPTS = [
    "a photo of an occluding object.",
    "a photo of a bag.",
    "a photo of a backpack.",
    "a photo of a bicycle.",
    "a photo of background.",
    "a photo of another pedestrian occluding the person.",
]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
_C.SOLVER = CN()
_C.SOLVER.SEED = 1234
_C.SOLVER.MARGIN = 0.3

# stage1
# ---------------------------------------------------------------------------- #
# Name of optimizer
_C.SOLVER.STAGE1 = CN()

_C.SOLVER.STAGE1.IMS_PER_BATCH = 64

_C.SOLVER.STAGE1.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.STAGE1.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.STAGE1.BASE_LR = 3e-4
# Momentum
_C.SOLVER.STAGE1.MOMENTUM = 0.9

# Settings of weight decay
_C.SOLVER.STAGE1.WEIGHT_DECAY = 0.0005
_C.SOLVER.STAGE1.WEIGHT_DECAY_BIAS = 0.0005

# warm up factor
_C.SOLVER.STAGE1.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.STAGE1.WARMUP_EPOCHS = 5
_C.SOLVER.STAGE1.WARMUP_LR_INIT = 0.01
_C.SOLVER.STAGE1.LR_MIN = 0.000016

_C.SOLVER.STAGE1.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.STAGE1.WARMUP_METHOD = "linear"

_C.SOLVER.STAGE1.COSINE_MARGIN = 0.5
_C.SOLVER.STAGE1.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.STAGE1.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.STAGE1.LOG_PERIOD = 100
# epoch number of validation
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
# _C.SOLVER.STAGE1.IMS_PER_BATCH = 64
_C.SOLVER.STAGE1.EVAL_PERIOD = 10
_C.SOLVER.STAGE1.SKIP = False
_C.SOLVER.STAGE1.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Solver
# stage1
# ---------------------------------------------------------------------------- #
_C.SOLVER.STAGE2 = CN()

_C.SOLVER.STAGE2.IMS_PER_BATCH = 64
# Name of optimizer
_C.SOLVER.STAGE2.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.STAGE2.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.STAGE2.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.STAGE2.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.STAGE2.BIAS_LR_FACTOR = 1
# Momentum
_C.SOLVER.STAGE2.MOMENTUM = 0.9
# Margin of triplet loss
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.STAGE2.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.STAGE2.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.STAGE2.WEIGHT_DECAY = 0.0005
_C.SOLVER.STAGE2.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.STAGE2.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STAGE2.STEPS = (40, 70)
# warm up factor
_C.SOLVER.STAGE2.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.STAGE2.WARMUP_EPOCHS = 5
_C.SOLVER.STAGE2.WARMUP_LR_INIT = 0.01
_C.SOLVER.STAGE2.LR_MIN = 0.000016


_C.SOLVER.STAGE2.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.STAGE2.WARMUP_METHOD = "linear"

_C.SOLVER.STAGE2.COSINE_MARGIN = 0.5
_C.SOLVER.STAGE2.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.STAGE2.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.STAGE2.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.STAGE2.EVAL_PERIOD = 10
_C.SOLVER.STAGE2.EVAL_EPOCHS = []
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
