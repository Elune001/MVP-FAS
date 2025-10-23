import os
from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = "MVP_FAS"
_C.MODEL.IMG_SIZE = 224
_C.MODEL.OUT_DIM = 256
_C.MODEL.TRAINABLE = True

_C.DATASET = CN()
_C.DATASET.Mean = [0.485, 0.456, 0.406]
_C.DATASET.Std = [0.229, 0.224, 0.225]

# OCIM-celeb
_C.DATASET.PATH = CN()
_C.DATASET.PATH.ROOT = 'D:/Anti_spoofing/dataset'#'./dataset'
_C.DATASET.PATH.O = 'oulu'
_C.DATASET.PATH.C = 'casia'
_C.DATASET.PATH.I = 'replay'
_C.DATASET.PATH.M = 'msu'
_C.DATASET.PATH.L = 'celeb'

_C.DATASET.PATH.S = 'surf'
_C.DATASET.PATH.F = 'cefa'
_C.DATASET.PATH.W = 'wmca'

_C.DATASET.USE_CELEB = False

_C.TRAIN = CN()
_C.TRAIN.NUM_WORKERS = 1
_C.TRAIN.SIMILARITY_ALPHA = 1.0
_C.TRAIN.PATCH_ALIGN_BETA = 1.0

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.LR = 1e-6# 1e-7
_C.TRAIN.WEIGHT_DECAY = 0.001
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.EPOCH = 30
_C.TRAIN.LR_STEP = []

_C.MCIO = CN()
_C.MCIO.Random_Horizontal_Flip = True
_C.MCIO.Random_Saturation = True
