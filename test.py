import os
import time
import datetime
import argparse
import random
import cv2
import numpy as np
import logging

import torch
from torch.utils.data import DataLoader

from configs.cfg import _C as cfg

from loaders.make_dataset import get_Dataset
from models.make_network import get_network
from losses.make_losses import get_loss_fucntion
from utils.metric import Metric

from torch.nn import functional as F

def set_pretrained_setting(net,weight_path):
    checkpoint_dict = torch.load(weight_path)
    checkpoint = checkpoint_dict['state_dict']
    last_epoch = checkpoint_dict['epoch']-1
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in net.module.state_dict().keys()}
    net.module.load_state_dict(pretrained_dict)
    return net, last_epoch

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_logger(filename='train_log.log'):
    logging.basicConfig(filename=filename, level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def get_eta(batch_time, batch_index, loader_len, this_epoch, max_epoch):
    # this epoch start 0
    this_epoch_eta = int(batch_time * (loader_len - (batch_index + 1)))
    left_epoch_eta = int(((max_epoch - (this_epoch + 1)) * batch_time * loader_len))
    eta = this_epoch_eta + left_epoch_eta
    return eta, this_epoch_eta


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Entry Fuction")

    parser.add_argument("--model", type=str, default="MVP_FAS", help="choose model")
    parser.add_argument("--save_name", type=str, default="MVP_FAS", help="choose subname of model")
    parser.add_argument("--batch_size", type=int, default=18, help="batch size for training")
    parser.add_argument("--seed", type=int, default=0, help="random seed for training")
    parser.add_argument("--resume", type=bool, default=True, help='resume')
    parser.add_argument("--checkpoint", type=str, default='best_model.pth', help='for resume')
    parser.add_argument("--setting", type=str, default='MCIO', help='DATASET SETTING [MCIO, SFW]')
    parser.add_argument("--train_dataset", type=str, default='MIO', help='TRAIN_DATASET')
    parser.add_argument("--test_dataset", type=str, default='C', help='TEST_DATASET')

    args = parser.parse_args()

    now_time = datetime.datetime.now()

    model_name = args.model
    save_name = args.save_name
    batch_size = args.batch_size
    seed = args.seed
    resume = args.resume
    checkpoint = args.checkpoint

    cfg['DATASET']['SETTING'] = args.setting
    cfg['DATASET']['TRAIN_DATASET'] = args.train_dataset
    cfg['DATASET']['TEST_DATASET'] = args.test_dataset

    set_seed(seed, deterministic=False)

    start_epoch = 0
    max_epoch = cfg.TRAIN.EPOCH
    save_folder = './save_model'
    save_folder = os.path.join(save_folder, model_name + '_' + save_name)
    createDirectory(save_folder)
    reference = './reference'

    logger_name = f'test_{save_name}.log'
    logger = create_logger(os.path.join(save_folder, logger_name))
    logger.info(
        f'##############################################################################################################\n'
        f'Experiment history\n'
        f'save_name: {save_name}\n'
        f'year: {now_time.year} month: {now_time.month} day: {now_time.day} hour: {now_time.hour} min: {now_time.minute}\n'
        f'##############################################################################################################')
    logger.info(cfg)

    last_epoch = -1
    validation = True
    best_val_loss = np.inf
    best_HTER = np.inf
    save_periodically = False
    period = 10
    PIN_MEMORY = True
    logger_interval = 10

    lr = cfg.TRAIN.LR

    Similarity_alpha = cfg.TRAIN.SIMILARITY_ALPHA
    # get dataset
    _, val_Dataset = get_Dataset(cfg, SETTING=cfg.DATASET.SETTING)
    net = get_network(cfg, net_name=model_name)


    if resume == True: net, last_epoch = set_pretrained_setting(net, os.path.join(reference, checkpoint))

    CE_loss, val_CE_loss = get_loss_fucntion(cfg, loss_name='CrossEntropy')

    val_batch_time = None
    batch_time = 0#None


    val_batch_iterator = iter(DataLoader(val_Dataset, batch_size, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=PIN_MEMORY))
    val_batch_iterator_len = val_batch_iterator.__len__()

    if validation == True:
        net.eval()
        val_total_loss_history, val_Sim_loss_history = [], []
        with torch.no_grad():
            val_metric = Metric()
            for val_batch_idx, (val_img, val_target) in enumerate(val_batch_iterator):
                val_start_time = time.time()

                val_img = val_img.cuda()
                val_Is_real = val_target['Is_real'].cuda()
                val_Domain = val_target['Domain']#.cuda()
                val_Attack_type = val_target['Attack_type']#.cuda()


                # forward
                #val_output_list, val_feature = net(val_img)
                val_results = net(val_img)
                val_output_list = val_results['similarity']


                val_Similarity_loss = val_CE_loss(val_output_list, val_Is_real).cpu().numpy()

                val_Sim_loss_history.append(val_Similarity_loss)
                val_Similarity_loss_mean = np.asarray(val_Sim_loss_history).mean()


                # metric
                # spoofing | real
                #     0       1
                prob = F.softmax(val_output_list, dim=-1).detach().cpu().numpy()[:, -1].tolist()
                val_acc, val_EER, val_HTER, val_auc, val_threshold, val_ACC_threshold, val_TPR_FPR_rate = val_metric(val_Is_real.cpu().numpy().tolist(),prob)
                if (val_batch_idx + 1) == 410:
                    print()

                val_end_time = time.time()
                val_batch_time = val_end_time - val_start_time

                val_eta, val_this_epoch_eta = get_eta(batch_time=val_batch_time, batch_index=val_batch_idx,
                                                      loader_len=val_batch_iterator_len, this_epoch=0,
                                                      max_epoch=1)
                val_eta = val_eta

                val_line = '[VAL][{}/{}] || iter: {}/{} || ' \
                           'This_iter: total_loss: {:.4f} || ' \
                           'This_epoch: Sim_loss: {:.4f} HTER: {:.4f} AUC: {:.4f} TPR@FPR: {:.4f} top-1: {:.4f} || ' \
                           'Batchtime: {:.4f} s || this_epoch: {} || ETA: {}'.format(
                    1, max_epoch, val_batch_idx + 1, val_batch_iterator_len,
                    val_Similarity_loss,
                    val_Similarity_loss_mean * Similarity_alpha, val_HTER * 100, val_auc*100, val_TPR_FPR_rate, val_acc,
                    val_batch_time, str(datetime.timedelta(seconds=val_this_epoch_eta)),
                    str(datetime.timedelta(seconds=val_eta)))
                logger.info(val_line)


    logging.shutdown()








