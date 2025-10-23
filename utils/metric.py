import math
import numpy as np
import torch

from scipy.integrate import simps
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def get_threshold(probs, grid_density):
    Min, Max = min(probs), max(probs)
    thresholds = []
    for i in range(grid_density + 1):
        thresholds.append(0.0 + i * 1.0 / float(grid_density))
    thresholds.append(1.1)
    return thresholds

def eval_state(probs, labels, thr):
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP

def get_EER_states(probs, labels, grid_density=10000):
    thresholds = get_threshold(probs, grid_density)
    min_dist = 1.0
    min_dist_states = []
    FRR_list = []
    FAR_list = []
    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)
        if (FN + TP == 0):
            FRR = TPR = 1.0
            FAR = FP / float(FP + TN)
            TNR = TN / float(TN + FP)
        elif (FP + TN == 0):
            TNR = FAR = 1.0
            FRR = FN / float(FN + TP)
            TPR = TP / float(TP + FN)
        else:
            FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)
            TNR = TN / float(TN + FP)
            TPR = TP / float(TP + FN)
        dist = math.fabs(FRR - FAR)
        FAR_list.append(FAR)
        FRR_list.append(FRR)
        if dist <= min_dist:
            min_dist = dist
            min_dist_states = [FAR, FRR, thr]
    EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
    thr = min_dist_states[2]
    return EER, thr, FRR_list, FAR_list

def calculate_threshold(probs, labels, threshold):
    TN, FN, FP, TP = eval_state(probs, labels, threshold)
    ACC = (TP + TN) / labels.shape[0]
    return ACC

def get_HTER_at_thr(probs, labels, thr):
    TN, FN, FP, TP = eval_state(probs, labels, thr)
    if (FN + TP == 0):
        FRR = 1.0
        FAR = FP / float(FP + TN)
    elif (FP + TN == 0):
        FAR = 1.0
        FRR = FN / float(FN + TP)
    else:
        FAR = FP / float(FP + TN)
        FRR = FN / float(FN + TP)
    HTER = (FAR + FRR) / 2.0
    return HTER

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Metric():
    def __init__(self):
        self.Acc_list = []
        self.label_list = []
        self.prob_list = []
    def __call__(self, GT, pred):

        auc_score = -1
        TPR_FPR_rate = -1

        #################### Metric
        # spoofing | real
        #     0       1
        # prob = F.softmax(output_list, dim=-1).cpu().data.numpy()[:, -1]
        self.label_list = self.label_list + GT
        self.prob_list = self.prob_list + pred

        acc_valid = ((np.array(self.prob_list)>0.5) == np.array(self.label_list)).sum()/len(self.label_list)

        cur_EER_valid, threshold, _, _ = get_EER_states(np.array(self.prob_list), np.array(self.label_list))
        ACC_threshold = calculate_threshold(np.array(self.prob_list), np.array(self.label_list), threshold)
        cur_HTER_valid = get_HTER_at_thr(np.array(self.prob_list), np.array(self.label_list), threshold)

        if len(np.unique(self.label_list)) > 1:
            auc_score = roc_auc_score(self.label_list, self.prob_list)
            # print(auc_score)
            fpr, tpr, thr = roc_curve(np.array(self.label_list), np.array(self.prob_list))
            tpr_filtered = tpr[fpr <= (1 / 100)]
            if len(tpr_filtered) == 0:
              TPR_FPR_rate = 0
            else:
              TPR_FPR_rate = tpr_filtered[-1]
        # print("TPR@FPR = ", TPR_FPR_rate)
        ####################

        return acc_valid, cur_EER_valid, cur_HTER_valid, auc_score, threshold, ACC_threshold * 100, TPR_FPR_rate

