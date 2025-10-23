import torch


def get_loss_fucntion(cfg, loss_name):
    if loss_name == 'CrossEntropy':
        train_loss = torch.nn.CrossEntropyLoss().cuda()
        val_loss = torch.nn.CrossEntropyLoss().cuda()

    else:
        raise Exception("Undefined loss function")
    return train_loss, val_loss
