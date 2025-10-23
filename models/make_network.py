import torch

from models.MVP_FAS import mspt
def get_network(cfg,net_name='MVP_FAS'):
    if net_name == 'MVP_FAS':
        net = mspt(cfg)
    net = torch.nn.DataParallel(net).cuda()
    return net

def set_pretrained_setting(net,optimizer,weight_path):
    checkpoint_dict = torch.load(weight_path)
    checkpoint = checkpoint_dict['state_dict']
    optim_checkpoint = checkpoint_dict['optimizer']
    last_epoch = checkpoint_dict['epoch']-1
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in net.module.state_dict().keys()}
    net.module.load_state_dict(pretrained_dict)
    optimizer.load_state_dict(optim_checkpoint)
    return net, optimizer, last_epoch