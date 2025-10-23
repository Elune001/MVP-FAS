import torchvision.transforms as transforms
from loaders.MCIO import MCIO_Dataset
from loaders.SFW import SFW_Dataset
def get_MCIO_dataset(cfg,train='CIO',test='M',img_size= (224, 224), normalize=None,):

    train_dataset = MCIO_Dataset(cfg=cfg,datasets=train,
                                transform=transforms.Compose([transforms.ToTensor(), normalize, transforms.Resize(img_size)]),is_train=True)
    val_dataset = MCIO_Dataset(cfg=cfg, datasets=test,
                                transform=transforms.Compose([transforms.ToTensor(), normalize, transforms.Resize(img_size)]), is_train=False)

    return train_dataset, val_dataset


def get_SFW_dataset(cfg,train='SF',test='W',img_size= (224, 224), normalize=None,):

    train_dataset = SFW_Dataset(cfg=cfg,datasets=train,
                                transform=transforms.Compose([transforms.ToTensor(), normalize, transforms.Resize(img_size)]),is_train=True)
    val_dataset = SFW_Dataset(cfg=cfg, datasets=test,
                                transform=transforms.Compose([transforms.ToTensor(), normalize, transforms.Resize(img_size)]), is_train=False)

    return train_dataset, val_dataset


def get_Dataset(cfg,SETTING="MCIO"):
    normalize = transforms.Normalize(mean=cfg.DATASET.Mean, std=cfg.DATASET.Std)
    if SETTING == "MCIO":
        train_dataset, val_dataset = get_MCIO_dataset(cfg,train=cfg.DATASET.TRAIN_DATASET,test=cfg.DATASET.TEST_DATASET,img_size= (cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE), normalize=normalize)
    elif SETTING == 'SFW':
        train_dataset, val_dataset = get_SFW_dataset(cfg, train=cfg.DATASET.TRAIN_DATASET,
                                                      test=cfg.DATASET.TEST_DATASET,
                                                      img_size=(cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE),
                                                      normalize=normalize)

    return train_dataset, val_dataset

