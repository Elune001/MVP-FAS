import os
import numpy as np
import copy
import cv2
from torch.utils.data import Dataset

class SFW_Dataset(Dataset):
    def __init__(self, cfg, datasets='SF', transform=None, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        self.image_size = cfg.MODEL.IMG_SIZE
        self.use_celeb = cfg.DATASET.USE_CELEB
        self.dataset_path_dict = cfg.DATASET.PATH

        if (self.use_celeb == True) and (is_train == True):
            datasets = datasets + 'L'
        # self.dataset_path = self.get_dataset_path_information(datasets,dataset_path_dict=self.dataset_path_dict)
        self.folder_name = 'train' if is_train is True else 'test'
        self.annotation_type = ['fake','real']

        Data_base = []
        for d in datasets:
            Data_base = self.get_file_information(Data_base, self.dataset_path_dict['ROOT'], d, is_train)
        self.database = Data_base

        self.init_aug()
        # torch transform
        self.Transform = transform
    def init_aug(self):
        self.Saturation = self.cfg.MCIO.Random_Saturation
        self.Flip = self.cfg.MCIO.Random_Horizontal_Flip

    def Flip_Saturation(self,Img, is_train=True):
        if is_train == True:
            if self.Flip is True:
                Flip_Flag = np.random.randint(0,2)
                if Flip_Flag ==1:
                    Img = self.Image_Flip(Img)
            if self.Saturation is True:
                Saturation_Flag = np.random.randint(0,2)
                if Saturation_Flag == 1:
                    Img = self.Image_Saturation(Img)
        return Img

    def Image_Flip(self, Img):
        Img = cv2.flip(Img, 1)
        return Img
    def Image_Saturation(self,Img):
        Img = Img.astype(np.float32)
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
        Img[..., 1] *= np.random.uniform(0.8, 1.2)
        Img = cv2.cvtColor(Img, cv2.COLOR_HSV2BGR).astype(np.uint8)
        return Img

    def get_type_name(self, type_dict, value):
        for k, v in type_dict.items():
            if value in v:
                return k

    # Reading Annotation
    def get_file_information(self,Data_base, img_path, domain ,is_train=True):

        for anno_type in self.annotation_type:
            files_list_txt = os.path.join(img_path, self.dataset_path_dict[domain], self.dataset_path_dict[domain] + '_' + anno_type + '_' + self.folder_name + '.txt')
            with open(files_list_txt, 'r') as f:
                files_list = f.readlines()

            file_paths = [os.path.join(img_path, s.strip()) for s in files_list]
            # frame_1_paths = [os.path.join(img_path, s.strip().replace('frame0', 'frame1')) for s in files_list]
            # file_paths = frame_0_paths#  + frame_1_paths


            is_real = 1 if anno_type == 'real' else 0
            for file_path in file_paths:
                # attack_type= None
                attack_type = 'real' if is_real else 'fake'

                name = file_path.split('/')[-1]

                # celeb
                if domain == 'L':
                    # subject_live_num
                    if is_real == True:
                        subject_num, live, num_ = name.split('_')
                        attack_type = 'real'
                    else:
                        subject_num, attack_type, num_ = name.split('_')

                    frame_num = -1

                Data_base.append({'Img': file_path,
                                  'is_real': is_real,
                                  'domain': domain,
                                  'attack_type': attack_type,
                                  })

        return Data_base


    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])

        Img_path = db_slic['Img']
        is_real = db_slic['is_real']
        domain = db_slic['domain']
        attack_type = db_slic['attack_type']

        Img = cv2.imread(Img_path)
        Img_shape = Img.shape
        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)

        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)

        if self.is_train:
            Img = self.Flip_Saturation(Img)

        # numpy to torch tensor and normalize
        if self.Transform is not None:
            Img = self.Transform(Img)

        meta = {
            'Img_path': Img_path,
            'Is_real': is_real,
            'Domain':  domain,
            'Attack_type': attack_type,
        }
        return Img, meta

if __name__ == '__main__':
    print()


