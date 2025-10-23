import os
import numpy as np
import copy
import cv2
from torch.utils.data import Dataset

class MCIO_Dataset(Dataset):
    def __init__(self, cfg, datasets='OCI', transform=None, is_train=True):
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

            frame_0_paths = [os.path.join(img_path, s.strip()) for s in files_list]
            frame_1_paths = [os.path.join(img_path, s.strip().replace('frame0', 'frame1')) for s in files_list]
            if domain == 'L':
                file_paths = frame_0_paths
            else:
                file_paths = frame_0_paths + frame_1_paths

            is_real = 1 if anno_type == 'real' else 0
            for file_path in file_paths:
                attack_type= None
                name = file_path.split('/')[-1]

                domain_attribute = {'O':['camera_type','light','subject_num','attack_type'],'C':['quality','attack_type']}
                if domain == 'O':
                    # camera_type -> 1-6
                    # light type -> 1-3
                    # subject num -> train 1-20 test 36-55
                    # attack type -> real 1 spoof 2-5
                    attack_type_dict = {'real': ['1'], 'print': ['2', '3'], 'replay': ['4', '5']}
                    camera_type, light, subject_num, attack_type = name.split('_')[:-1]
                    frame_num = name.split('.')[0][-1]

                    attack_type = self.get_type_name(attack_type_dict,attack_type)
                elif domain == 'C':
                    # attack_type: warp, cut, video
                    # subject_num -> train 1-20 test 1-30
                    # quality_dict = {'normal':[1,3,5,7],'low':[2,4,6,8],'high':['HR_1','HR_2','HR_3','HR_4']}
                    # attack_type_dict = {'real':[1,2,'HR_4'],'warped':[3,4,'HR_2'],'cut':[5,6,'HR_3'],'replay':[7,8,'HR_4']}
                    quality_dict = {'normal':['1','3','5','7'],'low':['2','4','6','8'],'high':['HR_1','HR_2','HR_3','HR_4']}
                    attack_type_dict = {'real':['1','2','HR_1'],'warped':['3','4','HR_2'],'cut':['5','6','HR_3'],'replay':['7','8','HR_4']}

                    if 'HR' in name:
                        subject_num, HR, HR_num = name.split('_')[:-1]
                        access_type = HR+'_'+HR_num
                    else:
                        subject_num, access_type = name.split('_')[:-1]
                    frame_num = name.split('.')[0][-1]

                    attack_type = self.get_type_name(attack_type_dict, access_type)
                    quality = self.get_type_name(quality_dict, access_type)

                elif domain == 'I':
                    # mobile -> iPhone 3GS
                    # Highdef -> iPad 1 generation
                    # print -> printed
                    if is_real == True:
                        real, subject_num, session_num, webcam, authenticate, background_type, background_num = name.split('_')[:-1]
                        attack_type = 'real'
                    else:
                        camera_type, attack, attack_type, subject_num, session_num, quality, is_video, background_type = name.split('_')[:-1]
                        if attack_type != 'print':
                            attack_type = is_video
                    frame_num = name.split('.')[0][-1]

                elif domain == 'M':
                    if is_real == True:
                        real, subject_num, device_os, SD, session_num = name.split('_')[:-1]
                        attack_type = 'real'
                    else:
                        real, subject_num, device_type, SD, camera_type, is_video, session_num = name.split('_')[:-1]
                        if is_video == 'photo':
                            # 'printed' photo
                            attack_type = camera_type
                        else:
                            # iPad|iPhone 'video'
                            attack_type = is_video
                    frame_num = name.split('.')[0][-1]

                # celeb
                elif domain == 'L':
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