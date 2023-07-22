from torchvision import transforms
from torch.utils import data
from PIL import Image
import os
import numpy as np
import json
import torch

def dataset_info(dt):  
    assert dt in ['EORSSD']
    if dt == 'EORSSD':
        dt_mean = [0.3412, 0.3798, 0.3583]
        dt_std = [0.1148, 0.1042, 0.0990]
    return dt_mean, dt_std


def random_aug_transform():  
    flip_h = transforms.RandomHorizontalFlip(p=1)
    flip_v = transforms.RandomVerticalFlip(p=1)
    angles = [0, 90, 180, 270]
    rot_angle = angles[np.random.choice(4)]
    rotate = transforms.RandomRotation((rot_angle, rot_angle))
    r = np.random.random()
    if r <= 0.25:
        flip_rot = transforms.Compose([flip_h, flip_v, rotate])
    elif r <= 0.5:
        flip_rot = transforms.Compose([flip_h, rotate])
    elif r <= 0.75:
        flip_rot = transforms.Compose([flip_v, flip_h, rotate])  
    else:
        flip_rot = transforms.Compose([flip_v, rotate])
    return flip_rot


class ORSI_SOD_Dataset(data.Dataset):
    def __init__(self, root, size=224, mode='train', aug=False):
        self.mode = mode 
        self.aug = aug 
        self.dt_mean, self.dt_std = dataset_info('EORSSD')
        self.prefixes = [line.strip() for line in open(os.path.join(root, mode+'.txt'))]
        self.image_paths = [os.path.join(root, 'images', prefix + '.jpg') for prefix in self.prefixes]
        self.label_paths = [os.path.join(root, 'labels', prefix + '.png') for prefix in self.prefixes]
        #self.edge_paths = [os.path.join(root, 'edges', prefix + '.png') for prefix in self.prefixes]
        self.image_transformation = transforms.Compose([transforms.Resize((size, size),Image.BILINEAR),transforms.ToTensor(), transforms.Normalize(self.dt_mean, self.dt_std)])
        self.label_transformation = transforms.Compose([transforms.Resize((size, size),Image.BILINEAR),transforms.ToTensor()])
        # read class_indict
        json_file = os.path.join(root, "scene_classes.json")
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        json_file = open(json_file, 'r')
        self.scene_dict = json.load(json_file)
        scene_file = os.path.join(root, "scene.txt")
        self.lines = [line.strip() for line in open(os.path.join(scene_file))]
        self.mapping = {}
        for line in self.lines:
            name, scene = line.split(',')
            self.mapping[name] = scene
        
            
    def __getitem__(self, index):
        if self.mode == "train": 
            if self.aug:
                flip_rot = random_aug_transform()
                image = self.image_transformation(flip_rot(Image.open(self.image_paths[index]).convert('RGB'))) 
                label = self.label_transformation(flip_rot(Image.open(self.label_paths[index]).convert('L')))
                #edge = self.label_transformation(flip_rot(Image.open(self.edge_paths[index])))
            else:
                image = self.image_transformation(Image.open(self.image_paths[index]).convert('RGB')) 
                label = self.label_transformation(Image.open(self.label_paths[index]).convert('L'))
                #edge = self.label_transformation(Image.open(self.edge_paths[index]))
        elif self.mode == "test": 
            image = self.image_transformation(Image.open(self.image_paths[index]).convert('RGB'))
            label = self.label_transformation(Image.open(self.label_paths[index]).convert('L'))
            #edge = self.label_transformation(Image.open(self.edge_paths[index]))
        name = self.prefixes[index]
        scene = self.mapping.get(name)
        scene_label = self.scene_dict[scene] 
        scene_label = torch.as_tensor(scene_label, dtype=torch.int64)    
        return image, label, scene_label, name 
        

    def __len__(self):
        return len(self.prefixes)
