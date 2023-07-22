import numpy as np
import torch
import os
from torch.utils.data import DataLoader

from ORSI_SOD_dataset import ORSI_SOD_Dataset
from tqdm import tqdm
from src.model import net as Net 
#from ablation_study.new_baseline import net as Net
#from ablation_study.baseline_scene import net as Net
#from ablation_study.baseline1 import net as Net
#from ablation_study.new_baseline3 import net as Net
#from ablation_study.baseline12 import net as Net
#from ablation_study.baseline12_BCE import net as Net
#from ablation_study.baseline_channel_concate_CAM import net as Net
#from ablation_study.baseline_channel_concate_Feat import net as Net
#from ablation_study.baseline_elem_mul_CAM import net as Net
#from ablation_study.baseline_elem_mul_Feat import net as Net
#from ablation_study.baseline_elem_sum_CAM import net as Net
#from ablation_study.baseline_elem_sum_Feat import net as Net
from evaluator import Eval_thread
from PIL import Image
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'





def unload(x):
    y = x.squeeze().cpu().data.numpy()
    return y
def convert2img(x):
    return Image.fromarray(x*255).convert('L')
def min_max_normalization(x):
    x_normed = (x - np.min(x)) / (np.max(x)-np.min(x))
    return x_normed
def save_smap(smap, path, negative_threshold=0.25):
    # smap: [1, H, W]
    if torch.max(smap) <= negative_threshold:
        smap[smap<negative_threshold] = 0
        smap = convert2img(unload(smap))
    else:
        smap = convert2img(min_max_normalization(unload(smap)))
    smap.save(path)



def getsmaps(dataset_name):
    #define dataset
    input_size = 448
    dataset_root  = "/data/iopen/lyf/SaliencyOD_in_RSIs/" + dataset_name + " dataset/"
    test_set = ORSI_SOD_Dataset(root = dataset_root,  size=input_size, mode = "test", aug = False)
    test_loader = DataLoader(test_set, batch_size = 1, shuffle = True, num_workers = 1)
    
    ##define network and load weights
    net = Net().cuda().eval() 
    if dataset_name == "ORSSD":
        net.load_state_dict(torch.load("./data/weights/ORSSD_weights.pth"))
    elif dataset_name == "EORSSD":
        net.load_state_dict(torch.load("./data/weights/EORSSD_weights.pth"))
    elif dataset_name == "ORS_4199":
        net.load_state_dict(torch.load("./data/weights/ORS_4199_weights.pth"))

    ##save saliency maps
    infer_time = 0
    for image, label, scene, name in tqdm(test_loader): 
        with torch.no_grad():
            image, label = image.cuda(), label.cuda()
            
            t1 = time.time()
            smap = net(image) 
            t2 = time.time()
            infer_time += (t2 - t1)
            
            ##if not exist then difine
            dirs = "./data/output/predict_smaps" +  "_SASOD_" + dataset_name
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            path = os.path.join(dirs, name[0] + "_SASOD" + '.png')  
            save_smap(smap, path)
    print(len(test_loader))
    print(infer_time)
    print(len(test_loader) / infer_time)  # inference speed (without I/O time),

if __name__ == "__main__":
    #define
    import torchvision.models as models
    net = models.resnet18(pretrained=False).cuda().eval()
    #compute Params and FLOPs
    from thop import profile
    from thop import clever_format
    x = torch.Tensor(1,3,448,448).cuda()
    macs, params = profile(net, inputs=(x, ), verbose = False)
    print('flops: ', f'{macs/1e9}GMac', 'params: ', f'{params/1e6}M')

    dataset = ["ORSSD", "EORSSD", "ORS_4199"] 
    for datseti in dataset:
        getsmaps(datseti)
