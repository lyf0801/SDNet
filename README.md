# SDNet
This repository is about the work "**Transcending Pixels: Boosting Saliency Detection via Scene Understanding From Aerial Imagery**" in IEEE TGRS 2023.

# Abstract
Existing remote sensing image salient object detection (RSI-SOD) methods widely perform object-level semantic understanding with pixel-level supervision, but ignore the image-level scene information. As a fundamental attribute of RSIs, the scene has a complex intrinsic correlation with salient objects, which may bring hints to improve saliency detection performance. However, existing RSI-SOD datasets lack both pixel- and image-level labels, and it is non-trivial to effectively transfer the scene domain knowledge for more accurate saliency localization. To address these challenges, we first annotate the image-level scene labels of three RSI-SOD datasets inspired by remote sensing scene classification. On top of it, we present a novel scene-guided dual-stream network (SDNet), which can perform cross-task knowledge distillation from the scene classification to facilitate accurate saliency detection. Specifically, a scene knowledge transfer module (SKTM) and a conditional dynamic guidance module (CDGM) are designed for extracting saliency key area as spatial attention from the scene subnet and guiding the saliency subnet to generate scene-enhanced saliency features, respectively. Finally, an object contour awareness module (OCAM) is introduced to enable the model to focus more on irregular spatial details of salient objects from the complicated background. Extensive experiments reveal that our SDNet outperforms over 20 state-of-the-art algorithms on three datasets. Moreover, we show that the proposed framework is model-agnostic, and its extension to six baselines can bring significant performance benefits.

# Methodology

## 1. Define 12 types of scene categories and manually annotate scene labels for three RSI-SOD datasets
![image](https://github.com/lyf0801/SDNet/assets/73867361/d5a20fcb-4530-42e6-b631-e7eb207dc1c1)

<div class="center">
  
|                               |                   |                    |                       |
|:-----------------------------:|:-----------------:|:------------------:|:---------------------:|
| **Scene Category**            | **ORSSD Dataset** | **EORSSD Dataset** | **ORSI-4199 Dataset** |
| **Airplane Facilities**      | 136               | 427                | 504                   |
| **Industrial Facilities**     | 49                | 57                 | 441                   |
| **Bridges**                  | 14                | 37                 | 429                   |
| **Ships**                    | 145               | 449                | 418                   |
| **Rural Buildings**           | 48                | 96                 | 634                   |
| **Transportation Facilities** | 30                | 71                 | 317                   |
| **Highways**                 | 49                | 229                | 571                   |
| **Rivers**                    | 81                | 105                | 70                    |
| **Lakes**                     | 104               | 192                | 303                   |
| **Islands**                   | 46                | 148                | 5                     |
| **Sports Facilities**         | 77                | 116                | 354                   |
| **Others**                   | 21                | 73                 | 153                   |
| **Total**                     | **800**               | **2000**               | **4199**                  |

</div>

  




## 2. Propose a multitask learning-based (MTL) sceneguided dual-branch network (SDNet)
![image](https://github.com/lyf0801/SDNet/assets/73867361/25332464-0263-4864-b27f-3e2c7c70e2d7)
![image](https://github.com/lyf0801/SDNet/assets/73867361/a395263e-4f74-406c-9771-1780249dd998)

## 3. Demonstrate the scene learning is effective
![image](https://github.com/lyf0801/SDNet/assets/73867361/89f9ce21-8486-471c-9b7b-d5ea8e560593)



## 4. Reveal the SDNet is model-agnostic
![image](https://github.com/lyf0801/SDNet/assets/73867361/2f8be427-98f3-4c47-a9bb-92037c9a37f6)



# How to use

## 1. Install newest versions of torch and torchdata
```
thop                      0.0.31
tqdm                      4.59.0
numpy                     1.20.2
timm                      0.4.12
tokenizers                0.12.1
torch                     1.8.1
torchvision               0.9.1
```
## 2. Download weights files from Google Drive

<https://drive.google.com/drive/folders/1zygarM13gu48gJQ1jPXCAUZEsuDB2Y0l>


## 3. Run getsmaps.py to generate the saliency maps
```
python getsmaps.py
```
![image](https://github.com/lyf0801/SDNet/assets/73867361/e1cb8f26-4192-4426-b46d-d3aa2fe82bec)

## 4. Run compute_metrics.py to calculate the qualititive results
```
python compute_metrics.py
```
![image](https://github.com/lyf0801/SDNet/assets/73867361/8e6ef5c0-3c46-4ccb-8b74-156100e016df)



# Citation (If you think this repository could help you, please cite:)
```BibTeX
@ARTICLE{SDNet2023,

  author={Liu, Yanfeng and Xiong, Zhitong and Yuan, Yuan and Wang, Qi},
  
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  
  title={Transcending Pixels: Boosting Saliency Detection via Scene Understanding From Aerial Imagery}, 
  
  year={2023},
  
  volume={61},
  
  number={},
  
  pages={1-16},

  doi={10.1109/TGRS.2023.3298661}
  }

@ARTICLE{SRAL2023,

  author={Liu, Yanfeng and Xiong, Zhitong and Yuan, Yuan and Wang, Qi},
  
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  
  title={Distilling Knowledge From Super-Resolution for Efficient Remote Sensing Salient Object Detection}, 
  
  year={2023},
  
  volume={61},
  
  number={},
  
  pages={1-16},
  
  doi={10.1109/TGRS.2023.3267271}
  
  }

@InProceedings{RSSOD2023,

  author = {Xiong, Zhitong and Liu, Yanfeng and Wang, Qi and Zhu, Xiao Xiang},

  title = {RSSOD-Bench: A Large-Scale Benchmark Dataset for Salient Object Detection in Optical Remote Sensing Imagery},

  booktitle = {Proc. IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},

  pages={},

  year = {2023}

}

@ARTICLE{HFANet2022,

  author={Wang, Qi and Liu, Yanfeng and Xiong, Zhitong and Yuan, Yuan},

  journal={IEEE Transactions on Geoscience and Remote Sensing},

  title={Hybrid Feature Aligned Network for Salient Object Detection in Optical Remote Sensing Imagery},

  year={2022},

  volume={60},

  number={},

  pages={1-15},

  doi={10.1109/TGRS.2022.3181062}

}
```

# Acknowledgment and our other works
1. <https://github.com/EarthNets/Dataset4EO>
2. <https://github.com/lyf0801/HFANet>
3. <https://github.com/lyf0801/SRAL>
4. <https://github.com/rmcong/DAFNet_TIP20>
5. <https://github.com/rmcong/EORSSD-dataset>
6. <https://github.com/rmcong/ORSSD-dataset>

