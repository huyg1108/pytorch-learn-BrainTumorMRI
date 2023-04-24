# Brain Tumor MRI - Training Model With Pytorch

## Dataset

#### Link to dataset: [Kaggle](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c)

#### Data structure
|  | T1 | T1C+ | T2 |
|--------------|-----|-----|-----|
| Astrocitoma | 176 | 233 | 171 |
| Carcinoma | 66 | 112 | 73 |
| Ependimoma | 45 | 48| 57 |
| Ganglioglioma | 20 | 18 | 23 |
| Germinoma | 27 | 40 | 33 |
| Glioblastoma | 55 | 94 | 55 | 
| Granuloma | 30 | 31 | 17 |
| Meduloblastoma | 23 | 67 | 41 |
| Meningioma | 272 | 369 | 233 |
| Neurocitoma | 130 | 223 | 104 |
| Oligodendroglioma | 86 | 72 | 66 |
| Papiloma | 66 | 108 | 63 |
| Schwannoma | 148 | 194 | 123 |
| Tuberculoma | 28 | 84 | 33 |
| _NORMAL | 251 | --- | 271 |

#### Preprocessing

I merged 3 groups T1, T2, T1C+ into 1 set, then split it with 70% training set (with data augmentation), 20% validation set and 10% testing set for evaluation. \
So that we had 15 classes to classify.

## Model structure

```
Layer (type:depth-idx)                   Param #
================================================
BrainTumorModelV0                        --
├─Sequential: 1-1                        --
│    └─Conv2d: 2-1                       896
│    └─BatchNorm2d: 2-2                  64
│    └─LeakyReLU: 2-3                    --
│    └─MaxPool2d: 2-4                    --
│    └─Conv2d: 2-5                       18,496
│    └─BatchNorm2d: 2-6                  128
│    └─LeakyReLU: 2-7                    --
│    └─MaxPool2d: 2-8                    --
│    └─Conv2d: 2-9                       73,856
│    └─BatchNorm2d: 2-10                 256
│    └─LeakyReLU: 2-11                   --
│    └─MaxPool2d: 2-12                   --
│    └─Conv2d: 2-13                      295,168
│    └─BatchNorm2d: 2-14                 512
│    └─LeakyReLU: 2-15                   --
│    └─MaxPool2d: 2-16                   --
├─Sequential: 1-2                        --
│    └─Linear: 2-17                      25,690,624
│    └─BatchNorm1d: 2-18                 1,024
│    └─LeakyReLU: 2-19                   --
│    └─Dropout: 2-20                     --
│    └─Linear: 2-21                      131,328
│    └─BatchNorm1d: 2-22                 512
│    └─LeakyReLU: 2-23                   --
│    └─Dropout: 2-24                     --
│    └─Linear: 2-25                      3,855
=============================================
Total params: 26,216,719
Trainable params: 26,216,719
Non-trainable params: 0
Total mult-adds (G): 24.51
```
## Training

- Loss function: Cross Entropy
- Optimizer: Adam
- Learning rate: 0.001
- During training, early stopping will be implemented such that if the validation loss does not improve for 10 consecutive epochs, the training will be terminated

## Learning Curve
![curve](https://user-images.githubusercontent.com/67747576/233972217-b4daede2-37cd-4002-b5f0-2bbeaa99aaf2.png)

## Evaluate in test set
```
Test Loss = 0.2563, Accuracy = 0.9438
```
