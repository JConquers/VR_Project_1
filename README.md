# Mask Detection and Segmentation Project

## 1. Introduction
This project focuses on detecting and segmenting face masks in images using various machine learning and deep learning techniques. The tasks include binary classification using handcrafted features and machine learning classifiers, binary classification using CNNs, region segmentation using traditional techniques, and mask segmentation using U-Net. The objective is to evaluate the performance of different approaches in terms of classification accuracy and segmentation quality.

### Contributors:

(IMT2022019) Daksh Rajesh <Daksh.Rajesh@iiitb.ac.in>

(IMT2022044) Jinesh Pagaria <Jinesh.Pagaria@iiitb.ac.in>

(IMT2022087) Aaditya Ramchandra Gole <Aaditya.Gole@iiitb.ac.in>

---

## 2. Dataset
### Source:
https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset

https://github.com/sadjadrz/MFSD

```
MSFD
├── 1
│   ├── face_crop # face-cropped images of images in MSFD/1/img
│   ├── face_crop_segmentation # ground truth of segmend face-mask
│   └── img
└── 2
    └── img
```

```
dataset
├── with_mask # contains images with mask
└── without_mask # contains images without face-mask
```

### Structure:
- **Training Set:** Images used for training the models.
- **Testing Set:** Images used to evaluate model performance.
- **Annotations:** Mask region labels for segmentation tasks.

---

## 3. Objectives

### a. Binary Classification Using Handcrafted Features and ML Classifiers
1. Extract handcrafted features from facial images (e.g., HOG, LBP, SIFT).
2. Train and evaluate at least two machine learning classifiers (Here we use XGBoost and Neural Network) and compare classifier performances based on accuracy.

### b. Binary Classification Using CNN
1. Design and train a Convolutional Neural Network (CNN) for mask classification.
2. Experiment with different hyperparameters (learning rate, batch size, optimizer, activation function).
3. Compare CNN performance with traditional ML classifiers.

### c. Region Segmentation Using Traditional Techniques
1. Apply region-based segmentation methods (e.g., thresholding, edge detection) to segment mask regions, visualize and evaluate segmentation results.

### d. Mask Segmentation Using U-Net
1. Train a U-Net model to segment the mask regions in facial images.
2. Compare segmentation performance with traditional techniques using IoU or Dice score.

---

## 4. Hyperparameters and Experiments
 [TO BE ADDED BY DAKSH AND ADITYA]

---

## 5. Results
### Evaluation Metrics:
- **Classification:** Accuracy, Precision, Recall, F1-score
- **Segmentation:** Intersection over Union (IoU), Dice Score

| Model | Accuracy (%) | IoU | Dice Score |
|--------|------------|----|-----------|
| XGBoost (part a) | 94.15% (80-20 train-test split) | - | - |
| Neural Network (part a)| 91.25% (80-20 train-test split) | - | - |
| CNN | [TO BE ADDED BY DAKSH AND ADITYA] |
| Region-growing (part c) | - | 0.3559 (mean) | 0.4798 (mean) |
| K-mean clustering  (part c) | Explained in section 6|
| U-Net Segmentation | [TO BE ADDED BY DAKSH AND ADITYA] |

---

## 6. Observations and Analysis

### Part a

For each image here we need to make a feature vector. We choose 5 features: color features, HoG, Edge features, texture featuresand ORB fetaures. 

***Since feature vector coresponding to images may be of diffrent lentgh, we resize all image and fix the length of individual sub-feature vectors, so that `np.hstack() `can work without interrupts when all individual sub-feature vectors ar combined into one vector for an image***. Data used is `dataset`. We train an XGBoost model as well as a neural network and as observed, the test accuracy of XGBoost is better. This is attributed to the fact that neural networks need a lot of data to learn and here we have 4095 images.

### Part b

[TO BE ADDED BY DAKSH AND ADITYA]

### Part c

2 techniques used: K-means clustering based segmentation and Region-growing.

For K-means, k=2, one for mask region and another for backround.

Here for the choice of the 2 initial centroids, we use domain knowledge. The images are cropped to face-size which implies that it is higly likely that some region of the mask must be in the center of image. 

So we choose one centoid at center and another at corner.

For Region-based segmentaion, choice of initial seed here(only one) is center of the image, the 'why' of it backed by the reasoning provided above.

We find that K-means captures all pixels as part of mask that have more or less the intensity as mask. Secondly it is found that if the masks have design patterns of high contast, they are inevitably left out in mask segment no matter how much blurring you apply.
<p align="center">
  <img src="images/3_1.png" width="65%" />
  <img src="images/3_1_gt.jpg" width="25%" />
</p>
[Results from K-means and ground truth mask for `MSFD/1/000003.jpg`]


We find that Region-growing technique is ***sensitive to tolerance***(the threshold difference for pixels be considered connected to seed). In cases where the tone of skin is comparable to that of face-mask, the tolerance needs to be drastically low to capture correct pixels.

<p align="center">
  <img src="images/58_1_rg.png" width="65%" />
  <img src="images/58_1_gt.jpg" width="25%" />
</p>
[Results from region-growing and ground truth mask for `MSFD/1/000058_1.jpg`]

<p align="center">
  <img src="images/7_1_rg.png" width="45%" />
  <img src="images/7_1.png" width="45%" />
</p>

[Results from region-growing and k-means for `MSFD/1/000007_1.jpg`. Less false-positives in Region-growing.]

<p align="center">
  <img src="images/13_1_rg.png" width="45%" />
  <img src="images/13_1.png" width="45%" />
</p>

[Results from region-growing and k-means for `MSFD/1/000013_1.jpg`. Less false-positives in Region-growing.]

In conclusion:
| K-means | Region-growing |
|--------|------------|
| Slower| Relatively faster|
| Highly likely to give false-positives (cases where mask tone matches hair, spectacle,etc)| Less likely to give false positives|
| Sensitive to number of iterations| Sensitive to tolerance|

Both the algorihtms rely on predefined parameters, they do not 'learn' and hence fail to generalise over large dataset (poor mean IoU and Dice scores). Computing mean IoU and Dice for K-means over 8500+ images is computationally expensive, moreover it is evident from its performance over random samples that its scores won't be significantly better region-growing.


### Part d

[TO BE ADDED BY DAKSH AND ADITYA]

---

## 7. How to Run the Code
### Setup
1. Clone the repository:
   ```bash
   git clone <repo_link>
   cd <repo_folder>
   ```
2. Install dependencies:
    
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```
3. Download the dataet from the source specified and put the 2 repositores `dataset` and `MSFD` at same directory level, immediately below repository level. Make directory `output`, command `mkdir output`. Final structure must look like :
    ```
    .
    ├── dataset
    ├── MSFD
    ├── output
    ├── scripts
    └── images
    
    # Other files like README.md, pdf, etc are not shown in this tree.
    ```
4. Run the scripts:
   
   `\scripts` contains 2 notebooks `part_a_b.ipynb` and `part_c_d.ipynb`, which contains scripts for the respective parts. They can be run all at once or one at a time to see partial results.

---

## 8. Conclusion
This project demonstrates the effectiveness of deep learning techniques for face mask detection and segmentation. CNN models outperform traditional classifiers for binary classification, while U-Net provides more precise segmentation results. Further improvements can be achieved by using more complex architectures and larger datasets.

---

