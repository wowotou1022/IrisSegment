# Iris-segmentation
Iris segmentation using feature channel optimization for noisy environments

# Requirements
Tensorflow 1.4.0  
Keras 2.2.0  
Python 3.5
# Results
R stands for recall rate, P stands for precision, and F-measure is a combination of the two. (unit: %)

 Dataset | F | R | P |Error rate(%)
 ---- | ----- | ------ | ------  | ------  
CASIA | 98.11 | 97.96 |98.27 | 0.81
 IITD | 97.84| 97.78 |97.91 |0.98
 
# Data
We use CASIA V4.0 Interval (Abbr. CASIA) dataset, and the IIT Delhi v1.0 (Abbr. IITD) dataset. The weights we provide (Model/CAV or Model/IITD) are the training results for Gaussian noise. We provide a noisy dataset with Gaussian noise. When training the model, we use a '.npy' file of the dataset.

# Run on GPU

* To test the model, you can run
```
python test_predict.py
```
 In 'Model/CAV', you can see the segmentation results.

* In order to measure the performance of the model with the RPF metric, you can run
```
python error_RPF.py
```  

* To train the model, you can run
```
python model.py
```
 The training results will be written to Model/CAV
# Citation
Please cite this paper if you think it is useful for you.  
Title: Iris segmentation using feature channel optimization for noisy environments  
Author: Kangli Hao · Guorui Feng · Yanli Ren · Xinpeng Zhang
