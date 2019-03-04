# MIDA-pytorch
**A pytorch implementation of "MIDA: Multiple Imputation using Denoising Autoencoders"   
Paper : https://arxiv.org/abs/1705.02737**

## Summary of the paper
1. Doing imputation with Overcomplete AutoEncoder for missing data
2. Using complete data for training
3. Dropout is used to generate artificial missings in the training session
4. Experimenting with two missing methods(MCAR/MNAR)
5. Simple but good

## Requirements
numpy==1.14.2   
pandas==0.22.0   
scikit-learn==0.19.1   
pytorch==1.0.0   

## Data
In the paper, 15 publicly available datasets used.   
In this code, only 'Boston Housing' data is used among 15.   
http://math.furman.edu/~dcs/courses/math47/R/library/mlbench/html/BostonHousing.html
