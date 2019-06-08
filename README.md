# cnn-facefeature-detection

## Overview

This is a quick 4-layers single pool CNN for detecting 15 face keypoints. 

## Dataset

The dataset is using Montreal dataset. It can be downloaded [here](https://www.kaggle.com/c/facial-keypoints-detection)

## Usage

1. Git clone
2. Install all dependancies:
  * Python 3.5 (or above)
  * Numpy
  * TensorFlow (1.13.x above)
  * Matplotlib
  * Pandas
3. Download the dataset
4. Configure test.py, adjust path
5. Build the model. Visualization is provided.

## Accuracy

Accuracy is rather low, lot of missing facial keypoints in the training data. Might need better dataset.
  
