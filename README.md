# Identifying Impressonist Painting Using Convolutional Neural Networks
Hyun Ko, Griffin McCauley, Eric Tria

## Project Overview

Using a dataset from Kaggle (https://www.kaggle.com/datasets/delayedkarma/impressionist-classifier-data), the aim of this project was to develop a deep learning ensemble model to swiftly and accurately predict the artist responsible for a given Impressionist work. Through exploring a variety of pre-trained models such as GoogLeNet, ResNet, and EfficientNet and conducting a number of optimization techniques such as hyperparameter tuning and data augmentation, we hoped to produce a model capable of generating reliable and robust results. While this approach was successful in the sense that it yielded a final ensemble with a test accuracy of 88.5%, more work will need to be done to fully understand how specific stylistic qualities embedded in the paintings can be associated with specific artists.

## Repository Manifest

* architectures    - this directory contains the architectures for the three models included in our ensemble
    * efficientnet.py
    * googlenet.py
    * resnet.py
* tuning    - this directory contains the notebooks used to perform the hyperparameter tuning experiments
    * Deep_Impressionist_Efficientnet.ipynb
    * GoogLeNet_Trial.ipynb
* .gitignore
* Deep_Impressionist_Ensemble.ipynb
* README.md
