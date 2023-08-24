# Identifying Impressonist Painters Using Convolutional Neural Networks
Hyun Ko, Griffin McCauley, Eric Tria

## Project Overview

Using a dataset from Kaggle (https://www.kaggle.com/datasets/delayedkarma/impressionist-classifier-data), the aim of this project was to develop a deep learning ensemble model to swiftly and accurately predict the artist responsible for a given Impressionist work. Through exploring a variety of pre-trained models such as GoogLeNet, ResNet, and EfficientNet and conducting a number of optimization techniques such as hyperparameter tuning and data augmentation, we hoped to produce a model capable of generating reliable and robust results. While this approach was successful in the sense that it yielded a final ensemble with a test accuracy of 88.5%, more work will need to be done to fully understand how specific stylistic qualities embedded in the paintings can be associated with specific artists.

**A video presentation outlining our methodology and results can be found [here](https://github.com/Griffin-McCauley/ImpressonistCNN/blob/main/DS%206050%20Project%20Presentation%20Video%20Recording.mp4).**

## Repository Manifest

* `architectures`    - this directory contains the architectures for the three models included in our ensemble
    * `efficientnet.py`
    * `googlenet.py`
    * `resnet.py`
* `tuning`    - this directory contains the notebooks used to perform the hyperparameter tuning experiments
    * `Efficientnet/Deep_Impressionist_Efficientnet.ipynb`
    * `GoogLeNet`
        - `GoogLeNet_BS.ipynb`
        - `GoogLeNet_DR.ipynb`
        - `GoogLeNet_LR.ipynb`
        - `GoogLeNet_WD.ipynb`
    * `ResNet/Deep_Impressionist.ipynb`
    
* .gitignore
* `DS 6050 Project Presentation Video Recording.mp4`
* `Deep_Impressionist_Ensemble.ipynb`
* `final_report.pdf`
* `README.md`

## Instructions
1. Fork the repository from Git
2. Clone your forked repository
3. Run each of the model in tuning directory, best models are already saved in `architectures` folder
4. Run `Deep_Impressionist_Ensemble.ipynb` and ensemble the results from each model 

