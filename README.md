# Deep-Sort-Tracker

This repository contains the code for a deep-learning based multi-object tracker (MoT) to track the pedestrian. With that being said, this MoT can be derived to track other objects like cars, animals, etc. There are detection and tracking parts in the MOT. The tracking part uses kamilar filter and Hungarian association for accurate tracking and data association, while the detection part is based on YOLOV7, which is well-trained for object detection. The tracking model is trained on [Market1501 dataset](https://www.kaggle.com/datasets/pengcw1/market-1501), which illustrated more than 90% accuracy. The YOLOV7 offical has provided us with different size of well-trained weights, so we did not train detection model on our own. These codes were developed on Nvidia Jetson Orin which supports cuda acceleration to make the detection and tracking faster.

# Dependency

# Dataset

Get the codes from github account:
```
git@github.com:Xiushishen/Deep-Sort-Tracker.git
```

Run bash script to download Market1501 dataset:
```
bash ./data_getter.sh
```
