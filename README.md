# Deep-Sort-Tracker

This repository contains the code for a deep-learning based multi-object tracker (MoT) to track the pedestrian. With that being said, this MoT can be derived to track other objects like cars, animals, etc. There are detection and tracking parts in the MOT. The tracking part uses kamilar filter and Hungarian association for accurate tracking and data association, while the detection part is based on YOLOV7, which is well-trained for object detection. The tracking model is trained on [Market1501 dataset](https://www.kaggle.com/datasets/pengcw1/market-1501), which illustrated more than 90% accuracy. The YOLOV7 offical has provided us with different size of well-trained weights, so we did not train detection model on our own. These codes were developed on Nvidia Jetson Orin which supports cuda acceleration to make the detection and tracking faster.

# Dependency

# Dataset

You can use the following commands to download the codes and prepare Market1501 dataset.
```
git@github.com:Xiushishen/Deep-Sort-Tracker.git
bash ./data_getter.sh
cd deep_sort/deep
python prepare.py
```
Now, you should see a folder named "Market1501/pytorch" including all the processed dataset for tracking model training and testing.
