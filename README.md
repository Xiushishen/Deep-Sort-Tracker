# Deep-Sort-Tracker

This repository contains the code for a deep-learning based multi-object tracker (MoT) to track the pedestrian. With that being said, this MoT can be derived to track other objects like cars, animals, etc. There are detection and tracking parts in the MOT. The tracking part uses kamilar filter and Hungarian association for accurate tracking and data association, while the detection part is based on YOLOV7, which is well-trained for object detection. The tracking model is trained on [Market1501 dataset](https://www.kaggle.com/datasets/pengcw1/market-1501), which illustrated more than 90% accuracy. The YOLOV7 offical has provided us with different size of well-trained weights, so we did not train detection model on our own. These codes were developed on Nvidia Jetson Orin which supports cuda acceleration to make the detection and tracking faster.

# Dependency
1 Create a virtual environment with Python >=3.8  
```
conda create -n sort python=3.8    
conda activate dsort  
```

2 Install pytorch = 2.1.0 to be suitable on Jetson Orin.
```
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
python3 -m pip install --upgrade pip;
python3 -m pip install numpy==1.19.5
python3 -m pip install --no-cache $TORCH_INSTALL
```

3 Install all dependencies
```
pip install -r requirements.txt
```

# Dataset

You can use the following commands to download the codes and prepare Market1501 dataset.
```
git@github.com:Xiushishen/Deep-Sort-Tracker.git
bash ./data_getter.sh
cd deep_sort/deep
python prepare.py
```
Now, you should see a folder named "Market1501/pytorch" including all the processed dataset for tracking model training and testing.

# Weights

The weights for tracking and detection model are already included in these repository. However, feel free to train new weights for better detection result and download other [YOLOV7 weights](https://github.com/ultralytics/yolov5).

# Run and Test
There is a demo video filed named "video.mp4" in the main folder. You can run main.py file to test tracking result on this video. The result will be stored under output folder.
```
# on video file
python main.py --input_path [VIDEO_FILE_NAME]

example: python main.py --input_path video.mp4

# on webcam 
python main.py --cam 0 --display

```
