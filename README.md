OpenPCDet ROS2 Running and Visualization
---

Effect pictures:


Test computer and System:

- Desktop setting: i5-10400, GPU 3080, CUDA 11.3
- System setting: Ubuntu 22.04, ROS2 Humble (Python 3.10)
- Test Date: 2025/07/25

If you don't want to destroy your env/change your system version, please directly <a href="#Docker">check the docker and build dockerfile</a>. But you still have to have a computer with NVIDIA-GPU and can install cuda. Check [Chinese blog Ubuntu下的NVIDIA显卡【驱动&CUDA 安装与卸载】](https://www.cnblogs.com/kin-zhang/p/17007246.html) 

Dataset: [2011_09_26_drive_0005](https://www.cvlibs.net/datasets/kitti/raw_data.php), ([synced+rectified data] [calibration]) 
tracking Label:[2011_09_26_drive_0005](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip) [0000.txt]
PS: Login account

## Build & RUN

Dependencies:

ros2_numpy:https://github.com/YoushaaMurhij/ros2_numpy.git

```bash
sudo apt install ros-humble-vision-msgs
pip install pyquaternion
```

And install the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) in the env
```bash
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install spconv-cu113
sudo apt-get install python-setuptools

pip install pyquaternion install numpy==1.23 pillow==8.4
# have some potential risks
sudo chown -R $USER /usr/local/lib/python3.10/
sudo chown -R $USER /usr/local/bin/
# have some potential risks

git clone https://github.com/Zack0307/openpcdet_ros.git
cd OpenPCDet && python3 setup.py develop
```

Run:
```bash
mkdir -p ~/workspace/OpenPCDet_ws/src
cd ~/workspace/OpenPCDet_ws/src
git clone https://github.com/zack0307/OpenPCDet_ros.git
cd .. && colcon build
```

One more step check your [Config file](launch/config.yaml)

```bash
source /opt/ros/humble/setup.bash
# before this step please change the model path in config file
ros2 launch openpcdet 3d_object_detector.launch
```

## Issue/TODO

- [ ] finish README.md`

- [ ] CaDDN模組 需要深度圖



## Other infos

This repo has a Chinese blog also to read through [【点云检测】OpenPCDet 教程系列 [1] 安装 与 ROS运行](https://www.cnblogs.com/kin-zhang/p/17002980.html)
kitti dataset: https://www.cvlibs.net/datasets/kitti/raw_data.php (raw data)

### Acknowledgement

All methods and models are from: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

Reference codes:

1. The first version of openpcdet-ros is from: [Cram3r95/OpenPCDet-ROS](https://github.com/Cram3r95/OpenPCDet-ROS)

2. For 3d box marker drawing: [Youtube AI葵](https://www.youtube.com/watch?v=nIiqo3ZuFCc&list=PLDV2CyUo4q-L4YlXUWDytZPz9a8cAWXST&index=11&ab_channel=AI%E8%91%B5) and [his code](https://github.com/kwea123/ROS_notes)


