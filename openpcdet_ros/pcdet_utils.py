import rclpy
import cv2 as cv
import math
import time
import glob
import torch
import os
import pandas as pd
import ros2_numpy
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2 as pcl2
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.node import Node
from cv_bridge import CvBridge
from PIL import Image as PILImage
from pathlib import Path
from openpcdet_ros.draw3d_utils import *
from openpcdet_ros.__init__ import *
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cv_bridge = CvBridge()
Image_DATA_PATH = '/home/zack/kitti/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync'
data_number = len(os.listdir(os.path.join(Image_DATA_PATH, 'image_02/data')))

# OpenPCDet imports
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
FRAME_ID = 'map'  # 依你的座標框架而定
TRACKING_COLUMN_NAMES = ['frame', 'track id', 'type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 
              'height', 'width', 'length', 'loc_x', 'loc_y', 'loc_z', 'rot_y' ]

#data
df = pd.read_csv('/home/zack/kitti/data_tracking_label_2/training/label_02/0000.txt', header=None, sep=' ')
df.columns = TRACKING_COLUMN_NAMES 
df.loc[df.type.isin(['Van','Truck','Tram']), 'type'] = 'Car'
df = df[df.type.isin(['Car','Pedestrian','Cyclist'])]

#config file
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
with open(f"{BASE_DIR}/launch/config.yaml", 'r') as f:
    try:
        para_cfg = yaml.safe_load(f, Loader=yaml.FullLoader)
    except:
        para_cfg = yaml.safe_load(f)

cfg_root = para_cfg["cfg_root"]
model_path = para_cfg["model_path"]
threshold = para_cfg["threshold"]
pointcloud_topic = para_cfg["pointcloud_topic"]
RATE_VIZ = para_cfg["viz_rate"]
inference_time_list = []

def cam_publish(frame, cam_pub):
    cap = cv.VideoCapture(0)
    ret, frame = cap.read()
          
    if ret == True:
        imgmsg = cv_inference(frame)
        cam_pub.publish(cv_bridge.cv2_to_imgmsg(frame, encoding = 'bgr8'))

def cam_subscribe_callback(self, data):
        # Convert ROS Image message to OpenCV image
        current_frame = cv_bridge.imgmsg_to_cv2(data, desired_encoding = 'bgr8')
        # Display image
        cv.imshow("camera", current_frame)
        cv.waitKey(1)

def image_publish(frame, cam_pub):
    img = cv.imread(os.path.join(Image_DATA_PATH, 'image_02/data/%010d.png'%frame))

    boxes = np.array(df[df.frame==frame][['bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom']])
    types = np.array(df[df.frame==frame]['type'])
 
    for typ, box in zip(types, boxes):
        top_left = int(box[0]), int(box[1])
        bottom_right = int(box[2]), int(box[3])
        cv.rectangle(img, top_left, bottom_right, DETECTION_COLOR_MAP[typ], 2)
        cv.putText(img, typ, (int(box[0]), int(box[1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cam_pub.publish(cv_bridge.cv2_to_imgmsg(img, encoding='bgr8'))

def publish_pcl(frame, PCL, clock):
    pcl = np.fromfile(os.path.join(Image_DATA_PATH, 'velodyne_points/data/%010d.bin' % frame), dtype=np.float32).reshape(-1, 4)
    header = Header()
    header.stamp = clock.now().to_msg()
    header.frame_id = FRAME_ID
    PCL.publish(pcl2.create_cloud_xyz32(header, pcl[:, :3]))

    
def cv_inference(frame):
        # img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # img = PILImage.fromarray(img)
        # img = transforms(img).unsqueeze(0)
        # if torch.cuda.is_available():
        #     img = img.to(device)
        # with torch.no_grad():
        #     output = model(img)['out'][0]   #['out']：取出主要輸出張量
        # output_predictions = output.argmax(0)
        # output_predictions = output_predictions.byte().cpu().numpy()
        # output_predictions = cv.applyColorMap(output_predictions, cv.COLORMAP_JET)
        # output_predictions = cv.cvtColor(output_predictions, cv.COLOR_RGB2BGR)
        # img_tomsg = cv_bridge.cv2_to_imgmsg(output_predictions, encoding='bgr8')
        # return img_tomsg
        pass

def lidar_callback(msg, proc_1):
    select_boxs, select_types = [],[]
    if proc_1.no_frame_id:
        proc_1.set_viz_frame_id(msg.header.frame_id)
        print(f"{bc.OKGREEN} setting marker frame id to lidar: {msg.header.frame_id} {bc.ENDC}")
        proc_1.no_frame_id = False

    frame = msg.header.seq # frame id -> not timestamp
    msg_cloud = ros2_numpy.point_cloud2.pointcloud2_to_array(msg)
    np_p = get_xyz_points(msg_cloud, True)
    scores, dt_box_lidar, types, pred_dict = proc_1.run(np_p, frame)
    for i, score in enumerate(scores):
        if score>threshold:
            select_boxs.append(dt_box_lidar[i])
            select_types.append(pred_dict['name'][i])
    if(len(select_boxs)>0):
        proc_1.pub_rviz.publish_3dbox(np.array(select_boxs), -1, pred_dict['name'])
        print_str = f"Frame id: {frame}. Prediction results: \n"
        for i in range(len(pred_dict['name'])):
            print_str += f"Type: {pred_dict['name'][i]:.3s} Prob: {scores[i]:.2f}\n"
        print(print_str)
    else:
        print(f"\n{bc.FAIL} No confident prediction in this time stamp {bc.ENDC}\n")
    print(f" -------------------------------------------------------------- ")


def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    """
    Create a sensor_msgs.PointCloud2 from an array of points.
    """
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
        # PointField('i', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg

