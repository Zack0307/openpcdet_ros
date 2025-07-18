import rclpy
import cv2
import math
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.node import Node
from cv_bridge import CvBridge
from PIL import Image as PILImage
from openpcdet_ros.pcdet_utils import *

class PcdetNode(Node):
    def __init__(self):
            super().__init__('CaDDN_node')
            self.frame = 0
            self.cam_pub = self.create_publisher(Image, '/camera_pub_image', 10)
            self.cam_sub = self.create_subscription(Image, '/camera_sub_image', cam_subscribe_callback,  10)
            self.PCL = self.create_publisher(PointCloud2, '/kitti_pcl', 10)
            self.marker = self.create_publisher(Marker, '/marker_gaze', 10)
            self.box = self.create_publisher(MarkerArray, '/kitti_3d_box', 10)
            # self.seg_pub = self.create_publisher(Image, '/seg_pub_image', 10)
            # self.cap = cv2.VideoCapture(0)
            # self.cvbr = CvBridge()
            self.model.eval()
            self.timer = self.create_timer(0.1, self.publish_video)  # 每 0.1 秒呼叫一次
    
    def publish_video(self):
            
            image_publish(self.frame, self.cam_pub)
            publish_pcl(self.frame, self.get_clock())
            cam_publish(self.frame, self.PCL)
            # publish_marker_array(self.marker, self.get_clock())
            # publish_imu(self.frame, self.imu, self.get_clock())
            # publish_gps(self.frame, self.gps, self.get_clock())
            # publish_3d_box(self.frame, self.box, self.get_clock())
            # self.frame += 1
            # if self.frame >= data_number:
            #     self.frame = 0
            self.get_logger().info(f'Publishing')


def main(args=None):
    rclpy.init(args=args)
    node = PcdetNode()
    rclpy.spin(node)                                
    rclpy.shutdown()

if __name__ == "__main__":
    main()




