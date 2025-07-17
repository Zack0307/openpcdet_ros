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
            self.cam_sub = self.create_subscription(Image, '/cam_sub_image', cam_subscribe_callback,  10)
            self.PCL = self.create_subscription(PointCloud2, '/kitti_pcl', pointcloud_topic[0], lidar_callback, 1)
            self.marker = self.create_publisher(Marker, '/marker_gaze', 10)
            self.box = self.create_publisher(MarkerArray, '/kitti_3d_box', 10)
            self.proc_1 = Processor_ROS(self, cfg_root, model_path)
            self.proc_1.initialize()
            self.timer = self.create_timer(0.1, self.publish_data)  # 每 0.1 秒呼叫一次
    
    def publish_data(self):
            
            image_publish(self.frame, self.cam_pub)
            cam_publish(self.frame, self.PCL, self.get_clock())
            lidar_callback(self.PCL, self.proc_1)
            self.proc_1.set_pub_rviz(self.box)
            # publish_marker_array(self.marker, self.get_clock())
            # publish_imu(self.frame, self.imu, self.get_clock())
            # publish_gps(self.frame, self.gps, self.get_clock())
            # publish_3d_box(self.frame, self.box, self.get_clock())
            # self.frame += 1
            # if self.frame >= data_number:
            #     self.frame = 0
            self.get_logger().info(f'Publishing')

class Processor_ROS:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None
        self.pub_rviz = None
        self.no_frame_id = True
        self.rate = RATE_VIZ

    def set_pub_rviz(self, box3d_pub, marker_frame_id = 'velodyne'):
        self.pub_rviz = Draw3DBox(box3d_pub, marker_frame_id, self.rate)
    
    def set_viz_frame_id(self, marker_frame_id):
        self.pub_rviz.set_frame_id(marker_frame_id)

    def initialize(self):
        self.read_config()
        
    def read_config(self):
        config_path = self.config_path
        cfg_from_yaml_file(self.config_path, cfg)
        self.logger = common_utils.create_logger()
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path("/home/kin/workspace/OpenPCDet/tools/000002.bin"),
            ext='.bin')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        print("Model path: ", self.model_path)
        self.net.load_params_from_file(filename=self.model_path, logger=self.logger, to_cpu=True)
        self.net = self.net.to(self.device).eval()

    def get_template_prediction(self, num_samples):
        ret_dict = {
            'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
            'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
            'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
            'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
            'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
        }
        return ret_dict

    def run(self, points, frame):
        t_t = time.time()
        num_features = 4 # X,Y,Z,intensity       
        self.points = points.reshape([-1, num_features])
        assert self.points.shape[0] > 0, "No points in lidar data, please check your lidar message."
        print(f"Total points: {self.points.shape[0]}")
        timestamps = np.empty((len(self.points),1))
        timestamps[:] = frame

        input_dict = {
            'points': self.points,
            'frame_id': frame,
        }

        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        torch.cuda.synchronize()
        t = time.time()

        pred_dicts, _ = self.net.forward(data_dict)
        
        torch.cuda.synchronize()
        inference_time = time.time() - t
        inference_time_list.append(inference_time)
        mean_inference_time = sum(inference_time_list)/len(inference_time_list)

        boxes_lidar = pred_dicts[0]["pred_boxes"].detach().cpu().numpy()
        scores = pred_dicts[0]["pred_scores"].detach().cpu().numpy()
        types = pred_dicts[0]["pred_labels"].detach().cpu().numpy()

        pred_boxes = np.copy(boxes_lidar)
        pred_dict = self.get_template_prediction(scores.shape[0])

        pred_dict['name'] = np.array(cfg.CLASS_NAMES)[types - 1]
        pred_dict['score'] = scores
        pred_dict['boxes_lidar'] = pred_boxes

        return scores, boxes_lidar, types, pred_dict

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def main(args=None):
    rclpy.init(args=args)
    node = PcdetNode()
    rclpy.spin(node)                                
    rclpy.shutdown()

if __name__ == "__main__":
    main()




