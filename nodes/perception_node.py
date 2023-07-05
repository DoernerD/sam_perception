#!/usr/bin/env python3
"""
Perception Node to run feature detection and estimation during docking.
"""

import os
import time
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from scipy.spatial.transform import Rotation
import tf2_ros

import rospkg
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose
from std_msgs.msg import Float64

from sam_perception.perception_ros_utils import vectorToPose, featurePointsToMsg
from sam_perception.perception import Perception
from sam_perception.feature_model import FeatureModel
from sam_perception.camera_model import Camera


class PerceptionNode(object):
    """
    Node to interact with the perception algorithms.
    Publishes an estimated pose of the docking station
    """
    def __init__(self):
        # Read external parameters
        feature_model_yaml = rospy.get_param("~feature_model_yaml")
        self.hz = rospy.get_param("~hz")
        hats_mode = rospy.get_param("~hats_mode", "valley")
        self.base_frame = rospy.get_param("~base_frame", "sam/base_link")
        self.camera_frame = rospy.get_param("~camera_frame", "camera_link")

        feature_model_yaml_path = os.path.join(rospkg.RosPack().get_path("sam_perception"),
                                               "feature_models/{}".format(feature_model_yaml))
        feature_model = FeatureModel.fromYaml(feature_model_yaml_path)

        self.camera = None
        self.image_msg = None
        self.bridge = CvBridge()
        self.camera_pose_vector = None  # This is some legacy code, needed, but stays none.
        self.estimated_ds_pose = None
        self.got_image = False
        self.estimation_error = 0.
        self.pose_aquired = False
        self.base2camera_mat = []

        # Messages to be published
        self.processed_image_drawing = Image()
        self.processed_image = Image()
        self.pose_image = Image()
        self.pose_array = PoseArray()
        self.estimated_pose = PoseWithCovarianceStamped()
        self.camera_pose = PoseWithCovarianceStamped()

        # Return values from the perception module
        self.processed_img = None
        self.pose_img = None

        # Initialize cameras. Has to be done before everything else.
        self.camera_info_topic = rospy.get_param("~camera_info_topic")
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo,
                                                self.camera_info_cb)

        while not rospy.is_shutdown() and self.camera is None:
            print("Waiting for camera info to be published")
            rospy.sleep(1)

        self.perception = Perception(self.camera, feature_model, hatsMode=hats_mode)

        # Initialize tf listener
        tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tf_buffer)

        try:
            rospy.loginfo("Waiting for transforms")
            base2camera_tf = tf_buffer.lookup_transform(self.camera_frame, self.base_frame,
                                               rospy.Time(0), rospy.Duration(60))
            self.base2camera_mat = self.transform_to_matrix(base2camera_tf)
            rospy.loginfo("[PN]: got transform %s to %s" % (self.base_frame, self.camera_frame))

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr("[PN]: Could not lookup transform %s to %s" 
                         % (self.base_frame, self.camera_frame))

        # Topics
        self.image_topic = rospy.get_param("~image_topic")
        self.image_processed_topic = rospy.get_param("~image_processed_topic")
        self.image_processed_drawing_topic = rospy.get_param("~image_processed_drawing_topic")
        self.image_pose_topic = rospy.get_param("~image_pose_topic")
        self.estimated_pose_topic = rospy.get_param("~estimated_pose_topic")
        self.estimated_camera_pose_topic = rospy.get_param("~estimated_camera_pose_topic")
        self.estimated_poses_array_topic = rospy.get_param("~estimated_poses_array_topic")   # FIXME: What's the difference to estimated_pose_topic?
        self.estimation_error_topic = rospy.get_param("~estimation_error_topic")

        # Subscribers
        rospy.Subscriber(self.image_topic, Image, self.image_cb)

        # Publishers
        self.image_processed_pub = rospy.Publisher(self.image_processed_topic, Image, queue_size=1)
        self.image_processed_drawing_pub = rospy.Publisher(self.image_processed_drawing_topic,
                                                           Image, queue_size=1)
        self.image_pose_pub = rospy.Publisher(self.image_pose_topic, Image, queue_size=1)
        self.estimated_pose_pub = rospy.Publisher(self.estimated_pose_topic,
                                                  PoseWithCovarianceStamped, queue_size=1)
        self.estimated_camera_pose_pub = rospy.Publisher(self.estimated_camera_pose_topic,
                                                         PoseWithCovarianceStamped, queue_size=10)
        self.estimated_poses_array_pub = rospy.Publisher(self.estimated_poses_array_topic,
                                                         PoseArray, queue_size=1)
        self.estimation_error_pub = rospy.Publisher(self.estimation_error_topic, Float64, queue_size=1)

        rate = rospy.Rate(self.hz)

        # Run
        while not rospy.is_shutdown():
            self.run_perception()

            self.publish_messages()

            rate.sleep()

    #region Callbacks
    def camera_info_cb(self, msg):
        """
        Use either K and D or just P
        https://answers.ros.org/question/119506/what-does-projection-matrix-provided-by-the-calibration-represent/
        https://github.com/dimatura/ros_vimdoc/blob/master/doc/ros-camera-info.txt
        """
        # Using only P (D=0), we should subscribe to the rectified image topic
        # camera = Camera(cameraMatrix=np.array(msg.P, dtype=np.float32).reshape((3,4))[:, :3],
                        # distCoeffs=np.zeros((1,4), dtype=np.float32),
                        # resolution=(msg.height, msg.width))
        # Using K and D, we should subscribe to the raw image topic
        self.camera = Camera(cameraMatrix=np.array(msg.K, dtype=np.float32).reshape((3,3)),
                       distCoeffs=np.array(msg.D, dtype=np.float32),
                       resolution=(msg.height, msg.width))

        # We only want one message
        self.camera_info_sub.unregister()


    def transform_to_matrix(self, transform):
        """
        Pose msg to 4x4 numpy array.
        """
        if transform._type == 'geometry_msgs/TransformStamped':
            transform = transform.transform

        trans = (transform.translation.x,
             transform.translation.y,
             transform.translation.z)
        quat = (transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w)

        rot_matrix = Rotation.from_quat(quat)
        translation = np.array([[trans[0]],[trans[1]],
                                [trans[2]]])

        transformation = np.eye(4)
        transformation[0:3,0:3] = rot_matrix.as_matrix()
        transformation[0:-1,-1:] = translation

        return transformation


    def image_cb(self, msg):
        """
        Callback for the image topic.
        """
        self.image_msg = msg
        self.got_image = True

    #endregion

    def run_perception(self):
        """
        Main loop for the perception node
        The if-statement is to ensure that we only run it when we have a new image.
        """
        if self.got_image:
            try:
                image_mat = self.bridge.imgmsg_to_cv2(self.image_msg, 'bgr8')
            except CvBridgeError as error_msg:
                print(error_msg)
            else:
                self.estimated_ds_pose, self.pose_aquired = self.estimate_pose(image_mat, self.estimated_ds_pose)

                # TODO: Do the outlier rejection here. If it fails, set the pose_aquired flag to false.
                self.transform_to_messages()

        self.got_image = False


    def estimate_pose(self, img_mat, estimated_ds_pose):
        """
        Update step for the estimation
        NOTE: We don't compute the covariance of the pose right now. The functions are there,
        but for computational reasons, we skip that part. 
        """
        estimation_start = time.time()

        (ds_pose, pose_aquired, _,
         self.processed_img, self.pose_img) = self.perception.estimatePose(img_mat, estimated_ds_pose,
                                                                 self.camera_pose_vector,colorCoeffs=None, calcCovariance=True)

        estimation_finished = time.time()

        self.annotate_images(estimation_start, estimation_finished)

        return ds_pose, pose_aquired


    def annotate_images(self, estimation_start, estimation_finished):
        """
        Function to print FPS on the image.
        TODO: Is this really necessary?
        """
        elapsed = estimation_finished - estimation_start
        virtual_frequency = 1./elapsed
        hz = min(self.hz, virtual_frequency)

        cv.putText(self.pose_img, "FPS {}".format(round(hz, 1)),
                   (int(self.pose_img.shape[1]*4/5), 25), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   color=(0,255,0), thickness=2, lineType=cv.LINE_AA)

        cv.putText(self.pose_img, "Virtual FPS {}".format(round(virtual_frequency, 1)),
                   (int(self.pose_img.shape[1]*4/5), 45), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   color=(0,255,0), thickness=2, lineType=cv.LINE_AA)


    def transform_to_messages(self):
        """
        Convert all images into publishable formats.
        Transform poses into sam/base_link
        """
        time_stamp = rospy.Time.now()

        self.processed_image_drawing = self.bridge.cv2_to_imgmsg(self.processed_img)
        self.pose_image = self.bridge.cv2_to_imgmsg(self.pose_img)
        self.processed_image = self.bridge.cv2_to_imgmsg(self.perception.featureExtractor.img)

        if self.pose_aquired:
            self.pose_array = featurePointsToMsg("docking_station_link",
                                                self.perception.featureModel.features, time_stamp)

            estimated_pose_docking_frame_pose = vectorToPose(self.image_msg.header.frame_id,
                                            self.estimated_ds_pose.translationVector,
                                            self.estimated_ds_pose.rotationVector,
                                            self.estimated_ds_pose.covariance, time_stamp)

            estimated_pose_docking_frame_mat = self.pose_to_matrix(estimated_pose_docking_frame_pose.pose)
            estimated_pose_base_frame_mat = self.express_in_base(estimated_pose_docking_frame_mat, self.base2camera_mat)
            self.estimated_pose = self.matrix_to_pose(estimated_pose_base_frame_mat, self.estimated_ds_pose.covariance,
                                                      time_stamp)

            self.camera_pose = vectorToPose("docking_station_link",
                                            self.estimated_ds_pose.camTranslationVector,
                                            self.estimated_ds_pose.camRotationVector,
                                            self.estimated_ds_pose.calcCamPoseCovariance(),
                                            time_stamp)

            self.estimation_error = self.estimated_ds_pose.rmse

    def pose_to_matrix(self, pose):
        """
        Pose msg to 4x4 numpy array.
        """
        rot_matrix = Rotation.from_quat([pose.pose.orientation.x,
                                     pose.pose.orientation.y,
                                     pose.pose.orientation.z,
                                     pose.pose.orientation.w])
        translation = np.array([[pose.pose.position.x],[pose.pose.position.y],
                                [pose.pose.position.z]])

        transformation = np.eye(4)
        transformation[0:3,0:3] = rot_matrix.as_matrix()
        transformation[0:-1,-1:] = translation

        return transformation

    def express_in_base(self, tf_to_ds, tf_to_base):
        """
        Express the docking station vector in sam/base_link frame.
        """
        # Create rotation matrices and translation vectors
        R_ds_to_cam = tf_to_ds[0:3,0:3]
        t_ds_to_cam = tf_to_ds[0:3,3]

        # Analytical inversion
        invers_tf_to_ds = np.zeros((4,4))
        invers_tf_to_ds[0:3,0:3] = R_ds_to_cam.transpose()
        invers_tf_to_ds[0:3,3] = -R_ds_to_cam.transpose().dot(t_ds_to_cam)
        invers_tf_to_ds[3,3] = 1.
        final_transformation = np.matmul(tf_to_base, invers_tf_to_ds)

        return final_transformation


    def matrix_to_pose(self, matrix, covariance, time_stamp):
        """
        Convert a matrix into a pose
        FIXME: Hard coded frame ID
        """
        # Get rotation and translation from matrix
        R = Rotation.from_matrix(matrix[0:3,0:3])
        quat = Rotation.as_quat(R)
        tran = matrix[0:3,3]

        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = '/sam/base_link'
        pose.header.stamp = time_stamp
        pose.pose.pose.position.x = tran[0]
        pose.pose.pose.position.y = tran[1]
        pose.pose.pose.position.z = tran[2]
        pose.pose.pose.orientation.x = quat[0]
        pose.pose.pose.orientation.y = quat[1]
        pose.pose.pose.orientation.z = quat[2]
        pose.pose.pose.orientation.w = quat[3]
        pose.pose.covariance = list(np.ravel(covariance))

        return pose



    def publish_messages(self):
        """
        Collect all messages and publish them
        """
        self.image_processed_drawing_pub.publish(self.processed_image_drawing)
        self.image_processed_pub.publish(self.processed_image)
        self.image_pose_pub.publish(self.pose_image)

        # Publish those only when you get a new one.
        if self.pose_aquired:
            self.estimation_error_pub.publish(self.estimation_error)
            self.estimated_poses_array_pub.publish(self.pose_array)
            self.estimated_pose_pub.publish(self.estimated_pose)
            self.estimated_camera_pose_pub.publish(self.camera_pose)


if __name__ == '__main__':

    rospy.init_node('perception_node')
    try:
        PerceptionNode()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch perception node")
