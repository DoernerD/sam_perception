#!/usr/bin/env python3
"""
Perception Node to run feature detection and estimation during docking.
"""

import os
import time
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

import rospkg
import rospy
import tf.msg
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray
from std_msgs.msg import Float32

import numpy as np

from sam_perception.perception_ros_utils import vectorToPose, vectorToTransform, poseToVector, lightSourcesToMsg, featurePointsToMsg
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
        self.cv_show = rospy.get_param("~cv_show", False)
        hats_mode = rospy.get_param("~hats_mode", "valley")

        feature_model_yaml_path = os.path.join(rospkg.RosPack().get_path("sam_perception"),
                                               "feature_models/{}".format(feature_model_yaml))
        feature_model = FeatureModel.fromYaml(feature_model_yaml_path)

        self.camera = None
        self.image_msg = None
        self.bridge = CvBridge()
        self.camera_pose_vector = None  # This is some legacy code, needed for the perception, but never changed.
        self.estimated_ds_pose = None
        self.got_image = False
        self.got_camera_pose = False

        self.processed_image_drawing = Image()
        self.processed_image = Image()
        self.pose_image = Image()

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

        # Topics
        self.image_topic = rospy.get_param("~image_topic")
        self.image_processed_topic = rospy.get_param("~image_processed_topic")
        self.image_processed_drawing_topic = rospy.get_param("~image_processed_drawing_topic")
        self.image_pose_topic = rospy.get_param("~image_pose_topic")
        self.associated_image_points_topic = rospy.get_param("~associated_image_points_topic")
        self.estimated_pose_topic = rospy.get_param("~estimated_pose_topic")
        self.estimated_camera_pose_topic = rospy.get_param("~estimated_camera_pose_topic")
        self.mahal_distance_topic = rospy.get_param("~mahal_distance_topic")
        self.estimated_poses_array_topic = rospy.get_param("~estimated_poses_array_topic")   # FIXME: What's the difference to estimated_pose_topic?

        # Subscribers
        rospy.Subscriber(self.image_topic, Image, self.image_cb)

        # Publishers
        self.image_processed_pub = rospy.Publisher(self.image_processed_topic, Image, queue_size=1)
        self.image_processed_drawing_pub = rospy.Publisher(self.image_processed_drawing_topic, Image, queue_size=1)
        self.image_pose_pub = rospy.Publisher(self.image_pose_topic, Image, queue_size=1)
        self.associated_image_points_pub = rospy.Publisher(self.associated_image_points_topic, PoseArray, queue_size=1) #TODO: rename this. It's a pose array and the name is misleading.
        self.estimated_pose_pub = rospy.Publisher(self.estimated_pose_topic, PoseWithCovarianceStamped, queue_size=1)
        self.estimated_camera_pose_pub = rospy.Publisher(self.estimated_camera_pose_topic, PoseWithCovarianceStamped, queue_size=10)
        self.mahal_distance_pub = rospy.Publisher(self.mahal_distance_topic, Float32, queue_size=1)
        self.estimated_poses_array_pub = rospy.Publisher(self.estimated_poses_array_topic, PoseArray, queue_size=1)

        # FIXME: This doesn't seem good to publish.
        self.transformPublisher = rospy.Publisher("/tf", tf.msg.tfMessage, queue_size=1)

        rate = rospy.Rate(self.hz)

        # Run
        while not rospy.is_shutdown():
            self.run_perception()

            self.transform_images()

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
        camera = Camera(cameraMatrix=np.array(msg.K, dtype=np.float32).reshape((3,3)),
                       distCoeffs=np.array(msg.D, dtype=np.float32),
                       resolution=(msg.height, msg.width))
        self.camera = camera

        # We only want one message
        self.camera_info_sub.unregister()



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
                (ds_pose, pose_aquired) = self.update(image_mat, self.estimated_ds_pose)

                # FIXME: Can you do that with an try-assert?
                # Or put that in the update function itself or so.
                # Bit weird with the circle of estimated_ds_pose
                if not pose_aquired:
                    self.estimated_ds_pose = None
                else:
                    self.estimated_ds_pose = ds_pose

        self.got_image = False


    def update(self, img_mat, estimated_ds_pose):
        """
        FIXME: Lots of stuff happening here. 
        - Extract the publishing into an individual function
        - take care of all the function arguments.
        - formatting in general. 
        """
        start = time.time()

        (ds_pose, pose_aquired, candidates,
         self.processed_img, self.pose_img) = self.perception.estimatePose(img_mat, estimated_ds_pose,
                                                                 self.camera_pose_vector)

        if ds_pose and ds_pose.covariance is None:
            ds_pose.calcCovariance()

        elapsed = time.time() - start   # FIXME: Dangerous, could be zero...
        virtual_frequency = 1./elapsed
        hz = min(self.hz, virtual_frequency)

        cv.putText(self.pose_img, "FPS {}".format(round(hz, 1)),
                   (int(self.pose_img.shape[1]*4/5), 25), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   color=(0,255,0), thickness=2, lineType=cv.LINE_AA)

        cv.putText(self.pose_img, "Virtual FPS {}".format(round(virtual_frequency, 1)),
                   (int(self.pose_img.shape[1]*4/5), 45), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   color=(0,255,0), thickness=2, lineType=cv.LINE_AA)

        time_stamp = rospy.Time.now()


        # publish pose if pose has been aquired

        self.processed_image_drawing = self.bridge.cv2_to_imgmsg(self.processed_img)
        self.pose_image = self.bridge.cv2_to_imgmsg(self.pose_img)
        self.processed_image = self.bridge.cv2_to_imgmsg(self.perception.featureExtractor.img)

        # FIXME: Check the if conditions, esp. the detection count. Might be good to reduce them.
        # TODO: Put al of this into the publishing function
        # TODO: put the conversion into an extra function, too.
        if pose_aquired and ds_pose.detectionCount >= 10:
            # publish transform
            ds_transform = vectorToTransform(self.image_msg.header.frame_id + "/perception",
                                            "docking_station_link",
                                            ds_pose.translationVector,
                                            ds_pose.rotationVector,
                                            timeStamp=time_stamp)
            self.transformPublisher.publish(tf.msg.tfMessage([ds_transform]))

            # Publish placement of the light sources as a PoseArray (published in the docking_station frame)
            pose_array = featurePointsToMsg("docking_station_link", self.perception.featureModel.features, timeStamp=time_stamp)
            self.estimated_poses_array_pub.publish(pose_array)

            # publish estimated pose
            self.estimated_pose_pub.publish(
                vectorToPose(self.image_msg.header.frame_id,
                ds_pose.translationVector,
                ds_pose.rotationVector,
                ds_pose.covariance,
                timeStamp=time_stamp))
            
            # publish mahalanobis distance
            if not ds_pose.mahaDist and estimated_ds_pose:
                ds_pose.calcMahalanobisDist(estimated_ds_pose)
                self.mahal_distance_pub.publish(Float32(ds_pose.mahaDist))

            self.estimated_camera_pose_pub.publish(
                vectorToPose("docking_station_link",
                ds_pose.camTranslationVector,
                ds_pose.camRotationVector,
                ds_pose.calcCamPoseCovariance(),
                timeStamp=time_stamp))

             
        if ds_pose:
            # if the light source candidates have been associated, we pusblish the associated candidates
            self.associated_image_points_pub.publish(lightSourcesToMsg(ds_pose.associatedLightSources, timeStamp=time_stamp))
        else:
            # otherwise we publish all candidates
            self.associated_image_points_pub.publish(lightSourcesToMsg(candidates, timeStamp=time_stamp))

        return ds_pose, pose_aquired
    
    def transform_images(self):
        """
        Convert all images into publishable formats.
        FIXME: Might not use this one
        """
        pass


    def publish_messages(self):
        """
        Collect all messages and publish them
        """
        # TODO: separate the conversion
        self.image_processed_drawing_pub.publish(self.processed_image_drawing)
        self.image_processed_pub.publish(self.processed_image)
        self.image_pose_pub.publish(self.pose_image)


if __name__ == '__main__':

    rospy.init_node('perception_node')
    try:
        PerceptionNode()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch perception node")
