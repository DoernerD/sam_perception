#!/usr/bin/env python3
"""
Wrapper node to select the estimate for underwater docking
David Doerner (ddorner@kth.se) 2023
"""

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float64
import tf2_ros

class VisualEstimateSelector(object):
    """
    Node to select a suitabel pose estimation based on visual perception
    """
    def __init__(self):
        # Read external parameters
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.loop_freq = rospy.get_param("~loop_freq", 20)

        
        # Topics
        cam_forward_estim_pose_topic = rospy.get_param('~cam_forward_estim_pose_topic')
        cam_forward_rsme_topic = rospy.get_param('~cam_forward_rsme_topic')
        cam_port_estim_pose_topic = rospy.get_param('~cam_port_estim_pose_topic')
        cam_port_rsme_topic = rospy.get_param('~cam_port_rsme_topic')
        cam_starboard_estim_pose_topic = rospy.get_param('~cam_starboard_estim_pose_topic')
        cam_starboard_rsme_topic = rospy.get_param('~cam_starboard_rsme_topic')
        self.selected_estimated_pose_topic = rospy.get_param('~selected_estimated_pose_topic')

        # Init variables
        self.cam_forward_estim_pos = PoseWithCovarianceStamped()
        self.cam_port_estim_pos = PoseWithCovarianceStamped()
        self.cam_starboard_estim_pos = PoseWithCovarianceStamped()
        self.cam_forward_rsme = 0.
        self.cam_port_rsme = 0.
        self.cam_starboard_rsme = 0.

        self.selected_pose = PoseWithCovarianceStamped()

        # TF (to transform the estimated pose into base frame)
        tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(tf_buffer)

        # TODO: Init transforms from the cameras to sam/base_link

        # Subscribers
        rospy.Subscriber(cam_forward_estim_pose_topic, PoseWithCovarianceStamped,
                         self.cam_forward_estim_pose_cb, queue_size=1)
        rospy.Subscriber(cam_port_estim_pose_topic, PoseWithCovarianceStamped,
                         self.cam_port_estim_pose_cb, queue_size=1)
        rospy.Subscriber(cam_starboard_estim_pose_topic, PoseWithCovarianceStamped,
                         self.cam_starboard_estim_pose_cb, queue_size=1)
        rospy.Subscriber(cam_forward_rsme_topic, Float64,
                         self.cam_forward_rsme_cb, queue_size=1)
        rospy.Subscriber(cam_port_rsme_topic, Float64,
                         self.cam_port_rsme_cb, queue_size=1)
        rospy.Subscriber(cam_starboard_rsme_topic, Float64,
                         self.cam_starboard_rsme_cb, queue_size=1)

        # Publishers
        self.selected_estimated_pose_pub = rospy.Publisher(self.selected_estimated_pose_topic,
                                                           PoseWithCovarianceStamped, queue_size=1)

        rate = rospy.Rate(self.loop_freq)

        # Run
        while not rospy.is_shutdown():

            self.select_pose()

            self.transform_selected_pose()

            self.publish_selected_pose()

            rate.sleep()


    #region Callbacks
    def cam_forward_estim_pose_cb(self, msg):
        """
        Callback for pose message
        """
        self.cam_forward_estim_pos = msg


    def cam_port_estim_pose_cb(self, msg):
        """
        Callback for pose message
        """
        self.cam_port_estim_pos = msg


    def cam_starboard_estim_pose_cb(self, msg):
        """
        Callback for pose message
        """
        self.cam_starboard_estim_pos = msg


    def cam_forward_rsme_cb(self, msg):
        """
        Callback for RSME
        """
        self.cam_forward_rsme = msg


    def cam_port_rsme_cb(self, msg):
        """
        Callback for RSME
        """
        self.cam_port_rsme = msg


    def cam_starboard_rsme_cb(self, msg):
        """
        Callback for RSME
        """
        self.cam_starboard_rsme = msg

    #endregion

    def select_pose(self):
        """
        Select the best estimate
        """
        pass

    def transform_selected_pose(self):
        """
        Transforms the selected pose into base_frame
        """
        pass

    def publish_selected_pose(self):
        """
        Publish the selected and transformed pose
        """
        self.selected_estimated_pose_pub.publish(self.selected_pose)




if __name__ == '__main__':

    rospy.init_node('visual_estimate_selector')
    try:
        VisualEstimateSelector()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch visual estimate selector")
