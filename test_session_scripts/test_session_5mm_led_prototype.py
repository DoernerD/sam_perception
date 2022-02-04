import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for python 2.7
from coordinate_system import CoordinateSystemArtist

def inverseTransform(pose):
    lightRotRelativeCamera = R.from_euler("YXZ", pose[3:], degrees=True).as_dcm()

    cameraTransRelativeLight = -np.matmul(lightRotRelativeCamera.transpose(), np.array(pose[:3]))
    cameraRotRelativeLight = lightRotRelativeCamera.transpose()
    cameraRotRelativeLight = R.from_dcm(cameraRotRelativeLight).as_euler("YXZ", degrees=True)
    return tuple(cameraTransRelativeLight) + tuple(cameraRotRelativeLight)

class TestSession:
    def __init__(self):
        pass

    def cameraRelativeLightTransform(self, pose):
        euler = [np.deg2rad(v) for v in pose[3:]]
        lightRotRelativeCamera = R.from_euler("YXZ", euler).as_dcm()

        cameraTransRelativeLight = -np.matmul(lightRotRelativeCamera.transpose(), np.array(pose[:3]))
        cameraRotRelativeLight = lightRotRelativeCamera.transpose()
        return cameraTransRelativeLight, cameraRotRelativeLight

    def calculateErrors(self, poses, posesTrue, camPoses, camPosesTrue):
        for i, (pose, poseTrue) in enumerate(zip(poses, posesTrue)):
            #print("Reference trans: {}".format(transLightToCameraRef))
            rangeErr = np.linalg.norm(poseTrue[:3]) - np.linalg.norm(pose[:3])
            transErr = np.array(poseTrue[:3]) - np.array(pose[:3])
            eulerErr = np.array(poseTrue[3:]) - np.array(pose[3:])
            print("Light pose {} err: range {} trans {} rot {}".format(i+1, rangeErr, transErr, eulerErr))

        for i, (pose, poseTrue) in enumerate(zip(poses, posesTrue)):
            camToLightTrans, camToLightRot = self.cameraRelativeLightTransform(pose)
            camToLightTransTrue, camToLightRotTrue = self.cameraRelativeLightTransform(poseTrue)
            camToLightRot = np.rad2deg(R.from_dcm(camToLightRot).as_euler("YXZ"))
            camToLightRotTrue = np.rad2deg(R.from_dcm(camToLightRotTrue).as_euler("YXZ"))

            rangeErr = np.linalg.norm(camToLightTransTrue) - np.linalg.norm(camToLightTrans)
            transErr = np.array(camToLightTransTrue) - np.array(camToLightTrans)
            eulerErr = np.array(camToLightRotTrue) - np.array(camToLightRot)
            print("Cam pose {} err: range {} trans {} rot {}".format(i+1, rangeErr, transErr, eulerErr))

    def getCoordinateSystems(self, poses, posesTrue, camPoses, camPosesTrue):
        # light prototype pose in world coordinates
        lightPrototypeTrans = [0, 0, 0]
        lightPrototypeRot = R.from_euler("XYZ", (-np.pi/2, np.pi/2, 0)).as_dcm()
        csLightPrototype = CoordinateSystemArtist(translation=lightPrototypeTrans, 
                                                  euler=(-np.pi/2, np.pi/2, 0))

        coordSystemsTrue = []  # camera relative to world
        coordSystemsLight = [] # light relative world (assuming measured 
                               # camera poses are at the true pose)

        for pose, poseTrue in zip(poses, posesTrue):

            # true poses in world coordinates
            cameraTransRelativeLight, cameraRotRelativeLight = self.cameraRelativeLightTransform(poseTrue)
            cameraTransRelativeWorld = np.matmul(lightPrototypeRot, cameraTransRelativeLight)
            cameraRotRelativeWorld = np.matmul(lightPrototypeRot, cameraRotRelativeLight)
    
            coordSystemsTrue.append(CoordinateSystemArtist(translation=cameraTransRelativeWorld, 
                                                        euler=R.from_dcm(cameraRotRelativeWorld).as_euler("YXZ"),
                                                        seq="YXZ"))

            # light relative world (assuming measured camera poses are at the true pose)
            lightTransRelativeWorld = cameraTransRelativeWorld + np.matmul(cameraRotRelativeWorld, np.array(pose[:3]))
            lightRotRelativeWorld = np.matmul(
                cameraRotRelativeWorld, 
                R.from_euler("YXZ", tuple(np.deg2rad(pose[3:]))).as_dcm()
                )
            coordSystemsLight.append(CoordinateSystemArtist(translation=lightTransRelativeWorld, 
                                                            euler=R.from_dcm(lightRotRelativeWorld).as_euler("YXZ"),
                                                            seq="YXZ"))


        camCcoordSystems = []
        camCoordSystemsTrue = []
        dockingStationCoordSystems = []
        for pose, camPose, camPoseTrue in zip(poses, camPoses, camPosesTrue):

            # measured camera poses in world coordinates
            cameraTransRelativeWorld = np.matmul(lightPrototypeRot, camPose[:3])
            cameraRotRelativeWorld = np.matmul(lightPrototypeRot, R.from_euler("YXZ", camPose[3:], degrees=True).as_dcm())

            camCcoordSystems.append(CoordinateSystemArtist(translation=cameraTransRelativeWorld, 
                                                    euler=R.from_dcm(cameraRotRelativeWorld).as_euler("YXZ"),
                                                    seq="YXZ"))

            # true camera poses in world coordinates
            cameraTransRelativeWorldTrue = np.matmul(lightPrototypeRot, camPoseTrue[:3])
            cameraRotRelativeWorldTrue = np.matmul(lightPrototypeRot, R.from_euler("YXZ", camPoseTrue[3:], degrees=True).as_dcm())
    
            camCoordSystemsTrue.append(CoordinateSystemArtist(translation=cameraTransRelativeWorldTrue, 
                                                        euler=R.from_dcm(cameraRotRelativeWorldTrue).as_euler("YXZ"),
                                                        seq="YXZ"))

            # light relative world (assuming measured camera poses are at the true pose)
            lightTransRelativeWorld = cameraTransRelativeWorldTrue + np.matmul(cameraRotRelativeWorldTrue, np.array(pose[:3]))
            lightRotRelativeWorld = np.matmul(
                cameraRotRelativeWorldTrue, 
                R.from_euler("YXZ", tuple(np.deg2rad(pose[3:]))).as_dcm()
                )
            dockingStationCoordSystems.append(CoordinateSystemArtist(translation=lightTransRelativeWorld, 
                                                            euler=R.from_dcm(lightRotRelativeWorld).as_euler("YXZ"),
                                                            seq="YXZ"))

        return csLightPrototype, camCcoordSystems, camCoordSystemsTrue, dockingStationCoordSystems
        #return csLightPrototype, coordSystems, coordSystemsTrue, coordSystemsLight

    def displayPoses(self, poses, posesTrue, camPoses, camPosesTrue):
        assert len(poses) == len(posesTrue), "Must have the same number of poses as ground truths"

        csLightPrototype, coordSystems, coordSystemsTrue, coordSystemsLight = self.getCoordinateSystems(poses, posesTrue, camPoses, camPosesTrue)

        fig = plt.figure()
        fig.add_subplot(projection="3d")
        ax = Axes3D(fig)

        s = 120
        ax.set_xlim(-s, 0)
        ax.set_ylim(-s/2, s/2)
        ax.set_zlim(-s/2, s/2)

        axisScale = 6
        poseIdx = 1
        for cs, csTrue, csLight in zip(coordSystems, coordSystemsTrue, coordSystemsLight):
            cs.draw(ax, scale=axisScale, color=("r", "r", "r"))
            csLight.draw(ax, scale=axisScale, color=("salmon", "lightgreen", "lightblue"))
            csTrue.draw(ax, scale=axisScale, color=("b", "b", "b"), text="Cam {}".format(poseIdx))
            poseIdx += 1
        csLightPrototype.draw(ax, scale=axisScale, text="Light")
        plt.show()

        fig = plt.figure()
        fig.add_subplot(projection="3d")
        ax = Axes3D(fig)

        while True:
            for csLightRef in coordSystemsLight:
                ax.clear()
                ax.set_xlim(-s, 0)
                ax.set_ylim(-s/2, s/2)
                ax.set_zlim(-s/2, s/2)

                csLightPrototype.draw(ax, scale=axisScale)
                for cs, csTrue, csLight in zip(coordSystems, coordSystemsTrue, coordSystemsLight):
                    cs.draw(ax, scale=axisScale, color=("r", "r", "r"))
                    if csLightRef == csLight:
                        csTrue.draw(ax, scale=axisScale, color=("g", "g", "g"))
                        csLight.draw(ax, scale=axisScale, color=("salmon", "lightgreen", "lightblue"))
                    else:
                        csTrue.draw(ax, scale=axisScale, color=("b", "b", "b"))
                plt.pause(1)
        plt.show()

if  __name__ == "__main__":
    # pose: x, y, z, yaw, pitch, roll
    poses1 = [
             # poses from long range (120 cm)
             (-3.58, -.12, 120.03, -4.25, 1.85, 4.45),
             (18.57, 2.16, 120.31, -2.75, 1.42, 4.77),
             (-25.42, 4.45, 120.02, -4.44, -1.21, 4.13),
             (39.75, 7.79, 120.41, -2.7, -0.61, 4.23),
             (-47.95, 2.98, 119.51, -4.45, -1.04, 4.08),
             # poses from mid range (60 cm)
             (-2.17, 2.45, 59.9, -4.31, -0.54, 3.88),
             (19.94, 3.39, 60.99, -2.71, -0.39, 4.44),
             (-22.9, 1.52, 60.86, -3.46, -0.52, 4.55),
             # pose from short range (20 cm)
             (-0.8, 0.16, 19.74, -3.08, 1.45, 1.47)
             ]

    yaw = 0
    posesTrue1 = [
                    # true poses from long range (120 cm)
                    (0, 0, 120, yaw, 0, 0),
                    (21.2, 0, 120, yaw, 0, 0),
                    (-21.2, 0, 120, yaw, 0, 0),
                    (43.7, 0, 120, yaw, 0, 0),
                    (-43.7, 0, 120, yaw, 0, 0),
                    # true poses from mid range (60 cm)
                    (0, 0, 60, yaw, 0, 0),
                    (21.8, 0, 60, yaw, 0, 0),
                    (-21.8, 0, 60, yaw, 0, 0),
                    # true pose from short range (20 cm)
                    (0, 0, 20, yaw, 0, 0)
                    ]

    camPoses1 = [inverseTransform(p) for p in poses1]
    camPosesTrue1 = [inverseTransform(p) for p in posesTrue1]

    # test 2
    poses2 = [
             # poses from long range (120 cm)
             (-.15, -.18, 119.88, -.13, .43, .74),
             (20.58, .37, 119.97, .62, -.17, .91),
             (-21.76, .03, 120.83, -.69, .48, .49),
             (44.02, -.36, 120.58, .55, 2.03, 1.58),
             (-44.57, .04, 121.71, -.5, .94, -.04),
             (-.12, -.46, 60.38, -.14, .89, -.01),
             (20.9, -.57, 60.28, 1.43, -.25, .18),
             (-22.63, -.75, 61.24, .78, .08, .78),
             (.09, .1, 19.84, 1.18, -1.51, -.36)
             ]

    camPoses2 = [
             # poses from long range (120 cm)
             (-.14, -.73, -119.88, .14, -.43, -.74),
             (-19.27, .27, -120.17, -.62, .15, -.91),
             (20.28, -1.21, -121.08, 0.7, -.47, -.5),
             (-42.95, -2.75, -120.93, -.5, -2.05, -1.56),
             (43.51, -2.01, -122.07, .5, -.94, .03),
             (-.03, -.48, -60.38, .14, -.89, .01),
             (-19.39, .9, -60.78, -1.43, .25, -.18),
             (23.47, .34, -60.93, -.78, -.09, -.77),
             (.32, .43, -19.83, -1.18, 1.52, .32)
             ] 

    yDisp = -0.9
    pitch = 1
    posesTrue2 = [
                    # true poses from long range (120 cm)
                    (0, yDisp, 120, 0, pitch, 0),
                    (21.2, yDisp, 120, 0, pitch, 0),
                    (-21.2, yDisp, 120, 0, pitch, 0),
                    (43.7, yDisp, 120, 0, pitch, 0),
                    (-43.7, yDisp, 120, 0, pitch, 0),
                    # true poses from mid range (60 cm)
                    (0, yDisp, 60, 0, pitch, 0),
                    (21.8, yDisp, 60, 0, pitch, 0),
                    (-21.8, yDisp, 60, 0, pitch, 0),
                    # true pose from short range (20 cm)
                    (0, yDisp, 20, 0, pitch, 0)
                    ]

    camPosesTrue2 = [inverseTransform(p) for p in posesTrue2]

    ts = TestSession()
    #poses = poses[:2]
    #posesTrue = posesTrue[:2]
    ts.calculateErrors(poses2, posesTrue2, camPoses2, camPosesTrue2)
    ts.displayPoses(poses2, posesTrue2, camPoses2, camPosesTrue2)
