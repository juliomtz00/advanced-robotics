import time
import argparse
# Universal standar to work with numerical data in Python
import numpy as np

parser = argparse.ArgumentParser(prog = 'Sequeantial Odometry',
                                 description = 'Real time odometry estimation')

parser.add_argument('-e', '--encoder_pulses',type = int, help = 'Use the following syntaxis: -e ENCODER_PULSES')
parser.add_argument('-p', '--pulses_per_second',type = int, nargs = 2, help = ' Use the following syntaxis: -p LEFT_PULSES RIGHT_PULSES')
parser.add_argument('-r', '--wheel_radius',type = float, help = ' Wheel radius, syntaxis: -r RADIUS')
parser.add_argument('-w', '--wheel_width',type = float, help = 'Wheel width, syntaxis: -w WIDTH')
parser.add_argument('-s', '--starting_pose',type = float, nargs = 3, help = 'Usar la sintaxis -s XG YG OG')
parser.add_argument('-t', '--elapsed_time', type = float, help = 'Time elapsed from start to end in seconds, syntaxis: -t TIME')

args = parser.parse_args()
inputs = vars(args)

# Pose class contains three arrays that will store vehicle position and orientation values
class Pose:
    def __init__(self, size):
        self.x = np.zeros(size)
        self.y = np.zeros(size)
        self.yaw = np.zeros(size)
    def update(self, newX, newY, newYaw):
        self.x = newX
        self.y = newY
        self.yaw = newYaw
# Robot class stores vehicle variables from its local frame: left and right speed & orientation
class Robot:
    def __init__(self, xSpeed,ySpeed,turningVel):
        self.xSpeed = xSpeed
        self.ySpeed = ySpeed
        self.turningVel = turningVel
    def update(self, newXSpeed, newYSpeed, newTurnSpeed):
        self.xSpeed = newXSpeed
        self.ySpeed = newYSpeed
        self.turningVel = newTurnSpeed

# This function contains the equation of the forward kinematic model for a differential drive vehicle
def computeOdometry(index2Updte, rotData, velocityData):
    # GET PREVIOUS POSE DATA FROM THE POSE VECTOR
    previousPose = np.array([
        [poseEstimation.x[index2Updte-1]],
        [poseEstimation.y[index2Updte-1]],
        [poseEstimation.yaw[index2Updte-1]]
    ])
    # SAVE THE CURRENT POSE DATA INTO THE POSE STRUCTURE
    velocititesFactor = samplingRate*np.dot(rotData,velocityData)
    poseEstimation.x[index2Updte] = previousPose[0] + velocititesFactor[0]
    poseEstimation.y[index2Updte] = previousPose[1] + velocititesFactor[1]
    poseEstimation.yaw[index2Updte] = previousPose[2] + velocititesFactor[2]
    
    
# Compute current vehicle speed and orientation variables
def vehicleParameters(rightWheelSpeed, leftWheelSpeed, wheelRadius, widthValue):
    wheelRadiusLeft = wheelRadius[0]
    wheelRadiusRight = wheelRadius[1]
    localXSpeed = (wheelRadiusRight*rightWheelSpeed/2)+(wheelRadiusLeft*leftWheelSpeed/2)
    localTurninigSpeed = (wheelRadiusRight*rightWheelSpeed/(2*widthValue))-(wheelRadiusLeft*leftWheelSpeed/(2*widthValue))
    myRobot.update(localXSpeed, 0, localTurninigSpeed)
    localRobotSpeed = np.array([
        [myRobot.xSpeed],
        [myRobot.ySpeed],
        [myRobot.turningVel]
    ])
    return localRobotSpeed

# Compute the new rotational matrix 
# Make sure the argument previousYaw is in radians 
def rotateAroundZ(previousYaw):
    cosResult = np.cos(previousYaw)
    sinResult = np.sin(previousYaw)
    rotZ = np.array([
        [cosResult, -sinResult, 0],
        [sinResult, cosResult, 0],
        [0, 0, 1]
    ]) #START ODOMETRY COMPUTATION
    return rotZ


# ==== USER INPUTS
encoder = inputs["encoder_pulses"]
pulses = inputs["pulses_per_second"]

pulses_left = np.random.normal(loc=pulses[0],scale = pulses[0]*0.05)
pulses_right = np.random.normal(loc=pulses[1],scale = pulses[1]*0.05)

rps_left = pulses_left/encoder
rps_right = pulses_right/encoder

leftSpeed = rps_left*2*np.pi
rightSpeed = rps_right*2*np.pi

startPose = inputs["starting_pose"]
initialX = startPose[0]
initialY = startPose[1]
initialYaw = startPose[2]

# ==== Physical Parameters of the Differential Mobile Robot SETUP ====
left_radius = np.random.normal(loc=inputs["wheel_radius"],scale = inputs["wheel_radius"]*0.05)
right_radius = np.random.normal(loc=inputs["wheel_radius"],scale = inputs["wheel_radius"]*0.05)
WHEELRADIUS = [left_radius,right_radius]

#Wheel circumference 
cir_left = 2*np.pi*left_radius
cir_right = 2*np.pi*right_radius

#Angular speed
left_ls = rps_left*cir_left
right_ls = rps_right*cir_right

WIDTHCONTACTPOINT = inputs["wheel_width"]
myRobot = Robot(0.0,0.0,0.0)

# ==== TIME REQUIREMENTS SETUP ====
# Setup elapsed time and sampling-rate for the test
elapsedTime = inputs["elapsed_time"]
samplingRate = 0.25

# ==== POSE VECTOR SIZE CALCULATION === 
vectorSize = elapsedTime/samplingRate
poseEstimation = Pose(int(vectorSize))
# ==== INITIAL POSE VALUES GOES TO 1st VALUE IN POSE VECTOR ====
# Variable of control to store into the Pose vector positions
currentIndex = 0
# Define a 3x3 Z-axis rotational matrix
angleRadians = np.radians(initialYaw)
# Starting pose vector values setup
poseEstimation.x[currentIndex] = initialX
poseEstimation.y[currentIndex] = initialY
poseEstimation.yaw[currentIndex] = angleRadians

# ==== DELTA-TIME REQUIREMTENS SETUP ====
# The algoritm is about to star so it nees to record past and curren time variables.
# Compute the finishing time for the test.
endTime = time.time() + elapsedTime # time.time() returns time values in seconds
# Initial past time
pastTime = (endTime - elapsedTime)

# === START ODOMETRY COMPUTATION ====
# As long as current time is less than endTime
while time.time() < endTime:
#    print("Pass time: ", pastTime)
#    print("Count: ", currentIndex)
#    print("Delta Time: ", time.time()-pastTime)
    currentTime = time.time()
    if (currentTime-pastTime) >= samplingRate:
        currentIndex += 1
        rotZMat = rotateAroundZ(poseEstimation.yaw[currentIndex-1])
        robotSpeedData = vehicleParameters(rightSpeed,leftSpeed,WHEELRADIUS, WIDTHCONTACTPOINT)
        computeOdometry(currentIndex, rotZMat, robotSpeedData)
        pastTime = currentTime
        print("Count: ", currentIndex)
    
# === LATELY, DISPLAY SAMPLED POSES ====
# Print final pose values
print("Pose Values: ")
print("x: ", np.round(poseEstimation.x,3))
print("y: ", np.round(poseEstimation.y,3))
print("yaw: ", np.round(poseEstimation.yaw,3))
print("left angular speed: ",np.round(leftSpeed,5))
print("right angular speed: ",np.round(rightSpeed,5))
print("left linear speed: ",np.round(left_ls,5))
print("right linear speed: ",np.round(right_ls,5))