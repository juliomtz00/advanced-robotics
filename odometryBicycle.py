import time
import numpy as np

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
    def __init__(self, xSpeed, ySpeed, turningVel):
        self.xSpeed = xSpeed
        self.ySpeed = ySpeed
        self.turningVel = turningVel

    def update(self, newXSpeed, newYSpeed, newTurnSpeed):
        self.xSpeed = newXSpeed
        self.ySpeed = newYSpeed
        self.turningVel = newTurnSpeed

# This function contains the equation of the forward kinematic model for a differential drive vehicle (Bicycle model)
def computeOdometry(index2Update, deltaAngle, velocityData):
    # GET PREVIOUS POSE DATA FROM THE POSE VECTOR
    previousPose = np.array([
        [poseEstimation.x[index2Update - 1]],
        [poseEstimation.y[index2Update - 1]],
        [poseEstimation.yaw[index2Update - 1]]
    ])
    # SAVE THE CURRENT POSE DATA INTO THE POSE STRUCTURE
    velocityFactor = samplingRate * np.dot(deltaAngle, velocityData)
    poseEstimation.x[index2Update] = previousPose[0] + velocityFactor[0]
    poseEstimation.y[index2Update] = previousPose[1] + velocityFactor[1]
    poseEstimation.yaw[index2Update] = previousPose[2] + velocityFactor[2]

# Compute current vehicle speed and orientation variables for the Bicycle model
def vehicleParameters(deltaAngle, wheelBase, rightWheelSpeed, leftWheelSpeed):
    # Calculate local X and Y speeds and turning velocity for the Bicycle model
    localXSpeed = (wheelBase / 2) * (rightWheelSpeed + leftWheelSpeed) * np.cos(deltaAngle)
    localYSpeed = (wheelBase / 2) * (rightWheelSpeed + leftWheelSpeed) * np.sin(deltaAngle)
    localTurninigSpeed = (wheelBase / 2) * (rightWheelSpeed - leftWheelSpeed) / wheelBase
    myRobot.update(localXSpeed, localYSpeed, localTurninigSpeed)
    localRobotSpeed = np.array([
        [myRobot.xSpeed],
        [myRobot.ySpeed],
        [myRobot.turningVel]
    ])
    return localRobotSpeed

# Compute the new rotational matrix
def rotateAroundZ(previousYaw):
    cosResult = np.cos(previousYaw)
    sinResult = np.sin(previousYaw)
    rotZ = np.array([
        [cosResult, -sinResult, 0],
        [sinResult, cosResult, 0],
        [0, 0, 1]
    ])
    return rotZ

# USER INPUTS
encoderPulses = float(input("Enter quantity of encoder pulses per revolution: "))
pulses = float(input("Enter quantity of pulses per second: "))
WHEELRADIUS = float(input("Enter vehicle wheel radius: "))
width = float(input("Enter vehicle width (W): "))
elapsedTime = float(input("Enter the elapsed time: "))
initialX = float(input("Enter Starting X position: "))
initialY = float(input("Enter Starting Y position: "))
initialYaw = np.radians(float(input("Enter Starting Heading Value (degrees): ")))
deltaAngle = np.radians(float(input("Enter Delta Angle (radians): ")))  # New input
wheelBase = float(input("Enter Wheel Base (L): "))  # New input

RPS = pulses / encoderPulses

leftSpeed = RPS * 2 * np.pi
rightSpeed = RPS * 2 * np.pi

# Physical Parameters of the Differential Mobile Robot SETUP
myRobot = Robot(0.0, 0.0, 0.0)

# TIME REQUIREMENTS SETUP
# Setup elapsed time and sampling-rate for the test
samplingRate = 0.25
# POSE VECTOR SIZE CALCULATION
vectorSize = int((elapsedTime / samplingRate) + 1)
poseEstimation = Pose(vectorSize)

# INITIAL POSE VALUES GO TO 1st VALUE IN POSE VECTOR
# Variable of control to store into the Pose vector positions
currentIndex = 0
# Starting pose vector values setup
poseEstimation.x[currentIndex] = initialX
poseEstimation.y[currentIndex] = initialY
poseEstimation.yaw[currentIndex] = initialYaw

# DELTA-TIME REQUIREMENTS SETUP
# Compute the finishing time for the test.
endTime = time.time() + elapsedTime
# Initial past time
pastTime = (endTime - elapsedTime)

# START ODOMETRY COMPUTATION
# As long as current time is less than or equal to endTime
while time.time() <= endTime:
    currentTime = time.time()
    if (currentTime - pastTime) >= samplingRate:
        rotZMat = rotateAroundZ(poseEstimation.yaw[currentIndex - 1])
        robotSpeedData = vehicleParameters(deltaAngle, wheelBase, rightSpeed, leftSpeed)
        computeOdometry(currentIndex, rotZMat, robotSpeedData)
        pastTime = currentTime
        currentIndex += 1

# LATELY, DISPLAY SAMPLED POSES
# Print final pose values
print("Pose Values:")
print("x: ", poseEstimation.x)
print("y: ", poseEstimation.y)
print("yaw: ", np.degrees(poseEstimation.yaw))  # Convert yaw back to degrees for display

# Calculate and display the total traveled distance
totalDistance = np.sum(np.sqrt(np.diff(poseEstimation.x)**2 + np.diff(poseEstimation.y)**2))
print("Total Traveled Distance:", totalDistance)