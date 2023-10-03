import numpy as np
import random

#Primera parte

#Wheel parameters
wheel_radius = 0.036 #m
wheel_diameter = 2*wheel_radius #Diameter of wheels
wheel_cir = np.pi*wheel_diameter

#Encoder parameters
encoder_pulses = 20
pulses_obtained = 80

#Other parameters
L = 0.22 #m
delta_time = 2 #s

#Initial Position
xg_past = 0
yg_past = 0
thetag_past = 0
yaw = np.deg2rad(0)

# Create our new coords array
new_coords = np.array([[xg_past],[yg_past],[thetag_past]]) 

# Calculate the rps, linear and angular speed
rps = pulses_obtained/encoder_pulses #rev/s
linear_speed = rps*wheel_cir
angular_vel = rps*2*np.pi
    
# Create the velocity matrix
vel_matrix = np.array([[(wheel_radius*angular_vel/2)+(wheel_radius*angular_vel/2)],[0],[((wheel_radius*angular_vel)/(2*L))-((wheel_radius*angular_vel)/(2*L))]])

# Create the rotational matrix
rot_matrix = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])

# Create our global position matrix
new_position = new_coords+delta_time*np.dot(rot_matrix,vel_matrix)

print("The position without uncertainties is: ", new_position, sep ='\n')

#Segunda Parte

# Creating the uncertainties in the wheel parameters
l_wheel_radius = np.random.normal(loc = wheel_radius,scale=wheel_radius*0.05)
r_wheel_radius = np.random.normal(loc=wheel_radius,scale = wheel_radius*0.05)
l_wheel_cir = 2*np.pi*l_wheel_radius
r_wheel_cir = 2*np.pi*r_wheel_radius

# Creating the uncertainties in the encoder parameters
l_wheel_pulses = random.uniform(60,98)
r_wheel_pulses = random.uniform(60,98)

# Calculate the rps, linear and angular speed for each wheel
l_rps = l_wheel_pulses/encoder_pulses #rev/s
r_rps = r_wheel_pulses/encoder_pulses #rev/s
    
l_linear_speed = l_rps*l_wheel_cir
r_linear_speed = r_rps*r_wheel_cir
   
l_angular_vel = l_rps*2*np.pi
r_angular_vel = r_rps*2*np.pi
        
# Create the velocity matrix
vel_matrix = np.array([[(l_wheel_radius*l_angular_vel/2)+(r_wheel_radius*r_angular_vel/2)],[0],[((l_wheel_radius*l_angular_vel)/(2*L))-((r_wheel_radius*r_angular_vel)/(2*L))]])

# Create the rotational matrix
rot_matrix = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])

# Create our new coords array
new_coords_uncertainties = np.array([[xg_past],[yg_past],[thetag_past]]) 

#Computing global position 
new_pos_uncertainties = new_coords_uncertainties + delta_time * np.dot(rot_matrix,vel_matrix)

print("The current position with uncertainties is: ",new_pos_uncertainties, sep ='\n')