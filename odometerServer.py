# Import the required modules
import socket
import network
from utime import sleep
import secret
from machine import Pin, PWM
from time import sleep
import RPi.GPIO as GPIO


# Define ports for motors
enableLeftMotorPort = 2
leftMotorPort1 = 3
leftMotorPort2 = 4

enableRightMotorPort = 6
rightMotorPort1 = 7
rightMotorPort2 = 8

# Define ports for encoder
encoderLeftPin = 5
encoderRightPin = 6
counterEncLeft = 0
counterEncRight = 0

# Initialize ports for encoder
GPIO.setmode(GPIO.BCM)
GPIO.setup(encoderLeftPin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(encoderRightPin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def motorMove(speed,direction,speedGP,cwGP,acwGP):
    # manage wrong speed values, keeping them in the 0-100 range:
    if speed > 100: speed=100
    if speed < 0: speed=0
    
    #manage the PWM PIN.
    Speed = PWM(Pin(speedGP))
    Speed.freq(50)
    
    # initialize cw (clockwise) and acw (anti-clockwise) PIN objects
    cw = Pin(cwGP, Pin.OUT)
    acw = Pin(acwGP, Pin.OUT)
    
    # DC motor speed is finally set using the duty cycle.
    Speed.duty_u16(int(speed/100*65536))
    
    # control dc motor rotation anticlockwise
    if direction < 0:
      cw.value(0)
      acw.value(1)
      
    # disable motors
    if direction == 0:
      cw.value(0)
      acw.value(0)
      
    # control dc motor rotation clockwise
    if direction > 0:
      cw.value(1)
      acw.value(0)

# Setup wifi Network
wlan = network.WLAN(network.STA_IF)
sleep(.5)
wlan.active(True)
sleep(.5)
wlan.connect(secret.ssid, secret.password)
# Add these lines to mainServer.py
print(f'SSID: {secret.ssid}')
print(f'Password: {secret.password}')

print(f'WLAN Status: {wlan.status()}')

# Wait for connection or fail
maxWaitAttempts = 1;
while wlan.isconnected() == False:
    print(f'Waiting for connection. Attempt: {maxWaitAttempts}')
    if wlan.status() < 0 or wlan.status() >= 3:
        print(wlan.status())
        break
    sleep(1)
    maxWaitAttempts += 1
# Manage connection error
if wlan.status() != 3:
    # If connection did not take place, send an error message
    raise RuntimeError('Network Connection Failed!!')
else:
    # If connection took place, send a message, gather some data, and set server
    print('Connected')
    status = wlan.ifconfig()
    # It will be good to reserve the given IP Address on the WIFI ROUTER. Check how to do it.
    # Everytime you run the program you will have the same IP Address otherwise
    print('ip = ' + status[0])
    print('subnet = ' + status[1])
    print('geteway = ' + status[2])
    print('dns = ' + status[3])
    # === SET UP SERVER PROFILE
    serverIP = status[0]
    serverPort = 2222
    bufferSize = 1024
    UDPServer = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    UDPServer.bind((serverIP,serverPort))
    UDPClient = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

while True:
    cmd,address = UDPServer.recvfrom(1024)
    cmdDecoded = cmd.decode('utf-8')
    if cmdDecoded == "1":
        motorMove(100,-1,enableLeftMotorPort,leftMotorPort1,leftMotorPort2)
        motorMove(100,-1,enableRightMotorPort,rightMotorPort1,rightMotorPort2)
        
        if GPIO.input(encoderLeftPin) == GPIO.HIGH:
            counterEncLeft += 1
        
        if GPIO.input(encoderRightPin) == GPIO.HIGH:
            counterEncRight += 1
            
        serverAddress = ('192.168.0.11',2222)
        while True:
            cmdEncoded = str(150) #aqui codigo para mandar RPS
            UDPClient.sendto(cmdEncoded,serverAddress)
        
    #-------------
    # Generar loop para mandar datos
