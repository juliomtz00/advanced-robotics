# CODIGO COMPUTADORA

import time
import math
import socket
import numpy as np
import cv2
from getAngleCamera import computeAngle

bufferSize = 1024
elapsedTime = 0
userInputGame = ''
coord_n = []
x_n = []
y_n = []
z_n = []
ang_n = []
c_n = []
doTurn = 0
doForward = 0

previousAngle = 0
previousCoordU = 0
previousCoordV = 0

getOrigin = 0

signAng = 0

angGiro = 0
angN = 0

dist = 0

# ------------------------------------
# VARIABLES PARA LA VERSION CON CAMARA
# ------------------------------------
coordU = []  # coordsU[0] seria el origen en X en pixeles
coordV = []  # coordsV[0] seria el origen en Y en pixeles
coordAng = []  # coordsV[0] seria el origen en Y
# startU = 0 # Coordenada origen en X
# startV = 0 # Coordenada origen en Y
# startAng = 0
# prevU = 0
# prevV = 0
# prevAng = 0
centrU = 0  # Coordenada que se recibe de la camara, llegar al objetivo
centrV = 0  # Coordenada que se recibe de la camara, llegar al objetivo
centrAng = 0  # Angulo que se recibe de la camara, llegar al objetivo
# ------------------------------------


clientTCP = socket.socket()
# Conociendo que la direccion IP del Servidor la da la variable status[0] del codigo
# de configuracion del Servidor en el puerto establecido en el mismo -> 80, se
# verifica que haya conexion entre Cliente y Servidor
socketInfo = socket.getaddrinfo('192.168.4.1', 80)[0][-1]
clientTCP.connect(socketInfo)
print('Conectado al Servidor: ', str(socketInfo))

# Para el cv2 el cuadrante que se transforma sera la del eje Y
angle_values = list()

# Reading an image using OpenCV, extracting its height and width, and converting it to grayscale.
# Create a new window
#cv2.namedWindow("Camera Frame", cv2.WINDOW_NORMAL)

# Create a video capture object for video streaming
camera_index = 0  # this is the camera index
video_capture = cv2.VideoCapture(camera_index)
video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off

# Calculate the image measurements of the camera through the functions of the opencv library.
image_width = float(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
image_height = float(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(image_height, image_width)
print("IMAGE MEASUREMENTS: height: " + str(image_height) + ", width: " + str(image_width))

# Open the calibration file
calibration_file = open("/Users/juliomtz00/Documents/GitHub/advanced-robotics/rob-lab-calib/calibration-parameters.txt",
                        'r')

# Initialize variables to store mtx and dist
mtx = None
dist = None

# Loop through each line in the calibration file
for line in calibration_file:
    line = line.rstrip()

    # Check if the line contains information about mtx
    if line.startswith('mtx:'):
        # Extract the matrix values from the line and convert them to a NumPy array
        mtx_values = np.array(eval(line[4:]))
        mtx = mtx_values.reshape((3, 3))  # Assuming mtx is a 3x3 matrix

    # Check if the line contains information about dist
    elif line.startswith('dist:'):
        # Extract the matrix values from the line and convert them to a NumPy array
        dist_values = np.array(eval(line[5:]))
        dist = dist_values.reshape((1, -1))  # Assuming dist is a 1xN matrix

# Close the calibration file
calibration_file.close()

# If camera opens correctly
i = 0


def matrixHTM(centrU, centrV, centrAng):
    #     dst, centrAng, centrU, centrV = computeAngle(frame,mtx,dist)
    # #     heightScreen = 0
    # #     # widthScreen = 0

    # Definir la matriz de rotación para 90 grados en el plano XY
    theta = np.radians(90)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    # Definir el vector o la matriz que deseas rotar
    # Por ejemplo, un vector [x, y] o una matriz [[a, b], [c, d]]
    #     vector_or_matrix = np.array([5,7])  # Cambia esto según tus necesidades
    #     LLAMAR A LA FUNCION DE LA CAMARA QUE REGRESE LOS SIGUIENTES
    #     centrU, centrV, centrAng =
    vector_or_matrix = np.array([centrU, centrV])  # Cambia esto según tus necesidades

    # Realizar la multiplicación matricial para la rotación
    rotated_vector_or_matrix = np.dot(rotation_matrix, vector_or_matrix)

    #     # Imprimir el resultado
    #     print("Vector o matriz original:", vector_or_matrix)
    #     print("Vector o matriz rotado 90 grados:", rotated_vector_or_matrix)
    try:
        if coordU[-1]:
            pass
    except:
        coordU.append(rotated_vector_or_matrix[0])
        coordV.append(image_height - rotated_vector_or_matrix[1])
        coordAng.append(centrAng)
    #     return centrU, centrV, centrAng
    return rotated_vector_or_matrix[0], (image_height - rotated_vector_or_matrix[1]), centrAng


# ESTA FUNCION FUE REESTRUCTURADA
def uncompressInput(input_string):  # Ej. coordenadas multiples -> "(10,0); (-2,1); (1,-1)"
    coordGame = input_string.split(';')
    nIndex = -1
    indexPop = []
    adjust = 0
    adjustFirst = 1
    for item in coordGame:
        nIndex += 1
        try:
            # Remove parentheses and split by comma to get x and y
            x, y, z = item.strip('( )').split(',')
        except:
            #             indexPop.append(nIndex)
            try:
                # Remove parentheses and split by comma to get x and y
                x, y = item.strip('( )').split(',')
                try:
                    if len(z) < 1:
                        z = ''
                except:
                    z = ''
            except:
                indexPop.append(nIndex)
                continue
        if len(x) > 0 and len(y) > 0:
            try:
                if float(x):
                    pass
                if float(y):
                    pass
            except:
                indexPop.append(nIndex)
                continue
            # Convert x and y to float and append to respective lists
            x_n.append(float(x))
            y_n.append(float(y))
            if len(z) > 0:
                z_n.append(float(z))
            else:
                z_n.append(0)
            '''ang_n.append(math.degrees(math.atan2(float(y), float(x))))
            # c_values es la distancia de la diagonal
            c_n.append(np.sqrt(float(x) ** 2 + float(y) ** 2))'''
            try:
                # ang_n.append(math.degrees(math.atan2(float(y_n[-1] - y_n[-2]), float(x_n[-1] - x_n[-2]))))
                ang_n.append(math.degrees(math.atan2(float(y_n[-1] - y_n[-2]), float(x_n[-1] - x_n[-2]))) - ang_n[-1])
                c_n.append(np.sqrt(float(x_n[-1] - x_n[-2]) ** 2 + float(y_n[-1] - y_n[-2]) ** 2))
            except:
                ang_n.append(math.degrees(math.atan2(float(y_n[-1]), float(x_n[-1]))))
                c_n.append(np.sqrt(float(x_n[-1]) ** 2 + float(y_n[-1]) ** 2))
        else:
            indexPop.append(nIndex)
            continue
    for indPop in indexPop:
        coordGame.pop(indPop - adjust)
        adjust += 1
    if userInputGame == '1':
        for indexAll in range(1, len(coordGame)):
            coordGame.pop()
            x_n.pop()
            y_n.pop()
            z_n.pop()
            ang_n.pop()
            c_n.pop()
    coord_n.append(coordGame)
    return


# def sendRaspberry(x_n, y_n, ang_n, c_n, coordU, coordV, coordAng):
def sendRaspberry(ang_n, c_n, coordU, coordV, coordAng):
    # --------------------------------------------------
    # REESTRUCTURACION DE GIRO Y TRAYECTO AL MISMO NIVEL
    # --------------------------------------------------
    print()
    #     doOne = 0
    signAng = 1
    #     sendRasp = 0
    sendRasp = 4
    condStop = 0
    for cycle in range(2):
        doOneTime = 0
        while sendRasp != 0:
            #            LLAMAR A LA FUNCION DE LA CAMARA QUE REGRESE LOS SIGUIENTES
            #             centrU, centrV, angCamera =
            # Get the current frame and pass it on to 'frame''
            # If the current frame cannot be captured, ret = 0
            ret, frame = video_capture.read()

            # If ret=0
            if not ret:
                print("Frame missed!")

            dst, angle, centrU, centrV = computeAngle(frame, mtx, dist)
            print(angle, centrU, centrV)
            '''
            if angle == 360:
                angle = previousAngle
            previousAngle = angle

            if centrU > image_width or centrV > image_height:
                centrU = previousCoordU
                centrV = previousCoordV

            previousCoordU = centrU
            previousCoordV = centrV
            '''
            # --------------------------------------------------
            # --------------------------------------------------
            if (centrU > image_width) or (centrV > image_height) or (angle >= 360):
                sendRasp = 0
                print('NO TRIANGLE DETECTED')
                break
            # centrU, centrV, angle = matrixHTM(centrU, centrV, centrAng)
            angle_values.append(angle)
            if len(angle_values) > 10:
                angle_values.pop(0)

            average_angle = np.sum(angle_values) / len(angle_values)

            centrU, centrV, null = matrixHTM(centrU, centrV, average_angle)

            # cv2.imshow("Camera Frame", dst)
            # --------------------------------------------------
            if doOneTime == 0:
                if cycle == 0:
                    condStop = ang_n - coordAng[-1]
                    if condStop < 0:  # turnClockwise
                        signAng = -1
                        sendRasp = 1
                    elif condStop > 0:  # turnCounterClockwise
                        signAng = 1
                        sendRasp = 2
                    textCond = 'Angulo de giro'
                    textCondCompl = '°'
                elif cycle == 1:
                    condStop = c_n
                    if c_n > 0:
                        sendRasp = 3
                    textCond = 'Distancia'
                    textCondCompl = 'px'
                doOneTime += 1
            '''if doOneTime == 0:
                if cycle == 0:
                    try:
                        condStop = ang_n - coordAng[-1]
                        if condStop > 0:  # turnClockwise
                            sendRasp = 1
                        elif condStop < 0:  # turnCounterClockwise
                            sendRasp = 2
                    except:
                        #                     pass
                        condStop = ang_n
                        if condStop < 0:  # turnClockwise
                            signAng = -1
                            sendRasp = 1
                        elif condStop > 0:  # turnCounterClockwise
                            signAng = 1
                            sendRasp = 2
                    textCond = 'Angulo de giro'
                    textCondCompl = '°'
                elif cycle == 1:
                    condStop = c_n
                    if c_n > 0:
                        sendRasp = 3
                    textCond = 'Distancia'
                    textCondCompl = 'px'
                doOneTime += 1'''

            match sendRasp:
                case 3:
                    cond = np.sqrt(float(centrU - coordU[-1]) ** 2 + float(centrV - coordV[-1]) ** 2)
                case other:
                    cond = average_angle - coordAng[-1]
            # --------------------------------------------------
            # --------------------------------------------------
            # YA NO DEBE RECIBIR DE LA RASPBERRY
            # --------------------------------------------------
            #             recvData, address = clientTCP.recvfrom(1024)
            #             dataPrint = recvData.decode('utf-8')
            #             cond = int(dataPrint)
            # --------------------------------------------------
            # --------------------------------------------------
            print(textCond, ': ', str(cond), textCondCompl)
            if cond >= condStop:
                sendRasp = 0
            cmd = str(sendRasp)
            cmdEncoded = cmd.encode('utf-8')
            clientTCP.send(cmdEncoded)
        if sendRasp == 0:
            print()  # print('\n')
            try:
                print('Llegaste al objetivo. ', textCond, ' final: ', str(condStop * signAng), textCondCompl)
            except:
                pass
            print()
        print()
        time.sleep(1)
    # --------------------------------------------------
    try:
        if coordU[-1]:
            coordU.append(centrU)
            coordV.append(centrV)
            coordAng.append(average_angle)
    except:
        pass
    time.sleep(1)
    return


while True:
    print('\n')
    userInputGame = input(
        'Que tipo de juego: \n 1) Single-Waypoint Navigation. \n 2) Multiple-Waypoints Navigation. \n')
    # ------------------------------------------------------------------ #
    # ----------------------PRIMER TIPO DE JUEGO------------------------ #
    if userInputGame == "1":
        inGame = 1
    # ------------------------------------------------------------------ #
    # ----------------------SEGUNDO TIPO DE JUEGO----------------------- #
    elif userInputGame == "2":
        inGame = 1
    else:
        inGame = 0
        print()
        print("Vuelve a intentarlo")

    while inGame:
        getOrigin = 0
        # ------------------------
        # Reinicia las coordenadas
        # ------------------------
        coord_n = []
        x_n = []
        y_n = []
        z_n = []
        ang_n = []
        c_n = []
        # ------------------------
        # ------------------------
        print()
        input_string = input('Dame la cordenada: \n')
        uncompressInput(input_string)
        print('\n')
        print('Coordenadas: ', str(coord_n))
        print('x_n: ', str(x_n))
        print('y_n: ', str(y_n))
        print('z_n: ', str(z_n))
        print('ang_n: ', str(ang_n))
        print('c_n: ', str(c_n))

        # Guarda las coordenadas iniciales del carrito de acuerdo con un eje coordenado normal

        if len(coord_n[-1]) > 0:
            for nCoord in range(len(coord_n[-1])):
                # sendRaspberry(nCoord, x_n, y_n, z_n, ang_n, c_n):
                sendRaspberry(ang_n[nCoord], c_n[nCoord], coordU, coordV, coordAng)
            #                 prevU, prevV, prevAng = sendRaspberry(x_n[nCoord], y_n[nCoord], ang_n[nCoord], c_n[nCoord], coordU, coordV):
            inGame = 0
            break
        else:
            print()
            print('Vuelve a intentarlo\n')