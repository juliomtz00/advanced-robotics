from machine import Pin
import time

perfEncoder = 20
sensorA = Pin(18, Pin.IN)
sensorB = Pin(19, Pin.IN)

countA = 0
countB = 0
start = 0
end = 0

def set_start():
	global start
	start = time.ticks_ms()

def set_end():
	global end
	end = time.ticks_ms()

def get_rpmA(pinA):
	global countA
	
	if not countA:
		if not start:
			set_start()
		countA += 1
	else:
		set_end()
		countA += 1
	#print("COUNT A: ", countA)

def get_rpmB(pinB):
	global countB
	
	if not countB:
		if not start:
			set_start()
		countB += 1
	else:
		set_end()
		countB += 1
	#print("COUNT B: ", countB)

sensorA.irq(trigger=Pin.IRQ_RISING, handler=get_rpmA)
sensorB.irq(trigger=Pin.IRQ_RISING, handler=get_rpmB)

try:
	while True:
		try:
			if time.ticks_diff(end, start) >= 1000:
				rpmA = (60 / perfEncoder) * countA
				rpmB = (60 / perfEncoder) * countB
				print("Pin18, RPM: ", rpmA)
				print("Pin19, RPM: ", rpmB)
				start = end
				countA = 0
				countB = 0
				print('\n')
		except:
			pass
		time.sleep(0.1)
except KeyboardInterrupt:
	print("Quit")
