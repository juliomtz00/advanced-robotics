import numpy as np

a = np.rad2deg(np.arctan((0.5-1.5)/(3-2)))
a2 = np.rad2deg(np.arctan2(0.5-1.5,3-2))

b = np.rad2deg(np.arctan((0.5-2)/(2.8-2)))
b2 = np.rad2deg(np.arctan2(0.5-2,2.8-2))

c = np.rad2deg(np.arctan(-0.9/-1))
c2 = np.rad2deg(np.arctan2(-0.9,-1))

d = np.rad2deg(np.arctan(-1.5/2.1))
d2 = np.rad2deg(np.arctan2(-1.5,2.1))

print(a,a2)
print(b,b2)
print(c,c2)
print(d,d2)