import numpy as np

a = np.rad2deg(np.arctan(1.5/2.1))
a2 = np.rad2deg(np.arctan2(1.5,2.1))

b = np.rad2deg(np.arctan(0.9/-1))
b2 = np.rad2deg(np.arctan2(0.9,-1))

c = np.rad2deg(np.arctan(-0.9/-1))
c2 = np.rad2deg(np.arctan2(-0.9,-1))

d = np.rad2deg(np.arctan(-1.5/2.1))
d2 = np.rad2deg(np.arctan2(-1.5,2.1))

print(a,a2)
print(b,b2)
print(c,c2)
print(d,d2)