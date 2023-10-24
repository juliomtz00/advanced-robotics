# Import needed libraries
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import sympy as sp

# Use pandas to read the given csv file
baseballData = pd.read_csv('baseballDataset.csv', header=None)
angle = baseballData[0]
distance = baseballData[1] # In meters

# Convert the data from meters to feet
mToFt = 3.28084  # Conversion made as 1 meter = 3.28084 feet
distanceFt = distance * mToFt #pies

# Establish the coefficients
degree = 2
coeffs = np.polyfit(angle, distanceFt, degree)
polyFunction = np.poly1d(coeffs)


goalDistance = 180  # Switch the desired value to feet
y_pred = polyFunction(angle)

x = angle
y = distanceFt

x2 = np.power(x,2)
x3 = np.power(x,3)
x4 = np.power(x,4)
xy = x*y
x2y = x2*y

sumx = np.sum(x)
sumy = np.sum(y)
sumx2 = np.sum(x2)
sumx3 = np.sum(x3)
sumx4 = np.sum(x4)
sumxy = np.sum(xy)
sumx2y = np.sum(x2y)

a = sp.symbols('a')
b = sp.symbols('b')
c = sp.symbols('c')

matrix = sp.Matrix([[sumx4,sumx3,sumx2],[sumx3,sumx2,sumx],[sumx2,sumx,len(x)]])
vector = sp.Matrix([[a],[b],[c]])
final = sp.Matrix([[sumx2y],[sumxy],[sumy]])

equation = sp.Eq(matrix * vector, final)
solution = sp.solve(equation,[a,b,c])

SSE = [np.power(y[i] - solution[a]*x2[i] - solution[b]*x[i] - solution[c],2)for i in range(len(x))]
SST = [np.power(y[i]-np.average(y),2)for i in range(len(x))]

r2 = 1 - (np.sum(SSE)/np.sum(SST))

xd = sp.symbols('x')
yd = sp.symbols('y')

repPolyFunction = sp.Eq(solution[a]*xd**2 + solution[b]*xd + solution[c], yd)

realPolyFunction = sp.Eq(solution[a]*xd**2 + solution[b]*xd + solution[c], 180)
finalAngle = sp.nroots(realPolyFunction)

print("\n")
print("Polynomial Model:")
print(sp.latex(repPolyFunction))
print(f"R-squared (R²): {r2:.4f}")
print("\n")
print(f"El bateador tendra que darle entre: {finalAngle[1]:.4f} y {finalAngle[0]:.4f} grados")
print("\n")

plt.scatter(angle, distanceFt, label='Data Points', color='blue')
x_range = np.linspace(min(angle), max(angle), 100)
y_range = polyFunction(x_range)
plt.plot(x_range, y_range, label='Polynomial Model', color='red')

plt.xlabel('Angle')
plt.ylabel('Distance (m)')
plt.title('Baseball after being hit') 
plt.legend(loc='best')
plt.show()
