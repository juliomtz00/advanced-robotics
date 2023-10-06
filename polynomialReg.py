import numpy as np
import sympy as sp

x = np.array([3,4,5,6,7])
y = np.array([2.5,3.2,3.8,6.5,12])

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

print("X    Y   x2    x3    x4    xy    x2y")
for i in range(5):
    print(x[i],y[i],x2[i],x3[i],x4[i],xy[i],x2y[i])
print(sumx,sumy,sumx2,sumx3,sumx4,sumxy,sumx2y)

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

