import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import sympy as sp
import numpy as np

# Open the csv file for reading
Data1 = pd.read_csv('ASPHALT01_UDEMPLBLD01F28MOD.csv', header=None)
Data2 = pd.read_csv('EARTH01_UDEMSCFLDF20MOD.csv', header=None)
Data3 = pd.read_csv('GRASS01_UDEMSMBRDF22MOD.csv', header=None)



#print(asphaltData[1])

# Create histograms
plt.figure(figsize=(8,6))

# Histogram for Data 1
hist1, bins1, _ = plt.hist(Data1[1], bins=15, density=True, alpha=0.7, color='red', label='Asphalt')
hist2, bins2, _ = plt.hist(Data2[1], bins=15, density=True, alpha=0.7, color='blue', label='Earth')
hist3, bins3, _ = plt.hist(Data3[1], bins=15, density=True, alpha=0.7, color='green', label='Grass')

# Obtain the mean from the data
mean1 = np.sum(0.5 * (bins1[1:] + bins1[:-1]) * hist1)
mean2 = np.sum(0.5 * (bins2[1:] + bins2[:-1]) * hist2)
mean3 = np.sum(0.5 * (bins3[1:] + bins3[:-1]) * hist3)

# Compute the standard deviation
sd1 = np.sqrt(np.sum(hist1 * ((0.5 * (bins1[1:] + bins1[:-1]) - mean1) ** 2)))
sd2 = np.sqrt(np.sum(hist2 * ((0.5 * (bins2[1:] + bins2[:-1]) - mean2) ** 2)))
sd3 = np.sqrt(np.sum(hist3 * ((0.5 * (bins3[1:] + bins3[:-1]) - mean3) ** 2)))

# Fit a normal distribution to Data 1
params1 = norm.fit(Data1[1])
params2 = norm.fit(Data2[1])
params3 = norm.fit(Data3[1])

# Compute the particular equation
x,fx = sp.symbols('x,f(x)')
eq1 = sp.pretty(sp.Eq(fx,1/(sd1*sp.sqrt(2*np.pi))*sp.exp(-(1/2)*((x-mean1)/sd1)**2),simplify=False),use_unicode=True)
eq2 = sp.Eq(fx,1/(x*sd2*sp.sqrt(2*np.pi))*sp.exp(-(1/2)*((sp.log(x)-mean2)**2/(sd1)**2)),simplify=False)
eq3 = sp.Eq(fx,1/(sd3*sp.sqrt(2*np.pi))*sp.exp(-(1/2)*((x-mean3)/sd1)**2),simplify=False)


methodOne = True
if methodOne:
    pdf1 = norm.pdf(bins1, loc=params1[0], scale=params1[1])
    pdf2 = norm.pdf(bins2, loc=params2[0], scale=params2[1])
    pdf3 = norm.pdf(bins3, loc=params3[0], scale=params3[1])
    # Overlay the PDF curve for Data 1 on the histogram
    plt.plot(bins1, pdf1, 'r-', linewidth=2, label=f'PDF Asphalt, $\mu$:{round(mean1,3)}, $\sigma$:{round(sd1,3)}')
    plt.plot(bins2, pdf2, 'b-', linewidth=2, label=f'PDF Earth, $\mu$:{round(mean2,3)}, $\sigma$:{round(sd2,3)}')
    plt.plot(bins3, pdf3, 'g-', linewidth=2, label=f'PDF Grass, $\mu$:{round(mean3,3)}, $\sigma$:{round(sd3,3)}')

else:
    x1 = np.linspace(np.min(Data1[1]), np.max(Data1[1]))
    x2 = np.linspace(np.min(Data2[1]), np.max(Data2[1]))
    x3 = np.linspace(np.min(Data3[1]), np.max(Data3[1]))
    # Overlay the PDF curve for Data 1 on the histogram
    plt.plot(x1, norm.pdf(x1,params1[0],params1[1]),)
    plt.plot(x2, norm.pdf(x2,params2[0],params2[1]),)
    plt.plot(x3, norm.pdf(x3,params3[0],params3[1]),)

# Plot the equations
plt.title("Probability Density Fit Over LiDAR Data")
plt.legend()

plt.xlabel('Intensity')
plt.ylabel('Probability Density')
plt.show()
