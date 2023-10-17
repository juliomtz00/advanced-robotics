import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# Open the csv file for reading
Data1 = pd.read_csv('ASPHALT01_UDEMPLBLD01F28MOD.csv', header=None)
Data2 = pd.read_csv('EARTH01_UDEMSCFLDF20MOD.csv', header=None)
Data3 = pd.read_csv('GRASS01_UDEMSMBRDF22MOD.csv', header=None)

#print(asphaltData[1])

# Create histograms
plt.figure(figsize=(8,6))

# Histogram for Data 1
hist1, bins1, _ = plt.hist(Data1[1], bins=15, density=True, alpha=0.7, color='blue', label='Data 1')
hist2, bins2, _ = plt.hist(Data2[1], bins=15, density=True, alpha=0.7, color='green', label='Data 2')
hist3, bins3, _ = plt.hist(Data3[1], bins=15, density=True, alpha=0.7, color='red', label='Data 3')

# Fit a normal distribution to Data 1
params1 = norm.fit(Data1[1])
params2 = norm.fit(Data2[1])
params3 = norm.fit(Data3[1])

methodOne = True
if methodOne:
    pdf1 = norm.pdf(bins1, loc=params1[0], scale=params1[1])
    pdf2 = norm.pdf(bins2, loc=params2[0], scale=params2[1])
    pdf3 = norm.pdf(bins3, loc=params3[0], scale=params3[1])
    # Overlay the PDF curve for Data 1 on the histogram
    plt.plot(bins1, pdf1, 'b-', linewidth=2, label='PDF1 of Asphalt')
    plt.plot(bins2, pdf2, 'g-', linewidth=2, label='PDF2 of Asphalt')
    plt.plot(bins3, pdf3, 'r-', linewidth=2, label='PDF3 of Asphalt')
else:
    x1 = np.linspace(np.min(Data1[1]), np.max(Data1[1]))
    x2 = np.linspace(np.min(Data2[1]), np.max(Data2[1]))
    x3 = np.linspace(np.min(Data3[1]), np.max(Data3[1]))
    # Overlay the PDF curve for Data 1 on the histogram
    plt.plot(x1, norm.pdf(x1,params1[0],params1[1]),)
    plt.plot(x2, norm.pdf(x2,params2[0],params2[1]),)
    plt.plot(x3, norm.pdf(x3,params3[0],params3[1]),)
plt.xlabel('Intensity')
plt.ylabel('Probability Density')
plt.show()
