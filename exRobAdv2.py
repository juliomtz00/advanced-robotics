import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import sympy as sp
import numpy as np

# Open the csv file for reading
Data1 = pd.read_csv('ASPHALT01_UDEMPLBLD01F28MOD.csv', header=None)
Data2 = pd.read_csv('EARTH01_UDEMSCFLDF20MOD.csv', header=None)
Data3 = pd.read_csv('GRASS01_UDEMSMBRDF22MOD.csv', header=None)
newData = pd.read_csv('dataSetF12.csv')

# Histogram for Data 1
hist1, bins1, _ = plt.hist(Data1[1], bins=15, density=True, alpha=0.7, color='red', label='Asphalt')
hist2, bins2, _ = plt.hist(Data2[1], bins=15, density=True, alpha=0.7, color='blue', label='Earth')
hist3, bins3, _ = plt.hist(Data3[1], bins=15, density=True, alpha=0.7, color='green', label='Grass')

# Normalize the histograms
hist1 /= np.sum(hist1)
hist2 /= np.sum(hist2)
hist3 /= np.sum(hist3)

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

# Create the list to save the max percentage names
percentage = []
type = ["Asphalt","Earth","Grass"]

# Loop to check all the points inside the list from the column intensity
for point in newData['intensity']:

    # Check points with the generated equations accordingly
    prob1 = (1/(sd1*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((point-mean1)/sd1)**2)
    prob2 = 1/(sd2*sp.sqrt(2*np.pi))*sp.exp(-(1/2)*((point-mean2)/sd1)**2)
    prob3 = 1/(sd3*sp.sqrt(2*np.pi))*sp.exp(-(1/2)*((point-mean3)/sd1)**2)

    # Obtain the max value from all three and save it in a variable
    maxProb = max(prob1,prob2,prob3)
    
    # Check which of the obtained values and save the corresponding name to the variable percentage list.
    if maxProb == prob1:
        percentage.append(type[0])
    elif maxProb == prob2: 
        percentage.append(type[1])
    else:
        percentage.append(type[2])

# Calculate the amount of percentage that was obtained for each type
asphaltPercentage = sum(1 for x in percentage if x == "Asphalt")/len(newData['intensity'])*100
earthPercentage = sum(1 for x in percentage if x == "Earth")/len(newData['intensity'])*100
grassPercentage = sum(1 for x in percentage if x == "Grass")/len(newData['intensity'])*100

# Calculate the max value from the percentages
result = max(asphaltPercentage,earthPercentage,grassPercentage)

# Check which one corresponds to the obtained percentages and save the name as observedTerrain
if result == asphaltPercentage:
    observedTerrain = "Asfalto"
elif result == earthPercentage: 
    observedTerrain = "Tierra"
else:
    observedTerrain = "Pasto"

# Print requested values
print(f"Cantidad de puntos analizados: {len(newData['intensity'])}")
print(f"Porcentaje de asphalto: {asphaltPercentage}")
print(f"Porcentaje de tierra: {earthPercentage}")
print(f"Porcentaje de pasto: {grassPercentage}")
print(f"Terreno observado en el dataset: {observedTerrain}")