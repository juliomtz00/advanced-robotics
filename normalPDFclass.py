import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#Generar datos ficticios
np.random.seed(0)
dataSet1 = np.random.normal(loc=5,scale=5,size=1000) 
dataSet2 = np.random.normal(loc=10,scale=3,size=1000)

#Definir tamaño de imagen
plt.figure(figsize=(5,5))

histDataSet1, bins1, _ = plt.hist(dataSet1,bins=20,density=True,color="blue",label="Dataset 1")
plt.xlabel('Variable')
plt.ylabel('Probability Density')

paramsDataset1 = norm.fit(dataSet1)
pdf1 = norm.pdf(bins1, loc=paramsDataset1[0], scale=paramsDataset1[1])
plt.plot(bins1, pdf1, 'r-', linewidth =2, label='PDF of Dataset 1')

plt.show()