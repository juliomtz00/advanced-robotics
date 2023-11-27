import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

# a) Prepara una gráfica 3D tipo scatter plot de los datos.

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'JDC02_TRCF2.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Create a scatter plot
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Points:0'], df['Points:1'], df['Points:2'])

# Add labels and a title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('LiDAR scan data')

# Aplica una operación Monadic sobre los datos de la imagen tal que extraigas toda la información de la región central de la imagen LiDAR.

dist_filtered_df = df[df['distance_m']<=10]

# Create a scatter plot
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(dist_filtered_df['Points:0'], dist_filtered_df['Points:1'], dist_filtered_df['Points:2'])

# Add labels and a title
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.set_zlabel('Z-axis')
ax2.set_title('LiDAR Scan Data with Distance less than 10 meters')

# Show the plot
plt.show()

# Del nuevo conjunto de datos, usa la información que corresponde a todos los puntos del laser_id 5 para realizar lo siquiente.
# i. Aplica un filtro promediador de 1x3 a la columna de distancia

df['distance_filt3'] = df['distance_m'].rolling(window=3, center=True).mean()

# Al resultado del filtro promediador aplica un segundo filtro de gradiente simétrico de 1x7.

# Define the weights for the symmetric gradient 1x7 filter
weights = np.array([-1, -2, -1, 0, 0, 1, 2, 1])

# Apply a 1x7 symmetric gradient filter to the specified column
df['distance_gradient'] = df['distance_filt3'].rolling(window=7, center=True).apply(lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum(), raw=True)

# iii. Prepara el siguiente conjunto de gráficas 2D

# Create a 2x2 grid of subplots, and plot each graph in a different subplot
plt.figure(figsize=(10, 6))

# Plot the first graph
plt.subplot(2, 2, 1)
plt.plot(df['azimuth'], df['distance_m'], label='Distance')
plt.title('Azimuth vs Distance')
plt.legend()

# Plot the second graph
plt.subplot(2, 2, 2)
plt.plot(df['azimuth'], df['distance_filt3'], label='Distance with Average Filter', color='orange')
plt.title('Azimuth vs Distance with Average Filter')
plt.legend()

# Plot the third graph
plt.subplot(2, 2, 3)
plt.plot(df['azimuth'], df['distance_gradient'], label='Distance with Gradient', color='green')
plt.title('Azimuth vs Distance with Gradient')
plt.legend()

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()