from matplotlib import pyplot as plt
import matplotlib
import pandas as pd

df = pd.read_csv('./LaserTherm/lightmatter_cyclic.csv')
# Columns are Lane, Wavelength [nm], Insertion Loss [dB]
# Plot wavelengths for lane 0 to 15
matplotlib.use('Qt5Agg')
fig, ax = plt.subplots()

for i in range(df['Lane'].unique().size):
    ax.plot(df['Wavelength [nm]'][df['Lane'] == i], df['Insertion Loss [dB]'][df['Lane'] == i])

ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Insertion Loss (dB)')
ax.grid()
plt.show()