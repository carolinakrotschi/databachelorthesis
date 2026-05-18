# @sergio.revuelta@mpq.mpg.de

# Import libraries
import numpy as np
import laserbeamsize as lbs
import matplotlib.pyplot as plt

# function to read the file
def read_txt_data(file_path):
    data = np.loadtxt(file_path, skiprows=1)
    z1_all = data[:, 0] * 1e-3
    d1_all = data[:, 1] * 1e-3 #my data is in mm
    d2_all = data[:, 2] * 1e-3
    return z1_all, d1_all, d2_all

# Files paths
file_path = r"/Users/carolinakrotsch/Library/Mobile Documents/com~apple~CloudDocs/Studium/Bachelorarbeit/Lab/Messdaten/Msquared/rawdata/uniphase_1507p_v4.txt"
file_path_save = r'/Users/carolinakrotsch/Library/Mobile Documents/com~apple~CloudDocs/Studium/Bachelorarbeit/Lab/Messdaten/Msquared/results'

# Calling function to read the file
z1_all, d1_all, d2_all = read_txt_data(file_path)

# Central wavelength
lambda1 = 632e-9

# Getting values of M2 for both axis 
lbs.M2_radius_plot(z1_all, d1_all, lambda1, strict=True)
plt.show()
lbs.M2_radius_plot(z1_all, d2_all, lambda1, strict=True)
plt.show()

# Getting fits of the analyzed data
t_valueM2d1 = lbs.M2_fit(z1_all, d1_all, lambda1, strict=True)
t_valueM2d2 = lbs.M2_fit(z1_all, d2_all, lambda1, strict=True)

# Reading the values from the Matrix generated in the lbs code
a_fitd1 = t_valueM2d1[0]
a_fitd2 = t_valueM2d2[0]

v_dnewd1 = a_fitd1[0]
v_znewd1 = a_fitd1[1]
v_thetanewd1 = a_fitd1[2]
v_M2newd1 = a_fitd1[3]
v_zrnewd1 = a_fitd1[4]

v_dnewd2 = a_fitd2[0]
v_znewd2 = a_fitd2[1]
v_thetanewd2 = a_fitd2[2]
v_M2newd2 = a_fitd2[3]
v_zrnewd2 = a_fitd2[4]

# Calculating values from the the equations of gaussian beams
a_d1ne = (v_dnewd1**2) + (v_thetanewd1**2)*((z1_all-v_znewd1)**2)
a_d2ne = (v_dnewd2**2) + (v_thetanewd2**2)*((z1_all-v_znewd2)**2)

a_d1new = np.sqrt(a_d1ne)
a_d2new = np.sqrt(a_d2ne)

a_z1new = np.arange(np.min(z1_all), np.max(z1_all), ((z1_all[1]-z1_all[0])*1e-2))

a_d1newlong = np.interp(a_z1new,z1_all,a_d1new)
a_d2newlong = np.interp(a_z1new,z1_all,a_d2new)

plt.plot(a_z1new,a_d1newlong,'b-',z1_all,d1_all,'b.')
plt.plot(a_z1new,a_d2newlong,'r-',z1_all,d2_all,'r.')

# Saving values
with open('d1_variables.txt', 'w') as f:
    f.write("v_dnewd1\tv_znewd1\tv_thetanewd1\tv_M2newd1\tv_zrnewd1\n")  # Column headers
    f.write(f"{v_dnewd1:.6e}\t{v_znewd1:.6e}\t{v_thetanewd1:.6e}\t{v_M2newd1:.6e}\t{v_zrnewd1:.6e}\n")

# Save variables for d2 in column format
with open('d2_variables.txt', 'w') as f:
    f.write("v_dnewd2\tv_znewd2\tv_thetanewd2\tv_M2newd2\tv_zrnewd2\n")  # Column headers
    f.write(f"{v_dnewd2:.6e}\t{v_znewd2:.6e}\t{v_thetanewd2:.6e}\t{v_M2newd2:.6e}\t{v_zrnewd2:.6e}\n")

# Save arrays for d1
with open('d1_arrays.txt', 'w') as f:
    f.write("Arrays for d1:\n")
    f.write("z1_new (m)\t d1_new_long (m)\n")
    for z, d1 in zip(a_z1new, a_d1newlong):
        f.write(f"{z:.6e}\t{d1:.6e}\n")

# Save arrays for d2
with open('d2_arrays.txt', 'w') as f:
    f.write("Arrays for d2:\n")
    f.write("z1_new (m)\t d2_new_long (m)\n")
    for z, d2 in zip(a_z1new, a_d2newlong):
        f.write(f"{z:.6e}\t{d2:.6e}\n")
