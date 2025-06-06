#
#This script extracts vg_b_vol vector components and position into a .csv file from a .vlsv file. 
#Currently reads FHA run times 1001-1612
#Virtual spacecraft locations in meters.
#

import pytools as pt
import numpy as np
import csv
#Virtual spacecraft (SC) locations 
#Multiply by R_e is coordinates in R_e 
R_e = 6371000
#z-axis 135 deg
"""
sc1 = [ 38226000., -70081000.,0.]
sc2 = [ 33721022.6970605, -65576022.6970605,6371000.]
sc3 = [ 29819597.9092426, -69477447.4848784,-3185500.]
sc4 = [ 37622447.48487841, -61674597.90924259,-3185500.]
sc5 = [ 37582431.81386579, -69437431.81386578,910142.85714286]
sc6 = [ 37025085.41560608, -69994778.21212548,-455071.42857143]
sc7 = [ 38139778.21212549, -68880085.4156061,-455071.42857143]
"""
#Virtual Spacecraft Constellation
#z-axis 58 deg
sc1 = np.array([6.0, -11.0, 0.0]) * R_e
sc2 = np.array([6.52532199, -10.14909648, 1.0]) * R_e
sc3 = np.array([5.78841792, -9.69415429, -0.5]) * R_e
sc4 = np.array([7.26222606, -10.60403866, -0.5]) * R_e
sc5 = np.array([6.075046, -10.87844235, 0.14285714]) * R_e
sc6 = np.array([5.96977399, -10.81345061, -0.07142857]) * R_e
sc7 = np.array([6.18031801, -10.94343409, -0.07142857]) * R_e

points = [sc1,sc2,sc3,sc4,sc5,sc6,sc7]
#Create a linspace of points for each spacecrafts trajectory based on start and end point of mothercraft
#Start and end points given in R_e (6371km)
def generate_constellation(N, points, start_point, end_point):
    
    R_e = 6371000 
    # Define the start and end positions for spacecraf 1
    sc1_start = start_point * R_e
    sc1_end   = end_point * R_e
    
    # Original position
    sc1_original = points[0]
    sc2_original = points[1]
    sc3_original = points[2]
    sc4_original = points[3]
    sc5_original = points[4]
    sc6_original = points[5]
    sc7_original = points[6]
    # Calculate the offsets

    offsets = {
        "sc2": sc2_original - sc1_original,
        "sc3": sc3_original - sc1_original,
        "sc4": sc4_original - sc1_original,
        "sc5": sc5_original - sc1_original,
        "sc6": sc6_original - sc1_original,
        "sc7": sc7_original - sc1_original,
    }
    
    
    sc1_positions = np.linspace(sc1_start, sc1_end, N)
    #print(sc1_positions)
    constellation_positions = {"sc1": sc1_positions}
    for sc_name, offset in offsets.items():
        constellation_positions[sc_name] = sc1_positions + offset  # offset is applied to every generated position
    
    return constellation_positions

"""
for i in range(1001,1613):
    file =  f"/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1/bulk1.000{i}.vlsv"
    f = pt.vlsvfile.VlsvReader(file)
    vg_B_value = f.read_interpolated_variable('vg_b_vol', )
"""
"""  
#time = 1432

header = ['Timeframe']
for i in range(len(points)):
    header.extend([f'vg_B_x_point{i+1}', f'vg_B_y_point{i+1}', f'vg_B_z_point{i+1}'])
data = [header]

#Constant constellation over moving simulation

for t in range(1001,1613):
    #Open the .vlsv file 
    file = f"/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1/bulk1.000{t}.vlsv"
    print(file)
    vlsvfile = pt.vlsvfile.VlsvReader(file)
    vg_B_values = [t]
    #extract vg_b_vol values for all the points 
    for point in points:
        vg_B_value = vlsvfile.read_interpolated_variable('vg_v', point)
        vg_B_values.extend(vg_B_value)
    data.append(vg_B_values)


#create output .csv file path
output_filename = '/home/leeviloi/plas_obs_vir_vg_v_full_45deg.csv' #<--- replace with own filepath
with open(output_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
"""

time_step = 1432 
file = f"/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1/bulk1.000{time_step}.vlsv"
print(f"Simulation file: {file}")
vlsvfile = pt.vlsvfile.VlsvReader(file)


N = 100  #number of measurements
constellation = generate_constellation(100, points,np.array([6,-11,-1]),np.array([10,-5,-1]))


header = ['Position_Index']

sc_keys = sorted(constellation.keys(), key=lambda x: int(x.replace('sc','')))
for sc in sc_keys:
    header.extend([f'{sc}_pos_x', f'{sc}_pos_y', f'{sc}_pos_z',   # NEW
                   f'{sc}_vg_B_x', f'{sc}_vg_B_y', f'{sc}_vg_B_z'])

data = [header]

for i in range(N):
    row = [i]
    for sc in sc_keys:
        point = constellation[sc][i] 
        row.extend(point) 
        vg_B_value = vlsvfile.read_interpolated_variable('vg_b_vol', point)
        
        row.extend(vg_B_value)
    data.append(row)

output_filename = '/home/leeviloi/plas_obs_vg_b_full_1432_fly_up+pos_z=-1.csv'  
with open(output_filename, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

