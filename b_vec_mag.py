#
#This script extracts vg_b_vol vector components and position into a .csv file from a .vlsv file. 
#Currently reads FHA run times 1001-1612 or trajectory in selected file
#Virtual spacecraft locations in meters.
#

import analysator as pt
import numpy as np
import csv
from rotation_matrix_leevi import get_sc_locations
#Virtual spacecraft (SC) locations 
#Multiply by R_e is coordinates in R_e 
R_e = 6371000

#Get different points from SC_constellations.txt

"""
Add the rotation matrix script here to automate making the points
Then scale factor multiplier to constellation size.
"""
points = get_sc_locations(rotation=0,translation=[-27,3,0.5])
#tail right z=0.5
sc1 = np.array([-27.0, 3.0, 0.5]) * R_e
sc2 = np.array([-26.0, 3.0, 1.5]) * R_e
sc3 = np.array([-26.0, 3.86602540, 0.0]) * R_e
sc4 = np.array([-26.0, 2.13397460, 0.0]) * R_e
sc5 = np.array([-26.85714286, 3.0, 0.64285714]) * R_e
sc6 = np.array([-26.85714286, 3.12371791, 0.42857143]) * R_e
sc7 = np.array([-26.85714286, 2.87628209, 0.42857143]) * R_e


points = [sc1,sc2,sc3,sc4,sc5,sc6,sc7]
points = get_sc_locations(rotation=0,translation=[-27,3,0.5])
points = points
print(points)
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

def Timeseries(var = "vg_b_vol", start_time= 1001,end_time=1613):
    header = ['Timeframe']
    for sc in range(1,len(points)+1):
        header.extend([f"sc{sc}_vg_B_x", f"sc{sc}_vg_B_y", f"sc{sc}_vg_B_z"])
    
    data = [header]

    #Constant constellation over moving simulation

    for t in range(start_time,end_time+1):
        #Open the .vlsv file 
        file = f"/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1/bulk1.000{t}.vlsv"
        print(file)
        vlsvfile = pt.vlsvfile.VlsvReader(file)
        vg_B_values = [t]
        #extract vg_b_vol values for all the points 
        for point in points:
            vg_B_value = vlsvfile.read_interpolated_variable('vg_b_vol', point)
            vg_B_values.extend(vg_B_value)
        data.append(vg_B_values)


    #create output .csv file 
    output_filename = '/home/leeviloi/plas_obs_vg_b_timeseries_tail_right_z=0.5.csv'
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    return 

def staticTime(start_point,end_point,time_step = 1432, N=100):

    file = f"/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1/bulk1.000{time_step}.vlsv"
    print(f"Simulation file: {file}")
    vlsvfile = pt.vlsvfile.VlsvReader(file)

    #Current constellation total distance is 7.211 R_e
    #For consistency between measurements try keeping about same
    #Or distance between measurements at about ~459.41km
    constellation = generate_constellation(N, points,np.array(start_point),np.array(end_point))
    #fly up start end: [6,-11,-1],[10,-5,-1], 100 measurements
    #start_point=[-29.11,3,0.5],end_point=[-22,3,0.5] tail 100 measurements
    #z rotation = 0, fly through
    #next [3.5,-10,-1] to [7.827,-10,-1] 60 measurements

    header = ['Position_Index']

    sc_keys = sorted(constellation.keys(), key=lambda x: int(x.replace('sc','')))
    for sc in sc_keys:
        header.extend([f'{sc}_pos_x', f'{sc}_pos_y', f'{sc}_pos_z',   
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

    #Check variable!!
    output_filename = '/home/leeviloi/plas_obs_vg_b_full_1432_fly_up_high_res+pos_z=-1_inner_scale=0.14.csv'  
    with open(output_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

#staticTime(start_point=[6,-11,-1],end_point=[10,-5,-1], N=200)
#Timeseries(start_time=1340,end_time=1372)