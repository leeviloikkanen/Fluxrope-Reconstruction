#This script was created by Leevi Loikkanen to do error analysis between Vlasiator simulation files 
#and reconstructed magnetic fields from virtual spacecraft data using Radial Basis Functions  
#by means of point-wise error and Wasserstein distances 
#RBF method and error analysis methods used here outlined in paper: 
#https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2023EA003369
#Specifically timeseries reconstruction of FHA run between 1340-1372s
#Determined Bulk velocity from 4 outside spacecraft average to avoid bias of inner tetrahedron. 
#Timeseries start when spacecrafts come in contact with flux rope structure and ends when last spacecraft loses contact
#Taylor's hypothesis is assumed to hold (at least for the fluxrope)
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors
import analysator as pt
import matplotlib as mpl

vg_v_x = -739256.9
vg_v_y = -268152.8
vg_v_z = 147101.5
vel_bulk = np.array([vg_v_x,vg_v_y,vg_v_z])

#TIP for future Leevi KEEP EVERYTHING IN METERS
#Time series reconstruction code structure
"""
Only main differences to static case: 
    -No constant vlasiator time to compare to (still have to figure this out in terms of coordinates and stuff)
    -Instead of having the centers ready from the file using the bulk velocity calculating the
     centers for each timestep.
    -other than that everything prettymuch the same. 
    IDEA: for first problem on list: select constant Vlasiator timestep where fluxrope is the best. have that basis for coordinates
    TODO first build timeseries RBFs and get basic 3 slice plots. everything else is after that. 
"""