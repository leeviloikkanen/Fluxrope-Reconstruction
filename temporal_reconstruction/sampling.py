"""

Move/adapt sampling scripts from RBF_time_reconstruction.py such that
changes in sampling techniques are easy to modify

"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/leeviloi/analysator-dev")
import analysator as pt; print(pt.__file__)
#TODO Needs sc_inits


def sample_slice(coord1, coord2, const_coord, plane, nx, ny, rbf):
    """
    Samples a slice of the RBF reconstruction at give coordinates
    :kword coord1: array of x coordinates
    :kword coord2: array of y coordinates
    :kword const_coord: Constant coordinate, last location coordinate of plane
    :kword plane: Plane wanted to be sliced
    
    Ex. use: 
        xs = np.linspace(bary[0]-L_m, bary[0]+L_m, nx)
        ys = np.linspace(bary[1]-L_m, bary[1]+L_m, ny)

        XY = sample_slice(xs, ys, bary[2], "xy", nx, ny)
    """
    if plane == "xy":
        X, Y = np.meshgrid(coord1, coord2)
        pts  = np.column_stack([X.ravel(), Y.ravel(),
                                np.full(X.size, const_coord)])
        Bxyz = rbf(pts)
        Bx, By, Bz = [Bxyz[:,i].reshape(ny,nx) for i in range(3)]
        return X, Y, Bx, By, Bz

    elif plane == "xz":
        X, Z = np.meshgrid(coord1, coord2)
        pts  = np.column_stack([X.ravel(),
                                np.full(X.size, const_coord),
                                Z.ravel()])
        Bxyz = rbf(pts)
        Bx, By, Bz = [Bxyz[:,i].reshape(ny,nx) for i in range(3)]
        return X, Z, Bx, Bz, By

    elif plane == "yz":
        Y, Z = np.meshgrid(coord1, coord2)
        pts  = np.column_stack([np.full(Y.size, const_coord),
                                Y.ravel(), Z.ravel()])
        Bxyz = rbf(pts)
        Bx, By, Bz = [Bxyz[:,i].reshape(ny,nx) for i in range(3)]
        return Y, Z, By, Bz, Bx
    else:
        raise "Invalid Plane, Options: xy, xz, yz"     

def sample_slice_vlas(vlsvfile = None, plane = None, time = None, nx = 200, ny = 200,L_Re = 1.2):
    
    #file
    if time != None:
        file = f"/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1/bulk1.000{time}.vlsv"
        print(file)
        vlsvfile = pt.vlsvfile.VlsvReader(file)
    elif vlsvfile != None:
        vlsvfile = vlsvfile
    else:
        raise "Provide vlasiator file or time"
    if plane == None:
        raise "Provide plane to slice"
    init_pts = np.vstack(list(sc_init.values()))
    bary = init_pts.mean(axis=0)
   
    """
    main thing to note about this function is that the output
    order of coordinates is dependant on chosen plane
    ex. yz plane will output coordinates as Y, Z, By, Bz, Bx
    Out of plane component will always be last
    """
    L_m = L_Re*R_e
    if plane == "xy":
        coord1 = np.linspace(bary[0]-L_m,bary[0]+L_m,nx)
        coord2 = np.linspace(bary[1]-L_m,bary[1]+L_m,ny)
        
        const_coord = bary[2]
        X, Y = np.meshgrid(coord1, coord2)                 
        pts  = np.column_stack([X.ravel(), Y.ravel(),
                                np.full(X.size, const_coord)])
        Bxyz = vlsvfile.read_interpolated_variable("vg_b_vol", pts)
        Bx, By, Bz = (Bxyz[:, i].reshape(nx, ny) for i in range(3))
        return X, Y, Bx, By, Bz                            
    elif plane == "xz":
        coord1 = np.linspace(bary[0]-L_m,bary[0]+L_m,nx)
        coord2 = np.linspace(bary[2]-L_m,bary[2]+L_m,ny)
        const_coord = bary[1]
        X, Z = np.meshgrid(coord1, coord2)                 
        pts  = np.column_stack([X.ravel(),
                                np.full(X.size, const_coord),
                                Z.ravel()])
        Bxyz = vlsvfile.read_interpolated_variable("vg_b_vol", pts)
        Bx, By, Bz = (Bxyz[:, i].reshape(nx, ny) for i in range(3))
        return X, Z, Bx, Bz, By                            
    elif plane == "yz":
        coord1 = np.linspace(bary[1]-L_m,bary[1]+L_m,nx)
        coord2 = np.linspace(bary[2]-L_m,bary[2]+L_m,ny)
        const_coord = bary[0]
        Y, Z = np.meshgrid(coord1, coord2)                
        pts  = np.column_stack([np.full(Y.size, const_coord),
                                Y.ravel(), Z.ravel()])
        Bxyz = vlsvfile.read_interpolated_variable("vg_b_vol", pts)
        Bx, By, Bz = (Bxyz[:, i].reshape(nx, ny) for i in range(3))
        return Y, Z, By, Bz, Bx   
    else:
        raise "Invalid Plane, Options: xy, xz, yz"  

def sample_slice_vlas_coords(time, coord1, coord2, const_coord, plane, nx = 200, ny = 200):
    """
    main thing to note about this function is that the output
    order of coordinates is dependant on chosen plane
    ex. yz plane will output coordinates as Y, Z, By, Bz, Bx
    Out of plane component will always be last
    """
    file = f"/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1/bulk1.000{time}.vlsv"
    print(file)
    vlsvfile = pt.vlsvfile.VlsvReader(file)
    

    if plane == "xy":
        X, Y = np.meshgrid(coord1, coord2)                 
        pts  = np.column_stack([X.ravel(), Y.ravel(),
                                np.full(X.size, const_coord)])
        Bxyz = vlsvfile.read_interpolated_variable("vg_b_vol", pts)
        Bx, By, Bz = (Bxyz[:, i].reshape(nx, ny) for i in range(3))
        return X, Y, Bx, By, Bz                            
    elif plane == "xz":
        X, Z = np.meshgrid(coord1, coord2)                 
        pts  = np.column_stack([X.ravel(),
                                np.full(X.size, const_coord),
                                Z.ravel()])
        Bxyz = vlsvfile.read_interpolated_variable("vg_b_vol", pts)
        Bx, By, Bz = (Bxyz[:, i].reshape(nx, ny) for i in range(3))
        return X, Z, Bx, Bz, By                            
    elif plane == "yz":
        Y, Z = np.meshgrid(coord1, coord2)                
        pts  = np.column_stack([np.full(Y.size, const_coord),
                                Y.ravel(), Z.ravel()])
        Bxyz = vlsvfile.read_interpolated_variable("vg_b_vol", pts)
        Bx, By, Bz = (Bxyz[:, i].reshape(nx, ny) for i in range(3))
        return Y, Z, By, Bz, Bx                           


def sample_slice_any_plane():
    """
    
    """
    return