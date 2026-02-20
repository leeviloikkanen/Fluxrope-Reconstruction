import numpy as np

#define the vector to be rotated, the axis of rotation, and the angle in radians of how much to rotate it 

R_E = 6371000 
def rotate_vector(vector, axis, angle):
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])
        
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])
    
    return np.dot(rotation_matrix, vector)
def translate_vec(vector, translation):
    return vector+translation
#These are the coordinates of the tetra in R_e
x1, y1, z1 = 0, 0 ,0
x2, y2, z2 = 1, 0, 1
x3, y3, z3 = 1, np.sqrt(3)/2, -1/2
x4, y4, z4 = 1, -np.sqrt(3)/2, -1/2
#Scale of inner tetra compared to outer
in_scl = 7
x5,y5,z5, x6,y6,z6, x7,y7,z7=x2/in_scl, y2/in_scl, z2/in_scl, x3/in_scl, y3/in_scl, z3/in_scl, x4/in_scl, y4/in_scl, z4/in_scl
vecs = [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4],[x5,y5,z5],[x6,y6,z6],[x7,y7,z7]]

#GIVE ANGLE IN RADIANS NOT DEGREES
rotation = 0 #45 rad is about 58 degrees            
trans =[-27,3,0.5]

def get_sc_locations(rotation, translation,axis = "z", 
                     in_scl=7, scale_constellation=1.0):
   
    outer = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [1, np.sqrt(3)/2, -1/2],
        [1, -np.sqrt(3)/2, -1/2],
    ])
    
    inner = outer[1:] / in_scl

    all_pts = np.vstack((outer, inner)) * scale_constellation

    sc_locs = []
    for v in all_pts:
        #rotate vector and then traslate
        v_rot   = rotate_vector(v, axis, rotation)
        v_trans = translate_vec(v_rot, translation)
        sc_locs.append(v_trans)

    return np.vstack(sc_locs)

