import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from sklearn.neighbors import NearestNeighbors


df = pd.read_csv('plas_obs_vir_vg_b_full_45deg.csv', sep=',')
times = df['Timeframe'].to_numpy()        # shape (T,)
T = len(times)

Bcols = []
for i in range(1,8):
    Bcols += [f'vg_B_x_point{i}', f'vg_B_y_point{i}', f'vg_B_z_point{i}']
B = df[Bcols].to_numpy().reshape(T, 7, 3)  # (T, 7, 3)

R_e = 6371.0  # km
sc1 = np.array([ 6.0, -11.0,   0.0]) * R_e
sc2 = np.array([ 6.5253, -10.1491,  1.0]) * R_e
sc3 = np.array([ 5.7884,  -9.6942, -0.5]) * R_e
sc4 = np.array([ 7.2622, -10.6040, -0.5]) * R_e
sc5 = np.array([ 6.0750, -10.8784,  0.1429]) * R_e
sc6 = np.array([ 5.9698, -10.8135, -0.0714]) * R_e
sc7 = np.array([ 6.1803, -10.9434, -0.0714]) * R_e

static_pos = np.vstack([sc1, sc2, sc3, sc4, sc5, sc6, sc7])  # (7,3)

#Taylor’s hypothesis (spacecraft fixed, plasma flows by)

v_sw = np.array([0.25, 0.25, 0.0])  
dt   = 1                      # seconds per Timeframe tick

# Build spatial “centers” for RBF: shape (T*7, 3)
# r_eff[i,j] = static_pos[i] + v_sw * (times[j]*dt)
centers = (
    static_pos[None, :, :]                 # shape (1,7,3)
    + (times * dt)[:, None, None] * v_sw[None, None, :]
).reshape(-1, 3)
# Flatten B-field values to (T*7, 3)
values = B.reshape(-1, 3)

#pick epsilon
"https://www.math.iit.edu/~fass/Dolomites.pdf?" #nearest neighbor method mentioned
nbrs = NearestNeighbors(n_neighbors=2).fit(centers)
dists, _ = nbrs.kneighbors(centers)
epsilon = np.median(dists[:,1])
print(f"RBF epsilon = {epsilon:.3g} km")

#RBF interpolation
rbf = RBFInterpolator(
    centers, values,
    kernel='multiquadric',
    epsilon=epsilon,
    smoothing=0.0
)
#Plotting


from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D


def unit_B(r_point):
    B = rbf(r_point.reshape(1,3))[0]
    norm = np.linalg.norm(B)
    return (B / norm) if norm > 0 else np.zeros(3)


def trace_field_line(seed, s_max=2e4, ds=500.0):
    # Ode: dr/ds = B_hat(r)
    def ode(s, y):
        return unit_B(np.array([y[0], y[1], y[2]]))
    
    sol_f = solve_ivp(ode, [0, s_max], seed, max_step=ds)
    sol_b = solve_ivp(ode, [0, -s_max], seed, max_step=ds)
    
    pts = np.vstack((sol_b.y.T[::-1], sol_f.y.T))
    return pts  # shape (~M,3)
#Points other than spacecrafts
np.random.seed(42)              
n_extra = 4                    #seeds per spacecraft
r_seed  = 0.5 * R_e             
                            
def random_points_in_sphere(center, radius, n):
    u = np.random.normal(size=(n, 3))          
    u /= np.linalg.norm(u, axis=1)[:, None]    
    r = radius * np.cbrt(np.random.rand(n, 1))
    return center + r * u
extra_seeds = np.vstack([
    random_points_in_sphere(sc, r_seed, n_extra) for sc in static_pos
])                                              # shape (7·n_extra, 3)

seeds = np.vstack([static_pos, extra_seeds])    # original 7 + extra seeds


lines = [trace_field_line(seed) for seed in seeds]

fig = plt.figure(figsize=(9,7))
ax  = fig.add_subplot(111, projection='3d')

orig, extra = lines[:len(static_pos)], lines[len(static_pos):]

for pts in orig:
    ax.plot(pts[:,0], pts[:,1], pts[:,2], lw=2.2, color='tab:red',
            label='spacecraft seed' if 'spacecraft seed' not in ax.get_legend_handles_labels()[1] else "")

for pts in extra:
    ax.plot(pts[:,0], pts[:,1], pts[:,2], lw=1.0, alpha=0.5, color='steelblue',
            label='random seed' if 'random seed' not in ax.get_legend_handles_labels()[1] else "")

ax.scatter(static_pos[:,0], static_pos[:,1], static_pos[:,2],
           c='k', s=60, depthshade=False, label='spacecraft')

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('RBF Field Lines: spacecraft & nearby random seeds')
ax.legend(loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
"""
"""
seeds = static_pos.copy()

lines = [trace_field_line(seed) for seed in seeds]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Plot each field line
for pts in lines:
    ax.plot(pts[:,0], pts[:,1], pts[:,2], linewidth=1.5)

# Plot spacecraft
ax.scatter(static_pos[:,0], static_pos[:,1], static_pos[:,2],
           c='k', s=50, label='spacecraft')

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('3D Field Lines from RBF-Interpolated B')
plt.legend()
plt.tight_layout()
plt.show()

"""

#2D B_z slice in the X-Z plane
pad = 2000.0  # km margin
x_min, x_max = static_pos[:,0].min() - pad, static_pos[:,0].max() + pad
z_min, z_max = static_pos[:,2].min() - pad, static_pos[:,2].max() + pad

Y0 = static_pos[:,1].mean()

n = 200  # resolution
xs = np.linspace(x_min, x_max, n)
zs = np.linspace(z_min, z_max, n)
X, Z = np.meshgrid(xs, zs, indexing='xy')  # X.shape = Z.shape = (n,n)

pts = np.column_stack([
    X.ravel(),
    np.full(X.size, Y0),
    Z.ravel()
])

Bq = rbf(pts)            # shape (n*n, 3)
Bz = Bq[:, 2].reshape(X.shape)

#plot
plt.figure(figsize=(6,6))
cf = plt.contourf(xs, zs, Bz, levels=50, cmap='plasma')
plt.scatter(static_pos[:,0], static_pos[:,2],
            c='k', s=40, label='spacecraft')
plt.colorbar(cf, label='$B_z$')
plt.xlabel('X (km)')
plt.ylabel('Z (km)')
plt.title(f'RBF‐reconstructed $B_z$ at y={Y0:.0f} km')
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()

"""
"""

#2D B_z slice in X,Y plane 
pad = 2000.0  # km of extra margin
x_min = static_pos[:,0].min() - pad
x_max = static_pos[:,0].max() + pad
y_min = static_pos[:,1].min() - pad
y_max = static_pos[:,1].max() + pad

n = 200

xs = np.linspace(x_min, x_max, n)
ys = np.linspace(y_min, y_max, n)
X, Y = np.meshgrid(xs, ys, indexing='xy')

Z0 = static_pos[:,2].mean()
pts = np.column_stack([
    X.ravel(),
    Y.ravel(),
    np.full(X.size, Z0)
])

Bq = rbf(pts)
Bz = Bq[:,2].reshape(X.shape)

#plot
plt.figure(figsize=(6,6))
cf = plt.contourf(xs, ys, Bz, levels=50, cmap='plasma')
plt.scatter(static_pos[:,0], static_pos[:,1],
            c='k', s=40, label='spacecraft')
plt.colorbar(cf, label='$B_z$ (same units as file)')
plt.xlabel('X (km)'); plt.ylabel('Y (km)')
plt.axis('equal')
plt.title(f'RBF‐reconstructed $B_z$ at z={Z0:.0f} km')
plt.legend()
plt.tight_layout()
plt.show()
"""
"""
#x-y plane slice with stream lines
z_plane = 0.0                            # km
nx, ny  = 200, 200                       # resolution

buf   = 2.0 * R_e                        # plot extends ±2 R_E past craft
xmin, xmax = static_pos[:,0].min()-buf, static_pos[:,0].max()+buf
ymin, ymax = static_pos[:,1].min()-buf, static_pos[:,1].max()+buf

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

pts   = np.column_stack([X.ravel(), Y.ravel(),
                         np.full(X.size, z_plane)])
Bxyz  = rbf(pts)                         # shape (nx*ny , 3)
Bx, By, Bz = (Bxyz[:,i].reshape(nx, ny) for i in range(3))

fig, ax = plt.subplots(figsize=(7,6))

# filled contours of the out-of-plane component
cf = ax.contourf(X/1e3, Y/1e3, Bz, 30, cmap='coolwarm') 
cbar = fig.colorbar(cf, ax=ax, label='Bz  (nT)')

# stream-lines
speed = np.hypot(Bx, By)
strm = ax.streamplot(X/1e3, Y/1e3, Bx, By,
                     color=speed, cmap='magma', density=2)

# spacecraft markers
ax.scatter(static_pos[:,0]/1e3, static_pos[:,1]/1e3,
           c='k', s=40, label='spacecraft')

ax.set_xlabel('X  (10³ km)')
ax.set_ylabel('Y  (10³ km)')
ax.set_aspect('equal')
ax.set_title('Equatorial slice  (z = 0 km)')
ax.legend(loc='upper right', fontsize='small')
plt.tight_layout()
plt.show()
"""
"""
#x-z slice with stream lines
y_plane = np.median(static_pos[:, 1])      

# grid size
nx, nz = 200, 200

buf = 2.0 * R_e                            # 2 R_E margin
xmin, xmax = static_pos[:,0].min() - buf, static_pos[:,0].max() + buf
zmin, zmax = static_pos[:,2].min() - buf, static_pos[:,2].max() + buf

x = np.linspace(xmin, xmax, nx)            
z = np.linspace(zmin, zmax, nz)

#
X, Z = np.meshgrid(x, z)                  

pts   = np.column_stack([X.ravel(),
                         np.full(X.size, y_plane),   
                         Z.ravel()])
Bxyz  = rbf(pts)
Bx = Bxyz[:,0].reshape(Z.shape)
By = Bxyz[:,1].reshape(Z.shape)            
Bz = Bxyz[:,2].reshape(Z.shape)

# speed for colouring stream lines
speed = np.hypot(Bx, Bz)


fig, ax = plt.subplots(figsize=(7, 6))

cf = ax.contourf(X/1e3, Z/1e3, By, 30, cmap='coolwarm')
fig.colorbar(cf, ax=ax, label='B$_y$  (nT)')

# stream-lines of the in-plane field
strm = ax.streamplot(x/1e3, z/1e3, Bx, Bz,
                     color=speed, cmap='magma', density=2)

# spacecraft markers
ax.scatter(static_pos[:,0]/1e3, static_pos[:,2]/1e3,
           c='k', s=40, label='spacecraft')

ax.set_xlabel('X  (10³ km)')
ax.set_ylabel('Z  (10³ km)')
ax.set_aspect('equal')
ax.set_title(f'X–Z slice  (y = {y_plane/1e3:.2f}·10³ km)')
ax.legend(loc='upper right', fontsize='small')
plt.tight_layout()
plt.show()
"""
