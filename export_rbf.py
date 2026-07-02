"""
Script to extract the RBF interpolation recontruction domain
The data is exported in to a VTK structured grid which can be view in visualation softwares
such as VisIt, Paraview or Blender (with VTK plugin).
TODO: add ability to export more variables. 
"""
import numpy as np
import vtk
from vtk.util import numpy_support
from scipy.spatial import ConvexHull
import trimesh


R_E = 6.371e6  


def export_rbf_vtk(df,rbf,included_pos_cols,ny=100,nz=100,nx=200,padding_Re=0.5,output_path="rbf_reconstruction.vts",
                   export_sc_positions=True,use_convex_hull=False,hull_distance_Re=0.5,include_inside=True,):

    pad = padding_Re * R_E
    T = len(df)
    N_sc = len(included_pos_cols) // 3

 
    pos = df[included_pos_cols].to_numpy().reshape(T, N_sc, 3)
    
    #Defining grid
    all_x = pos[:, :, 0].ravel()        
    all_y  = pos[:, :, 1].ravel()
    all_z  = pos[:, :, 2].ravel()
    y_grid = np.linspace(all_y.min() - pad, all_y.max() + pad, ny)
    z_grid = np.linspace(all_z.min() - pad, all_z.max() + pad, nz)
    x_grid = np.linspace(all_x.min() - pad, all_x.max() + pad, nx)

    print(
        f"Data extent:\n"
        f"  x : {all_x.min()/R_E:.3f} - {all_x.max()/R_E:.3f} Re \n"
        f"  y : {y_grid[0]/R_E:.3f} - {y_grid[-1]/R_E:.3f} Re \n"
        f"  z : {z_grid[0]/R_E:.3f} - {z_grid[-1]/R_E:.3f} Re"
    )

 
    II, JJ, KK = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    ) 

    x_flat = x_grid[II].ravel(order="F")   
    y_flat = y_grid[JJ].ravel(order="F")
    z_flat = z_grid[KK].ravel(order="F")

    pts_xyz = np.ascontiguousarray(np.column_stack([x_flat, y_flat, z_flat])  )

    #Masking points to only near constellation
    if use_convex_hull:

        print("Computing convex hull")

        keep_mask = np.zeros(len(pts_xyz), dtype=bool)

        D_max = hull_distance_Re * R_E

        for t in range(T):

            sc_points = pos[t]

            hull = ConvexHull(sc_points)
            mesh = trimesh.Trimesh(vertices=sc_points,faces=hull.simplices,process=True)

            closest_pts, unsigned_dist, _ = mesh.nearest.on_surface(pts_xyz)
            inside = mesh.contains(pts_xyz)
            signed_dist = unsigned_dist * np.where(inside,-1.0,1.0)
            
            if include_inside:
                mask_t = signed_dist < D_max
            else:
                mask_t = ((signed_dist > 0)&(signed_dist < D_max))

            keep_mask |= mask_t

    else:
        keep_mask = np.ones(len(pts_xyz), dtype=bool)
      
    n_pts = keep_mask.sum()

    print(f"Evaluating RBF")
    #Extracting the data from the RBF interpolator

    B_vals = np.full((len(pts_xyz),3),0.0,dtype=np.float64)

    B_vals[keep_mask] = rbf(pts_xyz[keep_mask])
                  
    B_vals = np.ascontiguousarray(B_vals)

    #WRITING VTK
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(nx, ny, nz)

  
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(
        numpy_support.numpy_to_vtk(pts_xyz, deep=True, array_type=vtk.VTK_DOUBLE)
    )
    grid.SetPoints(vtk_pts)

    B_arr = numpy_support.numpy_to_vtk(B_vals, deep=True, array_type=vtk.VTK_DOUBLE)
    B_arr.SetName("B_rbf")
    grid.GetPointData().SetVectors(B_arr)  

    B_mag = np.ascontiguousarray(np.linalg.norm(B_vals, axis=1))
    B_mag_arr = numpy_support.numpy_to_vtk(B_mag, deep=True, array_type=vtk.VTK_DOUBLE)
    B_mag_arr.SetName("B_rbf_mag")
    grid.GetPointData().SetScalars(B_mag_arr)

    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(grid)
    writer.SetDataModeToBinary()         
    writer.Write()
    print(f"Written: {output_path}")

    if export_sc_positions:
        _export_sc_polydata(pos, output_path, df)

    return grid


def _export_sc_polydata(pos: np.ndarray, vts_path: str, df) -> None:
    """
    Function to extract alongside reconstruction domain the virtual spacecraft trajectories 
    that define the domain
    """
    T, N_sc, _ = pos.shape
    t_0 = df["TimeFrame"][0]
    vtp_path = vts_path.replace(
        ".vts",
        "_sc_positions.vtp"
    )
    poly = vtk.vtkPolyData()
    sc_pts_flat = np.ascontiguousarray(pos.reshape(-1, 3))
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_support.numpy_to_vtk(sc_pts_flat, deep=True, array_type=vtk.VTK_DOUBLE))
    poly.SetPoints(vtk_pts)

    lines = vtk.vtkCellArray()

    for sc in range(N_sc):

        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(T)
        for t in range(T):
            pid = t * N_sc + sc
            line.GetPointIds().SetId(t,pid)
        lines.InsertNextCell(line)

    poly.SetLines(lines)

    sc_id = np.tile(np.arange(N_sc),T).astype(np.int32)
    timestep = np.repeat(np.arange(T)+t_0,N_sc).astype(np.int32)

    sc_id_arr = numpy_support.numpy_to_vtk(sc_id,deep=True,array_type=vtk.VTK_INT)
    sc_id_arr.SetName("sc_id")
    poly.GetPointData().AddArray(sc_id_arr)
    
    t_id_arr = numpy_support.numpy_to_vtk(timestep, deep=True, array_type=vtk.VTK_INT)
    t_id_arr.SetName("t_release")
    poly.GetPointData().AddArray(t_id_arr)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(vtp_path)
    writer.SetInputData(poly)
    writer.SetDataModeToBinary()
    writer.Write()

    print("Done!")
   
if __name__ == "__main__":
    pass 