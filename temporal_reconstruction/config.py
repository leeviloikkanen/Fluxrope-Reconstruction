"""

Script where basic running parameters can be saved for different locations/times
Reason for this is that with multiple runs it will be difficult to remember all the settings

"""

from dataclasses import dataclass
import numpy as np

R_e = 6371000

@dataclass
class Config:

    vg_v_file: str = ""
    b_field_file: str = ""
    bulk_path: str = "/turso/group/spacephysics/vlasiator/data/L1/3D/FHA/bulk1/"
    output_dir: str = "./"

    t_ref: float
    start_time:float = None
    end_time: float = None 

    sc_init: dict

    rbf_kernel: str = "multiquadric"
    rbf_eps_method: str = "neighbour"
    rbf_smoothing: float = 0.0

    pass

def tail_config():
    """
    """
    return Config(
        vg_v_file="./data/plas_obs_vir_vg_v_full_tail_right_Z=0.5_GOOD.csv",
        b_field_file= "./data/plas_obs_vg_b_timeseries_tail_right_z=0.5.csv",
        output_dir="./",
        t_ref = 1372,
        sc_init = {
            "sc1": np.array([-27.0, 3.0, 0.5]) * R_e,
            "sc2": np.array([-26.0, 3.0, 1.5]) * R_e,
            "sc3": np.array([-26.0, 3.86602540, 0.0]) * R_e,
            "sc4": np.array([-26.0, 2.13397460, 0.0]) * R_e,
            "sc5": np.array([-26.85714286, 3.0, 0.64285714]) * R_e,
            "sc6": np.array([-26.85714286, 3.12371791, 0.42857143]) * R_e,
            "sc7": np.array([-26.85714286, 2.87628209, 0.42857143]) * R_e,
        },

    )
def magnetopause_config():
    """
    """
    return Config(
        vg_v_file="./data/plas_obs_vir_vg_v_full_magnetopause_z=-1_1400-1500_GOOD.csv",
        b_field_file="./data/plas_obs_vg_b_timeseries_magnetopause_z=-1_1400-1500s.csv",
        t_ref= 1452,
        start_time = 1420,
        end_time = 1452,
        sc_init = {
            "sc1": np.array([6.0, -11.0, -1.0]) * R_e,
            "sc2": np.array([6.52532199, -10.14909648,  0.0]) * R_e,
            "sc3": np.array([5.78841792,  -9.69415429, -1.5]) * R_e,
            "sc4": np.array([7.26222606, -10.60403866, -1.5]) * R_e,
            "sc5": np.array([6.07504600, -10.87844235, -0.85714286]) * R_e,
            "sc6": np.array([5.96977399, -10.81345061, -1.07142857]) * R_e,
            "sc7": np.array([6.18031801, -10.94343409, -1.07142857]) * R_e,
        },
    )