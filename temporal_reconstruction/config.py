"""

Script where basic running parameters can be saved for different locations/times
Reason for this is that with multiple runs it will be difficult to remember all the settings

"""

from dataclasses import dataclass

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
