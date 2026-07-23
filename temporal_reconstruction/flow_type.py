"""
Purpose of this script is be able to input different types of flows
into the time varying RBF reconstruction. 
"""

import numpy as np
import pandas as pd

class flow_type:
    def __init__(self, sc_init, velocity_path, B_field_path, start_time = None, end_time = None):

        df_v = pd.read_csv(velocity_path)
        df = pd.read_csv(B_field_path)

        if start_time is not None:
            df_v = df_v[df_v["Timeframe"] >= start_time]
            df = df[df["Timeframe"] >= start_time]

        if end_time is not None:
            df_v = df_v[df_v["Timeframe"] <= end_time]
            df = df[df["Timeframe"] <= end_time]

        self.df_v = df_v.reset_index(drop=True)
        self.df = df.reset_index(drop=True)

        self.sc_init = sc_init
        self.static_vel = None
        
        pass

    #THESE ARE TEMPORAL --> SPATIAL TRANSITIONS 
    #LOGIC OF FIRST TWO IS FLYING SC THROUGH STATIC STRUCTURE
    def static_bulk_velocity(self):
        """
        
        Some possibly usable bulk times for magnetotail fluxrope

        #SC1-4 overall means (from 1353 onwards):
        vg_v_x = -884632.1570458194
        vg_v_y = -332263.0899893005
        vg_v_z = 165187.1091635656
        """
        """
        #SC1-4 overall means pre 1353:
        vg_v_x = -544078.8175128276
        vg_v_y = -181801.77131761858
        vg_v_z = 121311.91965101559
        """

        #SC1-4 overall mean:
        #SC1-4 is the outer tetrahedron and likely good approximation of bulk velocity
        #as is not over weighted by the inner
        vg_v_x = -739256.9
        vg_v_y = -268152.8
        vg_v_z =  147101.5

        vel_bulk = -1*np.array([vg_v_x,vg_v_y,vg_v_z])

        self.df["dt"] = self.df["Timeframe"] - self.df["Timeframe"].iloc[0]

        #Make artificial spacecraft location for RBF reconstructions
        for sc, init_pos in self.sc_init.items():
            self.df[f"{sc}_pos_x"] = init_pos[0] + vel_bulk[0] * self.df["dt"]
            self.df[f"{sc}_pos_y"] = init_pos[1] + vel_bulk[1] * self.df["dt"]
            self.df[f"{sc}_pos_z"] = init_pos[2] + vel_bulk[2] * self.df["dt"]


        pos_cols = sum([[f"{sc}_pos_x", f"{sc}_pos_y", f"{sc}_pos_z"]
                        for sc in self.sc_init.keys()], [])
        B_cols   = sum([[f"{sc}_vg_B_x", f"{sc}_vg_B_y", f"{sc}_vg_B_z"]
                        for sc in self.sc_init.keys()], [])
        self.static_vel = np.array([vg_v_x, vg_v_y, vg_v_z])
        
        return pos_cols, B_cols

    def dynamic_bulk_velocity(self, sc_nums = range(1,5)):
    
        v_x_cols = [f"vg_v_x_point{n}" for n in sc_nums]
        v_y_cols = [f"vg_v_y_point{n}" for n in sc_nums]
        v_z_cols = [f"vg_v_z_point{n}" for n in sc_nums]
        self.df_v["v_bulk_x"] = self.df_v[v_x_cols].mean(axis=1)
        self.df_v["v_bulk_y"] = self.df_v[v_y_cols].mean(axis=1)
        self.df_v["v_bulk_z"] = self.df_v[v_z_cols].mean(axis=1)

        self.df["delta_t"] = self.df["Timeframe"].diff().fillna(0.0)
        

        self.df["disp_x"] = (-self.df_v["v_bulk_x"]*self.df["delta_t"]).cumsum()
        self.df["disp_y"] = (-self.df_v["v_bulk_y"]*self.df["delta_t"]).cumsum()
        self.df["disp_z"] = (-self.df_v["v_bulk_z"]*self.df["delta_t"]).cumsum()

        for sc, init_pos in self.sc_init.items():
            self.df[f"{sc}_pos_x"] = init_pos[0] + self.df["disp_x"]
            self.df[f"{sc}_pos_y"] = init_pos[1] + self.df["disp_y"]
            self.df[f"{sc}_pos_z"] = init_pos[2] + self.df["disp_z"]

        pos_cols = sum([[f"{sc}_pos_x", f"{sc}_pos_y", f"{sc}_pos_z"]
                        for sc in self.sc_init.keys()], [])
        B_cols   = sum([[f"{sc}_vg_B_x", f"{sc}_vg_B_y", f"{sc}_vg_B_z"]
                        for sc in self.sc_init.keys()], [])

        return pos_cols, B_cols
    
    #THESE ARE FLOW MOVING THE DATA POINTS ALONG WITH IT AND STRUCTURE ASSUMED TO BE STABLE
    
    def steady_flow_velocity(self, t_ref = None):
        

        times = self.df["Timeframe"].values
        if t_ref is None: 
            t_ref = times[-1]

        dt_to_ref = t_ref-times
        T = len(times)

        for n, (sc, init_pos) in zip(range(1,len(self.sc_init)+1), self.sc_init.items()):
            v_x = self.df_v[f"vg_v_x_point{n}"].values
            v_y = self.df_v[f"vg_v_y_point{n}"].values
            v_z = self.df_v[f"vg_v_z_point{n}"].values

            self.df[f"{sc}_pos_x"] = init_pos[0] + v_x*dt_to_ref
            self.df[f"{sc}_pos_y"] = init_pos[1] + v_y*dt_to_ref
            self.df[f"{sc}_pos_z"] = init_pos[2] + v_z*dt_to_ref
            
        pos_cols = sum([[f"{sc}_pos_x", f"{sc}_pos_y", f"{sc}_pos_z"]
                        for sc in self.sc_init.keys()], [])
        B_cols   = sum([[f"{sc}_vg_B_x", f"{sc}_vg_B_y", f"{sc}_vg_B_z"]
                        for sc in self.sc_init.keys()], [])
        return pos_cols, B_cols
    
    def steady_flow_velocity_mean(self, t_ref = None):
        sc_nums = range(1,len(self.sc_init)+1)
        v_x_cols = [f"vg_v_x_point{n}" for n in sc_nums]
        v_y_cols = [f"vg_v_y_point{n}" for n in sc_nums]
        v_z_cols = [f"vg_v_z_point{n}" for n in sc_nums]

        times = self.df["Timeframe"].values
        t_end = times[-1]
        dt_to_end = t_end-times

        T = len(times)
        v_x = self.df_v[v_x_cols].mean(axis = 1).values
        v_y = self.df_v[v_y_cols].mean(axis = 1).values
        v_z = self.df_v[v_z_cols].mean(axis = 1).values

        for n, (sc, init_pos) in zip(range(1,len(self.sc_init)+1), self.sc_init.items()):
            

            self.df[f"{sc}_pos_x"] = init_pos[0] + v_x*dt_to_end
            self.df[f"{sc}_pos_y"] = init_pos[1] + v_y*dt_to_end
            self.df[f"{sc}_pos_z"] = init_pos[2] + v_z*dt_to_end        
        pos_cols = sum([[f"{sc}_pos_x", f"{sc}_pos_y", f"{sc}_pos_z"]
                        for sc in self.sc_init.keys()], [])
        B_cols   = sum([[f"{sc}_vg_B_x", f"{sc}_vg_B_y", f"{sc}_vg_B_z"]
                        for sc in self.sc_init.keys()], [])
        return pos_cols, B_cols
    
        
if __name__ == "__main__":
    R_e = 6371000   
    vg_v_file = "/home/leeviloi/plas_obs_vir_vg_v_full_tail_right_Z=0.5_GOOD.csv"
    b_field_file = "/home/leeviloi/plas_obs_vg_b_timeseries_tail_right_z=0.5.csv"

    sc_init = {
    "sc1": np.array([-27.0, 3.0, 0.5]) * R_e,
    "sc2": np.array([-26.0, 3.0, 1.5]) * R_e,
    "sc3": np.array([-26.0, 3.86602540, 0.0]) * R_e,
    "sc4": np.array([-26.0, 2.13397460, 0.0]) * R_e,
    "sc5": np.array([-26.85714286, 3.0, 0.64285714]) * R_e,
    "sc6": np.array([-26.85714286, 3.12371791, 0.42857143]) * R_e,
    "sc7": np.array([-26.85714286, 2.87628209, 0.42857143]) * R_e,
    }
    flow_test = flow_type(sc_init,vg_v_file, b_field_file)
    flow_test.steady_flow_velocity()
