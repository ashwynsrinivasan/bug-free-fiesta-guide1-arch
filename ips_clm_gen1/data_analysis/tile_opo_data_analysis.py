import numpy as np 
import pandas as pd 
import scipy as sp 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from glob import glob



class tile_opo_build_analysis:
    def __init__(self, build_id):
        self.tp1_test_id = ['TP1_1_Test', 'TP1_2_LIV_Data', 'TP1_3_Data', 'TP_1_4_Laser_Data', 'TP1_4_VOA_Data']
        self.tp2_test_id = ['TP2_1_Laser_Data', 'TP2_1_VOA_Data', 'TP2_2_Temp_Scan_Data', 'TP2_3_Original_Ratio_Data', 'TP2_3_Ratio_Data', 'TP2_3_Scan_Data', 'TP2_3_Scan_Data_2']
        self.build_id = build_id
        
        self.frequency_target_THz = {
            '0' : {'0': 230.35,'1': 229.95,'2': 229.55,'3': 229.15,'4': 228.75,'5': 228.35,'6': 227.95,'7': 227.55},
            '1' : {'0': 230.15,'1': 229.75,'2': 229.35,'3': 228.95,'4': 228.55,'5': 228.15,'6': 227.75,'7': 227.35},
            }
        self.wavelength_target_nm = {
            '0' : {'0': 1301.47,'1': 1303.73,'2': 1306.01,'3': 1308.28,'4': 1310.57,'5': 1312.87,'6': 1315.17,'7': 1317.48,},
            '1': {'0': 1302.6,'1': 1304.87,'2': 1307.14,'3': 1309.43,'4': 1311.72,'5': 1314.02,'6': 1316.33,'7': 1318.64}
            }
            
    def find_t_op_single_ch(self):
        self.tp1_1_data = pd.read_csv('tile_opo/'+self.build_id + '_'+self.tp1_test_id[0]+'.csv')
        
    def plot_tp1_2_liv_data(self):
        self.tp1_2_data = pd.read_csv('tile_opo/'+self.build_id + '_'+self.tp1_test_id[1]+'.csv')
        