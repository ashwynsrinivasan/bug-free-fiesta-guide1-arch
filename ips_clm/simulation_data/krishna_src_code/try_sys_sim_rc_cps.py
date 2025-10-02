from LaserTherm.tec import tec
from LaserTherm.euler import euler
from LaserTherm.pid import pid
from LaserTherm.laser import  laser
from LaserTherm.mux import mux
from LaserTherm.thermistor import  thermistor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


is_heater = False

tc = 273+25

dt = 0.001 # time step
T = 90 # Seconds
Th = 273+35 # Hot side temperature, K
l = laser(tj=273+42.5,i=200e-3, N_array=8, delta_T=5, freq_spacing=200e9)
l.v = 1.8
l.freq_spacing = 200e9

m = mux(lambda_array=l.lambda_array0[:,0], il_db=1.2, bw_lambda=0.2e-9, sb_atten=50)
therm1 = thermistor(theta_jsm=1e3, tau=2.5)
tec1 = tec(Th=Th, dTmax=68, Qmax=45.1, Imax=8.4, Umax=8.8, ACR=1.06, time_constant=1)
#pid_tec = pid(kp=0.183, ki=0.128, kd=0.0, dt=dt)
pid_tec = pid(kp=0.3335, ki=0.18, kd=0.4378, dt=dt)
pid_heater = pid(kp=0.1, ki=1/(10*dt), kd=0.000, dt=dt)
if not is_heater:
    pid_laser_drive = pid(kp=0.1, ki=1/(10*dt), kd=0.000, dt=dt)
else:
    pid_laser_drive = pid(kp=0.1, ki=0.00, kd=0.000, dt=dt)
#pid_tec = pid(kp=0.01, ki=0.01, kd=0.0, dt=dt)
# Initialize
t = np.arange(0, T, dt)
q = np.zeros((len(t), 1))
# q ramps up after 30 seconds to 30 W
# q[int(30/dt):] = 30

Tref = tc# K
Pmux_ref = 29e-3

Ta = np.zeros((len(t), 1))
Ttherm = np.zeros((len(t), 1))
V = np.zeros((len(t), 1))
I = np.zeros((len(t), 1))

I_laser_array = np.zeros((len(t), l.N_array))
I_heater_array = np.zeros((len(t), l.N_array))
p_optical = np.zeros((len(t), l.N_array))
p_mux_out = np.zeros((len(t), l.N_array))
I_laser_array[t>30, 3] = 0.1 + 0.1*np.sin(2*np.pi*0.1*(t[t>30]-30))
#I_laser_array[t>30, :] = 0.2
#I_laser_array[t>60, :] = 0
lambda_err_array = np.zeros((len(t), l.N_array))

for step in range(len(t)):
    if step == 0:
        init = True
    else:
        init = False
    tec1.update(qa = q[step,0], i = I[step,0], dt=dt, init=init)
    Ta[step,0] = tec1.Ta
    V[step,0] = tec1.V
    #Thermistor update
    therm1.update(Ta[step,0], dt=dt, init=init)
    Ttherm[step] = therm1.T_therm
    # PID control
    error = -(Tref - Ttherm[step])
    pid_tec.update(error)
    if step < len(t)-1:
        I[step+1,0] = pid_tec.output
        if I[step+1,0] > tec1.Imax:
            I[step+1,0] = tec1.Imax

    # Laser update
    if step<len(t)-1 and t[step]>=30:
        p_optical[step, :], lambda_array_step, temp, p_loss = l.update(tc=Ta[step,0], i_array=I_laser_array[step,:].T, i_heater_array=I_heater_array[step,:].T)
        lambda_err_array[step, :] = (lambda_array_step - l.lambda_array0[:,0])*1e9
        q[step+1,0] = sum(p_loss)
        p_mux_out[step, :] = m.update(lambda_array_step, p_optical[step, :])
        err_mux_out = Pmux_ref - np.mean(p_mux_out[step, :])
        # if t[step] >= 35 and t[step] < 60:
        #     pid_heater.update(err_mux_out)
        #     pid_laser_drive.update(err_mux_out)
        #     if is_heater:
        #         I_heater_array[step+1, :] = pid_heater.output
        #     else:
        #         I_laser_array[step+1, :] = pid_laser_drive.output



# Plot Temperature, Load Power, Voltage, Current, and TEC Power
Nrow_fig = 5
Ncol_fig = 2
fignum = 1
#Set font size for all plots
matplotlib.rcParams.update({'font.size': 8})

plt.figure()
plt.subplot(Nrow_fig,Ncol_fig,fignum)
plt.plot(t, Ta-273, t, Ttherm-273)
plt.ylabel('Temperature (C)')
plt.xlabel('Time (s)')
plt.grid()

fignum = fignum + 1
plt.subplot(Nrow_fig,Ncol_fig,fignum)
plt.plot(t, q)
plt.ylabel('Load Power (W)')
plt.xlabel('Time (s)')
plt.grid()

fignum = fignum + 1
plt.subplot(Nrow_fig,Ncol_fig,fignum)
plt.plot(t, V)
plt.ylabel('TEC Voltage (V)')
plt.xlabel('Time (s)')
plt.grid()

fignum = fignum + 1
plt.subplot(Nrow_fig,Ncol_fig,fignum)
plt.plot(t, I)
plt.ylabel('TEC Current (A)')
plt.xlabel('Time (s)')
plt.grid()

fignum = fignum + 1
plt.subplot(Nrow_fig, Ncol_fig, fignum)
plt.plot(t, V*I)
plt.ylabel('TEC Power (W)')
plt.xlabel('Time (s)')
plt.grid()
#Wavelength error

fignum = fignum + 1
plt.subplot(Nrow_fig,Ncol_fig,fignum)
plt.plot(t, lambda_err_array)
plt.xlabel('Time (s)')
plt.ylabel('Wavelength Error (nm)')
plt.grid()

#Laser power
fignum = fignum + 1
plt.subplot(Nrow_fig,Ncol_fig,fignum)
plt.plot(t, p_optical*1e3)
plt.xlabel('Time (s)')
plt.ylabel('Laser Power (mW)')
plt.grid()

#Heater drive curr

fignum = fignum + 1
plt.subplot(Nrow_fig,Ncol_fig,fignum)
if is_heater:
    plt.plot(t, I_heater_array, t, I_laser_array)
    plt.xlabel('Time (s)')
    plt.ylabel('Drive Current (A)')
    #plt.legend(['Heater', 'Laser'])
    # Show legend
else:
    plt.plot(t, I_laser_array)
    plt.xlabel('Time (s)')
    plt.ylabel('Laser Drive Current (A)')

plt.grid()

# MUX transfer function
fignum = fignum + 1
plt.subplot(Nrow_fig,Ncol_fig,fignum)
plt.plot(m.lambda_sweep*1e9, 10*np.log10(m.tf_array.T))
plt.ylabel('MUX Insertion Loss (dB)')
plt.xlabel('Wavelength (nm)')
plt.grid()

# MUX output power
fignum = fignum + 1
plt.subplot(Nrow_fig,Ncol_fig,fignum)
plt.plot(t, p_mux_out*1e3)
plt.ylabel('MUX Output Power (mW)')
plt.xlabel('Time (s)')
plt.grid()


import json
with open('../data/LaserData/TEC_curr_volt_temp_sine_mod_20231211T132519.json') as f:
    data = json.load(f)

time = np.array(data['Time (s)'])
Ild = np.array(data['LD Driver Curr (mA)'])
Vtec = np.array(data['TEC Voltage (V)'])
Itec = np.array(data['TEC Current (A)'])
T = np.array(data['TEC Temp (C)'])

Ptec = Vtec*Itec
#Plot
plt.subplot(Nrow_fig,Ncol_fig,1)
plt.plot(time+30, T)
plt.subplot(Nrow_fig,Ncol_fig,3)
plt.plot(time+30, -Vtec)
plt.subplot(Nrow_fig,Ncol_fig,4)
plt.plot(time+30, -Itec)

plt.show()