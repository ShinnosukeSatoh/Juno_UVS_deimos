# %% Import
import spiceypy as spice
from multiprocessing import Pool
# from numba import jit
import numpy as np
import math

import datetime
import time
# import os
from scipy.io import readsav
import JupiterMag as jm

from Leadangle_fit_JunoUVS import Obsresults
from Leadangle_fit_JunoUVS import viewingangle
from Leadangle_fit_JunoUVS import calc_eqlead
from Leadangle_fit_JunoUVS import local_time_moon

spice.furnsh('kernel/cassMetaK.txt')
savpath = 'data/Satellite_FP_JRM33.sav'

jm.Internal.Config(Model="jrm33", CartesianIn=True,
                   CartesianOut=True, Degree=18)


# Input about Juno observation
TARGET_MOON = 'Europa'
TARGET_FP = ['MAW', 'TEB']
TARGET_HEM = 'both'
PJ_LIST = [1, 3]+np.arange(4, 68+1, 1).tolist()


# %% Constants
MU0 = 1.26E-6            # 真空中の透磁率
AMU2KG = 1.66E-27        # 原子質量をkgに変換するファクタ [kg amu^-1]
RJ = 71492E+3            # JUPITER RADIUS [m]
MJ = 1.90E+27            # JUPITER MASS [kg]
C = 2.99792E+8           # LIGHT SPEED [m/s]
G = 6.67E-11             # 万有引力定数  [m^3 kg^-1 s^-2]
e = 1.60218E-19          # 素電荷 [J]
me = 9.10E-31            # 電子質量 [kg]

Psyn_io = (12.89)*3600      # Moon's synodic period [sec]
Psyn_eu = (11.22)*3600      # Moon's synodic period [sec]
Psyn_ga = (10.53)*3600      # Moon's synodic period [sec]


# Select moon synodic orbital period
PJ_LIST = [1, 3]+np.arange(4, 68+1, 1).tolist()
if TARGET_MOON == 'Io':
    Psyn = Psyn_io
    PJ_LIST.pop(54-2)
    PJ_LIST.pop(55-3)
    PJ_LIST.pop(56-4)
    PJ_LIST.pop(57-5)
    PJ_LIST.pop(61-6)
    PJ_LIST.pop(63-7)
    PJ_LIST.pop(64-8)
    PJ_LIST.pop(65-9)
    PJ_LIST.pop(67-10)
    r_moon = 5.9*RJ
elif TARGET_MOON == 'Europa':
    Psyn = Psyn_eu
    PJ_LIST.pop(24-2)
    PJ_LIST.pop(43-3)
    PJ_LIST.pop(47-4)
    PJ_LIST.pop(49-5)
    PJ_LIST.pop(50-6)
    PJ_LIST.pop(51-7)
    PJ_LIST.pop(53-8)
    PJ_LIST.pop(55-9)
    PJ_LIST.pop(56-10)
    PJ_LIST.pop(60-11)
    PJ_LIST.pop(61-12)
    PJ_LIST.pop(63-13)
    PJ_LIST.pop(64-14)
    PJ_LIST.pop(65-15)
    PJ_LIST.pop(66-16)
    PJ_LIST.pop(67-17)
    PJ_LIST.pop(68-18)
    r_moon = 9.4*RJ
elif TARGET_MOON == 'Ganymede':
    Psyn = Psyn_ga
    PJ_LIST.pop(24-2)
    PJ_LIST.pop(31-3)
    PJ_LIST.pop(39-4)
    PJ_LIST.pop(43-5)
    PJ_LIST.pop(44-6)
    PJ_LIST.pop(45-7)
    PJ_LIST.pop(51-8)
    PJ_LIST.pop(52-9)
    PJ_LIST.pop(53-10)
    PJ_LIST.pop(54-11)
    PJ_LIST.pop(55-12)
    PJ_LIST.pop(56-13)
    PJ_LIST.pop(61-14)
    PJ_LIST.pop(62-15)
    PJ_LIST.pop(63-16)
    PJ_LIST.pop(64-17)
    PJ_LIST.pop(65-18)
    PJ_LIST.pop(66-19)
    PJ_LIST.pop(67-20)
    PJ_LIST.pop(68-21)
    r_moon = 15.0*RJ


# %% Load the observation data
wlon_fp, err_wlon_fp, lat_fp, err_lat_fp, moon_S3wlon, et_fp, hem_fp, pj_fp = Obsresults(
    PJ_LIST, TARGET_MOON, TARGET_FP, TARGET_HEM=TARGET_HEM, FLIP=False
)


# %% Calculate the lead angle
eqlead_fp, eqlead_fp_0, eqlead_fp_1, wlon_fp_eq = calc_eqlead(wlon_fp,
                                                              err_wlon_fp,
                                                              lat_fp,
                                                              err_lat_fp,
                                                              hem_fp,
                                                              moon_S3wlon,
                                                              TARGET_MOON)


# %% Load the viewing angle
view_angle = np.zeros(3)
for i in PJ_LIST:
    for j in TARGET_FP:
        view_angle1 = viewingangle(PJnum=i,
                                   target_moon=TARGET_MOON,
                                   target_fp=j,
                                   target_hem=TARGET_HEM,
                                   FLIP=False)
        view_angle = np.append(view_angle, view_angle1)
view_angle = view_angle[3:]


# %% 経度0度(y=0)平面のx-z対応テーブル (900km高度)
extradius = np.loadtxt('data/Alt_900km/rthetaphi.txt')
r_e = extradius[0, :]        # [RJ]
theta_e = np.radians(extradius[1, :])    # [rad]
phi_e = np.radians(extradius[2, :])      # [rad]


# %% Loop function
def calc(coef):
    start_loop = time.time()

    # 中央値
    rho_arr = np.zeros(wlon_fp.size)
    z_arr = np.zeros(wlon_fp.size)

    # 磁場モデルの設定
    mu_i_default = 139.6    # default: 139.6 [nT]
    jm.Con2020.Config(mu_i=mu_i_default*coef,
                      equation_type='analytic')

    latitude = lat_fp
    if error_num == 1:
        for i in range(lat_fp.size):
            if latitude[i] >= 0.0:
                latitude[i] = latitude[i] + err_lat_fp[i]
            else:
                latitude[i] = latitude[i] - err_lat_fp[i]
    elif error_num == 2:
        for i in range(lat_fp.size):
            if latitude[i] >= 0.0:
                latitude[i] = latitude[i] - err_lat_fp[i]
            else:
                latitude[i] = latitude[i] + err_lat_fp[i]

    theta = np.radians(90.0-latitude)
    phi = np.radians(360.0-wlon_fp)
    if error_num == 3:
        phi = np.radians(360.0-(wlon_fp+err_wlon_fp))
    elif error_num == 4:
        phi = np.radians(360.0-(wlon_fp-err_wlon_fp))

    x0_norm = np.sin(theta)*np.cos(phi)
    y0_norm = np.sin(theta)*np.sin(phi)
    z0_norm = np.cos(theta)

    for i in range(rho_arr.size):
        # テーブルを参照し距離を確定
        dis = np.abs(theta[i]-theta_e)
        idx = np.argmin(dis)
        r = r_e[idx]

        x0 = r*x0_norm[i]
        y0 = r*y0_norm[i]
        z0 = r*z0_norm[i]

        # create trace objects, pass starting position(s) x0,y0,z0
        T1 = jm.TraceField(x0, y0, z0,
                           IntModel='jrm33', ExtModel='Con2020',
                           MaxStep=0.0003,
                           MaxLen=800000, ErrMax=0.000001)

        x1 = T1.x[0][~np.isnan(T1.x[0])]
        y1 = T1.y[0][~np.isnan(T1.y[0])]
        z1 = T1.z[0][~np.isnan(T1.z[0])]
        rho = np.sqrt(x1**2 + y1**2 + z1**2)

        # Satellite orbital plane
        idx_z0 = np.argmin(np.abs(z1))

        # phi_z0 = np.arctan2(y1[idx_z0], x1[idx_z0])  # in SIII RH [rad]
        rho_arr[i] = rho[idx_z0]
        # z_arr = T1.z[0][idx_z0]

    print('------ Loop time [sec]:', round(time.time()-start_loop, 4))
    return rho_arr


# %% The main function
def main():
    print('Target moon:', TARGET_MOON)
    print('Target fp:', TARGET_FP)
    print('Target hemisphere:', TARGET_HEM)
    start_all = time.time()

    coef_arr = np.arange(0, 2.1, 0.01)
    coef_zipped = list(zip(coef_arr))
    with Pool(processes=parallel) as pool:
        results_list = list(pool.starmap(calc, coef_zipped))

    rho_arr = np.array(results_list)
    print(rho_arr.shape)

    coef_best = np.zeros(rho_arr.shape[1])
    for j in range(rho_arr.shape[1]):
        rho_j_arr = rho_arr[:, j]
        idx_rho_best = np.argmin(np.abs(rho_j_arr-r_moon/RJ))
        coef_best[j] = coef_arr[idx_rho_best]

    print('--- Total time [sec]:', round(time.time()-start_all, 4))

    np.savetxt('results/azimuthal_current_fit/'+TARGET_MOON[0:2]+'_coef_'+str(error_num)+'.txt',
               coef_best)
    return None


# %%
def calc2(x0, y0, z0):
    start_loop = time.time()

    coef_arr = np.arange(0, 2.1, 0.01)

    # 中央値
    rho_arr = np.zeros(coef_arr.size)
    z_arr = np.zeros(coef_arr.size)

    for i in range(coef_arr.size):
        coef = coef_arr[i]

        # 磁場モデルの設定
        mu_i_default = 139.6    # default: 139.6 [nT]
        jm.Con2020.Config(mu_i=mu_i_default*coef,
                          equation_type='analytic')

        # create trace objects, pass starting position(s) x0,y0,z0
        T1 = jm.TraceField(x0, y0, z0,
                           IntModel='jrm33', ExtModel='Con2020',
                           MaxStep=0.0003,
                           MaxLen=800000, ErrMax=0.000001)
        x1 = T1.x[0][~np.isnan(T1.x[0])]
        y1 = T1.y[0][~np.isnan(T1.y[0])]
        z1 = T1.z[0][~np.isnan(T1.z[0])]
        rho = np.sqrt(x1**2 + y1**2 + z1**2)

        # Satellite orbital plane
        idx_z0 = np.argmin(np.abs(z1))

        # phi_z0 = np.arctan2(y1[idx_z0], x1[idx_z0])  # in SIII RH [rad]
        rho_arr[i] = rho[idx_z0]
        # z_arr = T1.z[0][idx_z0]

    idx_rho_best = np.argmin(np.abs(rho_arr-r_moon/RJ))
    coef_best = coef_arr[idx_rho_best]

    print('------ Loop time [sec]:', round(time.time()-start_loop, 4))
    print('----------------- [RJ]:', round(rho_arr[idx_rho_best], 3))
    return coef_best


# %%
def main2():
    print('Target moon:', TARGET_MOON)
    print('Target fp:', TARGET_FP)
    print('Target hemisphere:', TARGET_HEM)
    start_all = time.time()

    latitude = lat_fp
    if error_num == 1:
        for i in range(lat_fp.size):
            if latitude[i] >= 0.0:
                latitude[i] = latitude[i] + err_lat_fp[i]
            else:
                latitude[i] = latitude[i] - err_lat_fp[i]
    elif error_num == 2:
        for i in range(lat_fp.size):
            if latitude[i] >= 0.0:
                latitude[i] = latitude[i] - err_lat_fp[i]
            else:
                latitude[i] = latitude[i] + err_lat_fp[i]

    theta = np.radians(90.0-latitude)
    phi = np.radians(360.0-wlon_fp)
    if error_num == 3:
        phi = np.radians(360.0-(wlon_fp+err_wlon_fp))
    elif error_num == 4:
        phi = np.radians(360.0-(wlon_fp-err_wlon_fp))

    x0_norm = np.sin(theta)*np.cos(phi)
    y0_norm = np.sin(theta)*np.sin(phi)
    z0_norm = np.cos(theta)

    x0 = np.zeros(lat_fp.size)
    y0 = np.zeros(lat_fp.size)
    z0 = np.zeros(lat_fp.size)
    for i in range(lat_fp.size):
        # テーブルを参照し距離を確定
        dis = np.abs(theta[i]-theta_e)
        idx = np.argmin(dis)
        r = r_e[idx]

        x0[i] = r*x0_norm[i]
        y0[i] = r*y0_norm[i]
        z0[i] = r*z0_norm[i]

    xyz0_zip = list(zip(x0, y0, z0))
    with Pool(processes=parallel) as pool:
        results_list = list(pool.starmap(calc2, xyz0_zip))

    coef_best_arr = np.array(results_list)
    print(coef_best_arr.shape)

    print('--- Total time [sec]:', round(time.time()-start_all, 4))

    np.savetxt('results/azimuthal_current_fit/'+TARGET_MOON[0:2]+'_coef_'+str(error_num)+'.txt',
               coef_best_arr)
    return None


# %% EXECUTE
if __name__ == '__main__':
    # Input about Juno observation
    error_num = 4

    # Number of parallel processes
    parallel = 35

    main2()
