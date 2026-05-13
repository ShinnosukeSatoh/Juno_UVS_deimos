""" RAW_trace.py

Created on May 7, 2026
@author: Shin Satoh

Description:
This code calculates the location of the reflective Alfvén wave spot(s)
by tracing the Alfvén waves from a Galilean satellite, all the way along
the magnetic field line defined by the magnetic field models, including
reflection(s) near the planet. The reflection altitude is set at 900 km.

Version
1.0.0 (May 7, 2026)

"""
# %% Import
import spiceypy as spice
from multiprocessing import Pool
# from numba import jit
import numpy as np
import math
import matplotlib.pyplot as plt
from UniversalColor import UniversalColor
from SharedX import ShareXaxis
from legend_shadow import legend_shadow

import Leadangle_wave as Wave
import datetime
import time
# import os
from scipy.io import readsav
import JupiterMag as jm

jm.Internal.Config(Model='jrm33', CartesianIn=True, CartesianOut=True)
jm.Con2020.Config(equation_type='analytic')

UC = UniversalColor()
UC.set_palette()

F = ShareXaxis()
F.fontsize = 20
F.fontname = 'Liberation Sans Narrow'
F.set_default()

spice.furnsh('kernel/cassMetaK.txt')
savpath = 'data/Satellite_FP_JRM33.sav'


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

Fllen_io = 14.78*RJ     # Field line length [m]
Fllen_eu = 24.22*RJ     # Field line length [m]
Fllen_ga = 39.17*RJ     # Field line length [m]


# %% Calculate position of the target moon using Spiceypy
def spice_moonS3(et: float, MOON: str):
    """
    Args:
        et (float): Time
        MOON (str): Name of the target moon (IO/EUROPA/GANYMEDE)

    Returns:
        Tuple of \\
        `posx` (float) \\
        `posy` (float) \\
        `posz` (float) \\
        `posr` (float) \\
        `postheta` (float) \\
        `posphi` (float) \\
        `S3wlon` (float) \\
    """
    # Juno's position seen from Jupiter in IAU_JUPITER coordinate.
    _, lightTimes = spice.spkpos(
        targ='JUNO', et=et, ref='IAU_JUPITER', abcorr='LT+S', obs='JUPITER'
    )

    # Moon's position seen from Jupiter in IAU_JUPITER coordinate.
    pos, _ = spice.spkpos(
        targ=MOON, et=et, ref='IAU_JUPITER', abcorr='none', obs='JUPITER'
    )

    # S3 right-hand coordinate [m]
    posx, posy, posz = pos[0]*1E+3, pos[1]*1E+3, pos[2]*1E+3

    posr = np.sqrt(posx**2 + posy**2 + posz**2)
    postheta = np.arccos(posz/posr)
    posphi = np.arctan2(posy, posx)
    if posphi < 0:
        S3wlon = np.degrees(-posphi)
    else:
        S3wlon = np.degrees(2*np.pi - posphi)

    return posx, posy, posz, posr, postheta, posphi, S3wlon


# %% Calculate local time of the moon
def local_time_moon(et: float, MOON: str, abcorr='none'):
    # Moon's position seen from Jupiter in IAU_JUPITER coordinate.
    pos_moon, _ = spice.spkpos(
        targ=MOON, et=et, ref='JUNO_JSO', abcorr=abcorr, obs='JUPITER'
    )

    # S3 right-hand coordinate [m]
    x_moon = pos_moon[0]*1E+3
    y_moon = pos_moon[1]*1E+3
    z_moon = pos_moon[2]*1E+3

    r_moon = np.sqrt(x_moon**2 + y_moon**2 + z_moon**2)
    theta_moon = np.arccos(z_moon/r_moon)
    phi_moon = np.arctan2(y_moon, x_moon)

    local_time = (24.0*phi_moon)/(2*np.pi) + 12.0   # [hour]
    # if local_time > 24:
    #     local_time += -24

    return local_time


# %% System III position of the target moon from et_fp array.
def moonS3wlon_arr(et_fp, moon: str):
    if moon == 'Io':
        target = 'IO'
    elif moon == 'Europa':
        target = 'EUROPA'
    elif moon == 'Ganymede':
        target = 'GANYMEDE'

    moon_x0 = np.zeros(et_fp.shape)
    moon_y0 = np.zeros(et_fp.shape)
    moon_z0 = np.zeros(et_fp.shape)
    moon_r0 = np.zeros(et_fp.shape)
    moon_theta0 = np.zeros(et_fp.shape)
    moon_phi0 = np.zeros(et_fp.shape)
    moon_S3wlon0 = np.zeros(et_fp.shape)
    for i in range(et_fp.size):
        x0, y0, z0, r0, theta0, phi0, S3wlon0 = spice_moonS3(
            et=et_fp[i], MOON=target)
        moon_x0[i] = x0
        moon_y0[i] = y0
        moon_z0[i] = z0
        moon_r0[i] = r0
        moon_theta0[i] = theta0
        moon_phi0[i] = phi0
        moon_S3wlon0[i] = S3wlon0
    return moon_x0, moon_y0, moon_z0, moon_r0, moon_theta0, moon_phi0, moon_S3wlon0


# %% Read the savfile
def read1savfile(PJnum: int, target_moon: str, target_fp: str, target_hem='both', FLIP=False):
    """
    Args:
        `PJnum` (int): Perijove number \\
        `moon` (str): Name of the target (Io/Europa/Ganymede) \\
        `footprint` (str): `MAW` or `TEB` \\

    Returns:
        Tuple of \\
        `wlon_MAW` (ndarray) \\
        `wlon_TEB` (ndarray) \\
        `err_wlon_MAW` (ndarray) \\
        `err_wlon_TEB` (ndarray) \\
        `lat_MAW` (ndarray) \\
        `lat_TEB` (ndarray) \\
        `err_lat_MAW` (ndarray) \\
        `err_lat_TEB` (ndarray) \\
        `wlon_moon` (ndarray) \\
        `et` (ndarray) \\
        `hem` (ndarray) \\
    """

    # Look for the file named
    # `IFP_info_v900km_fixed.sav` for Io footprint
    savpath = 'data/Output_v2_PJ01_PJ68/' + 'PJ' + \
        str(PJnum).zfill(2)+'/'+target_moon[0]+'FP_info_v900km_fixed.sav'

    # Read
    savdata = readsav(savpath)

    var = savdata['fp_info']

    # 'MIDTIME_ET'を用いてスライス位置を決定する
    MIDTIME_ET = np.array(var['MIDTIME_ET'][0])
    idx = np.where(MIDTIME_ET > 0)

    wlon_fp = np.array(var['LON_'+target_fp][0])[idx]
    err_wlon_fp = np.array(var['LON_'+target_fp+'_ERROR'][0])[idx]
    lat_fp = np.array(var['LAT_'+target_fp][0])[idx]
    err_lat_fp = np.array(var['LAT_'+target_fp+'_ERROR'][0])[idx]
    wlon_moon = np.array(var['SIII_LON'][0])[idx]
    et = np.array(var['MIDTIME_ET'][0])[idx]
    hem = var['HEMISPHERE'][0][idx]

    # Extract MAWs (exclude values -999.)
    fpvalues = np.where((wlon_fp > -100))
    wlon_fp = wlon_fp[fpvalues]
    err_wlon_fp = err_wlon_fp[fpvalues]
    lat_fp = lat_fp[fpvalues]
    err_lat_fp = err_lat_fp[fpvalues]
    wlon_moon = wlon_moon[fpvalues]
    et = et[fpvalues]
    hem = hem[fpvalues]

    # 磁場ベクトルに対する南北で判別する
    # North -> -1 / South -> 1
    # hem_N = np.where(hem == b'North')
    # hem_S = np.where(hem == b'South')
    # hem[hem_N] = -1
    # hem[hem_S] = 1

    # MAWとTEBの判別
    hem_N = np.where(hem == b'North')
    hem_S = np.where(hem == b'South')
    if target_fp == 'MAW':
        hem[hem_N] = -1
        hem[hem_S] = 1
        if FLIP is True:
            hem[hem_N] = -101
            hem[hem_S] = 101
    elif target_fp == 'TEB':
        hem[hem_N] = -101
        hem[hem_S] = 101
        if FLIP is True:
            hem[hem_N] = -1
            hem[hem_S] = -1

    # 北半球もしくは南半球だけを取り出す
    if target_hem == 'N':
        wlon_fp = wlon_fp[hem_N]
        err_wlon_fp = err_wlon_fp[hem_N]
        lat_fp = lat_fp[hem_N]
        err_lat_fp = err_lat_fp[hem_N]
        wlon_moon = wlon_moon[hem_N]
        et = et[hem_N]
        hem = hem[hem_N]
    elif target_hem == 'S':
        wlon_fp = wlon_fp[hem_S]
        err_wlon_fp = err_wlon_fp[hem_S]
        lat_fp = lat_fp[hem_S]
        err_lat_fp = err_lat_fp[hem_S]
        wlon_moon = wlon_moon[hem_S]
        et = et[hem_S]
        hem = hem[hem_S]

    return wlon_fp, err_wlon_fp, lat_fp, err_lat_fp, wlon_moon, et, hem


def load_best_fit():
    # Import the best-fit parameter
    chi2_1d = np.loadtxt('results/fit/'+exname+'/params_chi2.txt')
    Ai_1d = np.loadtxt('results/fit/'+exname+'/params_Ai.txt')
    ni_1d = np.loadtxt('results/fit/'+exname+'/params_ni.txt')
    Ti_1d = np.loadtxt('results/fit/'+exname+'/params_Ti.txt')
    H_1d = np.loadtxt('results/fit/'+exname+'/params_H.txt')
    eqlead_est = np.loadtxt('results/fit/'+exname+'/eqlead_est.txt')
    eqlead_obs = np.loadtxt('results/fit/'+exname+'/eqlead_obs.txt')
    sigma_obs = np.loadtxt('results/fit/'+exname+'/sigma_y.txt')
    hem_obs = np.loadtxt('results/fit/'+exname+'/hems_obs.txt')
    moon_S3wlon_obs = np.loadtxt('results/fit/'+exname+'/moon_S3wlon_obs.txt')

    chi2_3d = chi2_1d.reshape(ni_num, Ai_num, Ti_num)
    H_3d = H_1d.reshape(ni_num, Ai_num, Ti_num)
    Ai_3d = Ai_1d.reshape(ni_num, Ai_num, Ti_num)
    ni_3d = ni_1d.reshape(ni_num, Ai_num, Ti_num)
    Ti_3d = Ti_1d.reshape(ni_num, Ai_num, Ti_num)
    H_3d = H_1d.reshape(ni_num, Ai_num, Ti_num)
    eqlead_est_3d = eqlead_est[1].reshape(ni_num, Ai_num, Ti_num)

    # 保存されているカイ2乗値は自由度で割ってしまっているのでここで元に戻す
    chi2_3d = chi2_3d*(eqlead_est.shape[0]-3)

    # chi2_3dの最小値を探す
    min_idx = np.where(chi2_3d == np.min(chi2_3d))

    # delta_chi2
    delta_chi2 = chi2_3d-np.min(chi2_3d)

    # best-fit parameters
    Ai_best = Ai_3d[min_idx][0]
    ni_best = ni_3d[min_idx][0]
    Ti_best = Ti_3d[min_idx][0]
    Hp_best = H_3d[min_idx][0]

    return Ai_best, ni_best, Ti_best, Hp_best


# calc function
def calc(Ai, ni, Hp, r_t0, s3wlon_t0, z_t0, s_t0, hem, num_reflection):
    # Initialize the result list/array
    tau_list = []

    # Initital trace
    # -> MAW position at 900 km altitude
    tau_t1, rs_t1, s3wlon_t1, z_t1, s_t1, phi_jov_t1, theta_s3_t1 = Wave.Awave().trace3_reflect(r_t0,
                                                                                                s3wlon_t0,
                                                                                                z_t0,
                                                                                                s_t0,
                                                                                                Ai,
                                                                                                ni,
                                                                                                Hp,
                                                                                                hem)

    # 1st reflection
    # -> 1st RAW position at 900 km altitude on the opposite hemisphere
    tau_t2, rs_t2, s3wlon_t2, z_t2, s_t2, phi_jov_t2, theta_s3_t2 = Wave.Awave().trace3_reflect(rs_t1,
                                                                                                s3wlon_t1,
                                                                                                z_t1,
                                                                                                s_t1,
                                                                                                Ai,
                                                                                                ni,
                                                                                                Hp,
                                                                                                hem*(-1))

    # 2nd reflection
    # -> 1st RAW position at 900 km altitude on the same hemisphere as MAW
    tau_t3, rs_t3, s3wlon_t3, z_t3, s_t3, phi_jov_t3, theta_s3_t3 = Wave.Awave().trace3_reflect(rs_t2,
                                                                                                s3wlon_t2,
                                                                                                z_t2,
                                                                                                s_t2,
                                                                                                Ai,
                                                                                                ni,
                                                                                                Hp,
                                                                                                hem*(-1)**2)

    # Change to the Jovigraphic coordinate
    phi_jov_t2 = phi_jov_t2-s3wlon_t1+phi_jov_t1[-1]
    phi_jov_t3 = phi_jov_t3-s3wlon_t2+phi_jov_t2[-1]

    # Result lists
    tau_list = [tau_t1, tau_t2, tau_t3]
    phi_jov_list = [phi_jov_t1, phi_jov_t2, phi_jov_t3]
    theta_s3_list = [theta_s3_t1, theta_s3_t2, theta_s3_t3]

    if num_reflection >= 2:
        # 3rd reflection
        # -> 2nd RAW position at 900 km altitude on the opposite hemisphere
        tau_t4, rs_t4, s3wlon_t4, z_t4, s_t4, phi_jov_t4, theta_s3_t4 = Wave.Awave().trace3_reflect(rs_t3,
                                                                                                    s3wlon_t3,
                                                                                                    z_t3,
                                                                                                    s_t3,
                                                                                                    Ai,
                                                                                                    ni,
                                                                                                    Hp,
                                                                                                    hem*(-1)**3)
        # 4th reflection
        # -> 2nd RAW position at 900 km altitude on the same hemisphere as MAW
        tau_t5, rs_t5, s3wlon_t5, z_t5, s_t5, phi_jov_t5, theta_s3_t5 = Wave.Awave().trace3_reflect(rs_t4,
                                                                                                    s3wlon_t4,
                                                                                                    z_t4,
                                                                                                    s_t4,
                                                                                                    Ai,
                                                                                                    ni,
                                                                                                    Hp,
                                                                                                    hem*(-1)**4)

        # Change to the Jovigraphic coordinate
        phi_jov_t4 = phi_jov_t4-s3wlon_t3+phi_jov_t3[-1]
        phi_jov_t5 = phi_jov_t5-s3wlon_t4+phi_jov_t4[-1]

        tau_list += [tau_t4, tau_t5]
        phi_jov_list += [phi_jov_t4, phi_jov_t5]
        theta_s3_list += [theta_s3_t4, theta_s3_t5]

    # print('h [m]:', (rs_t1-1.0*RJ)*1E-3,
    #       (rs_t2-1.0*RJ)*1E-3, (rs_t3-1.0*RJ)*1E-3)
    # print('z [RJ]:', z_t1/RJ, z_t2/RJ, z_t3/RJ)
    print('s [RJ]:', s_t1/RJ, s_t2/RJ, s_t3/RJ)
    # print('phi [deg]:', math.degrees(s3wlon_t1),
    #       math.degrees(s3wlon_t2), math.degrees(s3wlon_t3), )
    # print('Array shape:', phi_jov_t1.shape,
    #       phi_jov_t2.shape,  phi_jov_t3.shape)

    return tau_list, phi_jov_list, theta_s3_list


# %% Main function
def main():
    # the initial SIII w-longitude of the moon
    s3wlon_t0 = np.radians(210.0)
    s3wlon_t0_arr = np.radians(np.arange(-45.0, 360.0+45, 20))
    arr_size = s3wlon_t0_arr.size

    Ai_best, ni_best, Ti_best, Hp_best = load_best_fit()

    S_A0 = Wave.Awave().tracefield(r_moon,
                                   s3wlon_t0,
                                   0
                                   )

    # Initial trace direction (-1: 北向き, 1: 南向き)
    NS = -1

    args = list(zip(Ai_best*np.ones(arr_size),
                    ni_best*np.ones(arr_size),
                    Hp_best*np.ones(arr_size),
                    r_moon*np.ones(arr_size),
                    s3wlon_t0_arr,
                    0*np.ones(arr_size),
                    S_A0*np.ones(arr_size),
                    NS*np.ones(arr_size, dtype=int),
                    num_reflection*np.ones(arr_size, dtype=int)))

    time_start = time.time()
    with Pool(processes=parallel) as pool:
        results_list = list(pool.starmap(calc, args))

    print('Number of CPU cores used:', parallel)
    print('Loop time [sec]:', round(time.time()-time_start, 4))

    # print('=== ALL ===')
    # print(results_list)
    print('=== results_list[0][1] ===')
    print(results_list[0][0])
    print(results_list[1][0])

    tau_t1_arr = np.zeros(arr_size)
    tau_t2_arr = np.zeros(arr_size)
    tau_t3_arr = np.zeros(arr_size)
    tau_t4_arr = np.zeros(arr_size)
    tau_t5_arr = np.zeros(arr_size)
    for i in range(arr_size):
        tau_t1_arr[i] = results_list[i][0][0]
        tau_t2_arr[i] = results_list[i][0][1]
        tau_t3_arr[i] = results_list[i][0][2]
        if num_reflection == 2:
            tau_t4_arr[i] = results_list[i][0][3]
            tau_t5_arr[i] = results_list[i][0][4]

    """fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xticks(np.arange(0, 360+1, 45))
    ax.set_yticks(np.arange(-90, 90+1, 30))
    ax.grid(color=UC.lightgray, linewidth=0.5)
    ax.set_xlabel('Jovigraphic east longitude [deg]')
    ax.set_ylabel('SIII latitude')
    color_list = [UC.red, UC.blue, UC.green, UC.yellow,
                  UC.pink, UC.lightblue, UC.orange, UC.purple]

    for i in range(len(color_list)):
        phi_jov_t1_arr = results_list[i][1][0]
        theta_s3_t1_arr = results_list[i][2][0]
        phi_jov_t2_arr = results_list[i][1][1]
        theta_s3_t2_arr = results_list[i][2][1]
        phi_jov_t3_arr = results_list[i][1][2]
        theta_s3_t3_arr = results_list[i][2][2]
        if num_reflection >= 2:
            phi_jov_t4_arr = results_list[i][1][3]
            phi_jov_t5_arr = results_list[i][1][4]
            theta_s3_t4_arr = results_list[i][2][3]
            theta_s3_t5_arr = results_list[i][2][4]

        ax.plot(np.degrees(phi_jov_t1_arr),
                90.0-np.degrees(theta_s3_t1_arr),
                color=color_list[i])
        ax.plot(np.degrees(phi_jov_t2_arr),
                90.0-np.degrees(theta_s3_t2_arr),
                color=color_list[i])
        ax.plot(np.degrees(phi_jov_t3_arr),
                90.0-np.degrees(theta_s3_t3_arr),
                color=color_list[i])
        if num_reflection >= 2:
            ax.plot(np.degrees(phi_jov_t4_arr),
                    90.0-np.degrees(theta_s3_t4_arr),
                    color=color_list[i])
            ax.plot(np.degrees(phi_jov_t5_arr),
                    90.0-np.degrees(theta_s3_t5_arr),
                    color=color_list[i])
    fig.tight_layout()
    fig.savefig('img/test.jpg')
    plt.show()"""

    # Lead angle plot
    eqlead_t1_arr = tau_t1_arr*360.0/Psyn
    eqlead_t2_arr = tau_t2_arr*360.0/Psyn+eqlead_t1_arr
    eqlead_t3_arr = tau_t3_arr*360.0/Psyn+eqlead_t2_arr
    if num_reflection >= 2:
        eqlead_t4_arr = tau_t4_arr*360.0/Psyn+eqlead_t3_arr
        eqlead_t5_arr = tau_t5_arr*360.0/Psyn+eqlead_t4_arr

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0.0, 360.0)
    # ax.set_ylim(0.0, 90.0)
    # ax.set_xticks(np.arange(0, 360+1, 45))
    # ax.set_yticks(np.arange(0, 90+1, 30))
    # ax.grid(color=UC.lightgray, linewidth=0.5)
    ax.set_xlabel('Io SIII longitude [deg]')
    ax.set_ylabel('Equatorial lead angle [deg]')
    ax.plot(np.degrees(s3wlon_t0_arr)+eqlead_t1_arr,
            eqlead_t1_arr, color=UC.blue)
    ax.plot(np.degrees(s3wlon_t0_arr)+eqlead_t2_arr,
            eqlead_t2_arr, color=UC.red)
    ax.plot(np.degrees(s3wlon_t0_arr)+eqlead_t3_arr,
            eqlead_t3_arr, color=UC.blue)
    if num_reflection >= 2:
        ax.plot(np.degrees(s3wlon_t0_arr)+eqlead_t4_arr,
                eqlead_t4_arr, color=UC.red)
        ax.plot(np.degrees(s3wlon_t0_arr)+eqlead_t5_arr,
                eqlead_t5_arr, color=UC.blue)
    fig.tight_layout()
    fig.savefig('img/test4.jpg')
    plt.show()
    return None


# %% EXECUTE
if __name__ == '__main__':
    exname = '003/20250516_047'
    TARGET_MOON = 'Io'
    target_fp = ['MAW', 'TEB']
    PJ_num = [3]
    hem = 'N'
    Ai_num = 3
    ni_num = 50
    Ti_num = 60
    Zi = 1.3                # Io: 1.3 / Eu: 1.4 / Ga: 1.3
    Te = 6.0                # Io: 6.0 [eV]/ Eu: 20.0 / Ga: 300.0
    num_reflection = 2      # 1 or 2

    # Number of parallel processes
    parallel = 6

    if TARGET_MOON == 'Io':
        Psyn = Psyn_io
        r_moon = 5.9*RJ
        xticks = np.array([1, 10, 100, 500, 1000, 5000])
    elif TARGET_MOON == 'Europa':
        Psyn = Psyn_eu
        r_moon = 9.4*RJ
        xticks = np.array([1, 10, 100, 500, 1000, 5000])
    elif TARGET_MOON == 'Ganymede':
        Psyn = Psyn_ga
        r_moon = 15.0*RJ
        xticks = np.array([1, 10, 100, 1000])

    main()
