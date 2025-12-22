""" Leadangle_fit_JunoUVS.py

Created on Apr 8, 2025
@author: Shin Satoh

Description:ƒ
Using the lead angle values measured in one single Perijove of Juno,
this program iterates the Alfven wave tracing along the magnetic
field line and estimate the transit time of the Alfven wave from the
satellite to the auroral footprint.

Version
1.0.0 (Apr 8, 2025)
1.1.0 (Apr 22, 2025)
2.0.0 (Apr 28, 2025) TEB transit time
2.0.1 (May 11, 2025) Local time
2.0.2 (May 17, 2025) Select north or south (only for Io)

"""
# %% Import
import spiceypy as spice
from multiprocessing import Pool
# from numba import jit
import numpy as np
import math
# import copy
# import matplotlib.pyplot as plt
# from matplotlib.ticker import AutoMinorLocator
# from matplotlib.colors import LinearSegmentedColormap  # colormapをカスタマイズする
# import matplotlib.patheffects as pe

import Leadangle_wave as Wave
import datetime
import time
# import os
from scipy.io import readsav
import JupiterMag as jm

jm.Internal.Config(Model='jrm33', CartesianIn=True, CartesianOut=True)
jm.Con2020.Config(equation_type='analytic')


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


# %% Read the backtraced data
def read2backtraced(pj_list, target_moon: str, target_fp: str, target_hem='both', FLIP=False):
    # 初期化
    rho_arr = np.zeros(3)
    phi_arr = np.zeros(3)
    et_fp = np.zeros(3)
    hem_arr = np.zeros(3)

    for i in range(len(pj_list)):
        dir = 'data/Backtraced/PJ' + \
            str(pj_list[i]).zfill(2)+'/' + \
            target_moon[0]+'FP_info_v900km_fixed.txt'
        f = np.loadtxt(dir)
        # print(pj_list[i])

        if f.ndim == 1:
            rho_arr = np.append(rho_arr, f[0])
            phi_arr = np.append(phi_arr, f[1])
            et_fp = np.append(et_fp, f[2])
            hem_arr = np.append(hem_arr, f[3])
        else:
            rho_arr = np.append(rho_arr, f[0, :])
            phi_arr = np.append(phi_arr, f[1, :])
            et_fp = np.append(et_fp, f[2, :])
            hem_arr = np.append(hem_arr, f[3, :])

    # 余計な部分を削除
    rho_arr = rho_arr[3:]
    phi_arr = phi_arr[3:]
    et_fp = et_fp[3:]
    hem_arr = hem_arr[3:]

    # 半球で場合分け
    if target_hem == 'N':
        hem_idx = np.where((hem_arr == -1) | (hem_arr == -101))
        rho_arr = rho_arr[hem_idx]
        phi_arr = phi_arr[hem_idx]
        et_fp = et_fp[hem_idx]
        hem_arr = hem_arr[hem_idx]
    elif target_hem == 'S':
        hem_idx = np.where((hem_arr == 1) | (hem_arr == 101))
        rho_arr = rho_arr[hem_idx]
        phi_arr = phi_arr[hem_idx]
        et_fp = et_fp[hem_idx]
        hem_arr = hem_arr[hem_idx]

    # フットプリントの種類で場合分け
    if target_fp == 'MAW':
        fp_idx = np.where(np.abs(hem_arr) == 1)
        if FLIP is True:
            fp_idx = np.where(np.abs(hem_arr) == 101)
        rho_arr = rho_arr[fp_idx]
        phi_arr = phi_arr[fp_idx]
        et_fp = et_fp[fp_idx]
        hem_arr = hem_arr[fp_idx]
    elif target_fp == 'TEB':
        fp_idx = np.where(np.abs(hem_arr) == 101)
        if FLIP is True:
            fp_idx = np.where(np.abs(hem_arr) == 1)
        rho_arr = rho_arr[fp_idx]
        phi_arr = phi_arr[fp_idx]
        et_fp = et_fp[fp_idx]
        hem_arr = hem_arr[fp_idx]

    return rho_arr, phi_arr, et_fp, hem_arr


# %% GANYMEDE ONLY === read the current constant
def read_current_coef():
    f = np.loadtxt('results/azimuthal_current_fit/' +
                   TARGET_MOON[0:2]+'/PJ'+str(PJ_LIST[0])+TARGET_HEM+'.txt')
    mu_i_coef_ave = f[0]
    mu_i_coef_1_ave = f[1]
    return mu_i_coef_ave, mu_i_coef_1_ave


# %% Read the viewing angle
def viewingangle(PJnum: int, target_moon: str, target_fp: str, target_hem='both', FLIP=False):
    """
    Args:
        `PJnum` (int): Perijove number \\
        `moon` (str): Name of the target (Io/Europa/Ganymede) \\
        `footprint` (str): `MAW` or `TEB` \\

    Returns:
        `incident_angle` (ndarray): incident angles \\
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

    # INCIDENCE_ANGLE_TEB
    incident_angle = np.array(var['INCIDENCE_ANGLE_'+target_fp][0])[idx]
    hem = var['HEMISPHERE'][0][idx]
    wlon_fp = np.array(var['LON_'+target_fp][0])[idx]

    # Extract MAWs (exclude values -999.)
    fpvalues = np.where((wlon_fp > -100))
    incident_angle = incident_angle[fpvalues]
    hem = hem[fpvalues]

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
        incident_angle = incident_angle[hem_N]
        hem = hem[hem_N]
    elif target_hem == 'S':
        incident_angle = incident_angle[hem_S]
        hem = hem[hem_S]

    return incident_angle


# %% Footprint position will be mapped on the equatorial plane
def S3EQ(fpwlon: float, fplat: float, hemisphere, MOON: str):
    """
    Args:
        `fpwlon`: System III longitude of footprint aurora at Jupiter's upper atmosphere [deg]
        `satmodel`: footprint model from magnetic field model
        `MOON`: select from IO, EUROPA, GANYMEDE

    Returns:
        `y`: System III longitude of instantaneous field line at the orbital plane
    """

    data = readsav(savpath)

    # Select the target moon
    if (MOON == 'IO') or (MOON == 'Io'):
        data_name = 'ifp_contour'

    elif (MOON == 'EUROPA') or (MOON == 'Europa'):
        data_name = 'efp_contour'

    elif (MOON == 'GANYMEDE') or (MOON == 'Ganymede'):
        data_name = 'gfp_contour'

    # Select the hemisphere of the target auroal footprint
    if hemisphere == -1:   # North MAW
        variable = data[data_name+'_n']
        eqwlon = variable[0:-1][:, 0]
        s3wlon = variable[0:-1][:, 1]
        s3lat = variable[0:-1][:, 2]

    elif hemisphere == 1:  # South MAW
        variable = data[data_name+'_s']
        eqwlon = variable[0:-1][:, 0]
        s3wlon = variable[0:-1][:, 1]
        s3lat = variable[0:-1][:, 2]

    elif hemisphere == -101:   # North TEB
        variable = data[data_name+'_n']
        eqwlon = variable[0:-1][:, 0]
        s3wlon = variable[0:-1][:, 1]
        s3lat = variable[0:-1][:, 2]

    elif hemisphere == 101:  # South TEB
        variable = data[data_name+'_s']
        eqwlon = variable[0:-1][:, 0]
        s3wlon = variable[0:-1][:, 1]
        s3lat = variable[0:-1][:, 2]

    # Search the System III index
    distance = (s3wlon[:-1]-fpwlon)**2 + (s3lat[:-1]-fplat)**2

    """# Cartesian version
    fp_r = math.cos(math.radians(fplat))
    fp_x = fp_r*math.cos(math.radians(-fpwlon))
    fp_y = fp_r*math.sin(math.radians(-fpwlon))
    fp_z = math.sin(math.radians(fplat))
    ref_r = np.cos(np.radians(s3lat[:-1]))
    ref_x = ref_r*np.cos(np.radians(-s3wlon[:-1]))
    ref_y = ref_r*np.sin(np.radians(-s3wlon[:-1]))
    ref_z = np.sin(np.radians(s3lat[:-1]))
    distance = (ref_x-fp_x)**2 + (ref_y-fp_y)**2 + (ref_z-fp_z)**2"""

    argsorted = np.argsort(distance, axis=0)

    s3_idx0 = argsorted[0]
    _, eqwlon0 = s3wlon[s3_idx0], eqwlon[s3_idx0]

    y = eqwlon0

    return y


# %% Footprint position will be mapped on the equatorial plane
def calc_eqlead(wlon_fp,
                err_wlon_fp,
                lat_fp,
                err_lat_fp,
                hem_fp,
                moon_S3wlon0,
                target_moon: str):

    # 初期化
    eqlead_fp = np.zeros(wlon_fp.shape)
    eqlead_fp_0 = np.zeros(wlon_fp.shape)
    eqlead_fp_1 = np.zeros(wlon_fp.shape)

    wlon_fp_eq = calc_eqmapping(
        wlon_fp, lat_fp, hem_fp, target_moon)
    wlon_fp_eq_0 = calc_eqmapping(
        wlon_fp+err_wlon_fp, lat_fp, hem_fp, target_moon)
    wlon_fp_eq_1 = calc_eqmapping(
        wlon_fp-err_wlon_fp, lat_fp, hem_fp, target_moon)
    wlon_fp_eq_2 = calc_eqmapping(
        wlon_fp+err_wlon_fp, lat_fp+err_lat_fp, hem_fp, target_moon)
    wlon_fp_eq_3 = calc_eqmapping(
        wlon_fp-err_wlon_fp, lat_fp-err_lat_fp, hem_fp, target_moon)

    for i in range(wlon_fp.size):
        eqlead_fp_0[i], _ = calc_eqerrors(
            wlon_fp_eq[i], wlon_fp_eq_0[i], wlon_fp_eq_1[i], wlon_fp_eq_2[i], wlon_fp_eq_3[i]
        )

        # Equatorial lead angle
        eqlead_fp[i] = moon_S3wlon0[i] - wlon_fp_eq[i]

        if eqlead_fp[i] < 0:
            eqlead_fp[i] += 360.
            # print(delta0, delta1, delta2, delta3)
        if wlon_fp_eq[i] < 0:
            wlon_fp_eq[i] += 360.

    return eqlead_fp, eqlead_fp_0, eqlead_fp_1, wlon_fp_eq


# %% Equatorial mapping using the table
def calc_eqmapping(wlon_fp, lat_fp, hem_fp, target_moon):
    wlon_fp_eq = np.zeros(wlon_fp.shape)

    for i in range(wlon_fp.size):
        wlon_fp_eq[i] = S3EQ(wlon_fp[i],
                             lat_fp[i],
                             hem_fp[i], target_moon)

    return wlon_fp_eq


# %%
def calc_eqerrors(center, wlon_fp_eq_0, wlon_fp_eq_1, wlon_fp_eq_2, wlon_fp_eq_3):
    delta0 = abs(center - wlon_fp_eq_0)
    delta1 = abs(center - wlon_fp_eq_1)
    delta2 = abs(center - wlon_fp_eq_2)
    delta3 = abs(center - wlon_fp_eq_3)

    if delta0 >= 360.0:
        delta0 = abs(delta0-360.0)
    if delta1 >= 360.0:
        delta1 = abs(delta1-360.0)
    if delta2 >= 360.0:
        delta2 = abs(delta2-360.0)
    if delta3 >= 360.0:
        delta3 = abs(delta3-360.0)

    # Errors in the observed lead angle
    eqlead_fp_1 = 0   # unused !
    eqlead_fp_0 = np.max([delta0, delta1, delta2, delta3])

    return eqlead_fp_0, eqlead_fp_1


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


# %% Calculate launch site of the Alfven waves
def Alfven_launch_site(et_fp, eqlead_fp, moon):
    t1_arr = np.zeros(et_fp.size)

    if moon == 'Io':
        Psyn = Psyn_io
    elif moon == 'Europa':
        Psyn = Psyn_eu
    elif moon == 'Ganymede':
        Psyn = Psyn_ga

    for i in range(et_fp.size):
        t0 = spice.et2datetime(et_fp[i])
        omg_syn = 360/Psyn  # [deg/sec]
        tau_A = -eqlead_fp[i]/omg_syn  # Alfven travel time [sec]
        dt = datetime.timedelta(seconds=tau_A)
        t1_arr[i] = spice.datetime2et(t0+dt)

    x, y, z, r, theta, phi, S3wlon = moonS3wlon_arr(t1_arr, moon)

    return x, y, z, r, theta, phi, S3wlon


# %% Calculate the plasma sheet scale height
def scaleheight(Ai, Zi, Ti, Te):
    """
    Args:
        These are parameters of ions in the plasma sheet
        `Ai` : Ion mass [amu]
        `Zi` : Ion charge [C]
        `Ti` : Ion temperature [eV]
        `Te` : Electron temperature [eV]

    Returns:
        H : Scale height of the plasma sheet [m] \\
    """
    H = 0.64*RJ*np.sqrt((Ti/Ai)*(1+(Zi*Te/Ti)))
    # H = 0.64*RJ*np.sqrt((Ti/Ai))
    return H


# %% Function to be in loop
def calc(Ai, ni, Hp, r_A0, S3wlon_A0, z_A0, hem, S_A0=0):
    if CURRENT_CONSTANT_OFFSET:
        current_coef, _ = read_current_coef()
        tau, _, _, _ = Wave.Awave().trace3_magnetodisk(
            r_A0,
            np.radians(S3wlon_A0),
            z_A0,
            S_A0,
            Ai,
            ni,
            Hp,
            hem,
            current_coef=current_coef,
        )

    else:
        tau, _, _, _ = Wave.Awave().trace3(
            r_A0,
            np.radians(S3wlon_A0),
            z_A0,
            S_A0,
            Ai,
            ni,
            Hp,
            hem
        )
    return tau


# %%
def Obsresults(PJ_LIST, TARGET_MOON, TARGET_FP, TARGET_HEM, FLIP):
    # 初期化
    wlon_fp = np.zeros(3)
    err_wlon_fp = np.zeros(3)
    lat_fp = np.zeros(3)
    err_lat_fp = np.zeros(3)
    wlon_moon_fp = np.zeros(3)
    et_fp = np.zeros(3)
    hem_fp = np.zeros(3)
    pj_fp = np.zeros(3)

    for i in PJ_LIST:
        for j in TARGET_FP:
            wlon_fp1, err_wlon_fp1, lat_fp1, err_lat_fp1, wlon_moon_fp1, et_fp1, hem_fp1 = read1savfile(
                PJnum=i, target_moon=TARGET_MOON, target_fp=j, target_hem=TARGET_HEM, FLIP=FLIP)

            wlon_fp = np.append(wlon_fp, wlon_fp1)
            err_wlon_fp = np.append(err_wlon_fp, err_wlon_fp1)
            lat_fp = np.append(lat_fp, lat_fp1)
            err_lat_fp = np.append(err_lat_fp, err_lat_fp1)
            wlon_moon_fp = np.append(wlon_moon_fp, wlon_moon_fp1)
            et_fp = np.append(et_fp, et_fp1)
            hem_fp = np.append(hem_fp, hem_fp1)
            pj_fp = np.append(pj_fp, i*np.ones(wlon_fp1.size))

    # 余計な部分を削除
    wlon_fp = wlon_fp[3:]
    err_wlon_fp = err_wlon_fp[3:]
    lat_fp = lat_fp[3:]
    err_lat_fp = err_lat_fp[3:]
    wlon_moon_fp = wlon_moon_fp[3:]
    et_fp = et_fp[3:]
    hem_fp = hem_fp[3:]
    pj_fp = pj_fp[3:]

    return wlon_fp, err_wlon_fp, lat_fp, err_lat_fp, wlon_moon_fp, et_fp, hem_fp, pj_fp


# %%
def Obsresults_back(PJ_LIST, TARGET_MOON, TARGET_FP, TARGET_HEM, FLIP):
    # 初期化
    rho_arr = np.zeros(3)
    phi_arr = np.zeros(3)
    et_fp = np.zeros(3)
    hem_arr = np.zeros(3)

    for i in PJ_LIST:
        for j in TARGET_FP:
            rho_arr1, phi_arr1, et_fp1, hem_arr1 = read2backtraced(
                [i], target_moon=TARGET_MOON, target_fp=j, target_hem=TARGET_HEM, FLIP=FLIP)

            rho_arr = np.append(rho_arr, rho_arr1)
            phi_arr = np.append(phi_arr, phi_arr1)
            et_fp = np.append(et_fp, et_fp1)
            hem_arr = np.append(hem_arr, hem_arr1)

    # 余計な部分を削除
    rho_arr = rho_arr[3:]
    phi_arr = phi_arr[3:]
    et_fp = et_fp[3:]
    hem_arr = hem_arr[3:]

    return rho_arr, phi_arr, et_fp, hem_arr


# %% Calculate the error for west longitude of the moon
def eqwlong_err(Psyn, dt):

    err = dt*360.0/Psyn     # [deg]

    return err


# %% Convert eV to speed in [m s-1]
def eV2speed(energy):
    """
    energy: [eV]
    """
    J = e*energy    # [J]
    v = math.sqrt(J/(0.5*me))     # [m s-1]
    return v


# %% Transit time of TEB from one to the other hemisphere.
def TEB_transit(r_moon, s3wlon, target_moon):
    if target_moon == 'Io':
        v_e = C
    elif target_moon == 'Europa':
        TEB_en = 3600.0  # TEB ENERGY [eV]
        v_e = eV2speed(TEB_en)
    elif target_moon == 'Ganymede':
        TEB_en = 3600.0  # TEB ENERGY [eV]
        v_e = eV2speed(TEB_en)

    phi = math.radians(360.0-200.0)    # [rad]
    x0 = (r_moon/RJ)*np.cos(phi)        # [RJ]
    y0 = (r_moon/RJ)*np.sin(phi)        # [RJ]
    z0 = 0                              # [RJ]

    # Position is always in RJ
    T2 = jm.TraceField(x0, y0, z0, Verbose=True,
                       IntModel='jrm33',
                       ExtModel='Con2020',
                       MaxLen=600000,
                       MaxStep=0.0003,
                       InitStep=0.00001,
                       MinStep=0.00001)

    Fllen = T2.equator.fllen*RJ
    transit_time = Fllen/v_e      # [sec]

    return transit_time


# %% Create argument mesh and make them 1d vectors
def create_argmesh(a0=1, a1=2, a_num=3, a_scale='linear',
                   b0=1, b1=2, b_num=3, b_scale='linear',
                   c0=-99.9, c1=-98.0, c_num=0, c_scale='linear'):

    # パラメータ空間の作成
    a_arr = 0     # 1st parameter
    b_arr = 0     # 2nd parameter
    c_arr = 0     # 3rd parameter

    if a_scale == 'linear':
        a_arr = np.linspace(a0, a1, a_num)
    elif a_scale == 'log':
        a_arr = np.linspace(np.log(a0), np.log(a1), a_num)
        a_arr = np.exp(a_arr)

    if b_scale == 'linear':
        b_arr = np.linspace(b0, b1, b_num)
    elif b_scale == 'log':
        b_arr = np.linspace(np.log(b0), np.log(b1), b_num)
        b_arr = np.exp(b_arr)

    if c_num >= 1:
        if c_scale == 'linear':
            c_arr = np.linspace(c0, c1, c_num)
        elif c_scale == 'log':
            c_arr = np.linspace(np.log(c0), np.log(c1), c_num)
            c_arr = np.exp(c_arr)

        a_mesh, b_mesh, c_mesh = np.meshgrid(a_arr, b_arr, c_arr)
        # -> shape is like (b_arr.size, a_arr.size, c_arr.size)

        a_1d = a_mesh.reshape(int(a_arr.size*b_arr.size*c_arr.size))
        b_1d = b_mesh.reshape(int(a_arr.size*b_arr.size*c_arr.size))
        c_1d = c_mesh.reshape(int(a_arr.size*b_arr.size*c_arr.size))
    else:
        a_mesh, b_mesh = np.meshgrid(a_arr, b_arr)
        a_1d = a_mesh.reshape(int(a_arr.size*b_arr.size))
        b_1d = b_mesh.reshape(int(a_arr.size*b_arr.size))
        c_1d = -999.9

    return a_1d, b_1d, c_1d, a_arr, b_arr, c_arr


# %% Main function
def main():
    # Select moon synodic orbital period
    if TARGET_MOON == 'Io':
        Psyn = Psyn_io
        Zi = 1.3    # ION CHARGE [C]
        Te = 6.0    # ELECTRON TEMPERATURE [eV]
    elif TARGET_MOON == 'Europa':
        Psyn = Psyn_eu
        Zi = 1.4    # ION CHARGE [C]
        Te = 20.0   # ELECTRON TEMPERATURE [eV]
    elif TARGET_MOON == 'Ganymede':
        Psyn = Psyn_ga
        Zi = 1.3    # ION CHARGE [C]
        Te = 300.0  # ELECTRON TEMPERATURE [eV]

    wlon_fp, err_wlon_fp, lat_fp, err_lat_fp, _, et_fp, hem_fp, _ = Obsresults(
        PJ_LIST, TARGET_MOON, TARGET_FP, TARGET_HEM, FLIP
    )

    # Time: t0, the observation time
    _, _, _, _, _, _, moon_S3wlon = moonS3wlon_arr(et_fp, TARGET_MOON)

    eqlead_fp, eqlead_fp_0, _, wlon_fp_eq = calc_eqlead(wlon_fp,
                                                        err_wlon_fp,
                                                        lat_fp,
                                                        err_lat_fp,
                                                        hem_fp,
                                                        moon_S3wlon,
                                                        TARGET_MOON)

    if USE_BACKTRACED:
        _, phi_arr, et_fp2, hem_arr = Obsresults_back(PJ_LIST,
                                                      TARGET_MOON,
                                                      TARGET_FP,
                                                      TARGET_HEM,
                                                      FLIP)

        eqlead_fp = moon_S3wlon - phi_arr
        for i in range(eqlead_fp.size):
            if eqlead_fp[i] < 0:
                eqlead_fp[i] += 360.

        print('hem_arr:', hem_arr)
        print('Eq map diff.: ', wlon_fp_eq-phi_arr)

    # Moon position when the Alfven waves launched (Time: t0-tau_A)
    _, _, z_A0, r_A0, _, _, S3wlon_A0 = Alfven_launch_site(et_fp,
                                                           eqlead_fp,
                                                           TARGET_MOON)

    # パラメータ空間(meshgrid → 1d)の作成
    Ai_1d, ni_1d, Ti_1d, _, _, _ = create_argmesh(Ai_0, Ai_1, Ai_num, Ai_scale,
                                                  ni_0, ni_1, ni_num, ni_scale,
                                                  Ti_0, Ti_1, Ti_num, Ti_scale)
    H_1d = scaleheight(Ai=Ai_1d, Zi=Zi, Ti=Ti_1d, Te=Te)
    arg_size = Ai_1d.size

    # 衛星本体の経度: Juno spin中の変位 (どのデータに対しても一定値)
    sigma_x = eqwlong_err(Psyn, dt=22.5)  # [deg]

    # 注意: iは観測データ点のインデックス
    i_size = wlon_fp.size
    y_obs = np.zeros((i_size, arg_size))
    sigma_total = np.zeros((i_size, arg_size))
    sigma_y = np.zeros(i_size)
    y_estimate = np.zeros((i_size, arg_size))
    print('PJ number:', PJ_LIST)
    print('Target moon:', TARGET_MOON)
    print('Target fp:', TARGET_FP)
    print('Target hemisphere:', TARGET_HEM)
    if FLIP is True:
        print('MAW -> TEB (flipped)')
    print('Number of data points used/total:', i_size, '/', wlon_fp.size)
    print('Param space shape:', ni_num, Ai_num, Ti_num)
    start_all = time.time()
    for i in range(i_size):
        # print('r_A0 [RJ]:', r_A0[i]/RJ)
        # print('S3wlon_A0 [deg]:', S3wlon_A0[i])

        start_1loop = time.time()
        S_A0 = Wave.Awave().tracefield(r_A0[i],
                                       np.radians(S3wlon_A0[i]),
                                       z_A0[i]
                                       )

        args = list(zip(
            Ai_1d,
            ni_1d,
            H_1d,
            r_A0[i]*np.ones(arg_size),
            S3wlon_A0[i]*np.ones(arg_size),
            z_A0[i]*np.ones(arg_size),
            hem_fp[i]*np.ones(arg_size),
            S_A0*np.ones(arg_size)
        ))
        with Pool(processes=parallel) as pool:
            results_list = list(pool.starmap(calc, args))
        tau = np.array(results_list)    # [sec]

        if (hem_fp[i] == 101) or (hem_fp[i] == -101):
            tau += TEB_transit(r_A0[i], S3wlon_A0[i], TARGET_MOON)

        print(str(i).zfill(2), '- Loop time [sec]:', round(
            time.time()-start_1loop, 4))

        y_obs[i, :] = eqlead_fp[i]*np.ones(arg_size)
        # sigma_y[i] = eqlead_fp_0[i]
        # sigma_total[i, :] = sigma_y[i]*np.ones(arg_size)
        # !!!!!
        # !!!!!
        # !!!!! 積分時間中に衛星本体の経度が変化することを考慮したsigma
        sigma_y[i] = eqlead_fp_0[i]+sigma_x
        sigma_total[i, :] = sigma_y[i]*np.ones(arg_size)
        # !!!!!
        # !!!!!
        # !!!!!
        y_estimate[i, :] = tau*360/Psyn
        # print('Diff in eqlead:', (y_obs[i, :]-y_estimate[i, :]))

    print('--- Total time [sec]:', round(time.time()-start_all, 4))

    # Chi square value
    chi2 = np.sum(((y_obs-y_estimate)/sigma_total)**2, axis=0)
    # !!!!!
    # !!!!!
    chi2 = chi2/(i_size-3)
    # !!!!!
    # !!!!!
    np.savetxt('results/fit/'+exname+'/params_chi2.txt',
               chi2)
    np.savetxt('results/fit/'+exname+'/params_Ai.txt',
               Ai_1d)
    np.savetxt('results/fit/'+exname+'/params_ni.txt',
               ni_1d)
    np.savetxt('results/fit/'+exname+'/params_Ti.txt',
               Ti_1d)
    np.savetxt('results/fit/'+exname+'/params_H.txt',
               H_1d)
    np.savetxt('results/fit/'+exname+'/eqlead_est.txt',
               y_estimate)
    np.savetxt('results/fit/'+exname+'/eqlead_obs.txt',
               eqlead_fp)
    np.savetxt('results/fit/'+exname+'/sigma_y.txt',
               sigma_y)
    np.savetxt('results/fit/'+exname+'/hems_obs.txt',
               hem_fp)
    np.savetxt('results/fit/'+exname+'/moon_S3wlon_obs.txt',
               moon_S3wlon)
    np.savetxt('results/fit/'+exname+'/et_obs.txt',
               et_fp)


# %% EXECUTE
if __name__ == '__main__':
    # Name of execution
    exname = '005/20251221_308'

    # Input about Juno observation
    TARGET_MOON = 'Ganymede'
    TARGET_FP = ['MAW', 'TEB']
    PJ_LIST = [12]
    TARGET_HEM = 'both'   # 'both', 'N', or 'S'
    FLIP = False          # ALWAYS FALSE! Flip the flag (TEB <-> MAW)
    USE_BACKTRACED = True           # True for '005'
    CURRENT_CONSTANT_OFFSET = True  # ALWAYS FALSE!

    # Input about the paremeter space
    Ai_0, Ai_1, Ai_num, Ai_scale = 12.0, 16.0, 3, 'linear'
    ni_0, ni_1, ni_num, ni_scale = 1.0, 100.0, 50, 'log'
    Ti_0, Ti_1, Ti_num, Ti_scale = 1.0, 200.0, 60, 'log'

    # Number of parallel processes
    parallel = 35

    main()


"""
for j in range(Ai_1d.size):
    Ai = Ai_1d[j]
    ni = ni_1d[j]
    Hp = H_1d[j]
    S_A0 = Wave.Awave().tracefield(r_A0[i],
                                   np.radians(S3wlon_A0[i]),
                                   z_A0[i]
                                   )
    tau, _, _, _ = Wave.Awave().trace3(r_A0[i],
                                       np.radians(S3wlon_A0[i]),
                                       z_A0[i],
                                       S_A0,
                                       Ai,
                                       ni,
                                       Hp,
                                       hem_MAW[i]
                                       )

    print('eqlead [deg]:', tau*360/Psyn_eu)
"""

# 作業メモ
# Juno/UVSデータ(フットプリントの緯度経度)を取り込む → done
# とりあえずMAWだけを使う方針で → done
# 衛星本体のz座標は必ずしも0ではない → Spiceで計算 done
# 各時刻の赤道リード角を計算する部分を追加 done
# トレースもほぼできた。伝播時間からリード角も計算OK。
# トレース終着点は観測されたフットプリントの経度と違っていいのか？
