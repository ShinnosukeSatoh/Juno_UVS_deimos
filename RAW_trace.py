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

from Leadangle_fit_JunoUVS import moonS3wlon_arr
import Leadangle_wave as Wave

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


# ------------------------------------------------------------
# Hardcoded Juno Perijove UTC times
# ------------------------------------------------------------
JUNO_PJ_TIMES = {
    0: "2016-07-05 02:47:31.851",
    1: "2016-08-27 12:50:44.060",
    2: "2016-10-19 18:10:53.669",
    3: "2016-12-11 17:03:40.665",
    4: "2017-02-02 12:57:08.935",
    5: "2017-03-27 08:51:51.552",
    6: "2017-05-19 06:00:46.945",
    7: "2017-07-11 01:54:42.322",
    8: "2017-09-01 21:48:50.499",
    9: "2017-10-24 17:42:31.438",
    10: "2017-12-16 17:56:58.997",
    11: "2018-02-07 13:51:29.722",
    12: "2018-04-01 09:45:42.502",
    13: "2018-05-24 05:39:50.502",
    14: "2018-07-16 05:17:21.832",
    15: "2018-09-07 01:11:40.519",
    16: "2018-10-29 21:05:59.956",
    17: "2018-12-21 16:59:48.319",
    18: "2019-02-12 17:34:30.940",
    19: "2019-04-06 12:14:22.473",
    20: "2019-05-29 08:08:18.282",
    21: "2019-07-21 04:02:43.348",
    22: "2019-09-12 03:40:44.422",
    23: "2019-11-03 22:18:13.850",
    24: "2019-12-26 17:36:12.571",
    25: "2020-02-17 17:51:55.133",
    26: "2020-04-10 13:47:40.171",
    27: "2020-06-02 10:20:02.882",
    28: "2020-07-25 06:15:27.223",
    29: "2020-09-16 02:10:52.328",
    30: "2020-11-08 01:49:42.104",
    31: "2020-12-30 21:45:44.567",
    32: "2021-02-21 17:40:34.245",
}


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


# %% Alfvén launch site
def calc_launch(r_moon, s3wlon_t0, z_orbit):
    S_A0 = Wave.Awave().tracefield(r_moon,
                                   s3wlon_t0,
                                   z_orbit)
    return S_A0


# %% calc function
def calc(Ai, ni, Hp, r_t0, s3wlon_t0, z_t0, s_t0, hem, num_reflection):
    # Initialize the result list/array
    tau_list = []

    # Initital trace
    # -> MAW position at 900 km altitude
    tau_t1, rs_t1, s3wlon_t1, theta_s3_t1, s_t1 = Wave.Awave().trace3_reflect(r_t0,
                                                                              s3wlon_t0,
                                                                              z_t0,
                                                                              s_t0,
                                                                              Ai,
                                                                              ni,
                                                                              Hp,
                                                                              hem)
    z_t1 = rs_t1*math.cos(theta_s3_t1[-1])

    # 1st reflection
    # -> 1st RAW position at 900 km altitude on the opposite hemisphere
    tau_t2, rs_t2, s3wlon_t2, theta_s3_t2, s_t2 = Wave.Awave().trace3_reflect(rs_t1,
                                                                              s3wlon_t1[-1],
                                                                              z_t1,
                                                                              s_t1,
                                                                              Ai,
                                                                              ni,
                                                                              Hp,
                                                                              hem*(-1))
    z_t2 = rs_t2*math.cos(theta_s3_t2[-1])

    # 2nd reflection
    # -> 1st RAW position at 900 km altitude on the same hemisphere as MAW
    tau_t3, rs_t3, s3wlon_t3, theta_s3_t3, s_t3 = Wave.Awave().trace3_reflect(rs_t2,
                                                                              s3wlon_t2[-1],
                                                                              z_t2,
                                                                              s_t2,
                                                                              Ai,
                                                                              ni,
                                                                              Hp,
                                                                              hem*(-1)**2)
    z_t3 = rs_t3*math.cos(theta_s3_t3[-1])

    # 3rd reflection
    # -> 2nd RAW position at 900 km altitude on the opposite hemisphere
    tau_t4, rs_t4, s3wlon_t4, theta_s3_t4, s_t4 = Wave.Awave().trace3_reflect(rs_t3,
                                                                              s3wlon_t3[-1],
                                                                              z_t3,
                                                                              s_t3,
                                                                              Ai,
                                                                              ni,
                                                                              Hp,
                                                                              hem*(-1)**3)
    z_t4 = rs_t4*math.cos(theta_s3_t4[-1])

    # 4th reflection
    # -> 2nd RAW position at 900 km altitude on the same hemisphere as MAW
    tau_t5, rs_t5, s3wlon_t5, theta_s3_t5, s_t5 = Wave.Awave().trace3_reflect(rs_t4,
                                                                              s3wlon_t4[-1],
                                                                              z_t4,
                                                                              s_t4,
                                                                              Ai,
                                                                              ni,
                                                                              Hp,
                                                                              hem*(-1)**4)
    z_t5 = rs_t5*math.cos(theta_s3_t5[-1])

    # 5th reflection
    # -> 3rd RAW position at 900 km altitude on the opposite hemisphere
    tau_t6, rs_t6, s3wlon_t6, theta_s3_t6, s_t6 = Wave.Awave().trace3_reflect(rs_t5,
                                                                              s3wlon_t5[-1],
                                                                              z_t5,
                                                                              s_t5,
                                                                              Ai,
                                                                              ni,
                                                                              Hp,
                                                                              hem*(-1)**5)
    z_t6 = rs_t6*math.cos(theta_s3_t6[-1])

    # 6th reflection
    # -> 3rd RAW position at 900 km altitude on the same hemisphere as MAW
    tau_t7, rs_t7, s3wlon_t7, theta_s3_t7, s_t7 = Wave.Awave().trace3_reflect(rs_t6,
                                                                              s3wlon_t6[-1],
                                                                              z_t6,
                                                                              s_t6,
                                                                              Ai,
                                                                              ni,
                                                                              Hp,
                                                                              hem*(-1)**6)
    z_t7 = rs_t7*math.cos(theta_s3_t7[-1])

    # Result lists
    tau_list = [tau_t1, tau_t2, tau_t3,
                tau_t4, tau_t5, tau_t6,
                tau_t7,]
    theta_s3_list = [theta_s3_t1, theta_s3_t2, theta_s3_t3,
                     theta_s3_t4, theta_s3_t5, theta_s3_t6,
                     theta_s3_t7]
    s3wlon_list = [s3wlon_t1, s3wlon_t2, s3wlon_t3,
                   s3wlon_t4, s3wlon_t5, s3wlon_t6,
                   s3wlon_t7]

    # print('h [m]:', (rs_t1-1.0*RJ)*1E-3,
    #       (rs_t2-1.0*RJ)*1E-3, (rs_t3-1.0*RJ)*1E-3)
    # print('z [RJ]:', z_t1/RJ, z_t2/RJ, z_t3/RJ)
    # print('s [RJ]:', s_t1/RJ, s_t2/RJ, s_t3/RJ)
    # print('phi [deg]:', math.degrees(s3wlon_t1),
    #       math.degrees(s3wlon_t2), math.degrees(s3wlon_t3), )
    # print('Array shape:', phi_jov_t1.shape,
    #       phi_jov_t2.shape,  phi_jov_t3.shape)

    return tau_list, theta_s3_list, s3wlon_list


# %% calc function
def calc_copy(Ai, ni, Hp, r_t0, s3wlon_t0, z_t0, s_t0, hem, reflections, alt):
    """
    Return:
        tau_list ... time [sec]
        theta_s3_list ... SIII colatitude [rad]
        s3wlon_list ... SIII west longitude [rad]
    """
    # Initialize the result list/array
    tau_list = []

    # Initital trace
    # -> MAW position at 900 km altitude
    tau_t1, rs_t1, s3wlon_t1, theta_s3_t1, s_t1 = Wave.Awave().trace3_reflect(r_t0,
                                                                              s3wlon_t0,
                                                                              z_t0,
                                                                              s_t0,
                                                                              Ai,
                                                                              ni,
                                                                              Hp,
                                                                              hem,
                                                                              altitude=alt)
    z_t1 = rs_t1*math.cos(theta_s3_t1[-1])

    # 1st reflection
    # -> 1st RAW position at 900 km altitude on the opposite hemisphere
    tau_t2, rs_t2, s3wlon_t2, theta_s3_t2, s_t2 = Wave.Awave().trace3_reflect(rs_t1,
                                                                              s3wlon_t1[-1],
                                                                              z_t1,
                                                                              s_t1,
                                                                              Ai,
                                                                              ni,
                                                                              Hp,
                                                                              hem *
                                                                              (-1),
                                                                              altitude=alt)

    # 2nd reflection
    # -> 1st RAW position at 900 km altitude on the same hemisphere as MAW
    tau_t3 = tau_t2[-1] - tau_t2[::-1]
    rs_t3 = rs_t1
    s3wlon_t3 = s3wlon_t2[::-1]
    theta_s3_t3 = theta_s3_t2[::-1]

    # 3rd reflection
    # -> 2nd RAW position at 900 km altitude on the opposite hemisphere
    tau_t4 = tau_t2
    rs_t4 = rs_t2
    s3wlon_t4 = s3wlon_t2
    theta_s3_t4 = theta_s3_t2

    # 4th reflection
    # -> 2nd RAW position at 900 km altitude on the same hemisphere as MAW
    tau_t5 = tau_t3
    rs_t5 = rs_t3
    s3wlon_t5 = s3wlon_t3
    theta_s3_t5 = theta_s3_t3

    # 5th reflection
    # -> 3rd RAW position at 900 km altitude on the opposite hemisphere
    tau_t6 = tau_t4
    rs_t6 = rs_t4
    s3wlon_t6 = s3wlon_t4
    theta_s3_t6 = theta_s3_t4

    # 6th reflection
    # -> 3rd RAW position at 900 km altitude on the same hemisphere as MAW
    tau_t7 = tau_t5
    rs_t7 = rs_t5
    s3wlon_t7 = s3wlon_t5
    theta_s3_t7 = theta_s3_t5

    # 7th reflection
    # -> 4th RAW position at 900 km altitude on the opposite hemisphere
    tau_t8 = tau_t6
    rs_t8 = rs_t6
    s3wlon_t8 = s3wlon_t6
    theta_s3_t8 = theta_s3_t6

    # 8th reflection
    # -> 4th RAW position at 900 km altitude on the same hemisphere as MAW
    tau_t9 = tau_t7
    rs_t9 = rs_t7
    s3wlon_t9 = s3wlon_t7
    theta_s3_t9 = theta_s3_t7

    # Result lists
    skip = 60
    tau_list = [
        np.hstack((tau_t1[::skip], tau_t1[-1])),
        np.hstack((tau_t2[::skip], tau_t2[-1])),
        np.hstack((tau_t3[::skip], tau_t3[-1])),
        np.hstack((tau_t4[::skip], tau_t4[-1])),
        np.hstack((tau_t5[::skip], tau_t5[-1])),
        np.hstack((tau_t6[::skip], tau_t6[-1])),
        np.hstack((tau_t7[::skip], tau_t7[-1])),
        np.hstack((tau_t8[::skip], tau_t8[-1])),
        np.hstack((tau_t9[::skip], tau_t9[-1])),
    ]
    theta_s3_list = [
        np.hstack((theta_s3_t1[::skip], theta_s3_t1[-1])),
        np.hstack((theta_s3_t2[::skip], theta_s3_t2[-1])),
        np.hstack((theta_s3_t3[::skip], theta_s3_t3[-1])),
        np.hstack((theta_s3_t4[::skip], theta_s3_t4[-1])),
        np.hstack((theta_s3_t5[::skip], theta_s3_t5[-1])),
        np.hstack((theta_s3_t6[::skip], theta_s3_t6[-1])),
        np.hstack((theta_s3_t7[::skip], theta_s3_t7[-1])),
        np.hstack((theta_s3_t8[::skip], theta_s3_t8[-1])),
        np.hstack((theta_s3_t9[::skip], theta_s3_t9[-1])),
    ]
    s3wlon_list = [
        np.hstack((s3wlon_t1[::skip], s3wlon_t1[-1])),
        np.hstack((s3wlon_t2[::skip], s3wlon_t2[-1])),
        np.hstack((s3wlon_t3[::skip], s3wlon_t3[-1])),
        np.hstack((s3wlon_t4[::skip], s3wlon_t4[-1])),
        np.hstack((s3wlon_t5[::skip], s3wlon_t5[-1])),
        np.hstack((s3wlon_t6[::skip], s3wlon_t6[-1])),
        np.hstack((s3wlon_t7[::skip], s3wlon_t7[-1])),
        np.hstack((s3wlon_t8[::skip], s3wlon_t8[-1])),
        np.hstack((s3wlon_t9[::skip], s3wlon_t9[-1])),
    ]

    # print('s [RJ]:', s_t1/RJ, s_t2/RJ)

    return tau_list, theta_s3_list, s3wlon_list


# %% Main function
def main():
    # the initial SIII w-longitude of the moon
    s3wlon_t0_arr = np.radians(np.arange(-95.0, 360.0+1.0, d_phi))
    s3wlon_180 = np.argmin(abs(s3wlon_t0_arr-np.pi))
    arr_size = s3wlon_t0_arr.size

    Ai_best, ni_best, Ti_best, Hp_best = load_best_fit()

    # Alfvén wave launch position in the field line coordinate
    args = list(zip(r_moon*np.ones(arr_size),
                    s3wlon_t0_arr,
                    0*np.ones(arr_size)))
    """S_A0_arr = np.zeros(arr_size)
    for i in range(arr_size):
        S_A0_arr[i] = Wave.Awave().tracefield(r_moon,
                                              s3wlon_t0_arr[i],
                                              0
                                              )"""
    with Pool(processes=parallel) as pool:
        S_A0_list = list(pool.starmap(calc_launch, args))
    S_A0_arr = np.array(S_A0_list)
    print('Alfvén launch site determined.')

    # Arguments zipped
    NS = -1  # Initial trace direction (-1: 北向き, 1: 南向き)
    args = list(zip(Ai_best*np.ones(arr_size),
                    ni_best*np.ones(arr_size),
                    Hp_best*np.ones(arr_size),
                    r_moon*np.ones(arr_size),
                    s3wlon_t0_arr,
                    0.0*np.ones(arr_size),
                    S_A0_arr,
                    NS*np.ones(arr_size, dtype=int),
                    reflections*np.ones(arr_size, dtype=int),
                    altitude*np.ones(arr_size, dtype=int)))

    # Parallelized
    time_start = time.time()
    with Pool(processes=parallel) as pool:
        results_N = list(pool.starmap(calc_copy, args))
    print('Number of CPU cores used:', parallel)
    print('Loop time [sec]:', round(time.time()-time_start, 2))

    # Arguments zipped
    NS = 1  # Initial trace direction (-1: 北向き, 1: 南向き)
    args = list(zip(Ai_best*np.ones(arr_size),
                    ni_best*np.ones(arr_size),
                    Hp_best*np.ones(arr_size),
                    r_moon*np.ones(arr_size),
                    s3wlon_t0_arr,
                    0.0*np.ones(arr_size),
                    S_A0_arr*np.ones(arr_size),
                    NS*np.ones(arr_size, dtype=int),
                    reflections*np.ones(arr_size, dtype=int),
                    altitude*np.ones(arr_size, dtype=int)))

    # Parallelized
    time_start = time.time()
    with Pool(processes=parallel) as pool:
        results_S = list(pool.starmap(calc_copy, args))
    print('Number of CPU cores used:', parallel)
    print('Loop time [sec]:', round(time.time()-time_start, 2))

    # print('=== ALL ===')
    # print(results_list)
    # >> [moon_position, out_put_variants, reflection_number, tau_step]

    # Create the wavefront data array
    # j = 0: tau_arr (Alfveen travel time array[sec])
    # j = 1: theta_s3_arr (SIII colatitude [rad])
    # j = 2: s3wlon_arr (SIII west longitude [rad])
    data_N0 = np.zeros((results_N[s3wlon_180][0][0].size, 3))  # N MAW
    data_N1 = np.zeros((results_N[s3wlon_180][0][1].size, 3))  # a reflection
    data_N2 = np.zeros((results_N[s3wlon_180][0][2].size, 3))  # 2 reflections
    data_N3 = np.zeros((results_N[s3wlon_180][0][3].size, 3))  # 3 reflections
    data_N4 = np.zeros((results_N[s3wlon_180][0][4].size, 3))  # 4 reflections
    data_N5 = np.zeros((results_N[s3wlon_180][0][5].size, 3))  # 5 reflections
    data_N6 = np.zeros((results_N[s3wlon_180][0][6].size, 3))  # 6 reflections
    data_N7 = np.zeros((results_N[s3wlon_180][0][7].size, 3))  # 7 reflections
    data_N8 = np.zeros((results_N[s3wlon_180][0][8].size, 3))  # 8 reflections
    data_S0 = np.zeros((results_S[s3wlon_180][0][0].size, 3))  # S MAW
    data_S1 = np.zeros((results_S[s3wlon_180][0][1].size, 3))  # a reflection
    data_S2 = np.zeros((results_S[s3wlon_180][0][2].size, 3))  # 2 reflections
    data_S3 = np.zeros((results_S[s3wlon_180][0][3].size, 3))  # 3 reflections
    data_S4 = np.zeros((results_S[s3wlon_180][0][4].size, 3))  # 4 reflections
    data_S5 = np.zeros((results_S[s3wlon_180][0][5].size, 3))  # 5 reflections
    data_S6 = np.zeros((results_S[s3wlon_180][0][6].size, 3))  # 6 reflections
    data_S7 = np.zeros((results_S[s3wlon_180][0][7].size, 3))  # 7 reflections
    data_S8 = np.zeros((results_S[s3wlon_180][0][8].size, 3))  # 8 reflections
    for j in range(3):
        data_N0[:, j] = results_N[s3wlon_180][j][0]
        data_N1[:, j] = results_N[s3wlon_180][j][1]
        data_N2[:, j] = results_N[s3wlon_180][j][2]
        data_N3[:, j] = results_N[s3wlon_180][j][3]
        data_N4[:, j] = results_N[s3wlon_180][j][4]
        data_N5[:, j] = results_N[s3wlon_180][j][5]
        data_N6[:, j] = results_N[s3wlon_180][j][6]
        data_N7[:, j] = results_N[s3wlon_180][j][7]
        data_N8[:, j] = results_N[s3wlon_180][j][8]
        data_S0[:, j] = results_S[s3wlon_180][j][0]
        data_S1[:, j] = results_S[s3wlon_180][j][1]
        data_S2[:, j] = results_S[s3wlon_180][j][2]
        data_S3[:, j] = results_S[s3wlon_180][j][3]
        data_S4[:, j] = results_S[s3wlon_180][j][4]
        data_S5[:, j] = results_S[s3wlon_180][j][5]
        data_S6[:, j] = results_S[s3wlon_180][j][6]
        data_S7[:, j] = results_S[s3wlon_180][j][7]
        data_S8[:, j] = results_S[s3wlon_180][j][8]

    # Equatorial lead angle array [deg]
    eq_N0_arr = data_N0[:, 0]*360.0/Psyn
    eq_N1_arr = data_N1[:, 0]*360.0/Psyn + eq_N0_arr[-1]
    eq_N2_arr = data_N2[:, 0]*360.0/Psyn + eq_N1_arr[-1]
    eq_N3_arr = data_N3[:, 0]*360.0/Psyn + eq_N2_arr[-1]
    eq_N4_arr = data_N4[:, 0]*360.0/Psyn + eq_N3_arr[-1]
    eq_N5_arr = data_N5[:, 0]*360.0/Psyn + eq_N4_arr[-1]
    eq_N6_arr = data_N6[:, 0]*360.0/Psyn + eq_N5_arr[-1]
    eq_N7_arr = data_N7[:, 0]*360.0/Psyn + eq_N6_arr[-1]
    eq_N8_arr = data_N8[:, 0]*360.0/Psyn + eq_N7_arr[-1]
    eq_S0_arr = data_S0[:, 0]*360.0/Psyn
    eq_S1_arr = data_S1[:, 0]*360.0/Psyn + eq_S0_arr[-1]
    eq_S2_arr = data_S2[:, 0]*360.0/Psyn + eq_S1_arr[-1]
    eq_S3_arr = data_S3[:, 0]*360.0/Psyn + eq_S2_arr[-1]
    eq_S4_arr = data_S4[:, 0]*360.0/Psyn + eq_S3_arr[-1]
    eq_S5_arr = data_S5[:, 0]*360.0/Psyn + eq_S4_arr[-1]
    eq_S6_arr = data_S6[:, 0]*360.0/Psyn + eq_S5_arr[-1]
    eq_S7_arr = data_S7[:, 0]*360.0/Psyn + eq_S6_arr[-1]
    eq_S8_arr = data_S8[:, 0]*360.0/Psyn + eq_S7_arr[-1]

    fig, ax = plt.subplots()
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(np.arange(0, 90+1, 10))
    ax.set_yticks(np.cos(np.pi*np.linspace(0, 3, 7)/3))
    ax.set_yticklabels(np.linspace(90, -90, 7))
    ax.grid(color=UC.lightgray, linewidth=0.5)
    ax.set_xlabel('Equatorial lead angle [deg]')
    ax.set_ylabel(r'S${\rm III}$ latitude [deg]')
    ax.plot(eq_N0_arr, np.cos(data_N0[:, 1]),
            color=UC.red)
    ax.plot(eq_N1_arr, np.cos(data_N1[:, 1]),
            color=UC.red)
    ax.plot(eq_N2_arr, np.cos(data_N2[:, 1]),
            color=UC.red)
    ax.plot(eq_N3_arr, np.cos(data_N3[:, 1]),
            color=UC.red)
    ax.plot(eq_N4_arr, np.cos(data_N4[:, 1]),
            color=UC.red)
    ax.plot(eq_N5_arr, np.cos(data_N5[:, 1]),
            color=UC.red)
    ax.plot(eq_N6_arr, np.cos(data_N6[:, 1]),
            color=UC.red)
    ax.plot(eq_N7_arr, np.cos(data_N7[:, 1]),
            color=UC.red)
    ax.plot(eq_N8_arr, np.cos(data_N8[:, 1]),
            color=UC.red)
    ax.plot(eq_S0_arr, np.cos(data_S0[:, 1]),
            color=UC.blue)
    ax.plot(eq_S1_arr, np.cos(data_S1[:, 1]),
            color=UC.blue)
    ax.plot(eq_S2_arr, np.cos(data_S2[:, 1]),
            color=UC.blue)
    ax.plot(eq_S3_arr, np.cos(data_S3[:, 1]),
            color=UC.blue)
    ax.plot(eq_S4_arr, np.cos(data_S4[:, 1]),
            color=UC.blue)
    ax.plot(eq_S5_arr, np.cos(data_S5[:, 1]),
            color=UC.blue)
    ax.plot(eq_S6_arr, np.cos(data_S6[:, 1]),
            color=UC.blue)
    ax.plot(eq_S7_arr, np.cos(data_S7[:, 1]),
            color=UC.blue)
    ax.plot(eq_S8_arr, np.cos(data_S8[:, 1]),
            color=UC.blue)
    fig.tight_layout()
    fig.savefig('img/reflect/'+exname+'/eqlead_vs_s3lat_'+alt_str+'.jpg')
    plt.close()

    del eq_N0_arr, eq_N1_arr, eq_N2_arr, eq_N3_arr, eq_N4_arr
    del eq_N5_arr, eq_N6_arr, eq_N7_arr, eq_N8_arr
    del eq_S0_arr, eq_S1_arr, eq_S2_arr, eq_S3_arr, eq_S4_arr
    del eq_S5_arr, eq_S6_arr, eq_S7_arr, eq_S8_arr

    # Save the auroral footprint data
    # j = 0: moon_s3_obs (at the time of the footprint observation, t = t_obs) [deg]
    # j = 1: S3 colatitude of N-MAW (t = t_obs)
    # j = 2: S3 w-longitude of N-MAW (t = t_obs)
    # j = 3: equatorial lead angle of N-MAW (t = t_obs)
    # j = 4: S3 colatitude of S-MAW (t = t_obs)
    # j = 5: S3 w-longitude of S-MAW (t = t_obs)
    # j = 6: equatorial lead angle of S-MAW (t = t_obs)
    # j = 4n+7: S3 colatitude of n-th RAW of the N-MAW (t = t_obs) (n = 1, 2, 3, ...)
    # j = 5n+7: S3 w-longitude of n-th RAW of the N-MAW (t = t_obs) (n = 1, 2, 3, ...)
    # j = 6n+7: equatorial lead angle of n-th RAW of N-MAW (t = t_obs) (n = 1, 2, 3, ...)

    # Equatorial lead angle for the footprint
    eq_N_fp = np.zeros((arr_size, 3+3*reflections))
    eq_S_fp = np.zeros((arr_size, 3+3*reflections))
    for i in range(arr_size):
        # Footprintまでの伝搬時間 [sec]
        tau_N0_fp = results_N[i][0][0][-1]
        tau_N1_fp = results_N[i][0][1][-1] + tau_N0_fp
        tau_N2_fp = results_N[i][0][2][-1] + tau_N1_fp
        tau_N3_fp = results_N[i][0][3][-1] + tau_N2_fp
        tau_N4_fp = results_N[i][0][4][-1] + tau_N3_fp
        tau_N5_fp = results_N[i][0][5][-1] + tau_N4_fp
        tau_N6_fp = results_N[i][0][6][-1] + tau_N5_fp
        tau_N7_fp = results_N[i][0][7][-1] + tau_N6_fp
        tau_N8_fp = results_N[i][0][8][-1] + tau_N7_fp

        # Footprintの赤道リード角を格納 [deg]
        eq_N_fp[i, 0] = tau_N0_fp*360.0/Psyn
        eq_N_fp[i, 3] = tau_N1_fp*360.0/Psyn
        eq_N_fp[i, 6] = tau_N2_fp*360.0/Psyn
        eq_N_fp[i, 9] = tau_N3_fp*360.0/Psyn
        eq_N_fp[i, 12] = tau_N4_fp*360.0/Psyn
        eq_N_fp[i, 15] = tau_N5_fp*360.0/Psyn
        eq_N_fp[i, 18] = tau_N6_fp*360.0/Psyn
        eq_N_fp[i, 21] = tau_N7_fp*360.0/Psyn
        eq_N_fp[i, 24] = tau_N8_fp*360.0/Psyn

        # FootprintのSIII余緯度を格納 [rad]
        for j in range(1+reflections):
            eq_N_fp[i, 3*j+1] = results_N[i][1][j][-1]

        # FootprintのSIII西経を格納 [rad]
        for j in range(1+reflections):
            eq_N_fp[i, 3*j+2] = results_N[i][2][j][-1]

        # Footprintまでの伝搬時間 [sec]
        tau_S0_fp = results_S[i][0][0][-1]
        tau_S1_fp = results_S[i][0][1][-1] + tau_S0_fp
        tau_S2_fp = results_S[i][0][2][-1] + tau_S1_fp
        tau_S3_fp = results_S[i][0][3][-1] + tau_S2_fp
        tau_S4_fp = results_S[i][0][4][-1] + tau_S3_fp
        tau_S5_fp = results_S[i][0][5][-1] + tau_S4_fp
        tau_S6_fp = results_S[i][0][6][-1] + tau_S5_fp
        tau_S7_fp = results_S[i][0][7][-1] + tau_S6_fp
        tau_S8_fp = results_S[i][0][8][-1] + tau_S7_fp

        # Footprintの赤道リード角を格納 [deg]
        eq_S_fp[i, 0] = tau_S0_fp*360.0/Psyn
        eq_S_fp[i, 3] = tau_S1_fp*360.0/Psyn
        eq_S_fp[i, 6] = tau_S2_fp*360.0/Psyn
        eq_S_fp[i, 9] = tau_S3_fp*360.0/Psyn
        eq_S_fp[i, 12] = tau_S4_fp*360.0/Psyn
        eq_S_fp[i, 15] = tau_S5_fp*360.0/Psyn
        eq_S_fp[i, 18] = tau_S6_fp*360.0/Psyn
        eq_S_fp[i, 21] = tau_S7_fp*360.0/Psyn
        eq_S_fp[i, 24] = tau_S8_fp*360.0/Psyn

        # FootprintのSIII余緯度を格納 [rad]
        for j in range(1+reflections):
            eq_S_fp[i, 3*j+1] = results_S[i][1][j][-1]

        # FootprintのSIII西経を格納 [rad]
        for j in range(1+reflections):
            eq_S_fp[i, 3*j+2] = results_S[i][2][j][-1]

    # Interpolate & make array
    # j = 0: moon_s3_obs (at the time of the footprint observation) [deg]
    # j = 1, 7: equatorial lead angle for MAW at NORTH and SOUTH, respectively
    # j = 2, 8: equatorial lead angle for a reflection at SOUTH and NORTH, respectively
    # j = 3, 9: equatorial lead angle for 2 reflections
    # j = 4, 10: equatorial lead angle for 3 reflections
    # j = 5, 11: equatorial lead angle for 4 reflections
    # j = 6, 12: equatorial lead angle for 5 reflections
    moon_s3_obs = np.linspace(0.0, 360.0, 1500)  # every 0.24 deg
    data_fp_interp = np.zeros((moon_s3_obs.size, 1+3*(1+reflections)*2))
    data_fp_interp[:, 0] = moon_s3_obs
    for j in range(1+reflections):
        # Interp前のSIII経度軸 [deg]
        eq_wlon_arr = np.degrees(s3wlon_t0_arr)+eq_N_fp[:, 3*j]

        # SIII経度の周期性で線形補間がバグるのでデカルト座標系で線形補間を行う
        # eq_N_fp[:, 3*j+2]は西経なので注意
        fp_x = np.cos(2*np.pi-eq_N_fp[:, 3*j+2])
        fp_y = np.sin(2*np.pi-eq_N_fp[:, 3*j+2])
        fp_x_interp = np.interp(moon_s3_obs,
                                eq_wlon_arr,
                                fp_x,
                                period=360.0)
        fp_y_interp = np.interp(moon_s3_obs,
                                eq_wlon_arr,
                                fp_y,
                                period=360.0)

        # FootprintのSIII余緯度 [rad]
        data_fp_interp[:, 3*j+1] = np.interp(moon_s3_obs,
                                             eq_wlon_arr,
                                             eq_N_fp[:, 3*j+1],
                                             period=360.0)

        # FootprintのSIII西経 [rad]
        # data_fp_interp[:, 3*j+2] = np.interp(moon_s3_obs,
        #                                      eq_wlon_arr,
        #                                      eq_N_fp[:, 3*j+2])
        data_fp_interp[:, 3*j+2] = 2*np.pi-np.arctan2(fp_y_interp, fp_x_interp)

        # Footprintの赤道リード角 [deg]
        data_fp_interp[:, 3*j+3] = np.interp(moon_s3_obs,
                                             eq_wlon_arr,
                                             eq_N_fp[:, 3*j],
                                             period=360.0)

        # 反対半球
        jj = j+(1+reflections)

        # Interp前のSIII経度軸 [deg]
        eq_wlon_arr = np.degrees(s3wlon_t0_arr)+eq_S_fp[:, 3*j]

        # SIII経度の周期性で線形補間がバグるのでデカルト座標系で線形補間を行う
        fp_x = np.cos(2*np.pi-eq_S_fp[:, 3*j+2])
        fp_y = np.sin(2*np.pi-eq_S_fp[:, 3*j+2])
        fp_x_interp = np.interp(moon_s3_obs,
                                eq_wlon_arr,
                                fp_x,
                                period=360.0)
        fp_y_interp = np.interp(moon_s3_obs,
                                eq_wlon_arr,
                                fp_y,
                                period=360.0)

        # FootprintのSIII余緯度 [rad]
        data_fp_interp[:, 3*jj+1] = np.interp(moon_s3_obs,
                                              eq_wlon_arr,
                                              eq_S_fp[:, 3*j+1],
                                              period=360.0)

        # FootprintのSIII西経 [rad]
        # data_fp_interp[:, 3*jj+2] = np.interp(moon_s3_obs,
        #                                       eq_wlon_arr,
        #                                       eq_S_fp[:, 3*j+2],
        #                                       period=360.0)
        data_fp_interp[:, 3*jj+2] = 2*np.pi - \
            np.arctan2(fp_y_interp, fp_x_interp)

        # Footprintの赤道リード角 [deg]
        data_fp_interp[:, 3*jj+3] = np.interp(moon_s3_obs,
                                              eq_wlon_arr,
                                              eq_S_fp[:, 3*j],
                                              period=360.0)

    # Lead angle plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xticks(np.arange(0, 360+1, 45))
    ax.set_yticks(np.arange(0, 100+1, 10))
    ax.grid(color=UC.lightgray, linewidth=0.5)
    ax.set_xlabel('Io SIII longitude [deg]')
    ax.set_ylabel('Equatorial lead angle [deg]')
    for j in range(1+reflections):
        # N-MAWとそのRAWs
        ax.plot(moon_s3_obs, data_fp_interp[:, 3*j+3],
                color=UC.red)
        # S-MAWとそのRAWs
        jj = j+(1+reflections)
        ax.plot(moon_s3_obs, data_fp_interp[:, 3*jj+3],
                color=UC.blue)

    fig.tight_layout()
    fig.savefig('img/reflect/'+exname+'/moons3wlon_vs_eqlead_'+alt_str+'.jpg')
    plt.close()

    # Save the data
    np.savetxt('results/reflect/'+exname+'/data_N0_arr_'+alt_str+'.txt',
               data_N0)
    np.savetxt('results/reflect/'+exname+'/data_N1_arr_'+alt_str+'.txt',
               data_N1)
    np.savetxt('results/reflect/'+exname+'/data_N2_arr_'+alt_str+'.txt',
               data_N2)
    np.savetxt('results/reflect/'+exname+'/data_N3_arr_'+alt_str+'.txt',
               data_N3)
    np.savetxt('results/reflect/'+exname+'/data_N4_arr_'+alt_str+'.txt',
               data_N4)
    np.savetxt('results/reflect/'+exname+'/data_N5_arr_'+alt_str+'.txt',
               data_N5)
    np.savetxt('results/reflect/'+exname+'/data_N6_arr_'+alt_str+'.txt',
               data_N6)
    np.savetxt('results/reflect/'+exname+'/data_N7_arr_'+alt_str+'.txt',
               data_N7)
    np.savetxt('results/reflect/'+exname+'/data_N8_arr_'+alt_str+'.txt',
               data_N8)
    np.savetxt('results/reflect/'+exname+'/data_S0_arr_'+alt_str+'.txt',
               data_S0)
    np.savetxt('results/reflect/'+exname+'/data_S1_arr_'+alt_str+'.txt',
               data_S1)
    np.savetxt('results/reflect/'+exname+'/data_S2_arr_'+alt_str+'.txt',
               data_S2)
    np.savetxt('results/reflect/'+exname+'/data_S3_arr_'+alt_str+'.txt',
               data_S3)
    np.savetxt('results/reflect/'+exname+'/data_S4_arr_'+alt_str+'.txt',
               data_S4)
    np.savetxt('results/reflect/'+exname+'/data_S5_arr_'+alt_str+'.txt',
               data_S5)
    np.savetxt('results/reflect/'+exname+'/data_S6_arr_'+alt_str+'.txt',
               data_S6)
    np.savetxt('results/reflect/'+exname+'/data_S7_arr_'+alt_str+'.txt',
               data_S7)
    np.savetxt('results/reflect/'+exname+'/data_S8_arr_'+alt_str+'.txt',
               data_S8)
    np.savetxt('results/reflect/'+exname+'/data_fp_interp_'+alt_str+'.txt',
               data_fp_interp)
    return None


# %% EXECUTE
if __name__ == '__main__':
    exname = '003/20250516_054'
    TARGET_MOON = 'Io'
    target_fp = ['MAW', 'TEB']
    PJ_num = [9]
    hem = 'N'
    Ai_num = 3
    ni_num = 50
    Ti_num = 60
    Zi = 1.3                # Io: 1.3 / Eu: 1.4 / Ga: 1.3
    Te = 6.0                # Io: 6.0 [eV]/ Eu: 20.0 / Ga: 300.0
    reflections = 8         # fixed at 8
    altitude = 400          # default 900 [km]

    # Number of parallel processes
    parallel = 9

    # Grid
    d_phi = 0.6    # [deg]

    # PJ et
    utc = JUNO_PJ_TIMES[PJ_num[0]]
    et_fp = spice.utc2et(utc)

    # Target select
    if TARGET_MOON == 'Io':
        Psyn = Psyn_io
        r_moon = 5.9*RJ
    elif TARGET_MOON == 'Europa':
        Psyn = Psyn_eu
        r_moon = 9.4*RJ
    elif TARGET_MOON == 'Ganymede':
        Psyn = Psyn_ga
        r_moon = 15.0*RJ

    # Orbital distance at the PJ time
    _, _, _, r_moon_obs, _, _, _ = moonS3wlon_arr(np.array([et_fp]),
                                                  TARGET_MOON)
    r_moon = r_moon_obs[0]
    print('Orbital distance [RJ]:', r_moon/RJ)

    alt_str = str(int(altitude))
    main()
