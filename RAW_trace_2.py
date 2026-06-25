""" RAW_trace_2.py

Created on Jun 19, 2026
@author: Shin Satoh

Description:
This code calculates the location of the reflective Alfvén wave spot(s)
by tracing the Alfvén waves from a Galilean satellite, all the way along
the magnetic field line defined by the magnetic field models, including
reflection(s) near the planet. The reflection altitude is set at 900 km.

Version
1.0.0 (May 7, 2026)
2.0.0 (Jun 19, 2026)

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
from Leadangle_fit_JunoUVS import TEB_transit
import Leadangle_wave as Wave

import time
# import os
from scipy.io import readsav
import JupiterMag as jm

jm.Internal.Config(Model='jrm33', CartesianIn=True,
                   CartesianOut=True, Degree=18)
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
    levels = {'1-sigma': 2.30,
              '2-sigma': 6.17,
              '3-sigma': 11.8}
    # Import the best-fit parameter
    chi2_1d = np.loadtxt('results/fit/'+exname+'/params_chi2.txt')
    Ai_1d = np.loadtxt('results/fit/'+exname+'/params_Ai.txt')
    ni_1d = np.loadtxt('results/fit/'+exname+'/params_ni.txt')
    Ti_1d = np.loadtxt('results/fit/'+exname+'/params_Ti.txt')
    H_1d = np.loadtxt('results/fit/'+exname+'/params_H.txt')
    eqlead_est = np.loadtxt('results/fit/'+exname+'/eqlead_est.txt')

    chi2_3d = chi2_1d.reshape(ni_num, Ai_num, Ti_num)
    H_3d = H_1d.reshape(ni_num, Ai_num, Ti_num)
    Ai_3d = Ai_1d.reshape(ni_num, Ai_num, Ti_num)
    ni_3d = ni_1d.reshape(ni_num, Ai_num, Ti_num)
    Ti_3d = Ti_1d.reshape(ni_num, Ai_num, Ti_num)

    # 保存されているカイ2乗値は自由度で割ってしまっているのでここで元に戻す
    chi2_3d = chi2_3d*(eqlead_est.shape[0]-3)

    # chi2_3dの最小値を探す
    min_idx = np.where(chi2_3d == np.min(chi2_3d))

    # 信頼区間の端を取得する
    d_chi2 = chi2_3d[:, :, :]-chi2_3d[min_idx]
    idx_1sigma = np.where(d_chi2 < levels['1-sigma'])
    Ai_3d_1 = Ai_3d[idx_1sigma]
    ni_3d_1 = ni_3d[idx_1sigma]
    Ti_3d_1 = Ti_3d[idx_1sigma]
    H_3d_1 = H_3d[idx_1sigma]

    # 低密高温の場合
    if retrieval == 'hot':
        idx_hot = np.argmax(Ti_3d_1)
        Ai_best = Ai_3d_1[idx_hot]
        ni_best = ni_3d_1[idx_hot]
        Ti_best = Ti_3d_1[idx_hot]
        Hp_best = H_3d_1[idx_hot]
    # 高密低温の場合
    elif retrieval == 'dense':
        idx_dense = np.argmax(ni_3d_1)
        Ai_best = Ai_3d_1[idx_dense]
        ni_best = ni_3d_1[idx_dense]
        Ti_best = Ti_3d_1[idx_dense]
        Hp_best = H_3d_1[idx_dense]
    # best-fit parameters
    elif retrieval == 'best':
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
def calc_copy(Ai, ni, Hp, r_t0, s3wlon_t0, z_t0, s_t0, hem, reflect_altitude):
    """
    Return:
        tau_list ... time [sec]
        theta_s3_list ... SIII colatitude [rad]
        s3wlon_list ... SIII west longitude [rad]
    """
    # Initialize the result list/array
    tau_list = []

    # Moon -> N-MAW
    tau_N, rs_N, s3wlon_N, theta_s3_N, s_N, alt_flag_N = Wave.Awave().trace3_reflect(r_t0,
                                                                                     s3wlon_t0,
                                                                                     z_t0,
                                                                                     s_t0,
                                                                                     Ai,
                                                                                     ni,
                                                                                     Hp,
                                                                                     -1,)

    # Moon -> S-MAW
    tau_S, rs_S, s3wlon_S, theta_s3_S, s_S, alt_flag_S = Wave.Awave().trace3_reflect(r_t0,
                                                                                     s3wlon_t0,
                                                                                     z_t0,
                                                                                     s_t0,
                                                                                     Ai,
                                                                                     ni,
                                                                                     Hp,
                                                                                     1,)

    # N-MAW -> S-MAW
    tau_N2S = np.hstack((tau_N[-1]-tau_N[::-1], tau_S+tau_N[-1]))
    s3wlon_N2S = np.hstack((s3wlon_N[::-1], s3wlon_S))
    theta_s3_N2S = np.hstack((theta_s3_N[::-1], theta_s3_S))
    alt_flag_N2S = np.hstack((alt_flag_N[::-1], alt_flag_S))

    # S-MAW ->N-MAW
    tau_S2N = np.hstack((tau_S[-1]-tau_S[::-1], tau_N+tau_S[-1]))
    s3wlon_S2N = s3wlon_N2S[::-1]
    theta_s3_S2N = theta_s3_N2S[::-1]
    alt_flag_S2N = alt_flag_N2S[::-1]

    # =======================================
    # 保存用arrayを作成 (Moon -> N-MAW)
    # =======================================
    skip = 50
    non_0 = np.array(np.where(alt_flag_N != 0)[0])
    # print('alt_flag_N[non_0]:', alt_flag_N[non_0][0],
    #       alt_flag_N[non_0][fp_alt_target])
    tau_N_skip = tau_N[0:non_0[0]][::skip]
    s3wlon_N_skip = s3wlon_N[0:non_0[0]][::skip]
    theta_s3_N_skip = theta_s3_N[0:non_0[0]][::skip]
    alt_flag_N_skip = alt_flag_N[0:non_0[0]][::skip]
    for i in range(int(non_0.size)):
        tau_N_skip = np.hstack((tau_N_skip,
                                tau_N[non_0[i]]))
        s3wlon_N_skip = np.hstack((s3wlon_N_skip,
                                   s3wlon_N[non_0[i]]))
        theta_s3_N_skip = np.hstack((theta_s3_N_skip,
                                     theta_s3_N[non_0[i]]))
        alt_flag_N_skip = np.hstack((alt_flag_N_skip,
                                     alt_flag_N[non_0[i]]))

    # =======================================
    # 保存用arrayを作成 (Moon -> S-MAW)
    # =======================================
    non_0 = np.array(np.where(alt_flag_S != 0)[0])
    tau_S_skip = tau_S[0:non_0[0]][::skip]
    s3wlon_S_skip = s3wlon_S[0:non_0[0]][::skip]
    theta_s3_S_skip = theta_s3_S[0:non_0[0]][::skip]
    alt_flag_S_skip = alt_flag_S[0:non_0[0]][::skip]
    for i in range(int(non_0.size)):
        tau_S_skip = np.hstack((tau_S_skip,
                                tau_S[non_0[i]]))
        s3wlon_S_skip = np.hstack((s3wlon_S_skip,
                                   s3wlon_S[non_0[i]]))
        theta_s3_S_skip = np.hstack((theta_s3_S_skip,
                                     theta_s3_S[non_0[i]]))
        alt_flag_S_skip = np.hstack((alt_flag_S_skip,
                                     alt_flag_S[non_0[i]]))

    # =======================================
    # 保存用arrayを作成 (N -> S)
    # =======================================
    non_0 = np.array(np.where(alt_flag_N2S != 0)[0])
    tau_N2S_skip = tau_N2S[0]               # -> deleted later
    s3wlon_N2S_skip = s3wlon_N2S[0]         # -> deleted later
    theta_s3_N2S_skip = theta_s3_N2S[0]     # -> deleted later
    alt_flag_N2S_skip = alt_flag_N2S[0]     # -> deleted later
    for i in range(int(non_0.size/2)):
        tau_N2S_skip = np.hstack((tau_N2S_skip,
                                  tau_N2S[non_0[i]]))
        s3wlon_N2S_skip = np.hstack((s3wlon_N2S_skip,
                                     s3wlon_N2S[non_0[i]]))
        theta_s3_N2S_skip = np.hstack((theta_s3_N2S_skip,
                                       theta_s3_N2S[non_0[i]]))
        alt_flag_N2S_skip = np.hstack((alt_flag_N2S_skip,
                                       alt_flag_N2S[non_0[i]]))
    tau_N2S_skip = np.hstack((tau_N2S_skip,
                              tau_N2S[non_0[i]+1:non_0[i+1]][::skip]))
    s3wlon_N2S_skip = np.hstack((s3wlon_N2S_skip,
                                 s3wlon_N2S[non_0[i]+1:non_0[i+1]][::skip]))
    theta_s3_N2S_skip = np.hstack((theta_s3_N2S_skip,
                                   theta_s3_N2S[non_0[i]+1:non_0[i+1]][::skip]))
    alt_flag_N2S_skip = np.hstack((alt_flag_N2S_skip,
                                   alt_flag_N2S[non_0[i]+1:non_0[i+1]][::skip]))
    for i in range(int(non_0.size/2)):
        i += int(non_0.size/2)
        tau_N2S_skip = np.hstack((tau_N2S_skip,
                                  tau_N2S[non_0[i]]))
        s3wlon_N2S_skip = np.hstack((s3wlon_N2S_skip,
                                     s3wlon_N2S[non_0[i]]))
        theta_s3_N2S_skip = np.hstack((theta_s3_N2S_skip,
                                       theta_s3_N2S[non_0[i]]))
        alt_flag_N2S_skip = np.hstack((alt_flag_N2S_skip,
                                       alt_flag_N2S[non_0[i]]))
    tau_N2S_skip = tau_N2S_skip[1:]
    s3wlon_N2S_skip = s3wlon_N2S_skip[1:]
    theta_s3_N2S_skip = theta_s3_N2S_skip[1:]
    alt_flag_N2S_skip = alt_flag_N2S_skip[1:]

    # =======================================
    # 保存用arrayを作成 (S -> N)
    # =======================================
    tau_S2N_skip = tau_N2S_skip[-1]-tau_N2S_skip[::-1]
    s3wlon_S2N_skip = s3wlon_N2S_skip[::-1]
    theta_s3_S2N_skip = theta_s3_N2S_skip[::-1]
    alt_flag_S2N_skip = alt_flag_N2S_skip[::-1]

    # =======================================
    # アイデア
    # =======================================
    #
    # N -> S と S -> N の 伝播時間arrayを作成
    # 高度のピンも打ってある
    # 反射高度はプロット作成時など後で決めることができるようになった
    # 1st TEBの位置もデータに組み込みたい

    N_MAW_list = [tau_N_skip, s3wlon_N_skip, theta_s3_N_skip, alt_flag_N_skip]
    S_MAW_list = [tau_S_skip, s3wlon_S_skip, theta_s3_S_skip, alt_flag_S_skip]
    N2S_list = [tau_N2S_skip, s3wlon_N2S_skip,
                theta_s3_N2S_skip, alt_flag_N2S_skip]
    S2N_list = [tau_S2N_skip, s3wlon_S2N_skip,
                theta_s3_S2N_skip, alt_flag_S2N_skip]

    return N_MAW_list, S_MAW_list, N2S_list, S2N_list


# %% Select one SIII wlongitude and extract the data
def select_wlon(results_list, s3wlon_target: int,
                reflect_alt_target: int, fp_alt_target: int,
                transit_time: float):
    # >> results_list[s3wlon_180][reflection_number][output_variation][tau_step]

    # =======================================
    # tau
    # =======================================
    tau_N_MAW = results_list[s3wlon_target][0][0][:]
    tau_S_MAW = results_list[s3wlon_target][1][0][:]
    tau_N2S = results_list[s3wlon_target][2][0][:]
    tau_S2N = results_list[s3wlon_target][3][0][:]

    # =======================================
    # a dummy array for TEB data
    # =======================================
    dummy_arr = np.ones(tau_N_MAW[reflect_alt_target:fp_alt_target+1].size)

    # =======================================
    # SIII wlong as a function of tau
    # =======================================
    s3wlon_N_MAW = results_list[s3wlon_target][0][1][:]
    s3wlon_S_MAW = results_list[s3wlon_target][1][1][:]
    s3wlon_N2S = results_list[s3wlon_target][2][1][:]
    s3wlon_S2N = results_list[s3wlon_target][3][1][:]
    s3wlon_N_0 = s3wlon_N_MAW[:fp_alt_target+1]
    s3wlon_N_1 = s3wlon_N2S[-reflect_alt_target:fp_alt_target+1]
    s3wlon_N_2 = s3wlon_S2N[-reflect_alt_target:fp_alt_target+1]
    s3wlon_N_3 = s3wlon_N2S[-reflect_alt_target:fp_alt_target+1]
    s3wlon_N_4 = s3wlon_S2N[-reflect_alt_target:fp_alt_target+1]
    s3wlon_N_5 = s3wlon_N2S[-reflect_alt_target:fp_alt_target+1]
    s3wlon_N_6 = s3wlon_S2N[-reflect_alt_target:fp_alt_target+1]
    s3wlon_N_7 = s3wlon_N2S[-reflect_alt_target:fp_alt_target+1]
    s3wlon_N_8 = s3wlon_S2N[-reflect_alt_target:fp_alt_target+1]
    s3wlon_S_0 = s3wlon_S_MAW[:fp_alt_target+1]
    s3wlon_S_1 = s3wlon_S2N[-reflect_alt_target:fp_alt_target+1]
    s3wlon_S_2 = s3wlon_N2S[-reflect_alt_target:fp_alt_target+1]
    s3wlon_S_3 = s3wlon_S2N[-reflect_alt_target:fp_alt_target+1]
    s3wlon_S_4 = s3wlon_N2S[-reflect_alt_target:fp_alt_target+1]
    s3wlon_S_5 = s3wlon_S2N[-reflect_alt_target:fp_alt_target+1]
    s3wlon_S_6 = s3wlon_N2S[-reflect_alt_target:fp_alt_target+1]
    s3wlon_S_7 = s3wlon_S2N[-reflect_alt_target:fp_alt_target+1]
    s3wlon_S_8 = s3wlon_N2S[-reflect_alt_target:fp_alt_target+1]
    s3wlon_N_TEB_0 = np.hstack((s3wlon_N_MAW[:reflect_alt_target+1],
                                s3wlon_N2S[reflect_alt_target:fp_alt_target]))
    s3wlon_N_TEB_1 = np.hstack((s3wlon_S2N[-reflect_alt_target],
                                s3wlon_S2N[reflect_alt_target:fp_alt_target+1]))
    s3wlon_S_TEB_0 = np.hstack((s3wlon_S_MAW[:reflect_alt_target+1],
                                s3wlon_S2N[reflect_alt_target:fp_alt_target]))
    s3wlon_S_TEB_1 = np.hstack((s3wlon_N2S[-reflect_alt_target],
                                s3wlon_N2S[reflect_alt_target:fp_alt_target+1]))
    if fp_alt_target+1 == 0:
        print('s3wlon_N_0.shape:', s3wlon_N_0.shape)
    # print('s3wlon_N_TEB.shape:', s3wlon_N_TEB_0.shape)
    # print('s3wlon_S_TEB.shape:', s3wlon_S_TEB_0.shape)

    # =======================================
    # colatitude as a function of tau
    # =======================================
    theta_N_MAW = results_list[s3wlon_target][0][2][:]
    theta_S_MAW = results_list[s3wlon_target][1][2][:]
    theta_N2S = results_list[s3wlon_target][2][2][:]
    theta_S2N = results_list[s3wlon_target][3][2][:]
    theta_N_0 = theta_N_MAW[:fp_alt_target+1]
    theta_N_1 = theta_N2S[-reflect_alt_target:fp_alt_target+1]
    theta_N_2 = theta_S2N[-reflect_alt_target:fp_alt_target+1]
    theta_N_3 = theta_N2S[-reflect_alt_target:fp_alt_target+1]
    theta_N_4 = theta_S2N[-reflect_alt_target:fp_alt_target+1]
    theta_N_5 = theta_N2S[-reflect_alt_target:fp_alt_target+1]
    theta_N_6 = theta_S2N[-reflect_alt_target:fp_alt_target+1]
    theta_N_7 = theta_N2S[-reflect_alt_target:fp_alt_target+1]
    theta_N_8 = theta_S2N[-reflect_alt_target:fp_alt_target+1]
    theta_S_0 = theta_S_MAW[:fp_alt_target+1]
    theta_S_1 = theta_S2N[-reflect_alt_target:fp_alt_target+1]
    theta_S_2 = theta_N2S[-reflect_alt_target:fp_alt_target+1]
    theta_S_3 = theta_S2N[-reflect_alt_target:fp_alt_target+1]
    theta_S_4 = theta_N2S[-reflect_alt_target:fp_alt_target+1]
    theta_S_5 = theta_S2N[-reflect_alt_target:fp_alt_target+1]
    theta_S_6 = theta_N2S[-reflect_alt_target:fp_alt_target+1]
    theta_S_7 = theta_S2N[-reflect_alt_target:fp_alt_target+1]
    theta_S_8 = theta_N2S[-reflect_alt_target:fp_alt_target+1]
    theta_N_TEB_0 = np.hstack((theta_N_MAW[:reflect_alt_target+1],
                               theta_N2S[reflect_alt_target:fp_alt_target+1]))
    theta_N_TEB_1 = np.hstack((theta_S2N[-reflect_alt_target],
                               theta_S2N[reflect_alt_target:fp_alt_target+1]))
    theta_S_TEB_0 = np.hstack((theta_S_MAW[:reflect_alt_target+1],
                               theta_S2N[reflect_alt_target:fp_alt_target+1]))
    theta_S_TEB_1 = np.hstack((theta_N2S[-reflect_alt_target],
                               theta_N2S[reflect_alt_target:fp_alt_target+1]))

    # ==========================================
    # Equatorial lead angle as a function of tau
    # ==========================================
    tau_hR_N_hR = (tau_N_MAW[-1]-tau_N_MAW[reflect_alt_target])*2
    tau_hR_S_hR = (tau_S_MAW[-1]-tau_S_MAW[reflect_alt_target])*2
    eqlead_N_0 = tau_N_MAW[:fp_alt_target+1]*360.0/Psyn
    eqlead_S_0 = tau_S_MAW[:fp_alt_target+1]*360.0/Psyn
    eqlead_N_1 = (tau_N_MAW[-1] - tau_hR_N_hR
                  + tau_N2S[-reflect_alt_target:fp_alt_target+1])*360.0/Psyn
    # print('tau_N2S[-reflect_alt_target:fp_alt_target]:',
    #       tau_N2S[-reflect_alt_target:fp_alt_target])
    # print('eqlead_N_1:', eqlead_N_1)
    # print('tau_hR_N_hR:', tau_hR_N_hR)
    # print('tau_S_MAW:', tau_S_MAW)
    eqlead_N_2 = (tau_N_MAW[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-reflect_alt_target:fp_alt_target+1])*360.0/Psyn
    eqlead_N_3 = (tau_N_MAW[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-reflect_alt_target:fp_alt_target+1])*360.0/Psyn
    eqlead_N_4 = (tau_N_MAW[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-reflect_alt_target:fp_alt_target+1])*360.0/Psyn
    eqlead_N_5 = (tau_N_MAW[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-reflect_alt_target:fp_alt_target+1])*360.0/Psyn
    eqlead_N_6 = (tau_N_MAW[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-reflect_alt_target:fp_alt_target+1])*360.0/Psyn
    eqlead_N_7 = (tau_N_MAW[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-reflect_alt_target:fp_alt_target+1])*360.0/Psyn
    eqlead_N_8 = (tau_N_MAW[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-reflect_alt_target:fp_alt_target+1])*360.0/Psyn
    eqlead_S_1 = (tau_S_MAW[-1] - tau_hR_S_hR
                  + tau_S2N[-reflect_alt_target:fp_alt_target+1])*360.0/Psyn
    eqlead_S_2 = (tau_S_MAW[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-reflect_alt_target:fp_alt_target+1])*360.0/Psyn
    eqlead_S_3 = (tau_S_MAW[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-reflect_alt_target:fp_alt_target+1])*360.0/Psyn
    eqlead_S_4 = (tau_S_MAW[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-reflect_alt_target:fp_alt_target+1])*360.0/Psyn
    eqlead_S_5 = (tau_S_MAW[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-reflect_alt_target:fp_alt_target+1])*360.0/Psyn
    eqlead_S_6 = (tau_S_MAW[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-reflect_alt_target:fp_alt_target+1])*360.0/Psyn
    eqlead_S_7 = (tau_S_MAW[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-reflect_alt_target:fp_alt_target+1])*360.0/Psyn
    eqlead_S_8 = (tau_S_MAW[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-1] - tau_hR_S_hR
                  + tau_S2N[-1] - tau_hR_N_hR
                  + tau_N2S[-reflect_alt_target:fp_alt_target+1])*360.0/Psyn
    eqlead_N_TEB_0 = np.hstack((tau_N_MAW[:reflect_alt_target+1],
                                (tau_N_MAW[reflect_alt_target]+transit_time)*dummy_arr))*360.0/Psyn
    eqlead_N_TEB_1 = (np.hstack((tau_N2S[reflect_alt_target],
                                 (tau_N2S[reflect_alt_target]+transit_time)*dummy_arr))
                      + tau_N_MAW[-1] - tau_hR_N_hR)*360.0/Psyn
    eqlead_S_TEB_0 = np.hstack((tau_S_MAW[:reflect_alt_target+1],
                                (tau_S_MAW[reflect_alt_target]+transit_time)*dummy_arr))*360.0/Psyn
    eqlead_S_TEB_1 = (np.hstack((tau_S2N[reflect_alt_target],
                                 (tau_S2N[reflect_alt_target]+transit_time)*dummy_arr))
                      + tau_S_MAW[-1] - tau_hR_S_hR)*360.0/Psyn
    # print('eqlead_N_TEB_0.shape:', eqlead_N_TEB_0.shape)
    # print('eqlead_S_TEB_0.shape:', eqlead_S_TEB_0.shape)
    # print('theta_N_TEB_1.shape:', theta_N_TEB_1.shape)
    # print('eqlead_N_TEB_1.shape:', eqlead_N_TEB_1.shape)

    theta_N_list = [theta_N_0, theta_N_1, theta_N_2,
                    theta_N_3, theta_N_4, theta_N_5,
                    theta_N_6, theta_N_7, theta_N_8,
                    theta_N_TEB_0, theta_N_TEB_1]
    theta_S_list = [theta_S_0, theta_S_1, theta_S_2,
                    theta_S_3, theta_S_4, theta_S_5,
                    theta_S_6, theta_S_7, theta_S_8,
                    theta_S_TEB_0, theta_S_TEB_1]
    eqlead_N_list = [eqlead_N_0, eqlead_N_1, eqlead_N_2,
                     eqlead_N_3, eqlead_N_4, eqlead_N_5,
                     eqlead_N_6, eqlead_N_7, eqlead_N_8,
                     eqlead_N_TEB_0, eqlead_N_TEB_1]
    eqlead_S_list = [eqlead_S_0, eqlead_S_1, eqlead_S_2,
                     eqlead_S_3, eqlead_S_4, eqlead_S_5,
                     eqlead_S_6, eqlead_S_7, eqlead_S_8,
                     eqlead_S_TEB_0, eqlead_S_TEB_1]
    s3wlon_N_list = [s3wlon_N_0, s3wlon_N_1, s3wlon_N_2,
                     s3wlon_N_3, s3wlon_N_4, s3wlon_N_5,
                     s3wlon_N_6, s3wlon_N_7, s3wlon_N_8,
                     s3wlon_N_TEB_0, s3wlon_N_TEB_1]
    s3wlon_S_list = [s3wlon_S_0, s3wlon_S_1, s3wlon_S_2,
                     s3wlon_S_3, s3wlon_S_4, s3wlon_S_5,
                     s3wlon_S_6, s3wlon_S_7, s3wlon_S_8,
                     s3wlon_S_TEB_0, s3wlon_S_TEB_1]
    return theta_N_list, theta_S_list, eqlead_N_list, eqlead_S_list, s3wlon_N_list, s3wlon_S_list


# %% Main function
def main():
    # the initial SIII w-longitude of the moon
    s3wlon_t0_arr = np.radians(np.arange(-95.0, 360.0+1.0, d_phi))
    s3wlon_target = np.argmin(
        abs(s3wlon_t0_arr-np.radians(s3wlon_moon_obs[0])))
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
                    alt_ref[reflect_alt_target]*np.ones(arr_size, dtype=int),))

    # Parallelized
    print('Trace North started.')
    time_start = time.time()
    with Pool(processes=parallel) as pool:
        results_list = list(pool.starmap(calc_copy, args))
    print('Number of CPU cores used:', parallel)
    print('Loop time [sec]:', round(time.time()-time_start, 2))

    # print('=== ALL ===')
    # print(results_list)
    # >> results_list[s3wlon_180][reflection_number][output_variation][tau_step]

    # TEB transit time [sec]
    transit_time = TEB_transit(r_moon,
                               s3wlon_t0_arr[s3wlon_target],
                               TARGET_MOON)

    theta_N_list, theta_S_list, eqlead_N_list, eqlead_S_list, _, _ = select_wlon(results_list,
                                                                                 s3wlon_target,
                                                                                 reflect_alt_target,
                                                                                 fp_alt_target,
                                                                                 transit_time)

    # =========================================================
    # Plot: Latitude as a function of the equatorial lead angle
    # =========================================================
    title = 'PJ'+str(PJ_num[0]).zfill(2)
    if hem != 'both':
        title += hem
    title += ' '+utc
    title += ' ('+r'$\lambda_{\rm III}^{\rm Io}=$'
    title += str(round(s3wlon_moon_obs[0], 2))
    title += '˚W)'
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.set_title(title)
    ax.set_xlim(-10.0, 90.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(np.arange(-10, 90+1, 10))
    ax.set_yticks(np.cos(np.pi*np.linspace(0, 3, 7)/3))
    ax.set_yticklabels(np.linspace(90, -90, 7))
    ax.grid(color=UC.lightgray, linewidth=0.5)
    ax.set_xlabel('Equatorial lead angle [deg]')
    ax.set_ylabel(r'S${\rm III}$ latitude [deg]')
    # MAWとRAW (solid lines)
    for i in range(1+reflections):
        ax.plot(eqlead_N_list[i], np.cos(theta_N_list[i]),
                linewidth=1.5, color=UC.red)
        ax.plot(eqlead_S_list[i], np.cos(theta_S_list[i]),
                linewidth=1.5, color=UC.blue)
    # TEB (dashed lines)
    for j in range(2):
        ax.plot(eqlead_N_list[i+1+j], np.cos(theta_N_list[i+1+j]),
                linewidth=1.5, color=UC.red, linestyle='--')
        ax.plot(eqlead_S_list[i+1+j], np.cos(theta_S_list[i+1+j]),
                linewidth=1.5, color=UC.blue, linestyle='--')
    fig.tight_layout()
    fig.savefig('img/reflect_2/'+exname+'/eqlead_vs_s3lat.jpg')
    plt.close()

    # ======================================================
    # Foorptint lead angle as a function of moon's SIII wlon
    # ======================================================
    for k in range(len(alt_ref)-2):
        # Target altitude index (be always negative)
        k += 2
        k = -k
        print('k:', k, '// Altitude [km]:', alt_ref[k])
        eq_N_fp = np.zeros((arr_size, 3*(3+reflections)))
        eq_S_fp = np.zeros((arr_size, 3*(3+reflections)))
        for i in range(arr_size):
            theta_N_list, theta_S_list, eqlead_N_list, eqlead_S_list, s3wlon_N_list, s3wlon_S_list = select_wlon(results_list,
                                                                                                                 i,
                                                                                                                 reflect_alt_target,
                                                                                                                 k,
                                                                                                                 transit_time)

            # Footprintの赤道リード角を格納 [deg]
            for j in range(3+reflections):
                # print('len(eqlead_N_list[j]):', len(eqlead_N_list[j]))
                eq_N_fp[i, 3*j] = eqlead_N_list[j][-1]
                eq_S_fp[i, 3*j] = eqlead_S_list[j][-1]

            # FootprintのSIII余緯度を格納 [rad]
            for j in range(3+reflections):
                eq_N_fp[i, 3*j+1] = theta_N_list[j][-1]
                eq_S_fp[i, 3*j+1] = theta_S_list[j][-1]

            # FootprintのSIII西経を格納 [rad]
            for j in range(3+reflections):
                eq_N_fp[i, 3*j+2] = s3wlon_N_list[j][-1]
                eq_S_fp[i, 3*j+2] = s3wlon_S_list[j][-1]

        # ======================================================
        # Equatorial lead angle interporation
        # ======================================================
        # j = 0: moon_s3_obs (at the time of the footprint observation) [deg]
        # j = 1, 9: equatorial lead angle for MAW at NORTH and SOUTH, respectively
        # j = 2, 10: equatorial lead angle for a reflection at SOUTH and NORTH, respectively
        # j = 3, 11: equatorial lead angle for 2 reflections
        # j = 4, 12: equatorial lead angle for 3 reflections
        # j = 5, 13: equatorial lead angle for 4 reflections
        # j = 6, 14: equatorial lead angle for 5 reflections
        # j = 7, 15: equatorial lead angle for TEB_0 at SOUTH and NORTH, respectively
        # j = 8, 16: equatorial lead angle for TEB_1 at SOUTH and NORTH, respectively
        new_moon_s3wlon = np.linspace(0.0, 360.0, 1500)  # every 0.24 deg
        data_fp_interp = np.zeros(
            (new_moon_s3wlon.size, 1+3*(3+reflections)*2))
        data_fp_interp[:, 0] = new_moon_s3wlon
        for j in range(3+reflections):
            # Interp前のSIII経度軸 [deg]
            eq_wlon_arr = np.degrees(s3wlon_t0_arr)+eq_N_fp[:, 3*j]

            # SIII経度の周期性で線形補間がバグるのでデカルト座標系で線形補間を行う
            # eq_N_fp[:, 3*j+2]は西経[rad]なので注意
            fp_x = np.cos(2*np.pi-eq_N_fp[:, 3*j+2])
            fp_y = np.sin(2*np.pi-eq_N_fp[:, 3*j+2])
            fp_x_interp = np.interp(new_moon_s3wlon,
                                    eq_wlon_arr,
                                    fp_x,
                                    period=360.0)
            fp_y_interp = np.interp(new_moon_s3wlon,
                                    eq_wlon_arr,
                                    fp_y,
                                    period=360.0)

            # FootprintのSIII余緯度 [rad]
            data_fp_interp[:, 3*j+1] = np.interp(new_moon_s3wlon,
                                                 eq_wlon_arr,
                                                 eq_N_fp[:, 3*j+1],
                                                 period=360.0)

            # FootprintのSIII西経 [rad]
            data_fp_interp[:, 3*j+2] = 2*np.pi - \
                np.arctan2(fp_y_interp, fp_x_interp)

            # Footprintの赤道リード角 [deg]
            data_fp_interp[:, 3*j+3] = np.interp(new_moon_s3wlon,
                                                 eq_wlon_arr,
                                                 eq_N_fp[:, 3*j],
                                                 period=360.0)

            # 反対半球
            jj = j+(3+reflections)

            # Interp前のSIII経度軸 [deg]
            eq_wlon_arr = np.degrees(s3wlon_t0_arr)+eq_S_fp[:, 3*j]

            # SIII経度の周期性で線形補間がバグるのでデカルト座標系で線形補間を行う
            fp_x = np.cos(2*np.pi-eq_S_fp[:, 3*j+2])
            fp_y = np.sin(2*np.pi-eq_S_fp[:, 3*j+2])
            fp_x_interp = np.interp(new_moon_s3wlon,
                                    eq_wlon_arr,
                                    fp_x,
                                    period=360.0)
            fp_y_interp = np.interp(new_moon_s3wlon,
                                    eq_wlon_arr,
                                    fp_y,
                                    period=360.0)

            # FootprintのSIII余緯度 [rad]
            data_fp_interp[:, 3*jj+1] = np.interp(new_moon_s3wlon,
                                                  eq_wlon_arr,
                                                  eq_S_fp[:, 3*j+1],
                                                  period=360.0)

            # FootprintのSIII西経 [rad]
            data_fp_interp[:, 3*jj+2] = 2*np.pi - \
                np.arctan2(fp_y_interp, fp_x_interp)

            # Footprintの赤道リード角 [deg]
            data_fp_interp[:, 3*jj+3] = np.interp(new_moon_s3wlon,
                                                  eq_wlon_arr,
                                                  eq_S_fp[:, 3*j],
                                                  period=360.0)

        np.savetxt('results/reflect_2/'+exname+'/data_'+TARGET_MOON[0]+'FP_interp_map_'+str(int(alt_ref[k]))+'km_'+retrieval+'.txt',
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
    alt_ref = [1500.0, 1200.0, 1000.0, 900.0,
               800.0, 700.0, 600.0, 500.0,
               400.0, 300.0, 200.0, 100.0,
               10.0, 5.0]
    reflect_alt_target = -len(alt_ref)  # ALWAYS NEGATIVE!!!
    fp_alt_target = -6                  # ALWAYS NEGATIVE!!!
    retrieval = 'best'      # 'best', 'hot', 'dense'

    # Number of parallel processes
    parallel = 10

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
    _, _, _, r_moon_obs, _, _, s3wlon_moon_obs = moonS3wlon_arr(np.array([et_fp]),
                                                                TARGET_MOON)
    r_moon = r_moon_obs[0]
    print('Orbital distance [RJ]:', r_moon/RJ)

    alt_str = str(int(alt_ref[reflect_alt_target]))
    print('Reflection altitude [km]:', alt_str)
    main()
