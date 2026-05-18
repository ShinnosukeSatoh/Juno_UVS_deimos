import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mplcolors
import matplotlib.ticker as ptick
import matplotlib.colorbar as mplcolorbar
import matplotlib.cm as mplcm
from matplotlib.ticker import AutoMinorLocator
import math
import datetime
import Leadangle_wave as Wave
from Leadangle_fit_JunoUVS import eqwlong_err
from Leadangle_fit_JunoUVS import TEB_transit
from Leadangle_fit_JunoUVS import create_argmesh
from column_mass import calc as column_calc
from UniversalColor import UniversalColor
from SharedX import ShareXaxis
from legend_shadow import legend_shadow

import spiceypy as spice
import JupiterMag as jm
from scipy.io import readsav

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


# %% Import the best-fit parameters (Ai, ni, Ti)
def load_best_fit():
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


# %% Import the footprint positions based on the best-fit parameters (Ai, ni, Ti)
def fp_traced(target_moon_s3_obs):
    """
    Args:
        target_moon_s3_obs (float): moon position at the time of the footprint observation [deg]

    Returns:
        _type_: _description_
    """
    interp = np.loadtxt('results/reflect/'+exname+'/data_fp_interp.txt')
    moon_s3_obs = interp[:, 0]
    idx = np.argmin(abs(moon_s3_obs-target_moon_s3_obs))

    positions = interp[idx, :]
    return positions


# %% Polar plot
def polar_plot(fp_traced_arr):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    for j in range((1+reflections)*2):
        colat = fp_traced_arr[3*j+1]    # [rad]
        wlon = fp_traced_arr[3*j+2]     # [rad]
        if 90.0-np.degrees(colat) >= 0:
            ax.scatter(
                np.sin(colat)*np.cos(2*np.pi-wlon),
                np.sin(colat)*np.sin(2*np.pi-wlon),
                c=UC.red,
            )
        else:
            ax.scatter(
                np.sin(colat)*np.cos(2*np.pi-wlon),
                np.sin(colat)*np.sin(2*np.pi-wlon),
                c=UC.blue,
            )
        print(90.0-np.degrees(colat), np.degrees(wlon))

    fig.tight_layout()
    fig.savefig('img/test_polar.jpg')
    plt.close()

    return 0


# %% the main function
def main():
    wlon_fp, err_wlon_fp, lat_fp, err_lat_fp, _, et_fp, hem_fp, _ = Obsresults(
        PJ_LIST, TARGET_MOON, TARGET_FP, TARGET_HEM, FLIP
    )

    _, _, _, _, _, _, moon_S3wlon0_arr = moonS3wlon_arr(et_fp, TARGET_MOON)

    fp_traced_arr = fp_traced(moon_S3wlon0_arr[1])      # [deg]

    polar_plot(fp_traced_arr)

    return None


# %% EXECUTE
if __name__ == '__main__':
    # Name of execution
    exname = '003/20250516_047'

    # Input about Juno observation
    TARGET_MOON = 'Io'
    TARGET_FP = ['MAW', 'TEB']
    PJ_LIST = [3]
    TARGET_HEM = 'both'
    FLIP = False            # ALWAYS FALSE! Flip the flag (TEB <-> MAW)
    Ai_num = 3
    ni_num = 150
    Ti_num = 1
    Zi = 1.3                # Io: 1.3 / Eu: 1.4 / Ga: 1.3
    Te = 300.0              # Io: 6.0 [eV]/ Eu: 20.0 / Ga: 300.0
    reflections = 6         # fixed at 6

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

    main()
