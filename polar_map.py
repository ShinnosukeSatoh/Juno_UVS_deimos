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


# %% Footprint position will be mapped on the equatorial plane
def ft_ref(hemisphere, MOON: str):
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

    return s3lat, s3wlon


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
    moon_s3_obs = interp[:, 0]      # [deg]
    idx = np.argmin(abs(moon_s3_obs-target_moon_s3_obs))

    positions = interp[idx, :]
    return positions


# %% Generate the footpath of MAW
def fp_path():
    interp = np.loadtxt('results/reflect/'+exname+'/data_fp_interp.txt')
    moon_s3_obs = interp[:, 0]

    # j=1: colatitude, j=2: w-longitude [rad]
    pos_N_MAW = interp[:, 1:3]
    pos_S_MAW = interp[:, 1+3*(1+reflections):3*(1+reflections)+3]

    pos_S_RAW1 = interp[:, 4:6]
    pos_N_RAW1 = interp[:, 4+3*(1+reflections):3*(1+reflections)+6]

    # x = np.sin(pos_N_MAW[:, 0])*np.cos(2*np.pi-pos_N_MAW[:, 1])
    # y = np.sin(pos_N_MAW[:, 0])*np.sin(2*np.pi-pos_N_MAW[:, 1])

    return moon_s3_obs, pos_N_MAW, pos_S_MAW, pos_N_RAW1, pos_S_RAW1


# %% Instantaneous footprint position
def instantaneous(target_moon_s3_obs):
    Ai_best, ni_best, _, Hp_best = load_best_fit()

    s3wlon_t0 = np.radians(target_moon_s3_obs)

    S_A0 = Wave.Awave().tracefield(r_moon,
                                   s3wlon_t0,
                                   0.0)

    # Initital trace
    # -> Instantaneous position at 900 km altitude
    hem = -1    # North
    _, rs_t1, s3wlon_t1, theta_s3_t1, _ = Wave.Awave().trace3_reflect(r_moon,
                                                                      s3wlon_t0,
                                                                      0.0,
                                                                      S_A0,
                                                                      Ai_best,
                                                                      ni_best,
                                                                      Hp_best,
                                                                      hem)

    insta_fp_pos_N = np.zeros(2)
    insta_fp_pos_N[0] = theta_s3_t1[-1]     # Colatitude [rad]
    insta_fp_pos_N[1] = s3wlon_t1[-1]       # West longitude [rad]

    # -> Instantaneous position at 900 km altitude
    hem = 1    # South
    _, rs_t1, s3wlon_t1, theta_s3_t1, _ = Wave.Awave().trace3_reflect(r_moon,
                                                                      s3wlon_t0,
                                                                      0.0,
                                                                      S_A0,
                                                                      Ai_best,
                                                                      ni_best,
                                                                      Hp_best,
                                                                      hem)

    insta_fp_pos_S = np.zeros(2)
    insta_fp_pos_S[0] = theta_s3_t1[-1]     # Colatitude [rad]
    insta_fp_pos_S[1] = s3wlon_t1[-1]       # West longitude [rad]

    return insta_fp_pos_N, insta_fp_pos_S


# %% Propagation plot
def propagation_plot():
    data_N0 = np.loadtxt('results/reflect/'+exname+'/data_N0_arr.txt')
    # data_N0[:,0] ... time [sec]
    # data_N0[:,1] ... SIII colatitude [rad]
    # data_N0[:,2] ... SIII west longitude [rad]
    data_N1 = np.loadtxt('results/reflect/'+exname+'/data_N1_arr.txt')
    data_N2 = np.loadtxt('results/reflect/'+exname+'/data_N2_arr.txt')
    data_N3 = np.loadtxt('results/reflect/'+exname+'/data_N3_arr.txt')
    data_N4 = np.loadtxt('results/reflect/'+exname+'/data_N4_arr.txt')
    data_N5 = np.loadtxt('results/reflect/'+exname+'/data_N5_arr.txt')
    data_N6 = np.loadtxt('results/reflect/'+exname+'/data_N6_arr.txt')
    data_N7 = np.loadtxt('results/reflect/'+exname+'/data_N7_arr.txt')
    data_N8 = np.loadtxt('results/reflect/'+exname+'/data_N8_arr.txt')
    data_S0 = np.loadtxt('results/reflect/'+exname+'/data_S0_arr.txt')
    data_S1 = np.loadtxt('results/reflect/'+exname+'/data_S1_arr.txt')
    data_S2 = np.loadtxt('results/reflect/'+exname+'/data_S2_arr.txt')
    data_S3 = np.loadtxt('results/reflect/'+exname+'/data_S3_arr.txt')
    data_S4 = np.loadtxt('results/reflect/'+exname+'/data_S4_arr.txt')
    data_S5 = np.loadtxt('results/reflect/'+exname+'/data_S5_arr.txt')
    data_S6 = np.loadtxt('results/reflect/'+exname+'/data_S6_arr.txt')
    data_S7 = np.loadtxt('results/reflect/'+exname+'/data_S7_arr.txt')
    data_S8 = np.loadtxt('results/reflect/'+exname+'/data_S8_arr.txt')
    print('data_N0.shape', data_N0.shape)

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

    eqlead_max = 90.0   # [deg]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0.0, eqlead_max)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(np.arange(0, eqlead_max+1, 10))
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

    title = 'PJ'+str(PJ_LIST[0]).zfill(2)
    if TARGET_HEM != 'both':
        title += TARGET_HEM
    title += ' ('+r'$\lambda_{\rm III}^{\rm Io}=$'
    title += str(round(np.mod(math.degrees(data_N0[0, 2]), 360.0), 1))
    title += '˚W)'

    ax2 = ax.twiny()
    eqlead_max = eqlead_max
    tau_max = eqlead_max/(360.0/Psyn)   # [sec]
    print('tau_max [min]:', tau_max/60.0)
    ax2.set_title(title)
    ax2.set_xlabel('Time [min]')
    ax2.set_xlim(0, tau_max/60.0)
    ax2.set_xticks(np.arange(0, 180+1, 20))
    ax2.set_xticklabels(np.arange(0, 180+1, 20))

    fig.tight_layout()
    fig.savefig('img/reflect/'+exname+'/eqlead_time_vs_s3lat.jpg')
    plt.close()

    return None


# %% Lead angle plot
def leadangle_plot():
    interp = np.loadtxt('results/reflect/'+exname+'/data_fp_interp.txt')
    moon_s3_obs = interp[:, 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xticks(np.arange(0, 360+1, 45))
    ax.set_yticks(np.arange(0, 100+1, 10))
    ax.grid(color=UC.lightgray, linewidth=0.5)
    ax.set_xlabel('Io SIII longitude [deg]')
    ax.set_ylabel('Equatorial lead angle [deg]')

    # j=1: colatitude [rad], j=2: w-longitude [rad]
    # j=3: equatorial lead angle [rad]
    pos_N_MAW = interp[:, 3]
    pos_S_MAW = interp[:, 3*(1+reflections)+3]
    ax.plot(moon_s3_obs, pos_N_MAW, color=UC.red)
    ax.plot(moon_s3_obs, pos_S_MAW, color=UC.blue, linestyle='--')

    # Reflections
    colors = [UC.red, UC.blue, UC.red]
    for i in range(reflections):
        pos_N_RAW = interp[:, 3*(i+1)+3]
        pos_S_RAW = interp[:, 3*(i+1+reflections+1)+3]
        ax.plot(moon_s3_obs, pos_N_RAW,
                color=colors[i % 2+1])
        ax.plot(moon_s3_obs, pos_S_RAW,
                color=colors[i % 2], linestyle='--')

    fig.tight_layout()
    fig.savefig('img/reflect/'+exname+'/moons3wlon_vs_eqlead2.jpg')
    plt.close()
    return None


# %% Polar plot
def polar_plot(fp_traced_arr,
               target_moon_s3_obs,
               target_wlon_fp,
               target_err_wlon_fp,
               target_lat_fp,
               target_err_lat_fp,
               target_et_fp):
    savename = 'polar_NS'

    F = ShareXaxis()
    F.fontsize = 23
    F.fontname = 'Liberation Sans Narrow'

    title = 'PJ'+str(PJ_LIST[0]).zfill(2)
    if TARGET_HEM != 'both':
        title += TARGET_HEM
    target_UTC = spice.et2utc(et=target_et_fp, format_str='C', prec=0)
    print(target_UTC)
    title += ' '+target_UTC
    title += ' ('+r'$\lambda_{\rm III}^{\rm Io}=$'
    title += str(round(target_moon_s3_obs, 2))
    title += '˚W)'

    F.set_figparams(nrows=1, figsize=(9, 9),
                    ticksize=1.5, dpi='L')
    F.initialize()

    F.ax.set_title(title)
    F.set_xaxis(label=r'$X$ [$R_{\rm J}$]',
                min=-0.82, max=0.82,
                ticks=np.linspace(-8, 8, 9)/10,
                ticklabels=np.linspace(-8, 8, 9)/10,
                minor_num=2)
    F.set_yaxis(ax_idx=0, label=r'$Y$ [$R_{\rm J}$]',
                min=-0.82, max=0.82,
                ticks=np.linspace(-8, 8, 9)/10,
                ticklabels=np.linspace(-8, 8, 9)/10,
                minor_num=2)

    for j in range(2*(1+reflections)):
        colat = fp_traced_arr[3*j+1]    # [rad]
        wlon = fp_traced_arr[3*j+2]     # [rad]
        if 90.0-np.degrees(colat) >= 0:
            F.ax.scatter(
                np.sin(colat)*np.cos(2*np.pi-wlon),
                np.sin(colat)*np.sin(2*np.pi-wlon),
                fc=UC.red, ec='w', s=20.0, zorder=2.0
            )
        else:
            F.ax.scatter(
                np.sin(colat)*np.cos(2*np.pi-wlon),
                np.sin(colat)*np.sin(2*np.pi-wlon),
                fc=UC.blue, ec='w', s=20.0, zorder=2.0,
            )
        # print(90.0-np.degrees(colat), np.degrees(wlon))

    # Foot path
    _, pos_N_MAW, pos_S_MAW, _, _ = fp_path()
    F.ax.scatter(
        np.sin(pos_N_MAW[:, 0])*np.cos(2*np.pi-pos_N_MAW[:, 1]),
        np.sin(pos_N_MAW[:, 0])*np.sin(2*np.pi-pos_N_MAW[:, 1]),
        s=0.05, c=UC.red, zorder=1.0,
    )
    F.ax.scatter(
        np.sin(pos_S_MAW[:, 0])*np.cos(2*np.pi-pos_S_MAW[:, 1]),
        np.sin(pos_S_MAW[:, 0])*np.sin(2*np.pi-pos_S_MAW[:, 1]),
        s=0.05, c=UC.blue, zorder=1.0,
    )

    # Instantaneous footprint positions
    insta_fp_pos_N, insta_fp_pos_S = instantaneous(target_moon_s3_obs)
    F.ax.scatter(
        math.sin(insta_fp_pos_N[0])*math.cos(2*np.pi-insta_fp_pos_N[1]),
        math.sin(insta_fp_pos_N[0])*math.sin(2*np.pi-insta_fp_pos_N[1]),
        marker='D', fc='k', ec='w', s=20.0,
    )
    F.ax.scatter(
        math.sin(insta_fp_pos_S[0])*math.cos(2*np.pi-insta_fp_pos_S[1]),
        math.sin(insta_fp_pos_S[0])*math.sin(2*np.pi-insta_fp_pos_S[1]),
        marker='D', fc='k', ec='w', s=20.0,
    )

    # Observed footprint position
    if target_wlon_fp > -990:
        x_obs = math.sin(math.radians(90-target_lat_fp)) * \
            math.cos(math.radians(360-target_wlon_fp))
        y_obs = math.sin(math.radians(90-target_lat_fp)) * \
            math.sin(math.radians(360-target_wlon_fp))
        x_obs_1 = math.sin(math.radians(90-target_lat_fp)) * \
            math.cos(math.radians(360-target_wlon_fp+target_err_wlon_fp))
        x_obs_2 = math.sin(math.radians(90-target_lat_fp)) * \
            math.cos(math.radians(360-target_wlon_fp-target_err_wlon_fp))
        x_obs_3 = math.sin(math.radians(90-target_lat_fp+target_err_lat_fp)
                           )*math.cos(math.radians(360-target_wlon_fp))
        x_obs_4 = math.sin(math.radians(90-target_lat_fp-target_err_lat_fp)
                           )*math.cos(math.radians(360-target_wlon_fp))
        y_obs_1 = math.sin(math.radians(90-target_lat_fp)) * \
            math.sin(math.radians(360-target_wlon_fp+target_err_wlon_fp))
        y_obs_2 = math.sin(math.radians(90-target_lat_fp)) * \
            math.sin(math.radians(360-target_wlon_fp-target_err_wlon_fp))
        y_obs_3 = math.sin(math.radians(90-target_lat_fp+target_err_lat_fp)
                           )*math.sin(math.radians(360-target_wlon_fp))
        y_obs_4 = math.sin(math.radians(90-target_lat_fp-target_err_lat_fp)
                           )*math.sin(math.radians(360-target_wlon_fp))
        F.ax.plot(
            [x_obs_1, x_obs_2], [y_obs_1, y_obs_2],
            color='k', linewidth=1.0,
            zorder=10,
        )
        F.ax.plot(
            [x_obs_3, x_obs_4], [y_obs_3, y_obs_4],
            color='k', linewidth=1.0,
            zorder=10,
        )
        savename += '_UVS'

    # Longitudinal grid
    s3wlon_grid = np.linspace(0, 360, 9)
    phi_grid = np.radians(360-s3wlon_grid)
    for i in range(phi_grid.size):
        F.ax.plot((0, np.cos(phi_grid[i])),
                  (0, np.sin(phi_grid[i])),
                  color=UC.lightgray, linestyle='--',
                  linewidth=1.0, zorder=0.5)
        if i in [7, phi_grid.size-1]:
            continue
        F.textbox(ax_idx=0,
                  x=0.73*np.cos(phi_grid[i]),
                  y=0.73*np.sin(phi_grid[i]),
                  text=str(int(s3wlon_grid[i]))+'˚W',
                  fontsize=F.fontsize*0.7,
                  horizontalalignment='center',
                  textshadow=False,
                  textcolor='k',
                  facealpha=0.0,
                  edgecolor=(0, 0, 0, 0), )

    # Latitudinal grid
    lat_grid = np.arange(0, 90+1, 15)
    for i in range(lat_grid.size):
        circle = plt.Circle(xy=(0, 0),
                            radius=math.cos(math.radians(90.0-lat_grid[i])),
                            fill=False, ec=UC.lightgray, linewidth=1,
                            linestyle='--', zorder=0.5)
        F.ax.add_patch(circle)
        if i in [0, 1, lat_grid.size-1]:
            continue
        F.textbox(ax_idx=0,
                  x=np.cos(math.radians(90.0-lat_grid[i]))/1.4142,
                  y=np.cos(math.radians(90.0-lat_grid[i]))/1.4142,
                  text=str(int(90-lat_grid[i]))+'˚N/S',
                  fontsize=F.fontsize*0.7,
                  horizontalalignment='center',
                  textshadow=False,
                  textcolor='k',
                  facealpha=0.0,
                  edgecolor=(0, 0, 0, 0), )

    F.fig.tight_layout()
    F.fig.savefig('img/reflect/'+exname+'/'+savename+'.jpg')
    plt.close()

    print('Equatorial lead angle [deg]: ==========')
    for j in range(2*(1+reflections)):
        eq_lead = fp_traced_arr[3*j+3]    # [deg]
        if j in [0, reflections+1]:
            print('  (MAW)  ', round(eq_lead, 3))
        else:
            print('  (RAW)  ', round(eq_lead, 3))
    print('At 900 km [lat, wlongitude] [deg]: ==========')
    for j in range(2*(1+reflections)):
        colat = fp_traced_arr[3*j+1]    # [rad]
        wlon = fp_traced_arr[3*j+2]     # [rad]
        if j in [0, reflections+1]:
            print(
                '  (MAW)  ',
                round(90.0-math.degrees(colat), 3),
                round(np.mod(math.degrees(wlon), 360.0), 3)
            )
        else:
            print(
                '  (RAW)  ',
                round(90.0-math.degrees(colat), 3),
                round(np.mod(math.degrees(wlon), 360.0), 3)
            )

    return 0


# %% the main function
def main():
    # Observed footprint positions
    wlon_fp, err_wlon_fp, lat_fp, err_lat_fp, _, et_fp, hem_fp, _ = Obsresults(
        PJ_LIST, TARGET_MOON, TARGET_FP, TARGET_HEM, FLIP
    )
    _, _, _, _, _, _, moon_S3wlon0_arr = moonS3wlon_arr(et_fp, TARGET_MOON)

    obs_select = 0      # Which obs time?
    fp_traced_arr = fp_traced(moon_S3wlon0_arr[obs_select])      # [deg]
    polar_plot(
        fp_traced_arr,
        moon_S3wlon0_arr[obs_select],
        wlon_fp[obs_select],
        err_wlon_fp[obs_select],
        lat_fp[obs_select],
        err_lat_fp[obs_select],
        et_fp[obs_select]
    )

    # 横軸をmoon_s3_wlonにする
    moon_s3_wlon, pos_N_MAW, pos_S_MAW, _, _ = fp_path()
    fig, ax = plt.subplots()
    ax.plot(moon_s3_wlon, np.degrees(pos_N_MAW[:, 0]))
    fig.tight_layout()
    fig.savefig('img/test_theta.jpg')
    plt.close()

    fig, ax = plt.subplots()
    # ax.set_xlim(185, 205)
    ax.plot(moon_s3_wlon, np.degrees(pos_N_MAW[:, 1]))
    fig.tight_layout()
    fig.savefig('img/test_phi.jpg')
    plt.close()

    for i in range(pos_N_MAW[:, 1].size):
        if math.degrees(pos_N_MAW[i, 1]) > 360.0:
            pos_N_MAW[i, 1] += -2*np.pi

    fig, ax = plt.subplots()
    # ax.set_xlim(185, 205)
    ax.plot(moon_s3_wlon, np.degrees(pos_N_MAW[:, 1]))
    fig.tight_layout()
    fig.savefig('img/test_phi2.jpg')
    plt.close()

    propagation_plot()
    leadangle_plot()

    if TARGET_ET is not False:
        _, _, _, _, _, _, moon_S3wlon0 = moonS3wlon_arr(
            TARGET_ET, moon=TARGET_MOON)
        print('moon_S3wlon0:', moon_S3wlon0)
        fp_traced_arr = fp_traced(moon_S3wlon0[0])      # [deg]
        polar_plot(
            fp_traced_arr,
            moon_S3wlon0[0],
            -9999.0,
            -9999.0,
            -9999.0,
            -9999.0,
            TARGET_ET[0]
        )

    return None


# %% EXECUTE
if __name__ == '__main__':
    # Name of execution
    exname = '003/20250516_065'

    # Input about Juno observation
    TARGET_MOON = 'Io'
    TARGET_FP = ['MAW']
    PJ_LIST = [16]
    TARGET_HEM = 'both'
    FLIP = False            # ALWAYS FALSE! Flip the flag (TEB <-> MAW)
    Ai_num = 3
    ni_num = 50
    Ti_num = 60
    Zi = 1.3                # Io: 1.3 / Eu: 1.4 / Ga: 1.3
    Te = 6.0                # Io: 6.0 [eV]/ Eu: 20.0 / Ga: 300.0
    reflections = 8         # fixed at 8

    # PJ03 2016-12-11T17:51:10
    target_et_pj3 = np.array([spice.utc2et('2016-12-11T17:51:10')])

    # PJ03 2016-12-11T18:18:27
    target_et_pj3 = np.array([spice.utc2et('2016-12-11T18:18:27')])

    # PJ07 2017-07-11T02:53:44
    target_et_pj7 = np.array([spice.utc2et('2017-07-11T02:53:44')])

    # PJ09 2017-10-24T16:47:54
    target_et_pj9 = np.array([spice.utc2et('2017-10-24T16:47:54')])

    # PJ09 2017-10-24T16:48:54
    target_et_pj9 = np.array([spice.utc2et('2017-10-24T16:48:54')])

    # PJ09 2017-10-24T19:05:59
    target_et_pj9 = np.array([spice.utc2et('2017-10-24T19:05:59')])

    # PJ09 2017-10-24T19:24:10
    target_et_pj9 = np.array([spice.utc2et('2017-10-24T19:24:10')])

    # PJ11 2018-02-07T13:16:12
    target_et_pj11 = np.array([spice.utc2et('2018-02-07T13:16:12')])

    # PJ11 2018-02-07T13:21:13
    target_et_pj11 = np.array([spice.utc2et('2018-02-07T13:21:13')])

    # PJ13 2018-05-24T06:54:42
    target_et_pj13 = np.array([spice.utc2et('2018-05-24T06:54:42')])

    # PJ16 2018-10-29T21:57:21
    target_et_pj16 = np.array([spice.utc2et('2018-10-29T21:57:21')])

    # PJ16 2018-10-29T21:58:21
    target_et_pj16 = np.array([spice.utc2et('2018-10-29T21:58:21')])

    # PJ16 2018-10-29T22:10:23
    target_et_pj16 = np.array([spice.utc2et('2018-10-29T22:10:23')])

    # TARGET_ET = np.array([721041971.3])     # False or ET
    TARGET_ET = target_et_pj16

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
