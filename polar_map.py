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
from MyPlotRecipe.UniversalColor import UniversalColor
from MyPlotRecipe.SharedX import ShareXaxis
from MyPlotRecipe.legend_shadow import legend_shadow

from RAW_trace_2 import load_best_fit

import spiceypy as spice
import JupiterMag as jm
from scipy.io import readsav

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


# %% Import the footprint positions based on the best-fit parameters (Ai, ni, Ti)
def fp_traced(target_moon_s3_obs):
    """
    Args:
        target_moon_s3_obs (float): moon position at the time of the footprint observation [deg]

    Returns:
        _type_: _description_
    """
    filename = 'data_'+TARGET_MOON[0]+'FP_interp_map_' + \
        str(int(alt_ref[fp_alt_target]))+'km_'+retrieval+'.txt'
    interp = np.loadtxt('results/reflect_2/'+exname+'/'+filename)
    moon_s3_obs = interp[:, 0]      # [deg]
    idx = np.argmin(abs(moon_s3_obs-target_moon_s3_obs))

    positions = interp[idx, :]
    print('interp.shape:', interp.shape)
    print('positions.shape:', positions.shape)
    return positions


# %% Generate the footpath of MAW
def fp_path():
    filename = 'data_'+TARGET_MOON[0]+'FP_interp_map_' + \
        str(int(alt_ref[fp_alt_target]))+'km_'+retrieval+'.txt'
    interp = np.loadtxt('results/reflect_2/'+exname+'/'+filename)
    moon_s3_obs = interp[:, 0]

    # j=1: colatitude, j=2: w-longitude [rad]
    pos_N_MAW = interp[:, 1:3]
    pos_S_MAW = interp[:, 1+3*(3+reflections):3*(3+reflections)+3]

    pos_S_RAW1 = interp[:, 4:6]
    pos_N_RAW1 = interp[:, 4+3*(3+reflections):3*(3+reflections)+6]

    return moon_s3_obs, pos_N_MAW, pos_S_MAW, pos_N_RAW1, pos_S_RAW1


# %% Instantaneous footprint position
def instantaneous(target_moon_s3_obs):
    Ai_best, ni_best, _, Hp_best = load_best_fit()

    s3wlon_t0 = np.radians(target_moon_s3_obs)

    S_A0 = Wave.Awave().tracefield(r_moon,
                                   s3wlon_t0,
                                   0.0)

    # Initital trace
    # -> Instantaneous position at a selected altitude
    hem = -1    # North
    _, rs_N, s3wlon_N, theta_s3_N, _, alt_flag_N = Wave.Awave().trace3_reflect(r_moon,
                                                                               s3wlon_t0,
                                                                               0.0,
                                                                               S_A0,
                                                                               Ai_best,
                                                                               ni_best,
                                                                               Hp_best,
                                                                               hem)
    non_0 = np.array(np.where(alt_flag_N != 0)[0])
    print('non_0:', non_0)
    print('alt_flag_N[non_0]:', alt_flag_N[non_0])
    insta_fp_pos_N = np.zeros(2)
    insta_fp_pos_N[0] = theta_s3_N[non_0][fp_alt_target]  # Colatitude [rad]
    insta_fp_pos_N[1] = s3wlon_N[non_0][fp_alt_target]    # W.longitude [rad]

    # -> Instantaneous position at a selected altitude
    hem = 1    # South
    _, rs_t1, s3wlon_t1, theta_s3_t1, _, alt_flag_S = Wave.Awave().trace3_reflect(r_moon,
                                                                                  s3wlon_t0,
                                                                                  0.0,
                                                                                  S_A0,
                                                                                  Ai_best,
                                                                                  ni_best,
                                                                                  Hp_best,
                                                                                  hem)
    non_0 = np.array(np.where(alt_flag_S != 0)[0])
    insta_fp_pos_S = np.zeros(2)
    insta_fp_pos_S[0] = theta_s3_t1[non_0][fp_alt_target]  # Colatitude [rad]
    insta_fp_pos_S[1] = s3wlon_t1[non_0][fp_alt_target]    # W.longitude [rad]

    return insta_fp_pos_N, insta_fp_pos_S


# %% Lead angle plot
def leadangle_plot():
    filename = 'data_'+TARGET_MOON[0]+'FP_interp_map_' + \
        str(int(alt_ref[fp_alt_target]))+'km_'+retrieval+'.txt'
    interp = np.loadtxt('results/reflect_2/'+exname+'/'+filename)
    moon_s3_obs = interp[:, 0]

    F = ShareXaxis()
    F.fontsize = 23
    F.fontname = 'Liberation Sans Narrow'

    title = 'PJ'+str(PJ_LIST[0]).zfill(2)
    if TARGET_HEM != 'both':
        title += TARGET_HEM

    F.set_figparams(nrows=1, figsize=(9, 9),
                    ticksize=1.5, dpi='L')
    F.initialize()

    F.ax.set_title(title)
    F.set_xaxis(label=r'SIII longitude $\lambda_{\rm III}^{\rm Io}$ [deg]',
                min=0.0, max=360.0,
                ticks=np.arange(0, 360+1, 45),
                ticklabels=np.arange(0, 360+1, 45, dtype=int),
                minor_num=3)
    F.set_yaxis(ax_idx=0, label=r'Equatorial lead angle [deg]',
                min=-5, max=100,
                ticks=np.arange(0, 100+1, 10),
                ticklabels=np.arange(0, 100+1, 10, dtype=int),
                minor_num=2)

    # j=1: colatitude [rad], j=2: w-longitude [rad]
    # j=3: equatorial lead angle [rad]
    pos_N_MAW = interp[:, 3]
    pos_S_MAW = interp[:, 3*(1+reflections+2)+3]
    # F.ax.plot(moon_s3_obs, pos_N_MAW, color=UC.red)
    # F.ax.plot(moon_s3_obs, pos_S_MAW, color=UC.blue, linestyle='--')

    # Reflections
    colors = [UC.red, UC.blue, UC.red]
    for i in range(1+reflections+2):
        pos_N_fp = interp[:, 3*i+3]
        if i in [1+reflections, 1+reflections+1]:
            F.ax.plot(moon_s3_obs, pos_N_fp,
                      color=colors[i % 2],
                      linewidth=1.4, linestyle=(0, (5, 10)))
        else:
            F.ax.plot(moon_s3_obs, pos_N_fp,
                      color=colors[i % 2],
                      linewidth=1.4)
    for i in range(1+reflections+2):
        pos_S_fp = interp[:, 3*(i+1+reflections+2)+3]
        if i in [1+reflections, 1+reflections+1]:
            F.ax.plot(moon_s3_obs, pos_S_fp,
                      color=colors[i % 2+1],
                      linewidth=1.4, linestyle=(0, (5, 10)))
        else:
            F.ax.plot(moon_s3_obs, pos_S_fp,
                      color=colors[i % 2+1],
                      linewidth=1.4)

    F.fig.tight_layout()
    F.fig.savefig('img/reflect_2/'+exname +
                  '/moons3wlon_vs_eqlead_'+retrieval+'.jpg')
    F.close()
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

    # MAW & TEB
    for j in range(2*(3+reflections)):
        colat = fp_traced_arr[3*j+1]    # [rad]
        wlon = fp_traced_arr[3*j+2]     # [rad]
        if j in [3+reflections-2, 3+reflections-1, 2*(3+reflections)-2, 2*(3+reflections)-1]:
            marker = 'D'
            print(90.0-np.degrees(colat), np.degrees(wlon))
        else:
            marker = 'o'
        if 90.0-np.degrees(colat) >= 0:
            F.ax.scatter(
                np.sin(colat)*np.cos(2*np.pi-wlon),
                np.sin(colat)*np.sin(2*np.pi-wlon),
                marker=marker,
                fc=UC.red, ec='w', s=10.0, zorder=2.0
            )
        else:
            F.ax.scatter(
                np.sin(colat)*np.cos(2*np.pi-wlon),
                np.sin(colat)*np.sin(2*np.pi-wlon),
                marker=marker,
                fc=UC.blue, ec='w', s=10.0, zorder=2.0,
            )

    # Foot path
    _, pos_N_MAW, pos_S_MAW, _, _ = fp_path()
    F.ax.scatter(
        np.sin(pos_N_MAW[:, 0])*np.cos(2*np.pi-pos_N_MAW[:, 1]),
        np.sin(pos_N_MAW[:, 0])*np.sin(2*np.pi-pos_N_MAW[:, 1]),
        s=0.04, c=UC.red, zorder=1.0,
    )
    F.ax.scatter(
        np.sin(pos_S_MAW[:, 0])*np.cos(2*np.pi-pos_S_MAW[:, 1]),
        np.sin(pos_S_MAW[:, 0])*np.sin(2*np.pi-pos_S_MAW[:, 1]),
        s=0.04, c=UC.blue, zorder=1.0,
    )

    # Instantaneous footprint positions
    insta_fp_pos_N, insta_fp_pos_S = instantaneous(target_moon_s3_obs)
    F.ax.scatter(
        math.sin(insta_fp_pos_N[0])*math.cos(2*np.pi-insta_fp_pos_N[1]),
        math.sin(insta_fp_pos_N[0])*math.sin(2*np.pi-insta_fp_pos_N[1]),
        marker='D', fc='k', ec='w', s=18.0,
    )
    F.ax.scatter(
        math.sin(insta_fp_pos_S[0])*math.cos(2*np.pi-insta_fp_pos_S[1]),
        math.sin(insta_fp_pos_S[0])*math.sin(2*np.pi-insta_fp_pos_S[1]),
        marker='D', fc='k', ec='w', s=18.0,
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

    savename += '_'+str(int(alt_ref[fp_alt_target]))+'km_'+retrieval
    F.fig.tight_layout()
    F.fig.savefig('img/reflect_2/'+exname+'/'+savename+'.jpg')
    plt.close()

    print('Equatorial lead angle [deg]: ==========')
    for j in range(2*(1+reflections+2)):
        eq_lead = fp_traced_arr[3*j+3]    # [deg]
        if j in [0, 1+reflections+2]:
            print('  (MAW)  ', round(eq_lead, 3))
        elif j in [1+reflections, 1+reflections+1, 2*(1+reflections+2)-2, 2*(1+reflections+2)-1]:
            print('  (TEB)  ', round(eq_lead, 3))
        else:
            print('  (RAW)  ', round(eq_lead, 3))
    print('At', str(int(alt_ref[fp_alt_target])),
          'km [lat, wlongitude] [deg]: ==========')
    for j in range(2*(1+reflections+2)):
        colat = fp_traced_arr[3*j+1]    # [rad]
        wlon = fp_traced_arr[3*j+2]     # [rad]
        if j in [0, 1+reflections+2]:
            print(
                '  (MAW)  ',
                round(90.0-math.degrees(colat), 3),
                round(np.mod(math.degrees(wlon), 360.0), 3)
            )
        elif j in [1+reflections, 1+reflections+1, 2*(1+reflections+2)-2, 2*(1+reflections+2)-1]:
            print(
                '  (TEB)  ',
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
    print('fp_traced_arr.shape:', fp_traced_arr.shape)
    polar_plot(
        fp_traced_arr,
        moon_S3wlon0_arr[obs_select],
        wlon_fp[obs_select],
        err_wlon_fp[obs_select],
        lat_fp[obs_select],
        err_lat_fp[obs_select],
        et_fp[obs_select]
    )

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
    exname = '003/20250516_054'

    # Input about Juno observation
    TARGET_MOON = 'Io'
    TARGET_FP = ['MAW']
    PJ_LIST = [9]
    TARGET_HEM = 'N'
    FLIP = False            # ALWAYS FALSE! Flip the flag (TEB <-> MAW)
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
    retrieval = 'best'                  # 'best', 'hot', 'dense'

    # PJ03 2016-12-11T17:51:10
    target_et_pj3 = np.array([spice.utc2et('2016-12-11T17:51:10')])

    # PJ03 2016-12-11T18:18:27
    target_et_pj3 = np.array([spice.utc2et('2016-12-11T18:18:27')])

    # PJ07 2017-07-11T02:53:44
    target_et_pj7 = np.array([spice.utc2et('2017-07-11T02:53:44')])

    # PJ09 2017-10-24T16:47:54
    target_et_pj9n = np.array([spice.utc2et('2017-10-24T16:47:54')])

    # PJ09 2017-10-24T16:48:54
    target_et_pj9n = np.array([spice.utc2et('2017-10-24T16:48:54')])

    # PJ09 2017-10-24T19:05:59
    target_et_pj9s = np.array([spice.utc2et('2017-10-24T19:05:59')])

    # PJ09 2017-10-24T19:24:10
    target_et_pj9s = np.array([spice.utc2et('2017-10-24T19:24:10')])

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
    TARGET_ET = target_et_pj9n

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
    _, _, _, r_moon_obs, _, _, s3wlon_moon_obs = moonS3wlon_arr(TARGET_ET,
                                                                TARGET_MOON)
    r_moon = r_moon_obs[0]
    print('Orbital distance [RJ]:', r_moon/RJ)

    main()
