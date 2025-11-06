import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.ticker as ptick
from matplotlib.ticker import FixedLocator
import matplotlib.dates as mdates
from SharedX import ShareXaxis
from UniversalColor import UniversalColor
from legend_shadow import legend_shadow
from scipy.io import readsav
import datetime
import time

import JupiterMag as jm

from multiprocessing import Pool

from Leadangle_fit_JunoUVS import Obsresults
from Leadangle_fit_JunoUVS import viewingangle
from Leadangle_fit_JunoUVS import calc_eqlead

import spiceypy as spice
spice.furnsh('kernel/cassMetaK.txt')

UC = UniversalColor()
UC.set_palette()

F = ShareXaxis()
F.fontsize = 23
F.fontname = 'Liberation Sans Narrow'
F.set_default()


# %% Input about Juno observation
TARGET_MOON = 'Ganymede'
TARGET_FP = ['MAW', 'TEB']
TARGET_HEM = 'both'
PJ_LIST = [62]
PJ_LIST = [1, 3]+np.arange(4, 68+1, 1).tolist()


# %% Constants
MU0 = 1.26E-6            # 真空中の透磁率
AMU2KG = 1.66E-27        # 原子質量をkgに変換するファクタ [kg]
RJ = 71492E+3            # JUPITER RADIUS [m]
MJ = 1.90E+27            # JUPITER MASS [kg]
C = 2.99792E+8           # LIGHT SPEED [m/s]
G = 6.67E-11             # 万有引力定数  [m^3 kg^-1 s^-2]

Psyn_io = (12.89)*3600      # Moon's synodic period [sec]
Psyn_eu = (11.22)*3600      # Moon's synodic period [sec]
Psyn_ga = (10.53)*3600      # Moon's synodic period [sec]


# Select moon synodic orbital period
if TARGET_MOON == 'Io':
    Psyn = Psyn_io
    r_moon = 5.9*RJ
elif TARGET_MOON == 'Europa':
    Psyn = Psyn_eu
    r_moon = 9.4*RJ
elif TARGET_MOON == 'Ganymede':
    Psyn = Psyn_ga
    r_moon = 15.0*RJ


# %% Select moon synodic orbital period
if len(PJ_LIST) > 1:
    if TARGET_MOON == 'Io':
        PJ_LIST.pop(54-2)
        PJ_LIST.pop(55-3)
        PJ_LIST.pop(56-4)
        PJ_LIST.pop(57-5)
        PJ_LIST.pop(61-6)
        PJ_LIST.pop(63-7)
        PJ_LIST.pop(64-8)
        PJ_LIST.pop(65-9)
        PJ_LIST.pop(67-10)
    elif TARGET_MOON == 'Europa':
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
    elif TARGET_MOON == 'Ganymede':
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


# %% Surface
def calc_r_surf(r0, r1, lat):
    """
    `lat` ... latitude [rad]
    """
    rs = r0*r1/np.sqrt((r0*np.sin(lat))**2+(r1*np.cos(lat))**2)  # [m]

    return rs


# Calc
def calc(x0, y0, z0, mu_i_coeff_j):
    mu_i_default = 139.6    # default: 139.6 [nT]
    jm.Con2020.Config(mu_i=mu_i_default*mu_i_coeff_j,
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
    rho_j = rho[idx_z0]

    return rho_j


# %% Plot
def plot(et_fp, mu_i_arr, N_idx, S_idx):
    F = ShareXaxis()
    F.fontsize = 21
    F.fontname = 'Liberation Sans Narrow'

    F.set_figparams(nrows=1, figsize=(8.5, 4.0), dpi='L')
    F.initialize()

    sxmin = '2016-01-01'
    sxmax = '2025-01-01'
    xmin = datetime.datetime.strptime(sxmin, '%Y-%m-%d')
    xmax = datetime.datetime.strptime(sxmax, '%Y-%m-%d')
    xticks = [datetime.datetime.strptime('2016-01-01', '%Y-%m-%d'),
              datetime.datetime.strptime('2017-01-01', '%Y-%m-%d'),
              datetime.datetime.strptime('2018-01-01', '%Y-%m-%d'),
              datetime.datetime.strptime('2019-01-01', '%Y-%m-%d'),
              datetime.datetime.strptime('2020-01-01', '%Y-%m-%d'),
              datetime.datetime.strptime('2021-01-01', '%Y-%m-%d'),
              datetime.datetime.strptime('2022-01-01', '%Y-%m-%d'),
              datetime.datetime.strptime('2023-01-01', '%Y-%m-%d'),
              datetime.datetime.strptime('2024-01-01', '%Y-%m-%d'),
              datetime.datetime.strptime('2025-01-01', '%Y-%m-%d')]
    xticklabels = ['2016', '2017',
                   '2018', '2019',
                   '2020', '2021',
                   '2022', '2023',
                   '2024', '2025']
    F.set_xaxis(label='Date',
                min=xmin, max=xmax,
                ticks=xticks,
                ticklabels=xticklabels,
                minor_num=12)
    ticklabels = F.ax.get_xticklabels()
    ticklabels[0].set_ha('center')

    PJax = F.ax.twiny()
    # PJax.set_title(r'Flux tube mass contents',
    #                fontsize=F.fontsize, weight='bold')
    xticks = [datetime.datetime.strptime('2016-08-27', '%Y-%m-%d'),
              datetime.datetime.strptime('2016-10-19', '%Y-%m-%d'),
              datetime.datetime.strptime('2016-12-11', '%Y-%m-%d'),
              datetime.datetime.strptime('2017-02-02', '%Y-%m-%d'),
              datetime.datetime.strptime('2017-03-27', '%Y-%m-%d'),
              datetime.datetime.strptime('2017-05-19 06:00', '%Y-%m-%d %H:%M'),
              datetime.datetime.strptime('2017-07-11', '%Y-%m-%d'),
              datetime.datetime.strptime('2017-09-01', '%Y-%m-%d'),
              datetime.datetime.strptime('2017-10-24', '%Y-%m-%d'),
              datetime.datetime.strptime('2017-12-16', '%Y-%m-%d'),
              datetime.datetime.strptime('2018-02-07', '%Y-%m-%d'),
              datetime.datetime.strptime('2018-04-01', '%Y-%m-%d'),
              datetime.datetime.strptime('2018-05-24', '%Y-%m-%d'),
              datetime.datetime.strptime('2018-07-16', '%Y-%m-%d'),
              datetime.datetime.strptime('2018-09-07', '%Y-%m-%d'),
              datetime.datetime.strptime('2018-10-29', '%Y-%m-%d'),
              datetime.datetime.strptime('2018-12-21', '%Y-%m-%d'),
              datetime.datetime.strptime('2019-02-12', '%Y-%m-%d'),
              datetime.datetime.strptime('2019-04-06', '%Y-%m-%d'),
              datetime.datetime.strptime('2019-05-29', '%Y-%m-%d'),
              datetime.datetime.strptime('2019-07-21', '%Y-%m-%d'),
              datetime.datetime.strptime('2019-09-12', '%Y-%m-%d'),
              datetime.datetime.strptime('2019-11-03', '%Y-%m-%d'),
              datetime.datetime.strptime('2019-12-26', '%Y-%m-%d'),
              datetime.datetime.strptime('2020-02-17', '%Y-%m-%d'),
              datetime.datetime.strptime('2020-04-10', '%Y-%m-%d'),
              datetime.datetime.strptime('2020-06-02', '%Y-%m-%d'),
              datetime.datetime.strptime('2020-07-25', '%Y-%m-%d'),
              datetime.datetime.strptime('2020-09-16', '%Y-%m-%d'),
              datetime.datetime.strptime('2020-11-08', '%Y-%m-%d'),
              datetime.datetime.strptime('2020-12-30', '%Y-%m-%d'),
              datetime.datetime.strptime('2021-02-21', '%Y-%m-%d'),
              datetime.datetime.strptime('2021-04-15', '%Y-%m-%d'),
              datetime.datetime.strptime('2021-06-08', '%Y-%m-%d'),
              datetime.datetime.strptime('2021-07-21', '%Y-%m-%d'),
              datetime.datetime.strptime('2021-09-02', '%Y-%m-%d'),
              datetime.datetime.strptime('2021-10-16', '%Y-%m-%d'),
              datetime.datetime.strptime('2021-11-29', '%Y-%m-%d'),
              datetime.datetime.strptime('2022-01-12', '%Y-%m-%d'),
              datetime.datetime.strptime('2022-02-25', '%Y-%m-%d'),
              datetime.datetime.strptime('2022-04-09', '%Y-%m-%d'),
              datetime.datetime.strptime('2022-05-23', '%Y-%m-%d'),
              datetime.datetime.strptime('2022-07-05', '%Y-%m-%d'),
              datetime.datetime.strptime('2022-08-17', '%Y-%m-%d'),
              datetime.datetime.strptime('2022-09-29', '%Y-%m-%d'),
              datetime.datetime.strptime('2022-11-06', '%Y-%m-%d'),  # PJ46
              datetime.datetime.strptime('2022-12-15', '%Y-%m-%d'),  # PJ47
              datetime.datetime.strptime('2023-01-22', '%Y-%m-%d'),
              datetime.datetime.strptime('2023-03-01', '%Y-%m-%d'),
              datetime.datetime.strptime('2023-04-08', '%Y-%m-%d'),
              datetime.datetime.strptime('2023-05-16', '%Y-%m-%d'),
              datetime.datetime.strptime('2023-06-23', '%Y-%m-%d'),
              datetime.datetime.strptime('2023-07-31', '%Y-%m-%d'),
              datetime.datetime.strptime('2023-09-07', '%Y-%m-%d'),  # PJ54
              datetime.datetime.strptime('2023-10-15', '%Y-%m-%d'),
              datetime.datetime.strptime('2023-11-22', '%Y-%m-%d'),
              datetime.datetime.strptime('2023-12-30', '%Y-%m-%d'),
              datetime.datetime.strptime('2024-02-04', '%Y-%m-%d'),  # PJ58
              datetime.datetime.strptime('2024-03-07', '%Y-%m-%d'),
              datetime.datetime.strptime('2024-04-09', '%Y-%m-%d'),
              datetime.datetime.strptime('2024-05-12', '%Y-%m-%d'),
              datetime.datetime.strptime('2024-06-14', '%Y-%m-%d'),
              datetime.datetime.strptime('2024-07-16', '%Y-%m-%d'),
              datetime.datetime.strptime('2024-08-18', '%Y-%m-%d'),  # PJ64
              datetime.datetime.strptime('2024-09-20', '%Y-%m-%d'),
              datetime.datetime.strptime('2024-10-23', '%Y-%m-%d'),
              ]
    xticklabels = ['PJ1', '', '', '', '',
                   '6', '', '', '', '',
                   '11', '', '', '', '',
                   '16', '', '', '', '',
                   '21', '', '', '', '',
                   '26', '', '', '', '',
                   '31', '', '', '', '',
                   '36', '', '', '', '',
                   '41', '', '', '', '',
                   '46', '', '', '', '',
                   '51', '', '', '', '',
                   '56', '', '', '', '',
                   '61', '', '', '', '',
                   '66', '', '', '', '',]
    PJax.set_xlim(xmin, xmax)
    PJax.set_xticks(xticks[::5])
    PJax.set_xticklabels(xticklabels[::5])
    PJax.xaxis.set_minor_locator(FixedLocator(mdates.date2num(xticks)))
    PJax.set_title(TARGET_MOON+' (JRM33+CON20)', weight='bold')

    # Shades in each 5 perijove
    F.ax.axvspan(xticks[0], xticks[5], fc=UC.gray, ec=None, alpha=0.13)
    F.ax.axvspan(xticks[10], xticks[15], fc=UC.gray, ec=None, alpha=0.13)
    F.ax.axvspan(xticks[20], xticks[25], fc=UC.gray, ec=None, alpha=0.13)
    F.ax.axvspan(xticks[30], xticks[35], fc=UC.gray, ec=None, alpha=0.13)
    F.ax.axvspan(xticks[40], xticks[45], fc=UC.gray, ec=None, alpha=0.13)
    F.ax.axvspan(xticks[50], xticks[55], fc=UC.gray, ec=None, alpha=0.13)
    F.ax.axvspan(xticks[60], xticks[65], fc=UC.gray, ec=None, alpha=0.13)

    ymin = -11
    ymax = 250
    yticks = np.arange(0, 250+1, 50)
    yticklabels = np.arange(0, 250+1, 50)
    F.set_yaxis(ax_idx=0,
                label=r'$\mu_0 I$/2 [nT]',
                min=ymin, max=ymax,
                ticks=yticks,
                ticklabels=yticklabels,
                minor_num=5)

    # et_fpをdatetimeに変換する
    datetime_fp = []
    for i in range(et_fp.size):
        d0 = spice.et2datetime(et_fp[i])
        datetime_fp += [d0]
    datetime_fp = np.array(datetime_fp)

    # Satellite orbit
    F.ax.axhline(y=0, linestyle='dashed', color=UC.lightgray, zorder=0.9)

    # Data
    F.ax.scatter(datetime_fp[N_idx], mu_i_arr[N_idx],
                 marker='*', s=3.5, c=UC.red, label='North')

    F.ax.scatter(datetime_fp[S_idx], mu_i_arr[S_idx],
                 marker='*', s=3.5, c=UC.blue, label='South')

    F.textbox(ax_idx=0,
              x=0.03, y=0.9,
              text='Emission angle < 30 deg',
              fontsize=F.fontsize*0.7,
              horizontalalignment='left',
              textshadow=False,
              textcolor='k',
              facealpha=0.0,
              edgecolor=(0, 0, 0, 0),
              transform=F.ax.transAxes,)

    legend = F.legend(ax_idx=0,
                      ncol=3, markerscale=4.0,
                      loc='upper right',
                      handlelength=1.0,
                      fontsize_scale=0.65, handletextpad=0.2)
    legend_shadow(legend=legend, fig=F.fig, ax=F.ax, d=0.7)

    return F, datetime_fp


# %%
def main():
    # Data
    wlon_fp, err_wlon_fp, lat_fp, err_lat_fp, moon_S3wlon, et_fp, hem_fp, pj_fp = Obsresults(
        PJ_LIST, TARGET_MOON, TARGET_FP, TARGET_HEM=TARGET_HEM, FLIP=False
    )

    eqlead_fp, eqlead_fp_0, eqlead_fp_1, wlon_fp_eq = calc_eqlead(wlon_fp,
                                                                  err_wlon_fp,
                                                                  lat_fp,
                                                                  err_lat_fp,
                                                                  hem_fp,
                                                                  moon_S3wlon,
                                                                  TARGET_MOON)

    # Emission angle
    view_angle = np.zeros(3)
    for i in PJ_LIST:
        for j in TARGET_FP:
            view_angle1 = viewingangle(PJnum=i,
                                       target_moon=TARGET_MOON,
                                       target_fp=j,
                                       target_hem=TARGET_HEM,
                                       FLIP=False)
            view_angle = np.append(view_angle, view_angle1)

    # 余計な部分を削除
    view_angle = view_angle[3:]

    # Emission angleで制限
    view_idx = np.where((view_angle <= 30.0))
    wlon_fp = wlon_fp[view_idx]
    err_wlon_fp = err_wlon_fp[view_idx]
    lat_fp = lat_fp[view_idx]
    err_lat_fp = err_lat_fp[view_idx]
    moon_S3wlon = moon_S3wlon[view_idx]
    et_fp = et_fp[view_idx]
    hem_fp = hem_fp[view_idx]
    pj_fp = pj_fp[view_idx]
    view_angle = view_angle[view_idx]

    print(wlon_fp.shape)

    # 経度0度(y=0)平面のx-z対応テーブル (900km高度)
    extradius = np.loadtxt('data/Alt_900km/rthetaphi.txt')
    r_e = extradius[0, :]        # [RJ]
    theta_e = np.radians(extradius[1, :])    # [rad]
    phi_e = np.radians(extradius[2, :])      # [rad]

    # 中央値
    rho_arr = np.zeros(wlon_fp.size)
    mu_i_arr = np.zeros(wlon_fp.size)

    # 磁場モデルの設定
    mu_i_default = 139.6    # default: 139.6 [nT]
    mu_i_coeff = np.arange(0, 1.8, 0.02)
    jm.Internal.Config(Model="jrm33", CartesianIn=True,
                       CartesianOut=True, Degree=18)

    for i in range(rho_arr.size):
        latitude = lat_fp[i]
        theta = np.radians(90.0-latitude)
        phi = np.radians(360.0-wlon_fp[i])

        # テーブルを参照し距離を確定
        dis = np.abs(theta-theta_e)
        idx = np.argmin(dis)
        r = r_e[idx]

        x0 = r*np.sin(theta)*np.cos(phi)
        y0 = r*np.sin(theta)*np.sin(phi)
        z0 = r*np.cos(theta)

        arg_size = mu_i_coeff.size
        args = list(zip(
            x0*np.ones(arg_size),
            y0*np.ones(arg_size),
            z0*np.ones(arg_size),
            mu_i_coeff
        ))

        with Pool(processes=25) as pool:
            results_list = list(pool.starmap(calc, args))
        rho_j_arr = np.array(results_list)

        j_index = np.argmin(np.abs(rho_j_arr-r_moon/RJ))

        rho_arr[i] = rho_j_arr[j_index]
        mu_i_arr[i] = mu_i_default * mu_i_coeff[j_index]

    # 高緯度側の誤差評価
    rho_arr_1 = np.zeros(wlon_fp.size)
    mu_i_arr_1 = np.zeros(wlon_fp.size)
    for i in range(rho_arr.size):
        latitude = lat_fp[i]
        if latitude >= 0.0:
            latitude = latitude + err_lat_fp[i]
        else:
            latitude = latitude - err_lat_fp[i]
        theta = np.radians(90.0-latitude)
        phi = np.radians(360.0-wlon_fp[i])

        # テーブルを参照し距離を確定
        dis = np.abs(theta-theta_e)
        idx = np.argmin(dis)
        r = r_e[idx]

        x0 = r*np.sin(theta)*np.cos(phi)
        y0 = r*np.sin(theta)*np.sin(phi)
        z0 = r*np.cos(theta)

        arg_size = mu_i_coeff.size
        args = list(zip(
            x0*np.ones(arg_size),
            y0*np.ones(arg_size),
            z0*np.ones(arg_size),
            mu_i_coeff
        ))

        with Pool(processes=25) as pool:
            results_list = list(pool.starmap(calc, args))
        rho_j_arr = np.array(results_list)

        j_index = np.argmin(np.abs(rho_j_arr-r_moon/RJ))

        rho_arr_1[i] = rho_j_arr[j_index]
        mu_i_arr_1[i] = mu_i_default * mu_i_coeff[j_index]

    # 時系列を作成する
    N_idx = np.where((hem_fp < 0) & (view_angle <= 30.0))
    S_idx = np.where((hem_fp > 0) & (view_angle <= 30.0))
    F, datetime_fp = plot(et_fp, mu_i_arr, N_idx, S_idx)

    # 誤差も入れる
    F.ax.errorbar(x=datetime_fp[N_idx],
                  y=mu_i_arr[N_idx],
                  yerr=np.abs(mu_i_arr[N_idx]-mu_i_arr_1[N_idx]),
                  elinewidth=0.6, linewidth=0.,
                  markersize=0, color=UC.red)
    F.ax.errorbar(x=datetime_fp[S_idx],
                  y=mu_i_arr[S_idx],
                  yerr=np.abs(mu_i_arr[S_idx]-mu_i_arr_1[S_idx]),
                  elinewidth=0.6, linewidth=0.,
                  markersize=0, color=UC.blue)

    # Save
    save_dir = 'img/'
    save_name = 'diskcurrent_'+str(TARGET_MOON[0:2])
    F.fig.savefig(save_dir+save_name+'.jpg', bbox_inches='tight')
    F.close()
    return 0


# %% Execute
if __name__ == '__main__':
    t0 = time.time()
    main()
    print('Time [sec]: ', time.time()-t0)
