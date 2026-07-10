import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mplcolors
import matplotlib.ticker as ptick
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator
import matplotlib.colorbar as mplcolorbar
import matplotlib.cm as mplcm
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as patches
import math
import datetime

from Leadangle_fit_JunoUVS_Backtrace_2 import eqwlong_err
from Leadangle_fit_JunoUVS_Backtrace_2 import viewingangle
from Leadangle_fit_JunoUVS_Backtrace_2 import moonS3wlon_arr
from Leadangle_fit_JunoUVS_Backtrace_2 import TEB_transit
import Leadangle_wave as Wave

from scipy.odr import ODR, Model, RealData
from scipy.stats import spearmanr
from scipy.stats import t

from MyPlotRecipe.UniversalColor import UniversalColor
from MyPlotRecipe.SharedX import ShareXaxis
from MyPlotRecipe.legend_shadow import legend_shadow

import JupiterMag as jm

jm.Internal.Config(Model='jrm33', CartesianIn=True,
                   CartesianOut=True, Degree=18)
jm.Con2020.Config(equation_type='analytic')

spice.furnsh('kernel/cassMetaK.txt')

UC = UniversalColor()
UC.set_palette()

F = ShareXaxis()
F.fontsize = 20
F.fontname = 'Liberation Sans Narrow'
F.set_default()


# %%
exdir = '006/20260626'
TARGET_MOON = 'Ganymede'
target_fp = ['MAW', 'TEB']
PJ_num = [32]
hem = 'S'
Ai_num = 3
ni_num = 50
Ti_num = 60
Zi = 1.3                # Io: 1.3 / Eu: 1.4 / Ga: 1.3
Te = 300.0              # Io: 6.0 [eV]/ Eu: 20.0 / Ga: 300.0


# %% Footprint obs. list (Ganymede)
PJ_LIST = [3, 4, 5, 6, 7,
           8, 11, 12, 13, 14,
           16, 17, 19, 20, 21,
           22, 25, 27, 30, 32,
           ]
HEM_LIST = ['S', 'both', 'S', 'both', 'S',
            'both', 'N', 'S', 'both', 'S',
            'S', 'S', 'S', 'N', 'S',
            'N', 'S', 'both', 'S', 'S',
            ]
EXNAME_LIST = ['101', '102', '103', '104', '105',
               '106', '107', '117', '109', '118',
               '111', '112', '113', '114', '119',
               '120', '122', '123', '124', '125',
               ]


# %% Constants
dchi_1s = 2.30     # デルタchi2の1シグマ区間
dchi_2s = 6.17     # デルタchi2の2シグマ区間
dchi_3s = 11.8     # デルタchi2の3シグマ区間

MU0 = 1.26E-6            # 真空中の透磁率
AMU2KG = 1.66E-27        # 原子質量をkgに変換するファクタ [kg]
RJ = 71492E+3            # JUPITER RADIUS [m]
MJ = 1.90E+27            # JUPITER MASS [kg]
OMGJ = 2*np.pi/(9.0*3600.0+55.5*60)  # JUPITER ANGULAR VELOCITY [rad/s]
C = 2.99792E+8           # LIGHT SPEED [m/s]
G = 6.67E-11             # 万有引力定数  [m^3 kg^-1 s^-2]

Psyn_io = (12.89)*3600      # Moon's synodic period [sec]
Psyn_eu = (11.22)*3600      # Moon's synodic period [sec]
Psyn_ga = (10.53)*3600      # Moon's synodic period [sec]

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
    xticks = np.array([1, 10, 100])


# %% Data from Connerney+2020: PJ index
con20_pj_idx = np.array([1, 3, 4, 5, 6,
                         7, 8, 9, 10, 11,
                         12, 13, 14, 15, 16,
                         17, 18, 19, 20, 21,
                         22, 23, 24], dtype=int)


# %% Data from Connerney+2020: Current constant [nT]
con20_mu_i_tot = np.array([150.1, 137.8, 127.2, 129.1, 130.1,
                           142.3, 140.1, 143.8, 137.0, 141.4,
                           124.2, 148.9, 145.3, 144.8, 149.9,
                           132.1, 133.5, 152.9, 138.5, 138.8,
                           156.1, 141.4, 146.3])


# %% weighted_percentile
def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix]  # sort data
    weights = weights[ix]  # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / \
        np.sum(weights)  # 'like' a CDF function
    return np.interp(perc, cdf, data)


# %% weighted_boxplot2
def weighted_boxplot2(ax, x0, quartile1, medians, quartile3,
                      min, max,
                      width=0.03, ec='k', fc='w', lw=1.0):
    # Lower box
    height = medians-quartile1
    patch = patches.Rectangle(xy=(x0-width/2, quartile1),
                              width=width,
                              height=height,
                              ec=ec,
                              lw=lw,
                              fc=fc)
    patch = ax.add_patch(patch)
    patch.set_zorder(2)

    # Upper box
    height = quartile3-medians
    patch = patches.Rectangle(xy=(x0-width/2, medians),
                              width=width,
                              height=height,
                              ec=ec,
                              lw=lw,
                              fc=fc)
    patch = ax.add_patch(patch)
    patch.set_zorder(2)

    # Vertical line
    ax.plot([x0, x0],
            [min, max],
            color=ec, linewidth=lw,
            zorder=1)

    # Min
    ax.plot([x0-width/2, x0+width/2],
            [min, min],
            color=ec, linewidth=lw,
            zorder=1)

    # Max
    ax.plot([x0-width/2, x0+width/2],
            [max, max],
            color=ec, linewidth=lw,
            zorder=1)

    return None


# %% weighted_boxplot2
def weighted_boxplot_h2(ax, y0, quartile1, medians, quartile3,
                        min, max,
                        width=0.03, ec='k', fc='w', lw=1.0):
    # Lower box
    height = medians-quartile1
    patch = patches.Rectangle(xy=(quartile1, y0-width/2),
                              width=height,
                              height=width,
                              ec=ec,
                              lw=lw,
                              fc=fc)
    patch = ax.add_patch(patch)
    patch.set_zorder(2)

    # Upper box
    height = quartile3-medians
    patch = patches.Rectangle(xy=(medians, y0-width/2),
                              width=height,
                              height=width,
                              ec=ec,
                              lw=lw,
                              fc=fc)
    patch = ax.add_patch(patch)
    patch.set_zorder(2)

    # Vertical line
    ax.plot([min,
             max],
            [y0, y0],
            color=ec, linewidth=lw,
            zorder=1)

    # Min
    ax.plot([min,
             min],
            [y0-width/2, y0+width/2],
            color=ec, linewidth=lw,
            zorder=1)

    # Max
    ax.plot([max,
             max],
            [y0-width/2, y0+width/2],
            color=ec, linewidth=lw,
            zorder=1)

    return None


# %% Load the retrival results
ni_best = np.zeros(len(PJ_LIST))
ni_err_0 = np.zeros(len(PJ_LIST))
ni_err_1 = np.zeros(len(PJ_LIST))
Ti_best = np.zeros(len(PJ_LIST))
H_best = np.zeros(len(PJ_LIST))
Ai_best = np.zeros(len(PJ_LIST))
azi_currnet_0_ave = np.zeros(len(PJ_LIST))
azi_currnet_1_ave = np.zeros(len(PJ_LIST))
azi_currnet_2_ave = np.zeros(len(PJ_LIST))
selected_time = []
et_fp_0 = []
ftmc_min_q1_median_q3_max_arr = np.zeros((len(PJ_LIST), 5))
data_dir = 'data/Backtraced_AZI_CURRENT/'
for i in range(len(PJ_LIST)):
    exname = exdir+'_'+EXNAME_LIST[i]
    chi2_1d = np.loadtxt('results/fit/'+exname+'/params_chi2.txt')
    Ai_1d = np.loadtxt('results/fit/'+exname+'/params_Ai.txt')
    ni_1d = np.loadtxt('results/fit/'+exname+'/params_ni.txt')
    Ti_1d = np.loadtxt('results/fit/'+exname+'/params_Ti.txt')
    H_1d = np.loadtxt('results/fit/'+exname+'/params_H.txt')
    eqlead_est = np.loadtxt('results/fit/'+exname+'/eqlead_est.txt')
    eqlead_obs = np.loadtxt('results/fit/'+exname+'/eqlead_obs.txt')
    sigma_total = np.loadtxt('results/fit/'+exname+'/sigma_y.txt')
    hem_obs = np.loadtxt('results/fit/'+exname+'/hems_obs.txt')
    moon_S3wlon_obs = np.loadtxt('results/fit/'+exname+'/moon_S3wlon_obs.txt')
    et_fp = np.loadtxt('results/fit/'+exname+'/et_obs.txt')
    sigma_obs = sigma_total
    # print('chi2_1d.shape:', chi2_1d.shape)
    # print('eqlead_est.shape:', eqlead_est.shape)

    chi2_3d = chi2_1d.reshape(ni_num, Ai_num, Ti_num)
    H_3d = H_1d.reshape(ni_num, Ai_num, Ti_num)
    Ai_3d = Ai_1d.reshape(ni_num, Ai_num, Ti_num)
    ni_3d = ni_1d.reshape(ni_num, Ai_num, Ti_num)
    Ti_3d = Ti_1d.reshape(ni_num, Ai_num, Ti_num)
    H_3d = H_1d.reshape(ni_num, Ai_num, Ti_num)
    eqlead_est_3d = eqlead_est[1].reshape(ni_num, Ai_num, Ti_num)

    # 保存されているカイ2乗値は自由度で割ってしまっているので
    # ここで元に戻す
    chi2_3d = chi2_3d*(eqlead_est.shape[0]-3)
    d_chi2_3d = chi2_3d - np.min(chi2_3d)

    print('Parameter ranges:')
    print('--- PJ', str(PJ_LIST[i]).zfill(2))
    print('--- Ai:', np.min(Ai_3d), np.max(Ai_3d))
    print('--- ni:', np.min(ni_3d), np.max(ni_3d))
    print('--- Ti:', np.min(Ti_3d), np.max(Ti_3d))
    print('--- Hi:', np.min(H_3d)/71492E+3, np.max(H_3d)/71492E+3)
    print('Degree of freedom:', (eqlead_est.shape[0]-3))

    sigma_x = eqwlong_err(Psyn, dt=22.5)*np.ones(sigma_obs.shape)

    # Load the azimuthal current intensity
    data = np.loadtxt(
        data_dir+'PJ'+str(PJ_LIST[i]).zfill(2)+'/'+TARGET_MOON[0]+'FP_info_v900km_0.txt')
    azi_currnet_0 = data[5, :]
    data = np.loadtxt(
        data_dir+'PJ'+str(PJ_LIST[i]).zfill(2)+'/'+TARGET_MOON[0]+'FP_info_v900km_1.txt')
    azi_currnet_1 = data[5, :]
    data = np.loadtxt(
        data_dir+'PJ'+str(PJ_LIST[i]).zfill(2)+'/'+TARGET_MOON[0]+'FP_info_v900km_2.txt')
    azi_currnet_2 = data[5, :]
    hem_ref = data[3, :]
    if HEM_LIST[i] == 'N':
        azi_currnet_0 = azi_currnet_0[np.where(hem_ref < 0)]
        azi_currnet_1 = azi_currnet_1[np.where(hem_ref < 0)]
        azi_currnet_2 = azi_currnet_2[np.where(hem_ref < 0)]
        hem_ref = hem_ref[np.where(hem_ref < 0)]
    elif HEM_LIST[i] == 'S':
        azi_currnet_0 = azi_currnet_0[np.where(hem_ref > 0)]
        azi_currnet_1 = azi_currnet_1[np.where(hem_ref > 0)]
        azi_currnet_2 = azi_currnet_2[np.where(hem_ref > 0)]
        hem_ref = hem_ref[np.where(hem_ref > 0)]

    # Viewing angle
    view = viewingangle(PJ_LIST[i], TARGET_MOON, 'MAW', HEM_LIST[i])
    view_TEB = viewingangle(PJ_LIST[i], TARGET_MOON, 'TEB', HEM_LIST[i])
    # if target_fp == ['MAW', 'TEB']:
    #     view = np.hstack((view, view_TEB))      # [deg]
    if EXNAME_LIST[i] not in ['117', '118', '119', '120']:
        view = np.hstack((view, view_TEB))      # [deg]
    else:
        azi_currnet_0 = azi_currnet_0[np.where(abs(hem_ref) == 1)]
        azi_currnet_1 = azi_currnet_1[np.where(abs(hem_ref) == 1)]
        azi_currnet_2 = azi_currnet_2[np.where(abs(hem_ref) == 1)]
    azi_currnet_0_ave[i] = np.average(
        azi_currnet_0[np.where(view <= 30.0)])
    azi_currnet_1_ave[i] = np.average(
        azi_currnet_1[np.where(view <= 30.0)])
    azi_currnet_2_ave[i] = np.average(
        azi_currnet_2[np.where(view <= 30.0)])
    selected_time += [spice.et2datetime(
        np.median(et_fp[np.where(view <= 30.0)]))]
    et_fp_0 += [np.median(et_fp[np.where(view <= 30.0)])]

    # Chi-squared map
    z_value = d_chi2_3d[:, 1, :].T
    x_value = ni_3d[:, 1, :].T
    y_value = Ti_3d[:, 1, :].T
    fig_id = 'SS260707.001'
    F = ShareXaxis()
    F.fontsize = 23
    F.fontname = 'Liberation Sans Narrow'
    F.set_figparams(nrows=1, figsize=(5.0, 3.5), ticksize=1.5,
                    dpi='L')
    F.initialize()
    F.set_xaxis(label=r'$n_i$ [cm$^{-3}$]',
                min=10, max=100,
                ticks=xticks,
                ticklabels=xticks,
                xscale='log')
    F.set_yaxis(ax_idx=0, label=r'$T_i$ [eV]',
                min=10, max=3000,
                ticks=np.array([10, 100, 1000, 3000]),
                ticklabels=np.array([10, 100, 1000, 3000]),
                yscale='log')
    cn = F.ax.contour(x_value, y_value, z_value,
                      levels=[2.30, 6.17, 11.8],
                      # levels=[21.36, 30.48, 40.29],
                      colors='w',
                      linewidths=1.0,
                      zorder=2)
    # Location of the chi2 minimum
    x_best = x_value[np.where(z_value == np.min(z_value))]  # ni
    y_best = y_value[np.where(z_value == np.min(z_value))]  # Ti
    ni_best[i] = x_best
    Ti_best[i] = y_best
    H_best[i] = H_3d[:, 1, :].T[np.where(z_value == np.min(z_value))]
    Ai_best[i] = Ai_3d[0, 1, 0]
    F.ax.scatter(x_best, y_best,
                 s=5, color='w', zorder=2.5)
    mp, pp = F.colormap(ax_idx=0, xdata=x_value, ydata=y_value, zdata=z_value,
                        vmin=0, vmax=500, colorbar_label=r'$\Delta\chi^2$',
                        adjust=True)
    pp.set_ticks([0, 100, 200, 300, 400, 500])
    # F.manage(ax_idx=0, id=fig_id, color=UC.lightgray)
    img_savedir = 'img/ftmc/'+TARGET_MOON[:2]+'/'+exdir + '/'
    F.fig.savefig(img_savedir+'chi2_map/PJ'+str(PJ_LIST[i]).zfill(2)+'.jpg',
                  bbox_inches='tight')

    # FTMC histogram
    FTMC_3d = Ai_3d*AMU2KG*ni_3d*1E+6*np.sqrt(np.pi)*H_3d
    FTMC_2d = FTMC_3d[:, 1, :].T
    dx_arr = x_value[:-1, 1:]-x_value[:-1, :-1]
    dy_arr = y_value[1:, :-1]-y_value[:-1, :-1]
    area = dx_arr * dy_arr
    weight = area/np.median(area)
    FTMC_2d_select = FTMC_2d[:-1, :-1][np.where(z_value[:-1, :-1] < 11.8)]
    weight = weight[np.where(z_value[:-1, :-1] < 11.8)]
    fig_id = 'SS260707.002'
    F = ShareXaxis()
    F.fontsize = 23
    F.fontname = 'Liberation Sans Narrow'
    F.set_figparams(nrows=1, figsize=(5.0, 3.5), ticksize=1.5,
                    dpi='L')
    F.initialize()
    F.set_xaxis(label=r'FTMC [10$^{-11}$ kg m$^{-2}$]',
                min=1, max=25,
                ticks=np.linspace(0, 25, 6),
                ticklabels=np.linspace(0, 25, 6),
                minor_num=5)
    F.set_yaxis(ax_idx=0, label='Counts [#]',
                min=0, max=60,
                ticks=np.arange(0, 60+1, 10),
                ticklabels=np.arange(0, 60+1, 10),
                minor_num=2)
    _, bins, hpatches = F.ax.hist(FTMC_2d_select*1E+11,
                                  bins=np.arange(0, 25.0+1, 0.5),
                                  weights=weight,
                                  color=UC.blue)
    quartile1, medians, quartile3 = weighted_percentile(data=FTMC_2d_select,
                                                        perc=[0.25, 0.5, 0.75],
                                                        weights=weight)
    weighted_boxplot_h2(F.ax, 57,
                        quartile1*1E+11,
                        medians*1E+11,
                        quartile3*1E+11,
                        np.min(FTMC_2d_select)*1E+11,
                        np.max(FTMC_2d_select)*1E+11,
                        width=1.2)
    F.fig.savefig(img_savedir+'ftmc_histogram/PJ'+str(PJ_LIST[i]).zfill(2)+'.jpg',
                  bbox_inches='tight')
    F.close()

    ftmc_min_q1_median_q3_max_arr[i, 0] = np.min(FTMC_2d_select)
    ftmc_min_q1_median_q3_max_arr[i, 1] = quartile1
    ftmc_min_q1_median_q3_max_arr[i, 2] = medians
    ftmc_min_q1_median_q3_max_arr[i, 3] = quartile3
    ftmc_min_q1_median_q3_max_arr[i, 4] = np.max(FTMC_2d_select)
    print('FTMC [10^-9 kg m-2]:', ftmc_min_q1_median_q3_max_arr[i, :]*1E+9)


# %% Juno's perijove times
JUNO_PJ_TIMES = [
    datetime.datetime.strptime(
        '2016-08-27 12:50', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2016-10-19 18:10', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2016-12-11 17:03', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2017-02-02 12:57', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2017-03-27 08:51', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2017-05-19 06:00', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2017-07-11 01:54', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2017-09-01 21:48', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2017-10-24 17:42', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2017-12-16 17:56', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2018-02-07 13:51', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2018-04-01 09:45', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2018-05-24 05:39', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2018-07-16 05:17', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2018-09-07 01:11', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2018-10-29 21:05', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2018-12-21 16:59', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2019-02-12 17:34', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2019-04-06 12:14', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2019-05-29 08:08', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2019-07-21 04:02', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2019-09-12 03:40', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2019-11-03 22:18', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2019-12-26 17:36', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2020-02-17 17:51', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2020-04-10 13:47', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2020-06-02 10:20', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2020-07-25 06:15', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2020-09-16 02:10', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2020-11-08 01:49', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2020-12-30 21:45', '%Y-%m-%d %H:%M'),
    datetime.datetime.strptime(
        '2021-02-21 17:40', '%Y-%m-%d %H:%M'),
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
    datetime.datetime.strptime('2024-10-23', '%Y-%m-%d'),  # PJ66
]
JUNO_PJ_LABELS = ['PJ1', '', '', '', '',
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
                  '66']


# %% ================================
# 横軸: date / 縦軸: FTMC
# ===================================
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(8.2, 4.0), dpi='L')
F.initialize()
# F.panelname = [' a. Io ', ' b. Europa ', ' c. Ganymede ']

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
xticklabels = ['2016', '2017', '2018', '2019', '2020',
               '2021', '2022', '2023', '2024', '2025']
F.set_xaxis(label='Date',
            min=xmin, max=xmax,
            ticks=xticks,
            ticklabels=xticklabels,
            minor_num=12)
F.ax.minorticks_off()
F.ax.xaxis.set_minor_locator(mdates.MonthLocator())
ticklabels = F.ax.get_xticklabels()
ticklabels[0].set_ha('center')
F.set_yaxis(ax_idx=0,
            label=r'$M$ [10$^{-9}$ kg m$^{-2}$]',
            min=0.0, max=0.25,
            ticks=np.linspace(0, 25, 6)/100,
            ticklabels=np.linspace(0, 25, 6)/100,
            minor_num=5)

for i in range(len(PJ_LIST)):
    d0 = selected_time[i]
    min = ftmc_min_q1_median_q3_max_arr[i, 0]*1E+9
    q1 = ftmc_min_q1_median_q3_max_arr[i, 1]*1E+9
    median = ftmc_min_q1_median_q3_max_arr[i, 2]*1E+9
    q3 = ftmc_min_q1_median_q3_max_arr[i, 3]*1E+9
    max = ftmc_min_q1_median_q3_max_arr[i, 4]*1E+9
    width = datetime.timedelta(seconds=60*60*24*20)
    weighted_boxplot2(F.ax, d0, q1, median, q3,
                      min,
                      max,
                      width=width,
                      ec=UC.blue, lw=1.1)

# PJ numbers on the top horizontal axis
PJax = F.ax.twiny()
PJax.set_xlim(xmin, xmax)
PJax.set_xticks(JUNO_PJ_TIMES[::5])
PJax.set_xticklabels(JUNO_PJ_LABELS[::5])
PJax.xaxis.set_minor_locator(FixedLocator(mdates.date2num(JUNO_PJ_TIMES)))
PJax.tick_params('y', grid_zorder=-10)

# Shades in each 5 perijove
F.ax.axvspan(JUNO_PJ_TIMES[0], JUNO_PJ_TIMES[5],
             fc=UC.gray, ec=None, alpha=0.10)
F.ax.axvspan(JUNO_PJ_TIMES[10], JUNO_PJ_TIMES[15],
             fc=UC.gray, ec=None, alpha=0.10)
F.ax.axvspan(JUNO_PJ_TIMES[20], JUNO_PJ_TIMES[25],
             fc=UC.gray, ec=None, alpha=0.10)
F.ax.axvspan(JUNO_PJ_TIMES[30], JUNO_PJ_TIMES[35],
             fc=UC.gray, ec=None, alpha=0.10)
F.ax.axvspan(JUNO_PJ_TIMES[40], JUNO_PJ_TIMES[45],
             fc=UC.gray, ec=None, alpha=0.10)
F.ax.axvspan(JUNO_PJ_TIMES[50], JUNO_PJ_TIMES[55],
             fc=UC.gray, ec=None, alpha=0.10)
F.ax.axvspan(JUNO_PJ_TIMES[60], JUNO_PJ_TIMES[65],
             fc=UC.gray, ec=None, alpha=0.10)

F.fig.savefig(img_savedir + '/ftmc_timeseries.jpg',
              bbox_inches='tight')
F.close()


# %% ================================
# 横軸: FTMC / 縦軸: Current constant
# ===================================
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'
F.set_figparams(nrows=1, figsize=(5.4, 4.5), dpi='L')
F.initialize()
F.set_xaxis(label=r'$M$ [10$^{-9}$ kg m$^{-2}$]',
            min=0.0, max=0.25,
            ticks=np.linspace(0, 25, 6)/100,
            ticklabels=np.linspace(0, 25, 6)/100,
            minor_num=5)
F.set_yaxis(ax_idx=0,
            label=r'$\mu_0 I_{\varphi} / 2$ [nT]',
            min=50, max=200,
            ticks=np.linspace(50, 200, 4),
            ticklabels=np.linspace(50, 200, 4),
            minor_num=5)

for i in range(len(PJ_LIST)):
    x0 = azi_currnet_0_ave[i]
    min = ftmc_min_q1_median_q3_max_arr[i, 0]*1E+9
    q1 = ftmc_min_q1_median_q3_max_arr[i, 1]*1E+9
    median = ftmc_min_q1_median_q3_max_arr[i, 2]*1E+9
    q3 = ftmc_min_q1_median_q3_max_arr[i, 3]*1E+9
    max = ftmc_min_q1_median_q3_max_arr[i, 4]*1E+9
    width = 2.0
    weighted_boxplot_h2(F.ax, x0, q1, median, q3,
                        min,
                        max,
                        width=width,
                        ec=UC.blue, lw=1.1)

F.fig.savefig(img_savedir + '/ftmc_vs_azicurrent.jpg',
              bbox_inches='tight')
F.close()


# %% =========================================
# 横軸: SIII wlon / 縦軸: Equatorial lead angle
# ============================================
pj_select = np.where(np.array(PJ_LIST) == PJ_num[0])
exname = exdir+'_'+EXNAME_LIST[pj_select[0][0]]
eqlead_est = np.loadtxt('results/fit/'+exname+'/eqlead_est.txt')
eqlead_obs = np.loadtxt('results/fit/'+exname+'/eqlead_obs.txt')
sigma_total = np.loadtxt('results/fit/'+exname+'/sigma_y.txt')
hem_obs = np.loadtxt('results/fit/'+exname+'/hems_obs.txt')
moon_S3wlon_obs = np.loadtxt('results/fit/'+exname+'/moon_S3wlon_obs.txt')
et_fp = np.loadtxt('results/fit/'+exname+'/et_obs.txt')

chi2_1d = np.loadtxt('results/fit/'+exname+'/params_chi2.txt')
chi2_3d = chi2_1d.reshape(ni_num, Ai_num, Ti_num)
chi2_2d = chi2_3d[:, 1, :]
min_idx = np.where(chi2_2d == np.min(chi2_2d))

ni_1d = np.loadtxt('results/fit/'+exname+'/params_ni.txt')
ni_3d = ni_1d.reshape(ni_num, Ai_num, Ti_num)
ni_2d = ni_3d[:, 1, :]

H_1d = np.loadtxt('results/fit/'+exname+'/params_H.txt')
H_3d = H_1d.reshape(ni_num, Ai_num, Ti_num)
H_2d = H_3d[:, 1, :]
eqlead_est_best = np.zeros(et_fp.size)
for i in range(et_fp.size):
    eqlead_est_3d = eqlead_est[i, :].reshape(ni_num, Ai_num, Ti_num)
    eqlead_est_2d = eqlead_est_3d[:, 1, :]
    eqlead_est_best[i] = eqlead_est_2d[min_idx]

# Moon position when the Alfven waves launched (Time: t0-tau_A)
# Orbital distance at the PJ time
_, _, _, r_moon_obs, _, _, _ = moonS3wlon_arr(np.array([et_fp[0]]),
                                              TARGET_MOON)
r_A0_arr = r_moon_obs*np.ones(55)
S3wlon_A0_arr = np.linspace(-50, 370, r_A0_arr.size)

eqlead_best_MAW_N = np.zeros(r_A0_arr.size)
eqlead_best_MAW_S = np.zeros(r_A0_arr.size)
eqlead_best_TEB_N = np.zeros(r_A0_arr.size)
eqlead_best_TEB_S = np.zeros(r_A0_arr.size)
TEB_dt_arr = np.zeros(r_A0_arr.size)

mu_i_default = 139.6    # default: 139.6 [nT]
d_rj_default = 3.6      # default: 3.6 [RJ]
# print(azi_currnet_0_ave[pj_select]/mu_i_default)
Wave.Awave().update_Con2020(
    current_coef=azi_currnet_0_ave[pj_select]/mu_i_default
)
for i in range(r_A0_arr.size):
    r_A0 = r_A0_arr[i]
    S3wlon_A0 = S3wlon_A0_arr[i]
    S_A0 = Wave.Awave().tracefield(r_A0,
                                   np.radians(S3wlon_A0),
                                   0.0
                                   )
    tau, _, _, _ = Wave.Awave().trace3(r_A0,
                                       np.radians(S3wlon_A0),
                                       0.0,
                                       S_A0,
                                       Ai_best[pj_select],
                                       ni_best[pj_select],
                                       H_best[pj_select],
                                       -1
                                       )

    eqlead_best_MAW_N[i] = tau*360/Psyn     # [deg]
    TEB_dt_arr[i] = TEB_transit(r_A0, S3wlon_A0, TARGET_MOON)
    eqlead_best_TEB_S[i] = (tau+TEB_dt_arr[i])*360/Psyn     # [deg]

    tau, _, _, _ = Wave.Awave().trace3(r_A0,
                                       np.radians(S3wlon_A0),
                                       0.0,
                                       S_A0,
                                       Ai_best[pj_select],
                                       ni_best[pj_select],
                                       H_best[pj_select],
                                       1
                                       )

    eqlead_best_MAW_S[i] = tau*360/Psyn     # [deg]
    eqlead_best_TEB_N[i] = (tau+TEB_dt_arr[i])*360/Psyn     # [deg]

F = ShareXaxis()
F.fontsize = 21
F.fontname = 'Liberation Sans Narrow'
F.set_figparams(nrows=1, figsize=(6.0, 4.0), ticksize=1.5,
                dpi='L')
F.initialize()
F.set_xaxis(label=r'Moon System III longitude [deg]',
            min=0, max=360,
            ticks=np.arange(0, 360+1, 45),
            ticklabels=np.arange(0, 360+1, 45),
            minor_num=3)
F.set_yaxis(ax_idx=i, label=r'$\delta_{\rm eq}$ [deg]',
            min=-2, max=40,
            ticks=np.arange(0, 40+1, 10),
            ticklabels=np.arange(0, 40+1, 10),
            minor_num=10)

# Observations
for i in range(hem_obs.size):
    if hem_obs[i] == -1:
        color = UC.red
    elif hem_obs[i] == -101:
        color = UC.orange
    elif hem_obs[i] == 1:
        color = UC.blue
    elif hem_obs[i] == 101:
        color = UC.lightblue
    F.ax.scatter(moon_S3wlon_obs[i], eqlead_obs[i],
                 color=color, s=0.1,
                 marker='s',
                 zorder=0.9)
    F.ax.errorbar(moon_S3wlon_obs[i], eqlead_obs[i],
                  yerr=np.abs(sigma_total[i]),
                  linewidth=0., markersize=0,
                  elinewidth=0.8, color=color,
                  zorder=0.9)
    # F.ax.scatter(moon_S3wlon_obs[i], eqlead_est_best[i],
    #              color='k', s=1.1,
    #              marker='s',
    #              zorder=1.9)

# Dummy
labels = ['N MAW', 'S MAW', 'N TEB', 'S TEB']
colors = [UC.red, UC.blue, UC.orange, UC.lightblue]
for i in range(4):
    F.ax.errorbar(-999.9, -999.9,
                  xerr=5.0, yerr=5.0,
                  linewidth=0., markersize=2,
                  elinewidth=1.0, color=colors[i],
                  label=labels[i])

# Best fit lead angle
F.ax.plot(S3wlon_A0_arr+eqlead_best_MAW_N, eqlead_best_MAW_N,
          color=UC.red, linewidth=0.6)
F.ax.plot(S3wlon_A0_arr+eqlead_best_MAW_S, eqlead_best_MAW_S,
          color=UC.blue, linewidth=0.6)
F.ax.plot(S3wlon_A0_arr+eqlead_best_TEB_N, eqlead_best_TEB_N,
          color=UC.orange, linestyle='--', linewidth=0.6)
F.ax.plot(S3wlon_A0_arr+eqlead_best_TEB_S, eqlead_best_TEB_S,
          color=UC.lightblue, linestyle='--', linewidth=0.6)

fig_title = TARGET_MOON
fig_title += ' (PJ'+str(PJ_num[0]).zfill(2) + ')'
F.ax.set_title(fig_title, fontsize=F.fontsize, weight='bold')

legend = F.legend(ax_idx=0, loc='upper right', ncol=4, markerscale=4,
                  fontsize_scale=0.7, textcolor=False, handletextpad=0.2)
legend_shadow(fig=F.fig, ax=F.ax, legend=legend)

F.fig.savefig(img_savedir + '/eq_lead/PJ'+str(PJ_num[0]).zfill(2)+'.jpg',
              bbox_inches='tight')
F.close()
