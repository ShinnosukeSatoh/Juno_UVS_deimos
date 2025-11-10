import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import matplotlib.patches as patches
from matplotlib.ticker import FixedLocator
import matplotlib.dates as mdates
import math
import datetime
from UniversalColor import UniversalColor
from SharedX import ShareXaxis
from legend_shadow import legend_shadow

from Leadangle_fit_JunoUVS import local_time_moon
from Leadangle_fit_JunoUVS import calc_eqlead
from Leadangle_fit_JunoUVS import Obsresults

import spiceypy as spice
spice.furnsh('kernel/cassMetaK.txt')
savpath = 'data/Satellite_FP_JRM33.sav'

UC = UniversalColor()
UC.set_palette()


exdate = '003/20250516'
target_moon = 'Io'
target_fp = ['MAW', 'TEB']

exnum = ['050', '056']
PJ_list = [6, 10]

Ai_num = 3
ni_num = 50
Ti_num = 60

# Constants
MU0 = 1.26E-6            # 真空中の透磁率
AMU2KG = 1.66E-27        # 原子質量をkgに変換するファクタ [kg amu^-1]
RJ = 71492E+3            # JUPITER RADIUS [m]
MJ = 1.90E+27            # JUPITER MASS [kg]
C = 2.99792E+8           # LIGHT SPEED [m/s]
G = 6.67E-11             # 万有引力定数  [m^3 kg^-1 s^-2]

Psyn_io = (12.89)*3600      # Moon's synodic period [sec]
Psyn_eu = (11.22)*3600      # Moon's synodic period [sec]
Psyn_ga = (10.53)*3600      # Moon's synodic period [sec]

dchi_1s = 2.30     # デルタchi2の1シグマ区間
dchi_2s = 6.17     # デルタchi2の2シグマ区間
dchi_3s = 11.8     # デルタchi2の3シグマ区間


# Threshold scale height
if target_moon == 'Io':
    Psyn = Psyn_io
    H_thres = 3.0*RJ
    r_moon = 5.9*RJ
elif target_moon == 'Europa':
    Psyn = Psyn_eu
    H_thres = 6.0*RJ
    r_moon = 9.4*RJ
elif target_moon == 'Ganymede':
    Psyn = Psyn_ga
    H_thres = 10.0*RJ
    r_moon = 15.0*RJ
PJ_LIST = [3]+np.arange(4, 43+1, 1).tolist()


# deta_load
def data_load(exname):
    chi2_1d = np.loadtxt('results/fit/'+exname+'/params_chi2.txt')
    Ai_1d = np.loadtxt('results/fit/'+exname+'/params_Ai.txt')
    ni_1d = np.loadtxt('results/fit/'+exname+'/params_ni.txt')
    Ti_1d = np.loadtxt('results/fit/'+exname+'/params_Ti.txt')
    H_1d = np.loadtxt('results/fit/'+exname+'/params_H.txt')
    et_obs = np.loadtxt('results/fit/'+exname+'/et_obs.txt')
    eqlead_obs = np.loadtxt('results/fit/'+exname+'/eqlead_obs.txt')
    eqlead_est = np.loadtxt('results/fit/'+exname+'/eqlead_est.txt')
    moon_s3wlon = np.loadtxt('results/fit/'+exname+'/moon_S3wlon_obs.txt')

    # Reshape the data
    chi2_3d = chi2_1d.reshape(ni_num, Ai_num, Ti_num)
    H_3d = H_1d.reshape(ni_num, Ai_num, Ti_num)
    Ai_3d = Ai_1d.reshape(ni_num, Ai_num, Ti_num)
    ni_3d = ni_1d.reshape(ni_num, Ai_num, Ti_num)
    Ti_3d = Ti_1d.reshape(ni_num, Ai_num, Ti_num)
    chi2_3d = chi2_3d*(eqlead_est.shape[0]-3)

    # print('Parameter ranges:')
    # print('Ai:', np.min(Ai_3d), np.max(Ai_3d))
    # print('ni:', np.min(ni_3d), np.max(ni_3d))
    # print('Ti:', np.min(Ti_3d), np.max(Ti_3d))
    # print('Degree of freedom:', (eqlead_est.shape[0]-3))

    # Search the chi2-minimum
    min_idx = np.where(chi2_3d == np.min(chi2_3d))
    min_idx_Ai = 1
    print('Min chi2:', np.min(chi2_3d), 'at', min_idx)
    print('Scale height [RJ]:', H_3d[min_idx][0]/(71492*1E+3))

    # 1データのヒストグラムを作成する
    Ai_2d = Ai_3d[:, min_idx_Ai, :].T
    ni_2d = ni_3d[:, min_idx_Ai, :].T
    H_2d = H_3d[:, min_idx_Ai, :].T
    Ti_2d = Ti_3d[:, min_idx_Ai, :].T
    chi2_2d = chi2_3d[:, min_idx_Ai, :].T
    column_mass_2da = column_mass_3d[:, min_idx_Ai, :].T
    ftmc_mag_2da = ftmc_mag_3d[:, min_idx_Ai, :].T

    # グリッドの面積比を計算する
    x_grid = ni_2d    # shape -> (Ti.size, ni.size)
    y_grid = Ti_2d
    dx_arr = x_grid[:-1, 1:]-x_grid[:-1, :-1]
    dy_arr = y_grid[1:, :-1]-y_grid[:-1, :-1]

    area = dx_arr * dy_arr
    weight = area/np.max(area)

    # 面積arrayに形状を揃える
    column_mass_2da = column_mass_2da[:-1, :-1]
    chi2_2d = chi2_2d[:-1, :-1]

    # 3-sigma area
    d_chi2 = chi2_2d-np.min(chi2_2d)
    Ai_2d = Ai_2d[np.where(d_chi2 < dchi_3s)]
    ni_2d = ni_2d[np.where(d_chi2 < dchi_3s)]
    H_2d = H_2d[np.where(d_chi2 < dchi_3s)]
    Ti_2d = Ti_2d[np.where(d_chi2 < dchi_3s)]
    column_mass_2da = column_mass_2da[np.where(d_chi2 < dchi_3s)]
    weight = weight[np.where(d_chi2 < dchi_3s)]
    ftmc_mag_2da = ftmc_mag_2da[np.where(d_chi2 < dchi_3s)]
    chi2_2d = chi2_2d[np.where(d_chi2 < dchi_3s)]   # 一番最後に

    chi2_R = np.min(chi2_2d)/(eqlead_est.shape[0]-3)
    # print('chi2_R:', chi2_R)

    # 衛星ローカルタイムをリード角の分だけ補正
    moon_et = np.zeros(et_obs.size)
    for i in range(et_obs.size):
        t0 = spice.et2datetime(et_obs[i])
        omg_syn = 360/Psyn  # [deg/sec]
        tau_A = -eqlead_obs[i]/omg_syn  # Alfven travel time [sec]
        dt = datetime.timedelta(seconds=tau_A)
        moon_et[i] = spice.datetime2et(t0+dt)
    return column_mass_2da, chi2_R, moon_et, Ai_2d, moon_s3wlon, weight


# weighted_percentile
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


# weighted_percentile2
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
            [min,
             max],
            color=ec, linewidth=lw,
            zorder=1)

    # Min
    ax.plot([x0-width/2, x0+width/2],
            [min,
             min],
            color=ec, linewidth=lw,
            zorder=1)

    # Max
    ax.plot([x0-width/2, x0+width/2],
            [max,
             max],
            color=ec, linewidth=lw,
            zorder=1)

    return None


# local_time_moon2
def local_time_moon2(et: float, MOON: str, abcorr='none'):
    # Moon's position seen from Jupiter in IAU_JUPITER coordinate.
    pos_moon, _ = spice.spkpos(
        targ=MOON, et=et, ref='JUNO_JSS', abcorr=abcorr, obs='JUPITER'
    )

    # X axis pointing toward the Sun
    x_moon = pos_moon[0]*1E+3
    y_moon = pos_moon[1]*1E+3
    z_moon = pos_moon[2]*1E+3

    r_moon = np.sqrt(x_moon**2 + y_moon**2 + z_moon**2)
    theta_moon = np.arccos(z_moon/r_moon)
    phi_moon = np.arctan2(y_moon, x_moon)

    local_time = (24.0*phi_moon)/(2*np.pi) + 12.0   # [hour]

    return local_time


# %% フィッティング用の三角関数
def fit_func2(params, x):
    a, c, d = params
    return a * np.cos(2*np.pi*(x-c)/24.0) + d


# %% 横軸をPJ番号でプロットする(6)
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=2, figsize=(6.5, 5.5), height_ratios=[1, 0.8], dpi='L')
F.initialize()
# F.panelname = [' a. Io ', ' b. Europa ', ' c. Ganymede ']

sxmin = '2016-01-01'
sxmax = '2023-01-01'
xmin = datetime.datetime.strptime(sxmin, '%Y-%m-%d')
xmax = datetime.datetime.strptime(sxmax, '%Y-%m-%d')
xticks = [datetime.datetime.strptime('2016-01-01', '%Y-%m-%d'),
          datetime.datetime.strptime('2017-01-01', '%Y-%m-%d'),
          datetime.datetime.strptime('2018-01-01', '%Y-%m-%d'),
          datetime.datetime.strptime('2019-01-01', '%Y-%m-%d'),
          datetime.datetime.strptime('2020-01-01', '%Y-%m-%d'),
          datetime.datetime.strptime('2021-01-01', '%Y-%m-%d'),
          datetime.datetime.strptime('2022-01-01', '%Y-%m-%d'),
          datetime.datetime.strptime('2023-01-01', '%Y-%m-%d'),]
xticklabels = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
F.set_xaxis(label='Date',
            min=xmin, max=xmax,
            ticks=xticks,
            ticklabels=xticklabels,
            minor_num=12)
F.ax[1].minorticks_off()
F.ax[1].xaxis.set_minor_locator(mdates.MonthLocator())
ticklabels = F.ax[1].get_xticklabels()
ticklabels[0].set_ha('center')

target_moon_list = ['Io', 'Europa', 'Ganymede']

j = 0
target_moon = target_moon_list[j]

PJ_LIST = [1, 3]+np.arange(4, 43+1, 1).tolist()
if target_moon == 'Io':
    Psyn = Psyn_io
elif target_moon == 'Europa':
    Psyn = Psyn_eu
    PJ_LIST.pop(24-2)
    PJ_LIST.pop(43-3)
elif target_moon == 'Ganymede':
    Psyn = Psyn_ga
    PJ_LIST.pop(24-2)
    PJ_LIST.pop(31-3)
    PJ_LIST.pop(39-4)
    PJ_LIST.pop(43-5)

target_moon = target_moon_list[j]
if (exdate == '003/20250516') and (target_moon == 'Europa'):
    exnum = ['001', '002', '005', '006',
             '007', '010', '011',
             '012', '013', '014', '015', '016',
             '017', '018', '019', '020', '021',
             '022', '023', '024', '025', '026',
             '027', '028', '029', '030', '031',
             '032', '034', '035', '036',
             '037', '038',
             '039', '040', '041', '042'
             ]
    PJ_list = [3, 4, 8, 9,
               10, 13, 14,
               15, 16, 17, 18, 19,
               20, 21, 22, 23, 25,
               26, 27, 28, 29, 30,
               31, 32, 33, 34, 35,
               36, 38, 40, 41,
               7, 7,
               11, 11, 12, 12,
               ]
    Psyn = Psyn_eu
    ymax = 4.8
    ticks = np.arange(0, 4+1, 1)

elif (exdate == '003/20250516') and (target_moon == 'Io'):
    exnum = ['047', '048', '049', '050', '051',
             '052', '053', '054', '055', '056', '057',
             '058', '059',
             '062', '063', '064', '065', '066',
             '067', '068', '069', '070', '071',
             '072', '073', '074', '075', '076',
             '077', '078', '079', '080', '081',
             '083',
             '084', '085', '086', '088',
             '089', '090', '091', '092', '094',
             '095', '096', '097', '098',
             '114', '116',
             ]
    PJ_list = [3, 4, 5, 6, 7,
               8, 8, 9, 9, 10, 10,
               11, 11,
               13, 14, 15, 16, 17,
               18, 19, 20, 21, 22,
               23, 24, 25, 26, 27,
               27, 28, 29, 30, 31,
               32,
               33, 33, 34, 35,
               36, 37, 38, 39, 40,
               41, 42, 43, 43,
               18, 12,
               ]
    Psyn = Psyn_io
    ymax = 40
    ticks = np.arange(0, 40+1, 10)

elif (exdate == '003/20250516') and (target_moon == 'Ganymede'):
    exnum = ['100', '104', '105', '106', '107',
             '108', '109', '110', '111', '112',
             '113', '118', '119', '120', '121',
             '123', '124', '125', '126', '127',
             '128', '129', '131', '132', '134',
             '133', '136', '137', '138', '139',
             '140', '141', '142', '143',
             ]
    PJ_list = [3, 4, 5, 6, 7,
               8, 11, 12, 13, 14,
               15, 16, 17, 18, 19,
               20, 21, 22, 23, 25,
               26, 27, 29, 30, 32,
               33, 34, 34, 35, 37,
               38, 40, 41, 42,
               ]
    Psyn = Psyn_ga
    ymax = 0.24
    ticks = np.arange(0, 0.30, 0.10)

column_mass_1dN = np.loadtxt(
    'results/column_mass/'+exdate+'_'+target_moon+'/col_massdens_1dN.txt')
column_mass_1dS = np.loadtxt(
    'results/column_mass/'+exdate+'_'+target_moon+'/col_massdens_1dS.txt')
ftmc_mag_1dN = np.loadtxt(
    'results/column_mass/'+exdate+'_'+target_moon+'/ftmc_mag_1dN.txt')
ftmc_mag_1dS = np.loadtxt(
    'results/column_mass/'+exdate+'_'+target_moon+'/ftmc_mag_1dS.txt')
column_mass_1d = column_mass_1dN+column_mass_1dS
column_mass_3d = column_mass_1d.reshape(ni_num, Ai_num, Ti_num)
ftmc_mag_1d = ftmc_mag_1dN + ftmc_mag_1dS
ftmc_mag_3d = ftmc_mag_1d.reshape(ni_num, Ai_num, Ti_num)

F.set_yaxis(ax_idx=0,
            label='$M$ [10$^{-9}$ kg m$^{-2}$]',
            min=0, max=ymax,
            ticks=ticks,
            ticklabels=ticks,
            minor_num=5)

positions = np.arange(0, len(exnum)+1, 1)
colormap = plt.cm.get_cmap('turbo')
for i in range(len(exnum)):
    # %% Load the data
    exname = exdate+'_'+exnum[i]
    column_mass, chi2r, moon_et, _, moon_S3wlon, weight = data_load(
        exname)     # [kg m-2]
    column_mass *= 1E+9  # [10^-9 kg m-2]

    # Local time
    d0 = spice.et2datetime(moon_et[0])
    d0_list = []
    for ii in range(column_mass.size):
        d0_list += [d0]
    lt_arr = np.zeros(moon_et.size)
    for k in range(moon_et.size):
        lt_arr[k] = local_time_moon(moon_et[k], target_moon)

    lt_center = (lt_arr[0]+lt_arr[-1])/2

    q1, medians, q3 = weighted_percentile(data=column_mass,
                                          perc=[0.25, 0.5, 0.75],
                                          weights=weight)
    width = datetime.timedelta(seconds=60*60*24*20)
    weighted_boxplot2(F.ax[0], d0, q1, medians, q3,
                      np.min(column_mass),
                      np.max(column_mass), width=width,
                      ec=UC.blue, lw=1.1)


# 2nd axis
F.set_yaxis(ax_idx=1, label='Io LT [hour]',
            min=0, max=24,
            ticks=np.arange(0, 24, 6),
            ticklabels=np.arange(0, 24, 6),
            minor_num=6)

lt_med = np.zeros(len(PJ_LIST))
k = 0
for pj in PJ_LIST:
    wlon_fp, err_wlon_fp, lat_fp, err_lat_fp, moon_S3wlon, et_fp, hem_fp, pj_fp = Obsresults(
        [pj], target_moon, target_fp, TARGET_HEM='both', FLIP=False
    )

    eqlead_fp, eqlead_fp_0, eqlead_fp_1, wlon_TEB_eq = calc_eqlead(wlon_fp,
                                                                   err_wlon_fp,
                                                                   lat_fp,
                                                                   err_lat_fp,
                                                                   hem_fp,
                                                                   moon_S3wlon,
                                                                   target_moon)

    # 衛星ローカルタイムをリード角の分だけ補正
    moon_et = np.zeros(et_fp.size)
    for i in range(et_fp.size):
        t0 = spice.et2datetime(et_fp[i])
        omg_syn = 360/Psyn  # [deg/sec]
        tau_A = -eqlead_fp[i]/omg_syn  # Alfven travel time [sec]
        dt = datetime.timedelta(seconds=tau_A)
        moon_et[i] = spice.datetime2et(t0+dt)

    lt_arr = np.zeros(moon_et.size)
    d0 = spice.et2datetime(moon_et[0])
    d0_list = []
    for i in range(lt_arr.size):
        lt_arr[i] = local_time_moon2(moon_et[i], target_moon)
        d0_list += [d0]
    lt_med[k] = np.median(lt_arr)
    k += 1

    F.ax[1].scatter(d0_list, lt_arr,
                    s=5.0, marker='s', color=UC.blue,
                    linewidth=0.2, zorder=2.0)


# u_ax = F.upper_ax()
# u_ax.set_title(r'Flux tube mass contents',
#               fontsize=F.fontsize, weight='bold')

PJax = F.ax[0].twiny()
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
          datetime.datetime.strptime('2022-11-06', '%Y-%m-%d'),
          datetime.datetime.strptime('2022-12-15', '%Y-%m-%d'),]
xticklabels = ['PJ1', '', '', '', '',
               '6', '', '', '', '',
               '11', '', '', '', '',
               '16', '', '', '', '',
               '21', '', '', '', '',
               '26', '', '', '', '',
               '31', '', '', '', '',
               '36', '', '', '', '',
               '41', '', '', '', '',
               '46', '',]
PJax.set_xlim(xmin, xmax)
PJax.set_xticks(xticks[::5])
PJax.set_xticklabels(xticklabels[::5])
PJax.xaxis.set_minor_locator(FixedLocator(mdates.date2num(xticks)))
PJax.tick_params('y', grid_zorder=-10)

# Shades in each 5 perijove
for i in range(2):
    F.ax[i].axvspan(xticks[0], xticks[5], fc=UC.gray, ec=None, alpha=0.15)
    F.ax[i].axvspan(xticks[10], xticks[15], fc=UC.gray, ec=None, alpha=0.15)
    F.ax[i].axvspan(xticks[20], xticks[25], fc=UC.gray, ec=None, alpha=0.15)
    F.ax[i].axvspan(xticks[30], xticks[35], fc=UC.gray, ec=None, alpha=0.15)
    F.ax[i].axvspan(xticks[40], xticks[45], fc=UC.gray, ec=None, alpha=0.15)

F.fig.savefig('img/ftmc_lt_Io_r.pdf', bbox_inches='tight')
plt.show()


# %% Morgenthaler+ 2024の結果と比較
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=4, figsize=(7.5, 10.5), dpi='L')
F.initialize()
# F.panelname = [' a. Io ', ' b. Europa ', ' c. Ganymede ']

sxmin = '2016-01-01'
sxmax = '2024-01-01'
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
          datetime.datetime.strptime('2024-01-01', '%Y-%m-%d')]
xticklabels = ['2016', '2017', '2018', '2019', '2020',
               '2021', '2022', '2023', '2024']
F.set_xaxis(label='Date',
            min=xmin, max=xmax,
            ticks=xticks,
            ticklabels=xticklabels,
            minor_num=12)
F.ax[-1].minorticks_off()
F.ax[-1].xaxis.set_minor_locator(mdates.MonthLocator())
ticklabels = F.ax[-1].get_xticklabels()
ticklabels[0].set_ha('center')

target_moon_list = ['Io', 'Europa', 'Ganymede']

j = 0
target_moon = target_moon_list[j]

PJ_LIST = [1, 3]+np.arange(4, 43+1, 1).tolist()
if target_moon == 'Io':
    Psyn = Psyn_io
elif target_moon == 'Europa':
    Psyn = Psyn_eu
    PJ_LIST.pop(24-2)
    PJ_LIST.pop(43-3)
elif target_moon == 'Ganymede':
    Psyn = Psyn_ga
    PJ_LIST.pop(24-2)
    PJ_LIST.pop(31-3)
    PJ_LIST.pop(39-4)
    PJ_LIST.pop(43-5)

target_moon = target_moon_list[j]
if (exdate == '003/20250516') and (target_moon == 'Europa'):
    exnum = ['001', '002', '005', '006',
             '007', '010', '011',
             '012', '013', '014', '015', '016',
             '017', '018', '019', '020', '021',
             '022', '023', '024', '025', '026',
             '027', '028', '029', '030', '031',
             '032', '034', '035', '036',
             '037', '038',
             '039', '040', '041', '042'
             ]
    PJ_list = [3, 4, 8, 9,
               10, 13, 14,
               15, 16, 17, 18, 19,
               20, 21, 22, 23, 25,
               26, 27, 28, 29, 30,
               31, 32, 33, 34, 35,
               36, 38, 40, 41,
               7, 7,
               11, 11, 12, 12,
               ]
    Psyn = Psyn_eu
    ymax = 4.8
    ticks = np.arange(0, 4+1, 1)

elif (exdate == '003/20250516') and (target_moon == 'Io'):
    exnum = ['047', '048', '049', '050', '051',
             '052', '053', '054', '055', '056', '057',
             '058', '059',
             '062', '063', '064', '065', '066',
             '067', '068', '069', '070', '071',
             '072', '073', '074', '075', '076',
             '077', '078', '079', '080', '081',
             '083',
             '084', '085', '086', '088',
             '089', '090', '091', '092', '094',
             '095', '096', '097', '098',
             '114', '116',
             ]
    PJ_list = [3, 4, 5, 6, 7,
               8, 8, 9, 9, 10, 10,
               11, 11,
               13, 14, 15, 16, 17,
               18, 19, 20, 21, 22,
               23, 24, 25, 26, 27,
               27, 28, 29, 30, 31,
               32,
               33, 33, 34, 35,
               36, 37, 38, 39, 40,
               41, 42, 43, 43,
               18, 12,
               ]
    Psyn = Psyn_io
    ymax = 40
    ticks = np.arange(0, 40+1, 10)

elif (exdate == '003/20250516') and (target_moon == 'Ganymede'):
    exnum = ['100', '104', '105', '106', '107',
             '108', '109', '110', '111', '112',
             '113', '118', '119', '120', '121',
             '123', '124', '125', '126', '127',
             '128', '129', '131', '132', '134',
             '133', '136', '137', '138', '139',
             '140', '141', '142', '143',
             ]
    PJ_list = [3, 4, 5, 6, 7,
               8, 11, 12, 13, 14,
               15, 16, 17, 18, 19,
               20, 21, 22, 23, 25,
               26, 27, 29, 30, 32,
               33, 34, 34, 35, 37,
               38, 40, 41, 42,
               ]
    Psyn = Psyn_ga
    ymax = 0.24
    ticks = np.arange(0, 0.30, 0.10)

column_mass_1dN = np.loadtxt(
    'results/column_mass/'+exdate+'_'+target_moon+'/col_massdens_1dN.txt')
column_mass_1dS = np.loadtxt(
    'results/column_mass/'+exdate+'_'+target_moon+'/col_massdens_1dS.txt')
ftmc_mag_1dN = np.loadtxt(
    'results/column_mass/'+exdate+'_'+target_moon+'/ftmc_mag_1dN.txt')
ftmc_mag_1dS = np.loadtxt(
    'results/column_mass/'+exdate+'_'+target_moon+'/ftmc_mag_1dS.txt')
column_mass_1d = column_mass_1dN+column_mass_1dS
column_mass_3d = column_mass_1d.reshape(ni_num, Ai_num, Ti_num)
ftmc_mag_1d = ftmc_mag_1dN + ftmc_mag_1dS
ftmc_mag_3d = ftmc_mag_1d.reshape(ni_num, Ai_num, Ti_num)

F.set_yaxis(ax_idx=0,
            label='$M$ [10$^{-9}$ kg m$^{-2}$]',
            min=0, max=ymax,
            ticks=ticks,
            ticklabels=ticks,
            minor_num=5)
F.set_yaxis(ax_idx=1,
            label='$\Delta_M$ [10$^{-9}$ kg m$^{-2}$]',
            min=-10.0, max=30.0,
            ticks=np.array([0, 10, 20, 30]),
            ticklabels=np.array([0, 10, 20, 30]),
            minor_num=5)

positions = np.arange(0, len(exnum)+1, 1)
colormap = plt.cm.get_cmap('turbo')
for i in range(len(exnum)):
    # %% Load the data
    exname = exdate+'_'+exnum[i]
    column_mass, chi2r, moon_et, _, moon_S3wlon, weight = data_load(
        exname)     # [kg m-2]
    column_mass *= 1E+9  # [10^-9 kg m-2]

    # Local time
    d0 = spice.et2datetime(moon_et[0])
    d0_list = []
    for ii in range(column_mass.size):
        d0_list += [d0]
    lt_arr = np.zeros(moon_et.size)
    for k in range(moon_et.size):
        lt_arr[k] = local_time_moon(moon_et[k], target_moon)

    lt_center = (lt_arr[0]+lt_arr[-1])/2

    # LT依存性ベストフィット
    a_best = 3.22
    b_best = 3.70
    c_best = 13.23
    popt = (a_best, b_best, c_best)
    y_bestfit = fit_func2(popt, lt_center)

    # Calc percentile
    q1, medians, q3 = weighted_percentile(data=column_mass,
                                          perc=[0.25, 0.5, 0.75],
                                          weights=weight)
    width = datetime.timedelta(seconds=60*60*24*20)
    weighted_boxplot2(F.ax[0], d0, q1, medians, q3,
                      np.min(column_mass),
                      np.max(column_mass),
                      width=width,
                      ec=UC.orange2, lw=1.1)

    q1, medians, q3 = weighted_percentile(data=column_mass-y_bestfit,
                                          perc=[0.25, 0.5, 0.75],
                                          weights=weight)
    width = datetime.timedelta(seconds=60*60*24*20)
    weighted_boxplot2(F.ax[1], d0, q1, medians, q3,
                      np.min(column_mass-y_bestfit),
                      np.max(column_mass-y_bestfit),
                      width=width,
                      ec=UC.orange2, lw=1.1)

# 3rd & 4th axis
F.set_yaxis(ax_idx=2, label='Brightnees [R]',
            min=0, max=300,
            ticks=np.arange(50, 300+1, 50),
            ticklabels=np.arange(50, 300+1, 50),
            minor_num=5)
F.set_yaxis(ax_idx=3, label='Brightnees [R]',
            min=0, max=300,
            ticks=np.arange(0, 300+1, 50),
            ticklabels=np.arange(0, 300+1, 50),
            minor_num=5)
datapath = "/home/shinnosukesatoh/shinnosukesatoh/JunoUVS/data/jpm_data/Torus.ecsv"
data = Table.read(datapath, format='ascii.ecsv')
ansa_left_brightness = data['ansa_left_surf_bright']
ansa_left_berr = data['ansa_left_surf_bright_err']
ansa_left_SIII = data['Jupiter_PDObsLon']+90.0
ansa_left_SIII = np.where(ansa_left_SIII > 360.0,
                          ansa_left_SIII-360.0, ansa_left_SIII)

ansa_right_brightness = data['ansa_right_surf_bright']
ansa_right_berr = data['ansa_right_surf_bright_err']
ansa_right_SIII = data['Jupiter_PDObsLon']-90.0
ansa_right_SIII = np.where(ansa_right_SIII < 0.0,
                           ansa_right_SIII+360.0, ansa_right_SIII)
data_len = data['tavg'].size
tavg_datetime = []
for i in range(data_len):
    t = data['tavg'][i]
    tavg_datetime += [t.tt.datetime]

F.ax[2].scatter(tavg_datetime, ansa_right_brightness,
                s=1, marker='.', fc=UC.red, label='Dusk')
F.ax[2].scatter(tavg_datetime, ansa_left_brightness, s=1,
                marker='.', fc=UC.blue, label='Dawn')
F.ax[2].errorbar(x=tavg_datetime, y=ansa_right_brightness,
                 yerr=ansa_right_berr,
                 elinewidth=0.3, linewidth=0., markersize=0,
                 color=UC.red)
F.ax[2].errorbar(x=tavg_datetime, y=ansa_left_brightness,
                 yerr=ansa_left_berr,
                 elinewidth=0.3, linewidth=0., markersize=0,
                 color=UC.blue)


datapath = "/home/shinnosukesatoh/shinnosukesatoh/JunoUVS/data/jpm_data/Na_nebula.ecsv"
data = Table.read(datapath, format='ascii.ecsv')
r_60rj_biweight = data['biweight_largest_sub_annular_sb_6.0_jupiterRad']
r_120rj_biweight = data['biweight_largest_sub_annular_sb_12.0_jupiterRad']
r_240rj_biweight = data['biweight_largest_sub_annular_sb_24.0_jupiterRad']
data2_len = data['tavg'].size
tavg_datetime2 = []
for i in range(data2_len):
    t = data['tavg'][i]
    tavg_datetime2 += [t.tt.datetime]
cmap_turbo = plt.get_cmap('turbo')
F.ax[3].scatter(tavg_datetime2, r_60rj_biweight,
                s=1, marker='.', fc=cmap_turbo(30),
                label=r'6.0 $R_{\rm J}$')
F.ax[3].scatter(tavg_datetime2, r_120rj_biweight,
                s=1, marker='.', fc=cmap_turbo(100),
                label=r'12.0 $R_{\rm J}$')
F.ax[3].scatter(tavg_datetime2, r_240rj_biweight,
                s=1, marker='.', fc=cmap_turbo(170),
                label=r'24.0 $R_{\rm J}$')

PJax = F.ax[0].twiny()
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
          # datetime.datetime.strptime('2023-12-30', '%Y-%m-%d'),
          # datetime.datetime.strptime('2024-02-04', '%Y-%m-%d'),  # PJ58
          # atetime.datetime.strptime('2024-03-07', '%Y-%m-%d'),
          # datetime.datetime.strptime('2024-04-09', '%Y-%m-%d'),
          # datetime.datetime.strptime('2024-05-12', '%Y-%m-%d'),
          # datetime.datetime.strptime('2024-06-14', '%Y-%m-%d'),
          # datetime.datetime.strptime('2024-07-16', '%Y-%m-%d'),
          # datetime.datetime.strptime('2024-08-18', '%Y-%m-%d'),  # PJ64
          # datetime.datetime.strptime('2024-09-20', '%Y-%m-%d'),
          # datetime.datetime.strptime('2024-10-23', '%Y-%m-%d'),
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
               '56']
PJax.set_xlim(xmin, xmax)
PJax.set_xticks(xticks[::5])
PJax.set_xticklabels(xticklabels[::5])
PJax.xaxis.set_minor_locator(FixedLocator(mdates.date2num(xticks)))
PJax.tick_params('y', grid_zorder=-10)

for i in range(2):
    F.textbox(ax_idx=i, x=0.13, y=0.88,
              text='Io FTMC', fontsize=F.fontsize*0.63,
              horizontalalignment='center',
              facealpha=1.0, facecolor=UC.orange2,
              transform=F.ax[i].transAxes,
              edgecolor=(0, 0, 0, 1), )
F.textbox(ax_idx=1, x=0.50, y=0.89,
          text=r'$\Delta_M \equiv M-(3.22\cos(\frac{2\pi}{24}(LT-3.70))+13.23)$',
          fontsize=F.fontsize*0.6,
          horizontalalignment='center',
          textcolor='k',
          textshadow=False,
          facealpha=0.0,
          transform=F.ax[1].transAxes,
          edgecolor=(0, 0, 0, 0), )
F.textbox(ax_idx=2, x=0.13, y=0.88,
          text='Ribbon SII', fontsize=F.fontsize*0.63,
          horizontalalignment='center',
          facealpha=0.5, facecolor='k',
          transform=F.ax[2].transAxes,
          edgecolor=(0, 0, 0, 1.0), )
F.textbox(ax_idx=3, x=0.13, y=0.88,
          text='Na nebula', fontsize=F.fontsize*0.63,
          horizontalalignment='center',
          facealpha=0.5, facecolor='k',
          transform=F.ax[3].transAxes,
          edgecolor=(0, 0, 0, 1.0), )

# Shades in each 5 perijove
for i in range(4):
    F.ax[i].axvspan(xticks[0], xticks[5], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[10], xticks[15], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[20], xticks[25], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[30], xticks[35], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[40], xticks[45], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[50], xticks[55], fc=UC.gray, ec=None, alpha=0.10)

legend = F.legend(ax_idx=2,
                  ncol=3, markerscale=10.0,
                  loc='upper right',
                  handlelength=1.0,
                  fontsize_scale=0.65, handletextpad=0.2)
legend_shadow(legend=legend, fig=F.fig, ax=F.ax[2], d=0.7)

legend = F.legend(ax_idx=3,
                  ncol=3, markerscale=10.0,
                  loc='upper right',
                  handlelength=1.0,
                  fontsize_scale=0.65, handletextpad=0.2)
legend_shadow(legend=legend, fig=F.fig, ax=F.ax[3], d=0.7)

F.fig.savefig('img/ftmc_Io_jpm2024.pdf', bbox_inches='tight')
plt.show()
