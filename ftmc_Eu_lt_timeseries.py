import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.patches as patches
import matplotlib.ticker as ptick
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
from Leadangle_fit_JunoUVS import spice_moonS3

from scipy.odr import ODR, Model, RealData
from scipy.stats import spearmanr
from scipy.stats import t

import spiceypy as spice
spice.furnsh('kernel/cassMetaK.txt')
savpath = 'data/Satellite_FP_JRM33.sav'

UC = UniversalColor()
UC.set_palette()


exdate = '005/20250923'
target_moon = 'Ganymede'
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
    # print('Min chi2:', np.min(chi2_3d), 'at', min_idx)
    # print('Scale height [RJ]:', H_3d[min_idx][0]/(71492*1E+3))

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


# weighted_percentile2
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


# %% フィッティング用の一次関数
def fit_linear(params, x):
    a, b = params
    return a * x + b


#
#
#
#
#
#
#
#
#
# %% 横軸をPJ番号でプロットする(6)
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=2, figsize=(8.2, 6.5), dpi='L')
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
for i in range(F.ax.size):
    F.ax[i].minorticks_off()
    F.ax[i].xaxis.set_minor_locator(mdates.MonthLocator())
ticklabels = F.ax[i].get_xticklabels()
ticklabels[0].set_ha('center')

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
    FTMC_HEM = ['S', 'S', 'both', 'S',
                'N', 'N', 'S',
                'S', 'S', 'both', 'S', 'both',
                'both', 'S', 'both', 'S', 'S',
                'both', 'N', 'N', 'both', 'S',
                'S', 'both', 'S', 'S', 'S',
                'N', 'S', 'S', 'S',
                'N', 'S',
                'N', 'S', 'N', 'S']
    Psyn = Psyn_eu
    ymax = 4.0
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
    FTMC_HEM = ['both', 'both', 'both', 'both', 'both',
                'N', 'S', 'N', 'S', 'N', 'S',
                'N', 'S',
                'N', 'S', 'N', 'both', 'both',
                'N', 'both', 'both', 'both', 'both',
                'S', 'S', 'S', 'both', 'N',
                'S', 'S', 'S', 'S', 'S',
                'N',
                'N', 'S', 'S', 'S',
                'S', 'both', 'both', 'both', 'both',
                'both', 'both', 'N', 'S',
                'S', 'both']
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
    FTMC_HEM = ['both', 'both', 'S', 'both', 'S',
                'both', 'both', 'both', 'both', 'both',
                'N', 'S', 'S', 'S', 'both',
                'both', 'both', 'both', 'both', 'S',
                'both', 'both', 'S', 'both', 'both',
                'both', 'N', 'S', 'both', 'both',
                'S', 'S', 'N', 'N']
    Psyn = Psyn_ga
    ymax = 0.24
    ticks = np.arange(0, 0.30, 0.10)

elif (exdate == '005/20250923') and (target_moon == 'Europa'):
    exnum = ['024', '025', '003', '004', '008',
             '009', '007', '010', '011', '012',
             '051', '053', '014', '015',
             # '016', '017',    # ここは本当に使わない
             '018', '020', '021', '022',
             '023', '026', '029', '030', '052',
             # '032',   # ここは本当に使わない
             '033', '034', '035', '036', '037', '040',
             '041', '042', '043', '044', '045', '046',
             # '047',   # ここは本当に使わない
             '048', '050',
             ]
    PJ_list = [3, 4, 5, 7, 8,
               8, 9, 10, 11, 11,
               12, 12, 13, 14,
               # 15, 16,    # ここは本当に使わない
               17, 18, 19, 20,
               21, 22, 23, 25, 26,
               # 27,    # ここは本当に使わない
               28, 29, 30, 31, 32, 33,
               34, 35, 36, 38, 40, 41,
               # 45,    # ここは本当に使わない
               48, 62,
               ]
    FTMC_HEM = ['S', 'S', 'N', 'S', 'N',
                'S', 'S', 'N', 'N', 'S',
                'N', 'S', 'N', 'S',
                # 'S', 'S',    # ここは本当に使わない
                'both', 'S', 'both', 'both',
                'S', 'both', 'S', 'S', 'S',
                # 'N',    # ここは本当に使わない
                'N', 'both', 'S', 'S', 'both', 'S',
                'S', 'S', 'N', 'S', 'S', 'S',
                # 'S',    # ここは本当に使わない
                'S', 'S',
                ]
    Psyn = Psyn_eu
    ymax = 3.0
    ticks = np.arange(0, 3+1, 1)

elif (exdate == '005/20250923') and (target_moon == 'Ganymede'):
    exnum = ['054', '055', '056', '057', '058',
             '059', '063', '065', '068',  # '070',
             '071', '072', '073', '075', '076',
             '078', '079', '081', '082', '083',
             '084', '085', '086', '087', '088',
             '089', '090', '091', '092', '093',
             '094', '095', '096', '097', '098',
             '099', '100', '101', '102', '103',
             '104', '105', '106',
             # '107',
             # '108', '109',
             ]
    PJ_list = [3, 4, 5, 6, 7,
               8, 11, 12, 13,  # 14,
               15, 16, 17, 19, 20,
               21, 21, 22, 23, 25,
               26, 27, 29, 30, 32,
               32, 33, 34, 34, 35,
               37, 38, 40, 41, 42,
               46, 47, 48, 49, 50,
               58, 59, 60,
               # 22,
               # 3, 3,
               ]
    FTMC_HEM = ['both', 'both', 'S', 'both', 'S',
                'both', 'N', 'both', 'both',  # 'S',
                'N', 'S', 'S', 'S', 'N',
                'N', 'S', 'N', 'both', 'S',
                'S', 'both', 'S', 'S', 'N',
                'S', 'both', 'N', 'S', 'both',
                'S', 'S', 'S', 'N', 'N',
                'S', 'S', 'S', 'S', 'S',
                'S', 'S', 'N',
                # 'S',
                # 'N', 'S',
                ]
    Psyn = Psyn_ga
    ymax = 0.2
    ticks = np.round(np.arange(0, 0.20+0.01, 0.05), 2)

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

x_arr = np.zeros(3)
y_arr = np.zeros(3)
x_err_arr = np.zeros(3)
medians_arr = np.zeros(len(exnum))
q05_arr = np.zeros(len(exnum))
q25_arr = np.zeros(len(exnum))
q75_arr = np.zeros(len(exnum))
q95_arr = np.zeros(len(exnum))
medians_x = np.zeros(len(exnum))
medians_x_err = np.zeros(len(exnum))
y_sigma_arr = np.zeros(len(exnum))
jj = 0
for ii in range(2):
    for i in range(len(exnum)):
        # %% Load the data
        exname = exdate+'_'+exnum[i]
        column_mass, chi2r, moon_et, _, moon_S3wlon, weight = data_load(
            exname)     # [kg m-2]
        column_mass *= 1E+9  # [10^-9 kg m-2]

        # Local time
        lt_arr = np.zeros(moon_et.size)
        for k in range(moon_et.size):
            lt_arr[k] = local_time_moon(moon_et[k], target_moon)

        lt_center = ((lt_arr[0]+lt_arr[-1])/2)+24.0*ii
        lt_range = abs(lt_arr[0]-lt_arr[-1])

        q25, medians, q75 = weighted_percentile(data=column_mass,
                                                perc=[0.25, 0.5, 0.75],
                                                weights=weight)
        q05, medians, q95 = weighted_percentile(data=column_mass,
                                                perc=[0.05, 0.5, 0.95],
                                                weights=weight)

        if jj < len(exnum):
            x_arr = np.append(x_arr, lt_center*np.ones(column_mass.size))
            y_arr = np.append(y_arr, column_mass)
            x_err_arr = np.append(x_err_arr, lt_range *
                                  np.ones(column_mass.size))

            medians_arr[jj] = medians
            q05_arr[jj] = q05
            q25_arr[jj] = q25
            q75_arr[jj] = q75
            q95_arr[jj] = q95
            medians_x[jj] = lt_center
            medians_x_err[jj] = lt_range
            y_sigma_arr[jj] = np.std(column_mass)
            jj += 1

x_arr = medians_x
y_arr = medians_arr
x_err_arr = medians_x_err/2
y_err_arr = y_sigma_arr

sort = np.argsort(x_arr)[::-1]
x_arr, y_arr, x_err_arr, y_err_arr = x_arr[sort], y_arr[sort], x_err_arr[sort], y_err_arr[sort]

# ODR 用データとモデルの設定
data = RealData(x_arr, y_arr, sx=x_err_arr, sy=y_err_arr)
model = Model(fit_func2)

# ODR 実行
odr_instance = ODR(data, model, beta0=[1.0, 1.0, 1.0])
output = odr_instance.run()

# フィッティング結果
popt = output.beta
perr = output.sd_beta

F.set_yaxis(ax_idx=0,
            label='FTMC [10$^{-9}$ kg m$^{-2}$]',
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
if target_moon == 'Io':
    ymin = 5
    ymax = 8
    yticks = np.arange(ymin, ymax+1, 1)
    yticklabels = np.arange(ymin, ymax+1, 1)
    minor_num = 5
elif target_moon == 'Europa':
    ymin = 8
    ymax = 12
    yticks = np.arange(ymin, ymax+1, 1)
    yticklabels = np.arange(ymin, ymax+1, 1)
    minor_num = 5
elif target_moon == 'Ganymede':
    ymin = 11
    ymax = 25
    yticks = np.arange(ymin, ymax+1, 2)
    yticklabels = np.arange(ymin, ymax+1, 2)
    minor_num = 2
F.set_yaxis(ax_idx=1,
            label='M shell',
            min=ymin, max=ymax,
            ticks=yticks[:-1],
            ticklabels=yticklabels[:-1],
            minor_num=minor_num)

# Load the M shell data
rho_arr = np.loadtxt('results/Mshell/'+target_moon[0:2]+'/mshell.txt')
rho_arr_1 = np.loadtxt('results/Mshell/'+target_moon[0:2]+'/mshell_1.txt')
rho_arr_2 = np.loadtxt('results/Mshell/'+target_moon[0:2]+'/mshell_2.txt')
rho_arr_3 = np.loadtxt('results/Mshell/'+target_moon[0:2]+'/mshell_3.txt')
rho_arr_4 = np.loadtxt('results/Mshell/'+target_moon[0:2]+'/mshell_4.txt')
pj_fp_m = np.loadtxt('results/Mshell/'+target_moon[0:2]+'/pj_fp.txt')
hem_fp_m = np.loadtxt('results/Mshell/'+target_moon[0:2]+'/hem_fp.txt')
view_angle = np.loadtxt('results/Mshell/'+target_moon[0:2]+'/view_angle.txt')
et_fp_m = np.loadtxt('results/Mshell/'+target_moon[0:2]+'/et_fp.txt')
mu_i_coef = np.loadtxt('results/azimuthal_current_fit/' +
                       target_moon[0:2]+'_coef_0.txt')
mu_i_coef_1 = np.loadtxt('results/azimuthal_current_fit/' +
                         target_moon[0:2]+'_coef_1.txt')
mu_i_coef_2 = np.loadtxt('results/azimuthal_current_fit/' +
                         target_moon[0:2]+'_coef_2.txt')
mu_i_coef_3 = np.loadtxt('results/azimuthal_current_fit/' +
                         target_moon[0:2]+'_coef_3.txt')
mu_i_coef_4 = np.loadtxt('results/azimuthal_current_fit/' +
                         target_moon[0:2]+'_coef_4.txt')

# Data subsetごとに値をまとめる
d0_median = []
rho_ave_arr = np.zeros(len(PJ_list))
rho_1_ave_arr = np.zeros(len(PJ_list))
mu_i_coef_ave = np.zeros(len(PJ_list))
mu_i_coef_1_ave = np.zeros(len(PJ_list))
for i in range(len(PJ_list)):
    ftmc_pj = PJ_list[i]
    ftmc_hem = FTMC_HEM[i]
    pj_idx = np.where(pj_fp_m == ftmc_pj)

    if ftmc_hem == 'both':
        hem_subset = hem_fp_m[pj_idx]
        pj_subset = pj_fp_m[pj_idx]
        rho_arr_subset = rho_arr[pj_idx]
        rho_arr_1_subset = rho_arr_1[pj_idx]-rho_arr_subset
        rho_arr_2_subset = rho_arr_2[pj_idx]-rho_arr_subset
        rho_arr_3_subset = rho_arr_3[pj_idx]-rho_arr_subset
        rho_arr_4_subset = rho_arr_4[pj_idx]-rho_arr_subset
        d0_subset = et_fp_m[pj_idx]
        view_angle_subset = view_angle[pj_idx]
        mu_i_coef_subset = mu_i_coef[pj_idx]
        mu_i_coef_1_subset = mu_i_coef_1[pj_idx]-mu_i_coef_subset
        mu_i_coef_2_subset = mu_i_coef_2[pj_idx]-mu_i_coef_subset
        mu_i_coef_3_subset = mu_i_coef_3[pj_idx]-mu_i_coef_subset
        mu_i_coef_4_subset = mu_i_coef_4[pj_idx]-mu_i_coef_subset
    else:
        if ftmc_hem == 'S':
            hem_subset_idx = np.where(hem_fp_m[pj_idx] > 0)
        elif ftmc_hem == 'N':
            hem_subset_idx = np.where(hem_fp_m[pj_idx] < 0)
        hem_subset = hem_fp_m[pj_idx][hem_subset_idx]
        pj_subset = pj_fp_m[pj_idx][hem_subset_idx]
        rho_arr_subset = rho_arr[pj_idx][hem_subset_idx]
        rho_arr_1_subset = rho_arr_1[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_2_subset = rho_arr_2[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_3_subset = rho_arr_3[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_4_subset = rho_arr_4[pj_idx][hem_subset_idx]-rho_arr_subset
        d0_subset = et_fp_m[pj_idx][hem_subset_idx]
        view_angle_subset = view_angle[pj_idx][hem_subset_idx]
        mu_i_coef_subset = mu_i_coef[pj_idx][hem_subset_idx]
        mu_i_coef_1_subset = mu_i_coef_1[pj_idx][hem_subset_idx] - \
            mu_i_coef_subset
        mu_i_coef_2_subset = mu_i_coef_2[pj_idx][hem_subset_idx] - \
            mu_i_coef_subset
        mu_i_coef_3_subset = mu_i_coef_3[pj_idx][hem_subset_idx] - \
            mu_i_coef_subset
        mu_i_coef_4_subset = mu_i_coef_4[pj_idx][hem_subset_idx] - \
            mu_i_coef_subset

    view_angle_thres = np.where(view_angle_subset < 30.0)

    rho_arr_1_subset = np.average(np.abs(rho_arr_1_subset[view_angle_thres]))
    rho_arr_2_subset = np.average(np.abs(rho_arr_2_subset[view_angle_thres]))
    rho_arr_3_subset = np.average(np.abs(rho_arr_3_subset[view_angle_thres]))
    rho_arr_4_subset = np.average(np.abs(rho_arr_4_subset[view_angle_thres]))
    mu_i_coef_1_subset = np.average(
        np.abs(mu_i_coef_1_subset[view_angle_thres]))
    mu_i_coef_2_subset = np.average(
        np.abs(mu_i_coef_2_subset[view_angle_thres]))
    mu_i_coef_3_subset = np.average(
        np.abs(mu_i_coef_3_subset[view_angle_thres]))
    mu_i_coef_4_subset = np.average(
        np.abs(mu_i_coef_4_subset[view_angle_thres]))

    rho_ave_arr[i] = np.average(rho_arr_subset[view_angle_thres])
    rho_1_ave_arr[i] = np.average(
        [rho_arr_1_subset, rho_arr_2_subset, rho_arr_3_subset, rho_arr_4_subset])
    d0_median += [spice.et2datetime(np.median(d0_subset))]
    mu_i_coef_ave[i] = np.average(mu_i_coef_subset[view_angle_thres])
    mu_i_coef_1_ave[i] = np.average(
        [mu_i_coef_1_subset, mu_i_coef_2_subset, mu_i_coef_3_subset, mu_i_coef_4_subset])

# et_fpをdatetimeに変換する
datetime_fp = []
for i in range(et_fp_m.size):
    d0 = spice.et2datetime(et_fp_m[i])
    datetime_fp += [d0]
datetime_fp = np.array(datetime_fp)

# Satellite orbit
F.ax[1].axhline(y=r_moon/RJ, linestyle='dashed',
                color=UC.lightgray, zorder=0.9)

# Data
F.ax[1].scatter(d0_median, rho_ave_arr,
                marker='s', s=5.0, c=UC.blue, label='North')
F.ax[1].errorbar(x=d0_median, y=rho_ave_arr,
                 yerr=rho_1_ave_arr,
                 elinewidth=1.1, linewidth=0., markersize=0,
                 color=UC.blue)

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
               '66']
PJax.set_xlim(xmin, xmax)
PJax.set_xticks(xticks[::5])
PJax.set_xticklabels(xticklabels[::5])
PJax.xaxis.set_minor_locator(FixedLocator(mdates.date2num(xticks)))
PJax.tick_params('y', grid_zorder=-10)

# Shades in each 5 perijove
for i in range(2):
    F.ax[i].axvspan(xticks[0], xticks[5], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[10], xticks[15], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[20], xticks[25], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[30], xticks[35], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[40], xticks[45], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[50], xticks[55], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[60], xticks[65], fc=UC.gray, ec=None, alpha=0.10)

savedir = 'img/ftmc/'+target_moon[0:2]+'/'+exdate
F.fig.savefig(savedir+'/ftmc_lt_'+target_moon[0:2]+'_r.jpg',
              bbox_inches='tight')
F.close()
plt.show()


#
#
#
#
#
#
#
#
#
#
# %% 横軸 FTMC & 縦軸 M shell
# Color code: local time
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(6.5, 5.5), dpi='L')
F.initialize()
# F.panelname = [' a. Io ', ' b. Europa ', ' c. Ganymede ']

if target_moon == 'Io':
    xmin = 0
    xmax = 40
    ymin = 5
    ymax = 8
    xticks = np.arange(xmin, xmax+1, 10)
    xticklabels = np.arange(xmin, xmax+1, 10)
    yticks = np.arange(ymin, ymax+1, 1)
    yticklabels = np.arange(ymin, ymax+1, 1)
    minor_num = 5
    boxplot_width = 0.03
elif target_moon == 'Europa':
    xmin = 0
    xmax = 3
    ymin = 8
    ymax = 12
    xticks = np.arange(xmin, xmax+1, 1)
    xticklabels = np.arange(xmin, xmax+1, 1)
    yticks = np.arange(ymin, ymax+1, 1)
    yticklabels = np.arange(ymin, ymax+1, 1)
    minor_num = 5
    boxplot_width = 0.05
elif target_moon == 'Ganymede':
    xmin = 0
    xmax = 0.20
    ymin = 11
    ymax = 25
    xticks = np.linspace(xmin, xmax, 5)
    xticklabels = np.round(np.linspace(xmin, xmax, 5), 2)
    yticks = np.arange(ymin, ymax+1, 2)
    yticklabels = np.arange(ymin, ymax+1, 2)
    minor_num = 2
    boxplot_width = 0.20

F.set_xaxis(label=target_moon+'-FTMC [10$^{-9}$ kg m$^{-2}$]',
            min=xmin, max=xmax,
            ticks=xticks,
            ticklabels=xticklabels,
            minor_num=5)
F.set_yaxis(ax_idx=0,
            label=r'M shell',
            min=ymin, max=ymax,
            ticks=yticks,
            ticklabels=yticklabels,
            minor_num=minor_num)

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


# Data subsetごとに値をまとめる
d0_median = []
rho_ave_arr = np.zeros(len(PJ_list))
rho_1_ave_arr = np.zeros(len(PJ_list))
for i in range(len(PJ_list)):
    ftmc_pj = PJ_list[i]
    ftmc_hem = FTMC_HEM[i]
    pj_idx = np.where(pj_fp_m == ftmc_pj)

    if ftmc_hem == 'both':
        hem_subset = hem_fp_m[pj_idx]
        pj_subset = pj_fp_m[pj_idx]
        rho_arr_subset = rho_arr[pj_idx]
        rho_arr_1_subset = rho_arr_1[pj_idx]-rho_arr_subset
        rho_arr_2_subset = rho_arr_2[pj_idx]-rho_arr_subset
        rho_arr_3_subset = rho_arr_3[pj_idx]-rho_arr_subset
        rho_arr_4_subset = rho_arr_4[pj_idx]-rho_arr_subset
        rho_arr_1_subset = np.average(np.abs(rho_arr_1_subset))
        rho_arr_2_subset = np.average(np.abs(rho_arr_2_subset))
        rho_arr_3_subset = np.average(np.abs(rho_arr_3_subset))
        rho_arr_4_subset = np.average(np.abs(rho_arr_4_subset))
        d0_subset = et_fp_m[pj_idx]
        view_angle_subset = view_angle[pj_idx]
    else:
        if ftmc_hem == 'S':
            hem_subset_idx = np.where(hem_fp_m[pj_idx] > 0)
        elif ftmc_hem == 'N':
            hem_subset_idx = np.where(hem_fp_m[pj_idx] < 0)
        hem_subset = hem_fp_m[pj_idx][hem_subset_idx]
        pj_subset = pj_fp_m[pj_idx][hem_subset_idx]
        rho_arr_subset = rho_arr[pj_idx][hem_subset_idx]
        rho_arr_1_subset = rho_arr_1[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_2_subset = rho_arr_2[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_3_subset = rho_arr_3[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_4_subset = rho_arr_4[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_1_subset = np.average(np.abs(rho_arr_1_subset))
        rho_arr_2_subset = np.average(np.abs(rho_arr_2_subset))
        rho_arr_3_subset = np.average(np.abs(rho_arr_3_subset))
        rho_arr_4_subset = np.average(np.abs(rho_arr_4_subset))
        d0_subset = et_fp_m[pj_idx][hem_subset_idx]
        view_angle_subset = view_angle[pj_idx][hem_subset_idx]

    view_angle_thres = np.where(view_angle_subset < 30.0)
    rho_ave_arr[i] = np.average(rho_arr_subset[view_angle_thres])
    rho_1_ave_arr[i] = np.average(
        [rho_arr_1_subset, rho_arr_2_subset, rho_arr_3_subset, rho_arr_4_subset])
    d0_median += [spice.et2datetime(np.median(d0_subset))]


# Satellite orbit
F.ax.axhline(y=r_moon/RJ, linestyle='dashed',
             color=UC.lightgray, zorder=0.9)

cmap_turbo = plt.get_cmap('turbo')
N_color = 9
dN = int(256/N_color-1)
color_list = []
for i in range(N_color):
    color_list += [cmap_turbo(i*dN)]
cmap = mplcolors.ListedColormap(color_list)
norm = mplcolors.Normalize(vmin=0, vmax=360)

median_arr = np.zeros(len(exnum))
median_error = np.zeros(len(exnum))
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
    s3_center = ((moon_S3wlon[0]+moon_S3wlon[-1])/2)

    plotcolor = cmap(norm(s3_center))
    q1, medians, q3 = weighted_percentile(data=column_mass,
                                          perc=[0.25, 0.5, 0.75],
                                          weights=weight)

    weighted_boxplot_h2(F.ax, rho_ave_arr[i], q1, medians, q3,
                        np.min(column_mass),
                        np.max(column_mass), width=boxplot_width,
                        ec=plotcolor, lw=1.1)

    F.ax.errorbar(x=medians, y=rho_ave_arr[i],
                  yerr=rho_1_ave_arr[i],
                  elinewidth=1.1, linewidth=0., markersize=0,
                  color=plotcolor)

    median_arr[i] = medians
    median_error[i] = q3-q1
    if q3-q1 == 0.:
        median_error[i] = 0.001

# Dummy
sc = F.ax.scatter(-999, -999, s=1.0, c=90.0, cmap=cmap, vmin=0, vmax=360)

# Setting for color bar
cax = F.fig.colorbar(sc, ax=F.ax)
cax.ax.set_yticks(np.linspace(0, 360, 10))
cax.ax.set_yticklabels(np.linspace(0, 360, 10, dtype=int),
                       fontsize=F.fontsize*0.8)
# cax.ax.yaxis.set_minor_locator(ptick.AutoMinorLocator(3))
cax.ax.set_ylabel(r'SIII longitude [deg]', fontsize=F.fontsize*0.8)

# ここでまず画像保存
F.ax.set_title(target_moon,
               fontsize=F.fontsize, weight='bold')
savedir = 'img/ftmc/'+target_moon[0:2]+'/'+exdate
F.fig.savefig(savedir+'/ftmc_'+target_moon[0:2]+'_Mshell.jpg',
              bbox_inches='tight')

# 相関係数
isnan = np.isnan(rho_ave_arr)
correlation, pvalue = spearmanr(median_arr[~isnan], rho_ave_arr[~isnan])
print('Correlation coeff: ', correlation)

# t検定
n_data = median_arr[~isnan].size
t_value = correlation*math.sqrt((n_data-2)/(1-correlation**2))
print('t value:', t_value)
print('n_data:', n_data)

# 両側p値
p_two_sided = 2*(1-t.cdf(np.abs(t_value), n_data-2))
print('p value:', p_two_sided)

# ODR 用データとモデルの設定
data = RealData(median_arr[~isnan],
                rho_ave_arr[~isnan],
                sx=median_error[~isnan],
                sy=rho_1_ave_arr[~isnan]
                )
model = Model(fit_linear)
# print(median_error[~isnan])
# print(rho_1_ave_arr[~isnan])

# ODR 実行
odr_instance = ODR(data, model, beta0=[1.0, 1.0])
output = odr_instance.run()

# フィッティング結果
popt_li = output.beta
perr_li = output.sd_beta

print("Parameters:", popt_li)
x_fit = np.linspace(-1, 20, 10)
y_fit = fit_linear(popt_li, x_fit)
label_corrcoef = r'$\rho =$'+str(round(correlation, 2))
label_tvalue = r'$t =$'+str(round(t_value, 2))
label_pvalue = r'$p =$'+str(round(p_two_sided, 4))
label_linearfit = r'$y=$' + \
    str(round(popt_li[0], 2))+r'$x+$'+str(round(popt_li[1], 2))
F.ax.plot(x_fit, y_fit, color='k', zorder=0.1)

# Dummy
F.ax.plot([-999, -998], [-999, -998], color='w',
          label=label_corrcoef)
F.ax.plot([-999, -998], [-999, -998], color='w',
          label=label_tvalue)
F.ax.plot([-999, -998], [-999, -998], color='w',
          label=label_pvalue)

# ODR fit
F.ax.text(0.5, 0.01,
          'ODR fit: '+label_linearfit,
          color='k',
          horizontalalignment='center',
          verticalalignment='bottom',
          transform=F.ax.transAxes,
          fontsize=F.fontsize*0.5)

legend = F.legend(ax_idx=0,
                  ncol=3, markerscale=1.0,
                  loc='upper right',
                  handlelength=0.01,
                  textcolor=False,
                  title='Rank correlation',
                  fontsize_scale=0.65,
                  handletextpad=0.2)
legend_shadow(legend=legend, fig=F.fig, ax=F.ax, d=0.7)
savedir = 'img/ftmc/'+target_moon[0:2]+'/'+exdate
F.fig.savefig(savedir+'/ftmc_'+target_moon[0:2]+'_Mshell_coef.jpg',
              bbox_inches='tight')
F.close()
plt.show()


#
#
#
#
#
#
#
#
#
#
# %% 横軸 FTMC & 縦軸 M shell
# === WITHOUT COLOR BAR ===
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(5.2, 5.0), dpi='L')
F.initialize()
# F.panelname = [' a. Io ', ' b. Europa ', ' c. Ganymede ']

if target_moon == 'Io':
    xmin = 0
    xmax = 40
    ymin = 5
    ymax = 8
    xticks = np.arange(xmin, xmax+1, 10)
    xticklabels = np.arange(xmin, xmax+1, 10)
    yticks = np.arange(ymin, ymax+1, 1)
    yticklabels = np.arange(ymin, ymax+1, 1)
    minor_num = 5
    boxplot_width = 0.03
elif target_moon == 'Europa':
    xmin = 0
    xmax = 3
    ymin = 8
    ymax = 12
    xticks = np.arange(xmin, xmax+1, 1)
    xticklabels = np.arange(xmin, xmax+1, 1)
    yticks = np.arange(ymin, ymax+1, 1)
    yticklabels = np.arange(ymin, ymax+1, 1)
    minor_num = 5
    boxplot_width = 0.05
elif target_moon == 'Ganymede':
    xmin = 0
    xmax = 0.18
    ymin = 11
    ymax = 25
    xticks = np.linspace(xmin, xmax, 7)
    xticklabels = np.round(np.linspace(xmin, xmax, 7), 2)
    yticks = np.arange(ymin, ymax+1, 2)
    yticklabels = np.arange(ymin, ymax+1, 2)
    minor_num = 2
    boxplot_width = 0.20

F.set_xaxis(label=target_moon+'-FTMC [10$^{-9}$ kg m$^{-2}$]',
            min=xmin, max=xmax,
            ticks=xticks,
            ticklabels=xticklabels,
            minor_num=5)
F.set_yaxis(ax_idx=0,
            label=r'M shell',
            min=ymin, max=ymax,
            ticks=yticks,
            ticklabels=yticklabels,
            minor_num=minor_num)

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


# Data subsetごとに値をまとめる
d0_median = []
rho_ave_arr = np.zeros(len(PJ_list))
rho_1_ave_arr = np.zeros(len(PJ_list))
for i in range(len(PJ_list)):
    ftmc_pj = PJ_list[i]
    ftmc_hem = FTMC_HEM[i]
    pj_idx = np.where(pj_fp_m == ftmc_pj)

    if ftmc_hem == 'both':
        hem_subset = hem_fp_m[pj_idx]
        pj_subset = pj_fp_m[pj_idx]
        rho_arr_subset = rho_arr[pj_idx]
        rho_arr_1_subset = rho_arr_1[pj_idx]-rho_arr_subset
        rho_arr_2_subset = rho_arr_2[pj_idx]-rho_arr_subset
        rho_arr_3_subset = rho_arr_3[pj_idx]-rho_arr_subset
        rho_arr_4_subset = rho_arr_4[pj_idx]-rho_arr_subset
        rho_arr_1_subset = np.average(np.abs(rho_arr_1_subset))
        rho_arr_2_subset = np.average(np.abs(rho_arr_2_subset))
        rho_arr_3_subset = np.average(np.abs(rho_arr_3_subset))
        rho_arr_4_subset = np.average(np.abs(rho_arr_4_subset))
        d0_subset = et_fp_m[pj_idx]
        view_angle_subset = view_angle[pj_idx]
    else:
        if ftmc_hem == 'S':
            hem_subset_idx = np.where(hem_fp_m[pj_idx] > 0)
        elif ftmc_hem == 'N':
            hem_subset_idx = np.where(hem_fp_m[pj_idx] < 0)
        hem_subset = hem_fp_m[pj_idx][hem_subset_idx]
        pj_subset = pj_fp_m[pj_idx][hem_subset_idx]
        rho_arr_subset = rho_arr[pj_idx][hem_subset_idx]
        rho_arr_1_subset = rho_arr_1[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_2_subset = rho_arr_2[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_3_subset = rho_arr_3[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_4_subset = rho_arr_4[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_1_subset = np.average(np.abs(rho_arr_1_subset))
        rho_arr_2_subset = np.average(np.abs(rho_arr_2_subset))
        rho_arr_3_subset = np.average(np.abs(rho_arr_3_subset))
        rho_arr_4_subset = np.average(np.abs(rho_arr_4_subset))
        d0_subset = et_fp_m[pj_idx][hem_subset_idx]
        view_angle_subset = view_angle[pj_idx][hem_subset_idx]

    view_angle_thres = np.where(view_angle_subset < 30.0)
    rho_ave_arr[i] = np.average(rho_arr_subset[view_angle_thres])
    rho_1_ave_arr[i] = np.average(
        [rho_arr_1_subset, rho_arr_2_subset, rho_arr_3_subset, rho_arr_4_subset])
    d0_median += [spice.et2datetime(np.median(d0_subset))]


# Satellite orbit
F.ax.axhline(y=r_moon/RJ, linestyle='dashed',
             color=UC.lightgray, zorder=0.9)

median_arr = np.zeros(len(exnum))
median_error = np.zeros(len(exnum))
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
    s3_center = ((moon_S3wlon[0]+moon_S3wlon[-1])/2)

    q1, medians, q3 = weighted_percentile(data=column_mass,
                                          perc=[0.25, 0.5, 0.75],
                                          weights=weight)

    weighted_boxplot_h2(F.ax, rho_ave_arr[i], q1, medians, q3,
                        np.min(column_mass),
                        np.max(column_mass), width=boxplot_width,
                        ec=UC.blue, lw=1.1)

    F.ax.errorbar(x=medians, y=rho_ave_arr[i],
                  yerr=rho_1_ave_arr[i],
                  elinewidth=1.1, linewidth=0., markersize=0,
                  color=UC.blue)

    median_arr[i] = medians
    median_error[i] = q3-q1
    if q3-q1 == 0.:
        median_error[i] = 0.001

# 相関係数
isnan = np.isnan(rho_ave_arr)
correlation, pvalue = spearmanr(median_arr[~isnan], rho_ave_arr[~isnan])
print('Correlation coeff: ', correlation)

# t検定
n_data = median_arr[~isnan].size
t_value = correlation*math.sqrt((n_data-2)/(1-correlation**2))
print('t value:', t_value)
print('n_data:', n_data)

# 両側p値
p_two_sided = 2*(1-t.cdf(np.abs(t_value), n_data-2))
print('p value:', p_two_sided)

# ODR 用データとモデルの設定
data = RealData(median_arr[~isnan],
                rho_ave_arr[~isnan],
                sx=median_error[~isnan],
                sy=rho_1_ave_arr[~isnan]
                )
model = Model(fit_linear)
# print(median_error[~isnan])
# print(rho_1_ave_arr[~isnan])

# ODR 実行
odr_instance = ODR(data, model, beta0=[1.0, 1.0])
output = odr_instance.run()

# フィッティング結果
popt_li = output.beta
perr_li = output.sd_beta

print("Parameters:", popt_li)
x_fit = np.linspace(-1, 20, 10)
y_fit = fit_linear(popt_li, x_fit)
label_corrcoef = r'$\rho =$'+str(round(correlation, 2))
label_tvalue = r'$t =$'+str(round(t_value, 2))
label_pvalue = r'$p =$'+str(round(p_two_sided, 4))
label_linearfit = r'$y=$' + \
    str(round(popt_li[0], 2))+r'$x+$'+str(round(popt_li[1], 2))
F.ax.plot(x_fit, y_fit, color='k', zorder=0.1)

# Dummy
F.ax.plot([-999, -998], [-999, -998], color='w',
          label=label_corrcoef)
F.ax.plot([-999, -998], [-999, -998], color='w',
          label=label_tvalue)
F.ax.plot([-999, -998], [-999, -998], color='w',
          label=label_pvalue)

# ODR fit
F.ax.text(0.5, 0.01,
          'ODR fit: '+label_linearfit,
          color='k',
          horizontalalignment='center',
          verticalalignment='bottom',
          transform=F.ax.transAxes,
          fontsize=F.fontsize*0.5)

F.ax.set_title(target_moon,
               fontsize=F.fontsize, weight='bold')

legend = F.legend(ax_idx=0,
                  ncol=3, markerscale=1.0,
                  loc='upper right',
                  handlelength=0.01,
                  textcolor=False,
                  title='Rank correlation',
                  fontsize_scale=0.65,
                  handletextpad=0.2)
legend_shadow(legend=legend, fig=F.fig, ax=F.ax, d=0.7)
savedir = 'img/ftmc/'+target_moon[0:2]+'/'+exdate
F.fig.savefig(savedir+'/ftmc_'+target_moon[0:2]+'_Mshell_coef_nocolor.jpg',
              bbox_inches='tight')
F.close()
plt.show()


#
#
#
#
#
#
#
#
#
#
# %% 横軸 FTMC & 縦軸 Azimuthal current constant
# === WITHOUT COLOR BAR ===
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(5.3, 5.0), dpi='L')
F.initialize()
# F.panelname = [' a. Io ', ' b. Europa ', ' c. Ganymede ']

if target_moon == 'Io':
    xmin = 0
    xmax = 40
    ymin = 5
    ymax = 8
    xticks = np.arange(xmin, xmax+1, 10)
    xticklabels = np.arange(xmin, xmax+1, 10)
    yticks = np.arange(ymin, ymax+1, 1)
    yticklabels = np.arange(ymin, ymax+1, 1)
    xminor_num = 5
    yminor_num = 5
    boxplot_width = 0.02
elif target_moon == 'Europa':
    xmin = 0
    xmax = 3
    ymin = 0
    ymax = 240
    xticks = np.arange(xmin, xmax+1, 1)
    xticklabels = np.arange(xmin, xmax+1, 1)
    yticks = np.arange(0, 200+1, 50)
    yticklabels = np.round(np.arange(0, 200+1, 50), 2)
    xminor_num = 5
    yminor_num = 5
    boxplot_width = 2.5
elif target_moon == 'Ganymede':
    xmin = 0
    xmax = 0.18
    ymin = 40
    ymax = 230
    xticks = np.linspace(xmin, xmax, 7)
    xticklabels = np.round(np.linspace(xmin, xmax, 7), 2)
    yticks = np.arange(50, 200+1, 50)
    yticklabels = np.round(np.arange(50, 200+1, 50), 2)
    xminor_num = 3
    yminor_num = 5
    boxplot_width = 2.5

F.set_xaxis(label=target_moon+'-FTMC [10$^{-9}$ kg m$^{-2}$]',
            min=xmin, max=xmax,
            ticks=xticks,
            ticklabels=xticklabels,
            minor_num=xminor_num)
F.set_yaxis(ax_idx=0,
            label=r'Current constant [nT]',
            min=ymin, max=ymax,
            ticks=yticks,
            ticklabels=yticklabels,
            minor_num=yminor_num)

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


# Data subsetごとに値をまとめる
d0_median = []
rho_ave_arr = np.zeros(len(PJ_list))
rho_1_ave_arr = np.zeros(len(PJ_list))
mu_i_coef_ave = np.zeros(len(PJ_list))
mu_i_coef_1_ave = np.zeros(len(PJ_list))
for i in range(len(PJ_list)):
    ftmc_pj = PJ_list[i]
    ftmc_hem = FTMC_HEM[i]
    pj_idx = np.where(pj_fp_m == ftmc_pj)

    if ftmc_hem == 'both':
        hem_subset = hem_fp_m[pj_idx]
        pj_subset = pj_fp_m[pj_idx]
        rho_arr_subset = rho_arr[pj_idx]
        rho_arr_1_subset = rho_arr_1[pj_idx]-rho_arr_subset
        rho_arr_2_subset = rho_arr_2[pj_idx]-rho_arr_subset
        rho_arr_3_subset = rho_arr_3[pj_idx]-rho_arr_subset
        rho_arr_4_subset = rho_arr_4[pj_idx]-rho_arr_subset
        d0_subset = et_fp_m[pj_idx]
        view_angle_subset = view_angle[pj_idx]
        mu_i_coef_subset = mu_i_coef[pj_idx]
        mu_i_coef_1_subset = mu_i_coef_1[pj_idx]-mu_i_coef_subset
        mu_i_coef_2_subset = mu_i_coef_2[pj_idx]-mu_i_coef_subset
        mu_i_coef_3_subset = mu_i_coef_3[pj_idx]-mu_i_coef_subset
        mu_i_coef_4_subset = mu_i_coef_4[pj_idx]-mu_i_coef_subset
    else:
        if ftmc_hem == 'S':
            hem_subset_idx = np.where(hem_fp_m[pj_idx] > 0)
        elif ftmc_hem == 'N':
            hem_subset_idx = np.where(hem_fp_m[pj_idx] < 0)
        hem_subset = hem_fp_m[pj_idx][hem_subset_idx]
        pj_subset = pj_fp_m[pj_idx][hem_subset_idx]
        rho_arr_subset = rho_arr[pj_idx][hem_subset_idx]
        rho_arr_1_subset = rho_arr_1[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_2_subset = rho_arr_2[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_3_subset = rho_arr_3[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_4_subset = rho_arr_4[pj_idx][hem_subset_idx]-rho_arr_subset
        d0_subset = et_fp_m[pj_idx][hem_subset_idx]
        view_angle_subset = view_angle[pj_idx][hem_subset_idx]
        mu_i_coef_subset = mu_i_coef[pj_idx][hem_subset_idx]
        mu_i_coef_1_subset = mu_i_coef_1[pj_idx][hem_subset_idx] - \
            mu_i_coef_subset
        mu_i_coef_2_subset = mu_i_coef_2[pj_idx][hem_subset_idx] - \
            mu_i_coef_subset
        mu_i_coef_3_subset = mu_i_coef_3[pj_idx][hem_subset_idx] - \
            mu_i_coef_subset
        mu_i_coef_4_subset = mu_i_coef_4[pj_idx][hem_subset_idx] - \
            mu_i_coef_subset

    view_angle_thres = np.where(view_angle_subset < 30.0)

    rho_arr_1_subset = np.average(np.abs(rho_arr_1_subset[view_angle_thres]))
    rho_arr_2_subset = np.average(np.abs(rho_arr_2_subset[view_angle_thres]))
    rho_arr_3_subset = np.average(np.abs(rho_arr_3_subset[view_angle_thres]))
    rho_arr_4_subset = np.average(np.abs(rho_arr_4_subset[view_angle_thres]))
    mu_i_coef_1_subset = np.average(
        np.abs(mu_i_coef_1_subset[view_angle_thres]))
    mu_i_coef_2_subset = np.average(
        np.abs(mu_i_coef_2_subset[view_angle_thres]))
    mu_i_coef_3_subset = np.average(
        np.abs(mu_i_coef_3_subset[view_angle_thres]))
    mu_i_coef_4_subset = np.average(
        np.abs(mu_i_coef_4_subset[view_angle_thres]))

    rho_ave_arr[i] = np.average(rho_arr_subset[view_angle_thres])
    rho_1_ave_arr[i] = np.average(
        [rho_arr_1_subset, rho_arr_2_subset, rho_arr_3_subset, rho_arr_4_subset])
    d0_median += [spice.et2datetime(np.median(d0_subset))]
    mu_i_coef_ave[i] = np.average(mu_i_coef_subset[view_angle_thres])
    mu_i_coef_1_ave[i] = np.average(
        [mu_i_coef_1_subset, mu_i_coef_2_subset, mu_i_coef_3_subset, mu_i_coef_4_subset])

    # print('PJ'+str(ftmc_pj)+ftmc_hem, view_angle_thres[0].size)
    if view_angle_thres[0].size > 0:
        np.savetxt('results/azimuthal_current_fit/'+target_moon[0:2]+'/PJ'+str(ftmc_pj)+ftmc_hem+'.txt',
                   np.array([mu_i_coef_ave[i], mu_i_coef_1_ave[i]]))


# Azimuthal current constant
mu_i_default = 139.6    # default: 139.6 [nT]
mu_i_ave = mu_i_coef_ave*mu_i_default
mu_i_1_ave = mu_i_coef_1_ave*mu_i_default
F.ax.axhline(y=mu_i_default, linestyle='dashed',
             color=UC.lightgray, zorder=0.9)

median_arr = np.zeros(len(exnum))
median_error = np.zeros(len(exnum))
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
    s3_center = ((moon_S3wlon[0]+moon_S3wlon[-1])/2)

    q1, medians, q3 = weighted_percentile(data=column_mass,
                                          perc=[0.25, 0.5, 0.75],
                                          weights=weight)

    weighted_boxplot_h2(F.ax, mu_i_ave[i], q1, medians, q3,
                        np.min(column_mass),
                        np.max(column_mass), width=boxplot_width,
                        ec=UC.blue, lw=1.1)

    F.ax.errorbar(x=medians, y=mu_i_ave[i],
                  yerr=mu_i_1_ave[i],
                  elinewidth=1.1, linewidth=0., markersize=0,
                  color=UC.blue)

    median_arr[i] = medians
    median_error[i] = q3-q1
    if q3-q1 == 0.:
        median_error[i] = 0.001

# 相関係数
isnan = np.isnan(mu_i_ave)
correlation, pvalue = spearmanr(median_arr[~isnan], mu_i_ave[~isnan])
print('Correlation coeff: ', correlation)

# t検定
n_data = median_arr[~isnan].size
t_value = correlation*math.sqrt((n_data-2)/(1-correlation**2))
print('t value:', t_value)
print('n_data:', n_data)

# 両側p値
p_two_sided = 2*(1-t.cdf(np.abs(t_value), n_data-2))
print('p value:', p_two_sided)

# ODR 用データとモデルの設定
data = RealData(median_arr[~isnan],
                mu_i_ave[~isnan],
                sx=median_error[~isnan],
                sy=mu_i_1_ave[~isnan]
                )
model = Model(fit_linear)
# print(median_error[~isnan])
# print(mu_i_coef_1_ave[~isnan])

# ODR 実行
odr_instance = ODR(data, model, beta0=[1.0, 1.0])
output = odr_instance.run()

# フィッティング結果
popt_li = output.beta
perr_li = output.sd_beta

print("Parameters:", popt_li)
x_fit = np.linspace(-1, 20, 10)
y_fit = fit_linear(popt_li, x_fit)
label_corrcoef = r'$\rho =$'+str(round(correlation, 2))
label_tvalue = r'$t =$'+str(round(t_value, 2))
label_pvalue = r'$p =$'+str(round(p_two_sided, 4))
label_linearfit = r'$y=$' + \
    str(round(popt_li[0], 2))+r'$x+$'+str(round(popt_li[1], 2))
F.ax.plot(x_fit, y_fit, color='k', zorder=0.1)

# Dummy
F.ax.plot([-999, -998], [-999, -998], color='w',
          label=label_corrcoef)
F.ax.plot([-999, -998], [-999, -998], color='w',
          label=label_tvalue)
F.ax.plot([-999, -998], [-999, -998], color='w',
          label=label_pvalue)

# ODR fit
F.ax.text(0.5, 0.01,
          'ODR fit: '+label_linearfit,
          color='k',
          horizontalalignment='center',
          verticalalignment='bottom',
          transform=F.ax.transAxes,
          fontsize=F.fontsize*0.5)

F.ax.set_title(target_moon,
               fontsize=F.fontsize, weight='bold')

legend = F.legend(ax_idx=0,
                  ncol=3, markerscale=1.0,
                  loc='upper right',
                  handlelength=0.01,
                  textcolor=False,
                  title='Rank correlation',
                  fontsize_scale=0.65,
                  handletextpad=0.2)
legend_shadow(legend=legend, fig=F.fig, ax=F.ax, d=0.7)
savedir = 'img/ftmc/'+target_moon[0:2]+'/'+exdate
F.fig.savefig(savedir+'/ftmc_'+target_moon[0:2]+'_mui_coef_nocolor.jpg',
              bbox_inches='tight')
F.close()
plt.show()


#
#
#
#
#
#
#
#
#
#
# %% 横軸 FTMC & 縦軸 M shell
# Color code: local time
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(6.5, 5.5), dpi='L')
F.initialize()
# F.panelname = [' a. Io ', ' b. Europa ', ' c. Ganymede ']

if target_moon == 'Io':
    xmin = 0
    xmax = 40
    ymin = 5
    ymax = 8
    xticks = np.arange(xmin, xmax+1, 10)
    xticklabels = np.arange(xmin, xmax+1, 10)
    yticks = np.arange(ymin, ymax+1, 1)
    yticklabels = np.arange(ymin, ymax+1, 1)
    minor_num = 5
    boxplot_width = 0.03
elif target_moon == 'Europa':
    xmin = 0
    xmax = 3
    ymin = 8
    ymax = 12
    xticks = np.arange(xmin, xmax+1, 1)
    xticklabels = np.arange(xmin, xmax+1, 1)
    yticks = np.arange(ymin, ymax+1, 1)
    yticklabels = np.arange(ymin, ymax+1, 1)
    minor_num = 5
    boxplot_width = 0.05
elif target_moon == 'Ganymede':
    xmin = 0
    xmax = 0.20
    ymin = 11
    ymax = 25
    xticks = np.arange(xmin, xmax+0.01, 0.2)
    xticklabels = np.arange(xmin, xmax+0.01, 0.2)
    yticks = np.arange(ymin, ymax+1, 2)
    yticklabels = np.arange(ymin, ymax+1, 2)
    minor_num = 2
    boxplot_width = 0.20

F.set_xaxis(label=target_moon+'-FTMC [10$^{-9}$ kg m$^{-2}$]',
            min=xmin, max=xmax,
            ticks=xticks,
            ticklabels=xticklabels,
            minor_num=5)
F.set_yaxis(ax_idx=0,
            label=r'M shell',
            min=ymin, max=ymax,
            ticks=yticks,
            ticklabels=yticklabels,
            minor_num=minor_num)

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


# Data subsetごとに値をまとめる
d0_median = []
rho_ave_arr = np.zeros(len(PJ_list))
rho_1_ave_arr = np.zeros(len(PJ_list))
for i in range(len(PJ_list)):
    ftmc_pj = PJ_list[i]
    ftmc_hem = FTMC_HEM[i]
    pj_idx = np.where(pj_fp_m == ftmc_pj)

    if ftmc_hem == 'both':
        hem_subset = hem_fp_m[pj_idx]
        pj_subset = pj_fp_m[pj_idx]
        rho_arr_subset = rho_arr[pj_idx]
        rho_arr_1_subset = rho_arr_1[pj_idx]-rho_arr_subset
        rho_arr_2_subset = rho_arr_2[pj_idx]-rho_arr_subset
        rho_arr_3_subset = rho_arr_3[pj_idx]-rho_arr_subset
        rho_arr_4_subset = rho_arr_4[pj_idx]-rho_arr_subset
        rho_arr_1_subset = np.average(np.abs(rho_arr_1_subset))
        rho_arr_2_subset = np.average(np.abs(rho_arr_2_subset))
        rho_arr_3_subset = np.average(np.abs(rho_arr_3_subset))
        rho_arr_4_subset = np.average(np.abs(rho_arr_4_subset))
        d0_subset = et_fp_m[pj_idx]
        view_angle_subset = view_angle[pj_idx]
    else:
        if ftmc_hem == 'S':
            hem_subset_idx = np.where(hem_fp_m[pj_idx] > 0)
        elif ftmc_hem == 'N':
            hem_subset_idx = np.where(hem_fp_m[pj_idx] < 0)
        hem_subset = hem_fp_m[pj_idx][hem_subset_idx]
        pj_subset = pj_fp_m[pj_idx][hem_subset_idx]
        rho_arr_subset = rho_arr[pj_idx][hem_subset_idx]
        rho_arr_1_subset = rho_arr_1[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_2_subset = rho_arr_2[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_3_subset = rho_arr_3[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_4_subset = rho_arr_4[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_1_subset = np.average(np.abs(rho_arr_1_subset))
        rho_arr_2_subset = np.average(np.abs(rho_arr_2_subset))
        rho_arr_3_subset = np.average(np.abs(rho_arr_3_subset))
        rho_arr_4_subset = np.average(np.abs(rho_arr_4_subset))
        d0_subset = et_fp_m[pj_idx][hem_subset_idx]
        view_angle_subset = view_angle[pj_idx][hem_subset_idx]

    view_angle_thres = np.where(view_angle_subset < 30.0)
    rho_ave_arr[i] = np.average(rho_arr_subset[view_angle_thres])
    rho_1_ave_arr[i] = np.average(
        [rho_arr_1_subset, rho_arr_2_subset, rho_arr_3_subset, rho_arr_4_subset])
    d0_median += [spice.et2datetime(np.median(d0_subset))]


# Satellite orbit
F.ax.axhline(y=r_moon/RJ, linestyle='dashed',
             color=UC.lightgray, zorder=0.9)

cmap_turbo = plt.get_cmap('turbo')
N_color = 8
dN = int(256/N_color-1)
color_list = []
for i in range(N_color):
    color_list += [cmap_turbo(i*dN)]
cmap = mplcolors.ListedColormap(color_list)
norm = mplcolors.Normalize(vmin=0, vmax=24)

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
    s3_center = ((moon_S3wlon[0]+moon_S3wlon[-1])/2)

    plotcolor = cmap(norm(lt_center))
    q1, medians, q3 = weighted_percentile(data=column_mass,
                                          perc=[0.25, 0.5, 0.75],
                                          weights=weight)
    width = 0.20
    weighted_boxplot_h2(F.ax, rho_ave_arr[i], q1, medians, q3,
                        np.min(column_mass),
                        np.max(column_mass), width=boxplot_width,
                        ec=plotcolor, lw=1.1)

    F.ax.errorbar(x=medians, y=rho_ave_arr[i],
                  yerr=rho_1_ave_arr[i],
                  elinewidth=1.1, linewidth=0., markersize=0,
                  color=plotcolor)

# Dummy
sc = F.ax.scatter(-999, -999, s=1.0, c=10.0, cmap=cmap, vmin=0, vmax=24)

# Setting for color bar
cax = F.fig.colorbar(sc, ax=F.ax)
cax.ax.set_yticks(np.linspace(0, 24, 9))
cax.ax.set_yticklabels(np.linspace(0, 24, 9, dtype=int),
                       fontsize=F.fontsize*0.8)
# cax.ax.yaxis.set_minor_locator(ptick.AutoMinorLocator(3))
cax.ax.set_ylabel(r'Local time [deg]', fontsize=F.fontsize*0.8)

F.ax.set_title(target_moon,
               fontsize=F.fontsize, weight='bold')

savedir = 'img/ftmc/'+target_moon[0:2]+'/'+exdate
F.fig.savefig(savedir+'/ftmc_'+target_moon[0:2]+'_Mshell_2.jpg',
              bbox_inches='tight')
F.close()
plt.show()


#
#
#
#
#
#
#
#
#
#
# %% 横軸 FTMC (LT subtracted) & 縦軸 M shell
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(6.5, 5.5), dpi='L')
F.initialize()
# F.panelname = [' a. Io ', ' b. Europa ', ' c. Ganymede ']

if target_moon == 'Io':
    xmin = 0
    xmax = 40
    ymin = 5
    ymax = 8
    xticks = np.arange(xmin, xmax+1, 10)
    xticklabels = np.arange(xmin, xmax+1, 10)
    yticks = np.arange(ymin, ymax+1, 1)
    yticklabels = np.arange(ymin, ymax+1, 1)
    minor_num = 5
    boxplot_width = 0.03
elif target_moon == 'Europa':
    xmin = -1
    xmax = 2
    ymin = 8
    ymax = 12
    xticks = np.arange(xmin, xmax+1, 1)
    xticklabels = np.arange(xmin, xmax+1, 1)
    yticks = np.arange(ymin, ymax+1, 1)
    yticklabels = np.arange(ymin, ymax+1, 1)
    minor_num = 5
    boxplot_width = 0.05
elif target_moon == 'Ganymede':
    xmin = -0.1
    xmax = 0.2
    ymin = 11
    ymax = 25
    xticks = np.arange(xmin, xmax+0.01, 0.05)
    xticklabels = np.round(np.arange(xmin, xmax+0.01, 0.05), 2)
    yticks = np.arange(ymin, ymax+1, 2)
    yticklabels = np.arange(ymin, ymax+1, 2)
    minor_num = 2
    boxplot_width = 0.20

F.set_xaxis(label=r'$\Delta_{\rm FTMC}$ [10$^{-9}$ kg m$^{-2}$]',
            min=xmin, max=xmax,
            ticks=xticks,
            ticklabels=xticklabels,
            minor_num=5)
F.set_yaxis(ax_idx=0,
            label=r'M shell',
            min=ymin, max=ymax,
            ticks=yticks,
            ticklabels=yticklabels,
            minor_num=minor_num)

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


# Data subsetごとに値をまとめる
d0_median = []
rho_ave_arr = np.zeros(len(PJ_list))
rho_1_ave_arr = np.zeros(len(PJ_list))
for i in range(len(PJ_list)):
    ftmc_pj = PJ_list[i]
    ftmc_hem = FTMC_HEM[i]
    pj_idx = np.where(pj_fp_m == ftmc_pj)

    if ftmc_hem == 'both':
        hem_subset = hem_fp_m[pj_idx]
        pj_subset = pj_fp_m[pj_idx]
        rho_arr_subset = rho_arr[pj_idx]
        rho_arr_1_subset = rho_arr_1[pj_idx]-rho_arr_subset
        rho_arr_2_subset = rho_arr_2[pj_idx]-rho_arr_subset
        rho_arr_3_subset = rho_arr_3[pj_idx]-rho_arr_subset
        rho_arr_4_subset = rho_arr_4[pj_idx]-rho_arr_subset
        rho_arr_1_subset = np.average(np.abs(rho_arr_1_subset))
        rho_arr_2_subset = np.average(np.abs(rho_arr_2_subset))
        rho_arr_3_subset = np.average(np.abs(rho_arr_3_subset))
        rho_arr_4_subset = np.average(np.abs(rho_arr_4_subset))
        d0_subset = et_fp_m[pj_idx]
        view_angle_subset = view_angle[pj_idx]
    else:
        if ftmc_hem == 'S':
            hem_subset_idx = np.where(hem_fp_m[pj_idx] > 0)
        elif ftmc_hem == 'N':
            hem_subset_idx = np.where(hem_fp_m[pj_idx] < 0)
        hem_subset = hem_fp_m[pj_idx][hem_subset_idx]
        pj_subset = pj_fp_m[pj_idx][hem_subset_idx]
        rho_arr_subset = rho_arr[pj_idx][hem_subset_idx]
        rho_arr_1_subset = rho_arr_1[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_2_subset = rho_arr_2[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_3_subset = rho_arr_3[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_4_subset = rho_arr_4[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_1_subset = np.average(np.abs(rho_arr_1_subset))
        rho_arr_2_subset = np.average(np.abs(rho_arr_2_subset))
        rho_arr_3_subset = np.average(np.abs(rho_arr_3_subset))
        rho_arr_4_subset = np.average(np.abs(rho_arr_4_subset))
        d0_subset = et_fp_m[pj_idx][hem_subset_idx]
        view_angle_subset = view_angle[pj_idx][hem_subset_idx]

    view_angle_thres = np.where(view_angle_subset < 30.0)
    rho_ave_arr[i] = np.average(rho_arr_subset[view_angle_thres])
    rho_1_ave_arr[i] = np.average(
        [rho_arr_1_subset, rho_arr_2_subset, rho_arr_3_subset, rho_arr_4_subset])
    d0_median += [spice.et2datetime(np.median(d0_subset))]


# Satellite orbit
F.ax.axhline(y=r_moon/RJ, linestyle='dashed',
             color=UC.lightgray, zorder=0.9)

cmap_turbo = plt.get_cmap('turbo')
N_color = 9
dN = int(256/N_color-1)
color_list = []
for i in range(N_color):
    color_list += [cmap_turbo(i*dN)]
cmap = mplcolors.ListedColormap(color_list)
norm = mplcolors.Normalize(vmin=0, vmax=360)

median_arr = np.zeros(len(exnum))
median_error = np.zeros(len(exnum))
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
    s3_center = ((moon_S3wlon[0]+moon_S3wlon[-1])/2)

    y_fit = fit_func2(popt, lt_center)

    plotcolor = cmap(norm(s3_center))
    q1, medians, q3 = weighted_percentile(data=column_mass-y_fit,
                                          perc=[0.25, 0.5, 0.75],
                                          weights=weight)
    weighted_boxplot_h2(F.ax, rho_ave_arr[i], q1, medians, q3,
                        np.min(column_mass)-y_fit,
                        np.max(column_mass)-y_fit, width=boxplot_width,
                        ec=plotcolor, lw=1.1)

    eb = F.ax.errorbar(x=medians, y=rho_ave_arr[i],
                       yerr=rho_1_ave_arr[i],
                       elinewidth=1.1, linewidth=0., markersize=0,
                       color=plotcolor)

    median_arr[i] = medians
    median_error[i] = q3-q1
    if q3-q1 == 0.:
        median_error[i] = 0.001

# Dummy
sc = F.ax.scatter(-999, -999, s=1.0, c=90.0, cmap=cmap, vmin=0, vmax=360)

# Setting for color bar
cax = F.fig.colorbar(sc, ax=F.ax)
cax.ax.set_yticks(np.linspace(0, 360, 10))
cax.ax.set_yticklabels(np.linspace(0, 360, 10, dtype=int),
                       fontsize=F.fontsize*0.8)
# cax.ax.yaxis.set_minor_locator(ptick.AutoMinorLocator(3))
cax.ax.set_ylabel(r'SIII longitude [deg]', fontsize=F.fontsize*0.8)

# ここでまず画像保存
F.ax.set_title(target_moon,
               fontsize=F.fontsize, weight='bold')
savedir = 'img/ftmc/'+target_moon[0:2]+'/'+exdate
F.fig.savefig(savedir+'/ftmc_'+target_moon[0:2]+'_Mshell_3.jpg',
              bbox_inches='tight')

# 相関係数
isnan = np.isnan(rho_ave_arr)
correlation, pvalue = spearmanr(median_arr[~isnan], rho_ave_arr[~isnan])
print('Correlation coeff: ', correlation)

# t検定
n_data = median_arr[~isnan].size
t_value = correlation*math.sqrt((n_data-2)/(1-correlation**2))
print('t value:', t_value)
print('n_data:', n_data)

# 両側p値
p_two_sided = 2*(1-t.cdf(np.abs(t_value), n_data-2))
print('p value:', p_two_sided)

# ODR 用データとモデルの設定
data = RealData(median_arr[~isnan],
                rho_ave_arr[~isnan],
                sx=median_error[~isnan],
                sy=rho_1_ave_arr[~isnan]
                )
model = Model(fit_linear)
# print(median_error[~isnan])
# print(rho_1_ave_arr[~isnan])

# ODR 実行
odr_instance = ODR(data, model, beta0=[1.0, 1.0])
output = odr_instance.run()

# フィッティング結果
popt_li = output.beta
perr_li = output.sd_beta

print("Parameters:", popt_li)
x_fit = np.linspace(-10, 20, 10)
y_fit = fit_linear(popt_li, x_fit)
label_corrcoef = r'$\rho =$'+str(round(correlation, 2))
label_tvalue = r'$t =$'+str(round(t_value, 2))
label_pvalue = r'$p =$'+str(round(p_two_sided, 4))
label_linearfit = r'$y=$' + \
    str(round(popt_li[0], 2))+r'$x+$'+str(round(popt_li[1], 2))
F.ax.plot(x_fit, y_fit, color='k', zorder=0.1)

# Dummy
F.ax.plot([-999, -998], [-999, -998], color='w',
          label=label_corrcoef)
F.ax.plot([-999, -998], [-999, -998], color='w',
          label=label_tvalue)
F.ax.plot([-999, -998], [-999, -998], color='w',
          label=label_pvalue)

legend = F.legend(ax_idx=0,
                  ncol=3, markerscale=1.0,
                  loc='upper right',
                  handlelength=0.01,
                  textcolor=False,
                  title='Rank correlation',
                  fontsize_scale=0.65,
                  handletextpad=0.2)
legend_shadow(legend=legend, fig=F.fig, ax=F.ax, d=0.7)
savedir = 'img/ftmc/'+target_moon[0:2]+'/'+exdate
F.fig.savefig(savedir+'/ftmc_'+target_moon[0:2]+'_Mshell_3_coef.jpg',
              bbox_inches='tight')

F.close()
plt.show()


#
#
#
#
#
#
#
#
#
#
# %% 横軸 FTMC (LT subtracted) & 縦軸 M shell
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(5.1, 5.0), dpi='L')
F.initialize()
# F.panelname = [' a. Io ', ' b. Europa ', ' c. Ganymede ']

if target_moon == 'Io':
    xmin = 0
    xmax = 40
    ymin = 5
    ymax = 8
    xticks = np.arange(xmin, xmax+1, 10)
    xticklabels = np.arange(xmin, xmax+1, 10)
    yticks = np.arange(ymin, ymax+1, 1)
    yticklabels = np.arange(ymin, ymax+1, 1)
    minor_num = 5
    boxplot_width = 0.03
elif target_moon == 'Europa':
    xmin = -1
    xmax = 2.5
    ymin = 8
    ymax = 12
    xticks = np.arange(xmin, 2+1, 1)
    xticklabels = np.arange(xmin, 2+1, 1)
    yticks = np.arange(ymin, ymax+1, 1)
    yticklabels = np.arange(ymin, ymax+1, 1)
    minor_num = 5
    boxplot_width = 0.05
elif target_moon == 'Ganymede':
    xmin = -0.1
    xmax = 0.2
    ymin = 11
    ymax = 25
    xticks = np.arange(xmin, xmax+0.01, 0.05)
    xticklabels = np.round(np.arange(xmin, xmax+0.01, 0.05), 2)
    yticks = np.arange(ymin, ymax+1, 2)
    yticklabels = np.arange(ymin, ymax+1, 2)
    minor_num = 2
    boxplot_width = 0.20

F.set_xaxis(label=r'$\Delta_{\rm FTMC}$ [10$^{-9}$ kg m$^{-2}$]',
            min=xmin, max=xmax,
            ticks=xticks,
            ticklabels=xticklabels,
            minor_num=5)
F.set_yaxis(ax_idx=0,
            label=r'M shell',
            min=ymin, max=ymax,
            ticks=yticks,
            ticklabels=yticklabels,
            minor_num=minor_num)

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


# Data subsetごとに値をまとめる
d0_median = []
rho_ave_arr = np.zeros(len(PJ_list))
rho_1_ave_arr = np.zeros(len(PJ_list))
for i in range(len(PJ_list)):
    ftmc_pj = PJ_list[i]
    ftmc_hem = FTMC_HEM[i]
    pj_idx = np.where(pj_fp_m == ftmc_pj)

    if ftmc_hem == 'both':
        hem_subset = hem_fp_m[pj_idx]
        pj_subset = pj_fp_m[pj_idx]
        rho_arr_subset = rho_arr[pj_idx]
        rho_arr_1_subset = rho_arr_1[pj_idx]-rho_arr_subset
        rho_arr_2_subset = rho_arr_2[pj_idx]-rho_arr_subset
        rho_arr_3_subset = rho_arr_3[pj_idx]-rho_arr_subset
        rho_arr_4_subset = rho_arr_4[pj_idx]-rho_arr_subset
        rho_arr_1_subset = np.average(np.abs(rho_arr_1_subset))
        rho_arr_2_subset = np.average(np.abs(rho_arr_2_subset))
        rho_arr_3_subset = np.average(np.abs(rho_arr_3_subset))
        rho_arr_4_subset = np.average(np.abs(rho_arr_4_subset))
        d0_subset = et_fp_m[pj_idx]
        view_angle_subset = view_angle[pj_idx]
    else:
        if ftmc_hem == 'S':
            hem_subset_idx = np.where(hem_fp_m[pj_idx] > 0)
        elif ftmc_hem == 'N':
            hem_subset_idx = np.where(hem_fp_m[pj_idx] < 0)
        hem_subset = hem_fp_m[pj_idx][hem_subset_idx]
        pj_subset = pj_fp_m[pj_idx][hem_subset_idx]
        rho_arr_subset = rho_arr[pj_idx][hem_subset_idx]
        rho_arr_1_subset = rho_arr_1[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_2_subset = rho_arr_2[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_3_subset = rho_arr_3[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_4_subset = rho_arr_4[pj_idx][hem_subset_idx]-rho_arr_subset
        rho_arr_1_subset = np.average(np.abs(rho_arr_1_subset))
        rho_arr_2_subset = np.average(np.abs(rho_arr_2_subset))
        rho_arr_3_subset = np.average(np.abs(rho_arr_3_subset))
        rho_arr_4_subset = np.average(np.abs(rho_arr_4_subset))
        d0_subset = et_fp_m[pj_idx][hem_subset_idx]
        view_angle_subset = view_angle[pj_idx][hem_subset_idx]

    view_angle_thres = np.where(view_angle_subset < 30.0)
    rho_ave_arr[i] = np.average(rho_arr_subset[view_angle_thres])
    rho_1_ave_arr[i] = np.average(
        [rho_arr_1_subset, rho_arr_2_subset, rho_arr_3_subset, rho_arr_4_subset])
    d0_median += [spice.et2datetime(np.median(d0_subset))]


# Satellite orbit
F.ax.axhline(y=r_moon/RJ, linestyle='dashed',
             color=UC.lightgray, zorder=0.9)

median_arr = np.zeros(len(exnum))
median_error = np.zeros(len(exnum))
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
    s3_center = ((moon_S3wlon[0]+moon_S3wlon[-1])/2)

    y_fit = fit_func2(popt, lt_center)

    q1, medians, q3 = weighted_percentile(data=column_mass-y_fit,
                                          perc=[0.25, 0.5, 0.75],
                                          weights=weight)
    weighted_boxplot_h2(F.ax, rho_ave_arr[i], q1, medians, q3,
                        np.min(column_mass)-y_fit,
                        np.max(column_mass)-y_fit, width=boxplot_width,
                        ec=UC.blue, lw=1.1)

    eb = F.ax.errorbar(x=medians, y=rho_ave_arr[i],
                       yerr=rho_1_ave_arr[i],
                       elinewidth=1.1, linewidth=0., markersize=0,
                       color=UC.blue)

    median_arr[i] = medians
    median_error[i] = q3-q1
    if q3-q1 == 0.:
        median_error[i] = 0.001

# 相関係数
isnan = np.isnan(rho_ave_arr)
correlation, pvalue = spearmanr(median_arr[~isnan], rho_ave_arr[~isnan])
print('Correlation coeff: ', correlation)

# t検定
n_data = median_arr[~isnan].size
t_value = correlation*math.sqrt((n_data-2)/(1-correlation**2))
print('t value:', t_value)
print('n_data:', n_data)

# 両側p値
p_two_sided = 2*(1-t.cdf(np.abs(t_value), n_data-2))
print('p value:', p_two_sided)

# ODR 用データとモデルの設定
data = RealData(median_arr[~isnan],
                rho_ave_arr[~isnan],
                sx=median_error[~isnan],
                sy=rho_1_ave_arr[~isnan]
                )
model = Model(fit_linear)
# print(median_error[~isnan])
# print(rho_1_ave_arr[~isnan])

# ODR 実行
odr_instance = ODR(data, model, beta0=[1.0, 1.0])
output = odr_instance.run()

# フィッティング結果
popt_li = output.beta
perr_li = output.sd_beta

print("Parameters:", popt_li)
x_fit = np.linspace(-10, 20, 10)
y_fit = fit_linear(popt_li, x_fit)
label_corrcoef = r'$\rho =$'+str(round(correlation, 2))
label_tvalue = r'$t =$'+str(round(t_value, 2))
label_pvalue = r'$p =$'+str(round(p_two_sided, 4))
label_linearfit = r'$y=$' + \
    str(round(popt_li[0], 2))+r'$x+$'+str(round(popt_li[1], 2))
F.ax.plot(x_fit, y_fit, color='k', zorder=0.1)

# Dummy
F.ax.plot([-999, -998], [-999, -998], color='w',
          label=label_corrcoef)
F.ax.plot([-999, -998], [-999, -998], color='w',
          label=label_tvalue)
F.ax.plot([-999, -998], [-999, -998], color='w',
          label=label_pvalue)

# ODR fit
F.ax.text(0.5, 0.01,
          'ODR fit: '+label_linearfit,
          color='k',
          horizontalalignment='center',
          verticalalignment='bottom',
          transform=F.ax.transAxes,
          fontsize=F.fontsize*0.5)

F.ax.set_title(target_moon,
               fontsize=F.fontsize, weight='bold')


legend = F.legend(ax_idx=0,
                  ncol=3, markerscale=1.0,
                  loc='upper right',
                  handlelength=0.01,
                  textcolor=False,
                  title='Rank correlation',
                  fontsize_scale=0.65,
                  handletextpad=0.2)
legend_shadow(legend=legend, fig=F.fig, ax=F.ax, d=0.7)
savedir = 'img/ftmc/'+target_moon[0:2]+'/'+exdate
F.fig.savefig(savedir+'/ftmc_'+target_moon[0:2]+'_Mshell_3_coef_nocolor.jpg',
              bbox_inches='tight')

F.close()
plt.show()
