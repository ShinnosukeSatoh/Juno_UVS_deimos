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

from scipy.odr import ODR, Model, RealData, Output
from scipy.stats import spearmanr
from scipy.stats import t

import spiceypy as spice
spice.furnsh('kernel/cassMetaK.txt')
savpath = 'data/Satellite_FP_JRM33.sav'

UC = UniversalColor()
UC.set_palette()


exdate_list = ['003/20250516', '005/20251221', '005/20251221']
target_moon_list = ['Io', 'Europa', 'Ganymede']
target_fp = ['MAW', 'TEB']

exnum = ['050', '056']
PJ_list = [6, 10]

view_angle_thres_degree = 30.0

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


# %% data_load
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
    ax.plot([min, max],
            [y0, y0],
            color=ec, linewidth=lw,
            zorder=1)

    # Min
    ax.plot([min, min],
            [y0-width/2, y0+width/2],
            color=ec, linewidth=lw,
            zorder=1)

    # Max
    ax.plot([max, max],
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


# %% フィッティング用のexp関数
def fit_exp(params, x):
    a, b = params
    return a * np.exp(-x/b)


# %% フィッティング用のべき乗関数
def fit_power(params, x):
    a, b = params
    return a * (x/5.9)**b


#
#
#
#
#
#
#
#
#
# %% 横軸: 木星中心からの動径距離 (planetcentric) / 縦軸: FTMC
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=3, figsize=(7.1, 8.5), dpi='L')
F.initialize()
# F.panelname = [' a. Io ', ' b. Europa ', ' c. Ganymede ']

xmin = 3.0
xmax = 17.0
xticks = np.arange(xmin, xmax+1.0, 2.0)
xticklabels = xticks
F.set_xaxis(label=r'Radial distance [$R_{\rm J}$]',
            min=xmin, max=xmax,
            ticks=xticks,
            ticklabels=xticklabels,
            minor_num=4)

F.set_yaxis(ax_idx=0,
            label='FTMC [kg m$^{-2}$]',
            min=1E-11, max=1E-7,
            ticks=None,
            ticklabels=None,
            minor_num=5,
            yscale='log',)
F.set_yaxis(ax_idx=1,
            label=r'FTMC [kg m$^{-2}$]',
            min=1E-11, max=1E-7,
            ticks=None,
            ticklabels=None,
            minor_num=5,
            yscale='log',)
F.set_yaxis(ax_idx=2,
            label=r'FTMC [kg m$^{-2}$]',
            min=1E-11, max=1E-7,
            ticks=None,
            ticklabels=None,
            minor_num=5,
            yscale='log',)
F.fig.subplots_adjust(hspace=0.2)

IEG_median_arr = np.zeros(3)
IEG_error_arr = np.zeros(3)
IEG_median_arr_LT1 = np.zeros(3)
IEG_error_arr_LT1 = np.zeros(3)
IEG_median_arr_LT2 = np.zeros(3)
IEG_error_arr_LT2 = np.zeros(3)
for idx_moon in range(3):
    exdate = exdate_list[idx_moon]
    target_moon = target_moon_list[idx_moon]

    # Retrieval result list
    PJ_LIST = [1, 3]+np.arange(4, 43+1, 1).tolist()
    if target_moon == 'Io':
        Psyn = Psyn_io
        r_moon = 5.9*RJ
    elif target_moon == 'Europa':
        Psyn = Psyn_eu
        PJ_LIST.pop(24-2)
        PJ_LIST.pop(43-3)
        r_moon = 9.4*RJ
    elif target_moon == 'Ganymede':
        Psyn = Psyn_ga
        PJ_LIST.pop(24-2)
        PJ_LIST.pop(31-3)
        PJ_LIST.pop(39-4)
        PJ_LIST.pop(43-5)
        r_moon = 15.0*RJ

    if (exdate == '003/20250516') and (target_moon == 'Io'):
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

    elif (exdate == '005/20251221') and (target_moon == 'Europa'):
        exnum = ['333', '334', '335', '336', '362',
                 '338', '339', '340', '341', '342',
                 '363', '344', '345', '346', '347',
                 '348', '349', '350', '351', '352',
                 '353', '354', '355', '356', '357',
                 '358', '359', '360', '361',
                 ]
        PJ_list = [4, 7, 8, 8, 9,
                   11, 12, 13, 14, 17,
                   18, 19, 20, 22, 23,
                   25, 26, 29, 30, 31,
                   32, 33, 34, 35, 36,
                   38, 41, 48, 62,
                   ]
        FTMC_HEM = ['S', 'S', 'N', 'S', 'S',
                    'S', 'N', 'N', 'S', 'both',
                    'S', 'both', 'both', 'both', 'S',
                    'S', 'S', 'both', 'S', 'S',
                    'both', 'S', 'S', 'S', 'N',
                    'S', 'S', 'S', 'S',
                    ]

    elif (exdate == '005/20251221') and (target_moon == 'Ganymede'):
        exnum = ['301', '302', '303', '304', '305',
                 '306', '307', '308', '309', '310',
                 '311', '312', '313', '314', '315',
                 '316', '317', '318', '319', '320',
                 '321', '322', '323', '324', '325',
                 '326', '327', '328', '329', '330',
                 '331', '332',
                 ]
        PJ_list = [3, 4, 5, 6,
                   7,
                   8, 11, 12, 13, 16,
                   17, 19, 20, 21, 22,
                   25, 27, 30, 32, 34,
                   34, 35, 37, 38, 40,
                   41, 42, 46, 47, 50,
                   59, 60,
                   ]
        FTMC_HEM = ['both', 'both', 'S', 'both', 'S',
                    'both', 'N', 'both', 'both', 'S',
                    'S', 'S', 'N', 'S', 'N',
                    'S', 'both', 'S', 'S', 'N',
                    'S', 'both', 'S', 'S', 'S',
                    'N', 'N', 'S', 'S', 'S',
                    'S', 'N',
                    ]

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
    odr_run = odr_instance.run()

    # フィッティング結果
    popt = odr_run.beta
    perr = odr_run.sd_beta

    ftmc_all_points = []
    ftmc_LT1_points = []
    ftmc_LT2_points = []
    weight_all_points = []
    median_all_points = []
    error0_all_points = []
    ltcenter_all_points = []
    for i in range(len(exnum)):
        # %% Load the data
        exname = exdate+'_'+exnum[i]
        column_mass, chi2r, moon_et, _, moon_S3wlon, weight = data_load(
            exname)     # [kg m-2]
        column_mass *= 1E+9  # [10^-9 kg m-2]

        ftmc_all_points += column_mass.tolist()
        weight_all_points += weight.tolist()

        # Local time
        d0 = spice.et2datetime(moon_et[0])
        d0_list = []
        for ii in range(column_mass.size):
            d0_list += [d0]
        lt_arr = np.zeros(moon_et.size)
        for k in range(moon_et.size):
            lt_arr[k] = local_time_moon(moon_et[k], target_moon)

        lt_center = (lt_arr[0]+lt_arr[-1])/2
        ltcenter_all_points += [lt_center]
        if (lt_center <= 3.70+6.0) or (lt_center >= 3.70+18.0):
            ftmc_LT1_points += column_mass.tolist()
        else:
            ftmc_LT2_points += column_mass.tolist()

        q1, medians, q3 = weighted_percentile(data=column_mass,
                                              perc=[0.01, 0.5, 0.99],
                                              weights=weight)
        error0 = max(abs(q3-medians), abs(medians-q1))

        median_all_points += [medians]
        error0_all_points += [error0]

    ftmc_all_points = np.array(ftmc_all_points)
    ftmc_LT1_points = np.array(ftmc_LT1_points)
    ftmc_LT2_points = np.array(ftmc_LT2_points)
    weight_all_points = np.array(weight_all_points)
    median_all_points = np.array(median_all_points)
    error0_all_points = np.array(error0_all_points)
    ltcenter_all_points = np.array(ltcenter_all_points)

    sc = F.ax[0].scatter(r_moon/RJ, np.average(median_all_points)*1E-9,
                         marker='d', s=8.0, c=UC.blue)
    F.ax[0].errorbar(x=r_moon/RJ, y=np.average(median_all_points)*1E-9,
                     yerr=np.average(error0_all_points)*1E-9,
                     elinewidth=1.2, linewidth=0., markersize=0,
                     color=UC.blue)
    q1, medians, q3 = weighted_percentile(data=ftmc_all_points,
                                          perc=[0.25, 0.5, 0.75],
                                          weights=weight_all_points)
    width = 0.15
    if target_moon == 'Ganymede':
        for i in range(2):
            sc = F.ax[i+1].scatter(r_moon/RJ, np.average(median_all_points)*1E-9,
                                   marker='d', s=8.0, c=UC.blue)
            F.ax[i+1].errorbar(x=r_moon/RJ, y=np.average(median_all_points)*1E-9,
                               yerr=np.average(error0_all_points)*1E-9,
                               elinewidth=1.2, linewidth=0., markersize=0,
                               color=UC.blue)
    else:
        LT1 = np.where((ltcenter_all_points <= 3.70+6.0) |
                       (ltcenter_all_points >= 3.70+18.0))
        LT2 = np.where((ltcenter_all_points > 3.70+6.0) &
                       (ltcenter_all_points < 3.70+18.0))
        sc = F.ax[1].scatter(r_moon/RJ,
                             np.average(median_all_points[LT1])*1E-9,
                             marker='d', s=8.0, c=UC.blue)
        F.ax[1].errorbar(x=r_moon/RJ,
                         y=np.average(median_all_points[LT1])*1E-9,
                         yerr=np.average(error0_all_points[LT1])*1E-9,
                         elinewidth=1.2, linewidth=0., markersize=0,
                         color=UC.blue)
        sc = F.ax[2].scatter(r_moon/RJ,
                             np.average(median_all_points[LT2])*1E-9,
                             marker='d', s=8.0, c=UC.blue)
        F.ax[2].errorbar(x=r_moon/RJ,
                         y=np.average(median_all_points[LT2])*1E-9,
                         yerr=np.average(error0_all_points[LT2])*1E-9,
                         elinewidth=1.2, linewidth=0., markersize=0,
                         color=UC.blue)

    IEG_median_arr[idx_moon] = np.average(median_all_points)
    IEG_error_arr[idx_moon] = np.average(error0_all_points)
    IEG_median_arr_LT1[idx_moon] = np.average(median_all_points[LT1])
    IEG_error_arr_LT1[idx_moon] = np.average(error0_all_points[LT1])
    IEG_median_arr_LT2[idx_moon] = np.average(median_all_points[LT2])
    IEG_error_arr_LT2[idx_moon] = np.average(error0_all_points[LT2])

# (All LT) べき乗関数でフィッティング
data = RealData(np.array([5.9, 9.4, 15.0]),
                IEG_median_arr*1E-9,
                sx=None,
                sy=IEG_error_arr*1E-9
                )
model = Model(fit_power)
odr_instance = ODR(data, model, beta0=[1.0, 1.0])
odr_run = odr_instance.run()

# フィッティング結果
res_var = odr_run.res_var
popt_li = odr_run.beta
perr_li = odr_run.sd_beta
cov = odr_run.cov_beta*res_var

print("All LT")
print("Parameters:", popt_li)
print("Errors:", perr_li)
print("cov_beta:", odr_run.cov_beta*res_var)
print("res_var:", res_var)
x_fit = np.linspace(2.0, 18.0, 80)
y_fit = fit_power(popt_li, x_fit)
F.ax[0].plot(x_fit, y_fit, linewidth=1.0, color='k', zorder=2)

# ヤコビアンの計算
J_f0 = (x_fit/5.9)**popt_li[1]
J_f1 = y_fit*np.log(x_fit/5.9)
J_f = np.array([J_f0, J_f1])
sigma_f = np.zeros(x_fit.size)
for i in range(x_fit.size):
    sigma_f[i] = np.sqrt((J_f[:, i]@cov)@J_f[:, i].T)
y_fit_up = y_fit + 3.0*sigma_f
y_fit_dw = y_fit - 3.0*sigma_f
# print(sigma_f)
lsigma = F.ax[0].fill_between(x_fit, y_fit_up, y_fit_dw,
                              color=UC.gray, alpha=0.2,
                              label='2',
                              zorder=0.01)
F.ax[0].text(0.97, 0.80,
             r'$y=1.18(x/5.9)^{-5.49}$ [10$^{-8}$ kg m$^{-2}$]',
             color='k',
             horizontalalignment='right',
             verticalalignment='top',
             transform=F.ax[0].transAxes,
             fontsize=F.fontsize*0.5)

# (LT1) べき乗関数でフィッティング
data = RealData(np.array([5.9, 9.4, 15.0]),
                IEG_median_arr_LT1*1E-9,
                sx=None,
                sy=IEG_error_arr_LT1*1E-9
                )
model = Model(fit_power)
odr_instance = ODR(data, model, beta0=[1.0, 1.0])
odr_run = odr_instance.run()

# フィッティング結果
res_var = odr_run.res_var
popt_li = odr_run.beta
perr_li = odr_run.sd_beta
cov = odr_run.cov_beta*res_var
print("LT1")
print("Parameters:", popt_li)
print("Errors:", perr_li)
print("cov_beta:", odr_run.cov_beta*res_var)
print("res_var:", res_var)
x_fit = np.linspace(2.0, 18.0, 80)
y_fit = fit_power(popt_li, x_fit)
F.ax[1].plot(x_fit, y_fit, linewidth=1.0, color='k', zorder=2)

# ヤコビアンの計算
J_f0 = (x_fit/5.9)**popt_li[1]
J_f1 = y_fit*np.log(x_fit/5.9)
J_f = np.array([J_f0, J_f1])
sigma_f = np.zeros(x_fit.size)
for i in range(x_fit.size):
    sigma_f[i] = np.sqrt((J_f[:, i]@cov)@J_f[:, i].T)
y_fit_up = y_fit + 3.0*sigma_f
y_fit_dw = y_fit - 3.0*sigma_f
lsigma = F.ax[1].fill_between(x_fit, y_fit_up, y_fit_dw,
                              color=UC.gray, alpha=0.2,
                              label='2',
                              zorder=0.01)
F.ax[1].text(0.97, 0.80,
             r'$y=1.40(x/5.9)^{-5.57}$ [10$^{-8}$ kg m$^{-2}$]',
             color='k',
             horizontalalignment='right',
             verticalalignment='top',
             transform=F.ax[1].transAxes,
             fontsize=F.fontsize*0.5)

# (LT2) べき乗関数でフィッティング
data = RealData(np.array([5.9, 9.4, 15.0]),
                IEG_median_arr_LT2*1E-9,
                sx=None,
                sy=IEG_error_arr_LT2*1E-9
                )
model = Model(fit_power)
odr_instance = ODR(data, model, beta0=[1.0, 1.0])
odr_run = odr_instance.run()

# フィッティング結果
res_var = odr_run.res_var
popt_li = odr_run.beta
perr_li = odr_run.sd_beta
cov = odr_run.cov_beta*res_var
print("LT2")
print("Parameters:", popt_li)
print("Errors:", perr_li)
print("cov_beta:", odr_run.cov_beta*res_var)
print("res_var:", res_var)
x_fit = np.linspace(2.0, 18.0, 80)
y_fit = fit_power(popt_li, x_fit)
F.ax[2].plot(x_fit, y_fit, linewidth=1.0, color='k', zorder=2)

# ヤコビアンの計算
J_f0 = (x_fit/5.9)**popt_li[1]
J_f1 = y_fit*np.log(x_fit/5.9)
J_f = np.array([J_f0, J_f1])
sigma_f = np.zeros(x_fit.size)
for i in range(x_fit.size):
    sigma_f[i] = np.sqrt((J_f[:, i]@cov)@J_f[:, i].T)
y_fit_up = y_fit + 3.0*sigma_f
y_fit_dw = y_fit - 3.0*sigma_f
lsigma = F.ax[2].fill_between(x_fit, y_fit_up, y_fit_dw,
                              color=UC.gray, alpha=0.2,
                              label='2',
                              zorder=0.01)
F.ax[2].text(0.97, 0.80,
             r'$y=0.90(x/5.9)^{-5.35}$ [10$^{-8}$ kg m$^{-2}$]',
             color='k',
             horizontalalignment='right',
             verticalalignment='top',
             transform=F.ax[2].transAxes,
             fontsize=F.fontsize*0.5)

# Text
F.ax[0].text(0.97, 0.95,
             'All local time',
             color='k',
             horizontalalignment='right',
             verticalalignment='top',
             transform=F.ax[0].transAxes,
             fontsize=F.fontsize*0.8)
F.ax[1].text(0.97, 0.95,
             'Io & Europa (3.7$\\pm$6.0) LT',
             color='k',
             horizontalalignment='right',
             verticalalignment='top',
             transform=F.ax[1].transAxes,
             fontsize=F.fontsize*0.8)
F.ax[2].text(0.97, 0.95,
             'Io & Europa (15.7$\\pm$6.0) LT',
             color='k',
             horizontalalignment='right',
             verticalalignment='top',
             transform=F.ax[2].transAxes,
             fontsize=F.fontsize*0.8)

savedir = 'img/ftmc'
F.fig.savefig(savedir+'/ftmc_radialprofile_r.jpg',
              bbox_inches='tight')
F.close()
plt.show()
