# %% Import
import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FixedLocator
import matplotlib.dates as mdates
import math
import datetime
from UniversalColor import UniversalColor
from SharedX import ShareXaxis
from legend_shadow import legend_shadow

import Leadangle_wave as Wave
from Leadangle_fit_JunoUVS import local_time_moon
from Leadangle_fit_JunoUVS import spice_moonS3

import JupiterMag as jm
jm.Internal.Config(Model='jrm33', CartesianIn=True, CartesianOut=True)
jm.Con2020.Config(equation_type='analytic')

spice.furnsh('kernel/cassMetaK.txt')
savpath = 'data/Satellite_FP_JRM33.sav'

UC = UniversalColor()
UC.set_palette()


# %% Settings
target_moon = ['Io', 'Europa', 'Ganymede']
target_fp = ['MAW', 'TEB']
thres_scaleheight = False
fit = True

Ai_num = 3
ni_num = 50
Ti_num = 60


# %% Constants
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


# %% Load the data
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
    min_idx_Ai = min_idx[1][0]
    print('Min chi2:', np.min(chi2_3d), 'at', min_idx)
    print('Scale height [RJ]:', H_3d[min_idx][0]/(71492*1E+3))

    # 1データのヒストグラムを作成する
    Ai_2d = Ai_3d[:, min_idx_Ai, :]
    ni_2d = ni_3d[:, min_idx_Ai, :]
    H_2d = H_3d[:, min_idx_Ai, :]
    Ti_2d = Ti_3d[:, min_idx_Ai, :]
    chi2_2d = chi2_3d[:, min_idx_Ai, :]
    column_mass_2da = column_mass_3d[:, min_idx_Ai, :]
    ftmc_mag_2da = ftmc_mag_3d[:, min_idx_Ai, :]

    d_chi2 = chi2_2d-np.min(chi2_2d)
    Ai_2d = Ai_2d[np.where(d_chi2 < dchi_3s)]
    ni_2d = ni_2d[np.where(d_chi2 < dchi_3s)]
    H_2d = H_2d[np.where(d_chi2 < dchi_3s)]
    Ti_2d = Ti_2d[np.where(d_chi2 < dchi_3s)]
    column_mass_2da = column_mass_2da[np.where(d_chi2 < dchi_3s)]
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
    return column_mass_2da, chi2_R, moon_et, Ai_2d, moon_s3wlon, ftmc_mag_2da


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


#
#
#
#
# %% 横軸をPJ番号でプロットする(2)
F = ShareXaxis()
F.fontsize = 20
F.fontname = 'Liberation Sans Narrow'

panel_num = len(target_moon)
F.set_figparams(nrows=panel_num,
                figsize=(7.5, 2.6*panel_num+0.1),
                dpi='L')
F.initialize()
F.panelname = [' a. Io ', ' b. Europa ', ' c. Ganymede '][0:panel_num]

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
          datetime.datetime.strptime('2025-01-01', '%Y-%m-%d'),]
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
ticklabels = F.ax[panel_num-1].get_xticklabels()
ticklabels[0].set_ha('center')

for j in range(len(target_moon)):
    target_moon[j]
    if target_moon[j] == 'Io':
        # data from 003/20250516
        exdate = '003/20250516'
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
                 '146', '147', '148', '149',  # '162',
                 '151',
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
                   45, 45, 46, 48,  # 58,
                   60,
                   ]
        Psyn = Psyn_io
        ymax = 50
        ticks = np.arange(0, 50+1, 10)
    elif target_moon[j] == 'Europa':
        # data from 005/20251221
        exdate = '005/20251221'
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
        ymax = 3.0
        ticks = np.arange(0, 3+0.1, 0.5)[:-1]
    elif target_moon[j] == 'Ganymede':
        # data from 005/20251221
        exdate = '005/20251221'
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
        ymax = 0.25
        ticks = np.round(np.arange(0, 0.25+0.01, 0.05), 2)[:-1]

    column_mass_1dN = np.loadtxt(
        'results/column_mass/'+exdate+'_'+target_moon[j]+'/col_massdens_1dN.txt')
    column_mass_1dS = np.loadtxt(
        'results/column_mass/'+exdate+'_'+target_moon[j]+'/col_massdens_1dS.txt')
    ftmc_mag_1dN = np.loadtxt(
        'results/column_mass/'+exdate+'_'+target_moon[j]+'/ftmc_mag_1dN.txt')
    ftmc_mag_1dS = np.loadtxt(
        'results/column_mass/'+exdate+'_'+target_moon[j]+'/ftmc_mag_1dS.txt')
    column_mass_1d = column_mass_1dN+column_mass_1dS
    column_mass_3d = column_mass_1d.reshape(ni_num, Ai_num, Ti_num)
    ftmc_mag_1d = ftmc_mag_1dN + ftmc_mag_1dS
    ftmc_mag_3d = ftmc_mag_1d.reshape(ni_num, Ai_num, Ti_num)

    F.set_yaxis(ax_idx=j,
                label='[10$^{-9}$ kg m$^{-2}$]',
                min=0, max=ymax,
                ticks=ticks,
                ticklabels=ticks,
                minor_num=5)

    positions = np.arange(0, len(exnum)+1, 1)
    medians_arr = np.zeros(len(exnum))
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
            lt_arr[k] = local_time_moon(moon_et[k], target_moon[j])

        lt_center = (lt_arr[0]+lt_arr[-1])/2

        q1, medians, q3 = weighted_percentile(data=column_mass,
                                              perc=[0.25, 0.5, 0.75],
                                              weights=weight)
        width = datetime.timedelta(seconds=60*60*24*20)
        weighted_boxplot2(F.ax[j], d0, q1, medians, q3,
                          np.min(column_mass),
                          np.max(column_mass), width=width,
                          ec=UC.blue, lw=1.1)

        medians_arr[i] = medians

    print(target_moon[j])
    print('Average over the Juno epoch:', np.average(medians))

PJax = F.ax[0].twiny()
PJax.set_title(r'Flux tube mass content',
               fontsize=F.fontsize, weight='bold')
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
          datetime.datetime.strptime('2024-11-24', '%Y-%m-%d'),
          datetime.datetime.strptime('2024-12-27', '%Y-%m-%d'),
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
               '66', '', '',]
PJax.set_xlim(xmin, xmax)
PJax.set_xticks(xticks[::5])
PJax.set_xticklabels(xticklabels[::5])
PJax.xaxis.set_minor_locator(FixedLocator(mdates.date2num(xticks)))
PJax.tick_params('y', grid_zorder=-10)

# Shades in each 5 perijove
for i in range(panel_num):
    F.ax[i].axvspan(xticks[0], xticks[5], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[10], xticks[15], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[20], xticks[25], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[30], xticks[35], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[40], xticks[45], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[50], xticks[55], fc=UC.gray, ec=None, alpha=0.10)
    F.ax[i].axvspan(xticks[60], xticks[65], fc=UC.gray, ec=None, alpha=0.10)

F.fig.savefig('timeseries.jpg', bbox_inches='tight')
F.close()
