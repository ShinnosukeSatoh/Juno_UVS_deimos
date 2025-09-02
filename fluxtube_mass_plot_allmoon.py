""" fluxtube_mass_plot.py

Created on May 9, 2025
@author: Shin Satoh

Description:
Make a plot of time variation in the flux tube mass contents
at the satellite orbit. The flux tube mass contents are calculated
from the retrieved ion parameters.

Version
1.0.0 (May 9, 2025)

"""
# %% Import
import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
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
exdate = '003/20250516'
target_moon = 'Ganymede'
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


# %% フィッティング用の三角関数
def fit_func(x, a, b, c, d):
    return a * np.cos(2*np.pi*(x-c)/24.0) + d


def fit_func2(params, x):
    a, c, d = params
    return a * np.cos(2*np.pi*(x-c)/24.0) + d


#
#
#
#
# %% 横軸をPJ番号でプロットする(2)
F = ShareXaxis()
F.fontsize = 23
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=3, figsize=(6.5, 7.5), dpi='L')
F.initialize()
F.panelname = [' a. Io ', ' b. Europa ', ' c. Ganymede ']

F.set_xaxis(label='PJ #',
            min=1, max=43,
            ticks=np.arange(1, 45+1, 4),
            ticklabels=np.arange(1, 45+1, 4),
            minor_num=4)

target_moon_list = ['Io', 'Europa', 'Ganymede']
for j in range(3):
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

    if target_moon == 'Io':
        ymax = 40
        ticks = np.arange(0, 40+1, 10)
    elif target_moon == 'Europa':
        ymax = 4.8
        ticks = np.arange(0, 4+1, 1)
    elif target_moon == 'Ganymede':
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

    F.set_yaxis(ax_idx=j,
                label='$M$\n[10$^{-9}$ kg m$^{-2}$]',
                min=0, max=ymax,
                ticks=ticks,
                ticklabels=ticks,
                minor_num=5)

    positions = np.arange(0, len(exnum)+1, 1)
    colormap = plt.cm.get_cmap('turbo')
    for i in range(len(exnum)):
        # %% Load the data
        exname = exdate+'_'+exnum[i]
        column_mass, chi2r, moon_et, _, moon_S3wlon, ftmc_mag_2da = data_load(
            exname)     # [kg m-2]
        column_mass *= 1E+9  # [10^-9 kg m-2]

        # Local time
        lt_arr = np.zeros(moon_et.size)
        for k in range(moon_et.size):
            lt_arr[k] = local_time_moon(moon_et[k], target_moon)

        lt_center = (lt_arr[0]+lt_arr[-1])/2

        _, medians, _ = np.percentile(column_mass, [25, 50, 75], axis=0)

        F.ax[j].scatter(PJ_list[i]*np.ones(column_mass.size),
                        column_mass,
                        s=2.75, color=UC.blue, alpha=0.11)
        F.ax[j].scatter(PJ_list[i], medians, s=13, marker='o',
                        edgecolor='k', facecolor='w', linewidth=0.5,
                        zorder=5)

        """# Local time
        F.ax[j].text(PJ_list[i], column_mass.min()*0.98,
                     str(round(lt_center, 1)),
                     color='k', fontsize=F.fontsize*0.35,
                     verticalalignment='top',
                     horizontalalignment='center',
                     zorder=6)"""

F.ax[0].set_title(r'Flux tube mass contents',
                  fontsize=F.fontsize, weight='bold')
save_dir = 'img/column_mass/'+exdate+'_'+target_moon+'/'
save_name = 'PJ'
if thres_scaleheight:
    save_name += '_H_thres'
save_name += '_sc_all'
F.fig.savefig(save_dir+save_name+'.jpg', bbox_inches='tight')
F.close()


#
#
#
#
# %% 横軸をPJ番号でプロットする(3) 単位: kg Wb-1
F = ShareXaxis()
F.fontsize = 23
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=3, figsize=(6.5, 7.5), dpi='L')
F.initialize()
F.panelname = [' a. Io ', ' b. Europa ', ' c. Ganymede ']

F.set_xaxis(label='PJ #',
            min=1, max=43,
            ticks=np.arange(1, 45+1, 4),
            ticklabels=np.arange(1, 45+1, 4),
            minor_num=4)

target_moon_list = ['Io', 'Europa', 'Ganymede']
for j in range(3):
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
        r_moon = 9.4*RJ
        B_0 = 370*1E-9     # [T]

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
        r_moon = 5.9*RJ
        B_0 = 1720*1E-9     # [T]

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
        r_moon = 15.0*RJ
        B_0 = 64*1E-9     # [T]

    if target_moon == 'Io':
        ymax = 20
        ticks = np.arange(0, 20+1, 5)
    elif target_moon == 'Europa':
        ymax = 12.0
        ticks = np.arange(0, 15, 5)
    elif target_moon == 'Ganymede':
        ymax = 3.4
        ticks = np.arange(0, 4, 1)

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

    F.set_yaxis(ax_idx=j,
                label='$\eta_B$\n[10$^{-3}$ kg Wb$^{-1}$]',
                min=0, max=ymax,
                ticks=ticks,
                ticklabels=ticks,
                minor_num=5)

    positions = np.arange(0, len(exnum)+1, 1)
    colormap = plt.cm.get_cmap('turbo')
    for i in range(len(exnum)):
        # %% Load the data
        exname = exdate+'_'+exnum[i]
        _, chi2r, moon_et, _, moon_S3wlon, ftmc_mag = data_load(
            exname)     # [kg m-2]

        # Magnetic field magnitude from community codes
        x = r_moon*np.cos(np.radians(360.0-np.median(moon_S3wlon)))
        y = r_moon*np.sin(np.radians(360.0-np.median(moon_S3wlon)))
        Bx0, By0, Bz0 = jm.Internal.Field(x/RJ, y/RJ, 0)  # [nT]
        Bx1, By1, Bz1 = jm.Con2020.Field(x/RJ, y/RJ, 0)   # [nT]
        Bx = (Bx0+Bx1)*1E-9     # [T]
        By = (By0+By1)*1E-9     # [T]
        Bz = (Bz0+Bz1)*1E-9     # [T]

        B0 = math.sqrt(Bx[0]**2+By[0]**2+Bz[0]**2)      # [T]

        # column_mass *= 1/B0  # [kg Wb-1]
        # column_mass *= 1000.0  # [10^-3 kg Wb-1]
        ftmc_mag *= 1000.0  # [10^-3 kg Wb-1]

        # Local time
        lt_arr = np.zeros(moon_et.size)
        for k in range(moon_et.size):
            lt_arr[k] = local_time_moon(moon_et[k], target_moon)

        lt_center = (lt_arr[0]+lt_arr[-1])/2

        _, medians, _ = np.percentile(ftmc_mag, [25, 50, 75], axis=0)

        F.ax[j].scatter(PJ_list[i]*np.ones(ftmc_mag.size),
                        ftmc_mag,
                        s=2.75, color=UC.blue, alpha=0.11)
        F.ax[j].scatter(PJ_list[i], medians, s=13, marker='o',
                        edgecolor='k', facecolor='w', linewidth=0.5,
                        zorder=5)

F.ax[0].set_title(r'Flux tube mass contents',
                  fontsize=F.fontsize, weight='bold')
save_dir = 'img/column_mass/'+exdate+'_'+target_moon+'/'
save_name = 'PJ'
if thres_scaleheight:
    save_name += '_H_thres'
save_name += '_sc_all_Wb'
F.fig.savefig(save_dir+save_name+'.jpg', bbox_inches='tight')
F.close()


#
#
#
#
# %% 横軸をPJ番号でプロットする(4)
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=3, figsize=(7.5, 7.5), dpi='L')
F.initialize()
F.panelname = [' a. Io ', ' b. Europa ', ' c. Ganymede ']

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
ticklabels = F.ax[2].get_xticklabels()
ticklabels[0].set_ha('center')

target_moon_list = ['Io', 'Europa', 'Ganymede']
for j in range(3):
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

    if target_moon == 'Io':
        ymax = 40
        ticks = np.arange(0, 40+1, 10)
    elif target_moon == 'Europa':
        ymax = 4.8
        ticks = np.arange(0, 4+1, 1)
    elif target_moon == 'Ganymede':
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

    if j == 1:
        F.set_yaxis(ax_idx=j,
                    label='Flux tube mass contents\n$M$\n[10$^{-9}$ kg m$^{-2}$]',
                    min=0, max=ymax,
                    ticks=ticks,
                    ticklabels=ticks,
                    minor_num=5)
    else:
        F.set_yaxis(ax_idx=j,
                    label='$M$\n[10$^{-9}$ kg m$^{-2}$]',
                    min=0, max=ymax,
                    ticks=ticks,
                    ticklabels=ticks,
                    minor_num=5)

    positions = np.arange(0, len(exnum)+1, 1)
    colormap = plt.cm.get_cmap('turbo')
    for i in range(len(exnum)):
        # %% Load the data
        exname = exdate+'_'+exnum[i]
        column_mass, chi2r, moon_et, _, moon_S3wlon, _ = data_load(
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

        _, medians, _ = np.percentile(column_mass, [25, 50, 75], axis=0)

        F.ax[j].scatter(d0_list,
                        column_mass,
                        s=2.75, color=UC.blue, alpha=0.11)
        F.ax[j].scatter(d0, medians, s=13, marker='o',
                        edgecolor='k', facecolor='w', linewidth=0.5,
                        zorder=5)

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

# Shades in each 5 perijove
for i in range(3):
    F.ax[i].axvspan(xticks[0], xticks[5], fc=UC.gray, ec=None, alpha=0.15)
    F.ax[i].axvspan(xticks[10], xticks[15], fc=UC.gray, ec=None, alpha=0.15)
    F.ax[i].axvspan(xticks[20], xticks[25], fc=UC.gray, ec=None, alpha=0.15)
    F.ax[i].axvspan(xticks[30], xticks[35], fc=UC.gray, ec=None, alpha=0.15)
    F.ax[i].axvspan(xticks[40], xticks[45], fc=UC.gray, ec=None, alpha=0.15)

save_dir = 'img/column_mass/'+exdate+'_'+target_moon+'/'
save_name = 'date'
if thres_scaleheight:
    save_name += '_H_thres'
save_name += '_sc_all'
F.fig.savefig(save_dir+save_name+'.jpg', bbox_inches='tight')
F.fig.savefig(save_dir+save_name+'.pdf', bbox_inches='tight')
F.close()


#
#
#
#
# %% 横軸をPJ番号でプロットする(5)
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(figsize=(7.5, 2.0), dpi='L')
F.initialize()

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
ticklabels = F.ax.get_xticklabels()
ticklabels[0].set_ha('center')

target_moon_list = ['Io', 'Europa', 'Ganymede']
for j in range(3):
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

    if target_moon == 'Io':
        ymax = 40
        ticks = np.arange(-10, 10+1, 2)
    elif target_moon == 'Europa':
        ymax = 4.8
        ticks = np.arange(0, 4+1, 1)
    elif target_moon == 'Ganymede':
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
                label='$M$ \n[10$^{-9}$ kg m$^{-2}$]',
                min=0.0, max=40.0,
                ticks=np.arange(0, 40+1, 5),
                ticklabels=np.arange(0, 40+1, 5),
                minor_num=5)

    positions = np.arange(0, len(exnum)+1, 1)
    colormap = plt.cm.get_cmap('turbo')
    for i in range(len(exnum)):
        # %% Load the data
        exname = exdate+'_'+exnum[i]
        column_mass, chi2r, moon_et, _, moon_S3wlon, _ = data_load(
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

        _, medians, _ = np.percentile(column_mass, [25, 50, 75], axis=0)

        F.ax.scatter(d0_list,
                     column_mass,
                     s=2.75, color=UC.blue, alpha=0.11)
        F.ax.scatter(d0, medians, s=13, marker='o',
                     edgecolor='k', facecolor='w', linewidth=0.5,
                     zorder=5)

    if j == 0:
        break

# u_ax = F.upper_ax()
# u_ax.set_title(r'Flux tube mass contents',
#               fontsize=F.fontsize, weight='bold')

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

# Shades in each 5 perijove
F.ax.axvspan(xticks[0], xticks[5], fc=UC.gray, ec=None, alpha=0.15)
F.ax.axvspan(xticks[10], xticks[15], fc=UC.gray, ec=None, alpha=0.15)
F.ax.axvspan(xticks[20], xticks[25], fc=UC.gray, ec=None, alpha=0.15)
F.ax.axvspan(xticks[30], xticks[35], fc=UC.gray, ec=None, alpha=0.15)
F.ax.axvspan(xticks[40], xticks[45], fc=UC.gray, ec=None, alpha=0.15)

save_dir = 'img/column_mass/'+exdate+'_Io/'
save_name = 'date'
if thres_scaleheight:
    save_name += '_H_thres'
save_name += '_sc_all_residual'
F.fig.savefig(save_dir+save_name+'.jpg', bbox_inches='tight')
F.close()
