""" fluxtube_mass_plot_Wb.py

Created on July 4, 2025
@author: Shin Satoh

Description:
Make a plot of time variation in the flux tube mass contents
at the satellite orbit. The flux tube mass contents are calculated
from the retrieved ion parameters.

Version
1.0.0 (July 4, 2025)

"""
# %% Import
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
from matplotlib.ticker import FormatStrFormatter
import math
import datetime
from UniversalColor import UniversalColor
from SharedX import ShareXaxis
from legend_shadow import legend_shadow

import Leadangle_wave as Wave
from Leadangle_fit_JunoUVS import local_time_moon
from Leadangle_fit_JunoUVS import spice_moonS3

from scipy.odr import ODR, Model, RealData

import spiceypy as spice
spice.furnsh('kernel/cassMetaK.txt')
savpath = 'data/Satellite_FP_JRM33.sav'

UC = UniversalColor()
UC.set_palette()


# %% Settings
exdate = '003/20250516'
target_moon = 'Io'
target_fp = ['MAW', 'TEB']
thres_scaleheight = False
fit = False

if (exdate == '002/20250427') and (target_moon == 'Io'):
    exnum = ['006',
             '082', '007', '041', '008',
             '013', '015', '017', '019', '028',
             '029', '056',
             '030', '031', '032', '033',
             '034', '035', '036', '037', '083', '084',
             '039', '040', '042', '043', '044',
             '045', '046', '047', '048', '049',
             '104', '105',
             '051', '052', '053', '085', '086', '055',
             '112', '113']
    PJ_list = [3,
               4, 5, 6, 7,
               9, 10, 11, 12, 13,
               14, 15,
               16, 17, 18, 19,
               20, 21, 22, 23, 24, 25,
               26, 27, 28, 29, 30,
               31, 32, 33, 34, 35,
               36, 37,
               38, 39, 40, 41, 42, 43,
               8, 8]
    # めも: PJ15, 35は怪しい

if (exdate == '002/20250427') and (target_moon == 'Europa'):
    exnum = ['003', '004', '063', '001', '057',
             '002', '005', '060', '061',
             '069', '070', '062', '071',
             '064', '065', '066', '067', '068',
             '072', '073', '087', '088',
             '074', '075', '076',
             '077', '078', '079', '080', '081',
             ]
    PJ_list = [3, 4, 5.5, 7, 8.5,
               10, 11, 12, 13.5,
               15, 16, 17, 18,
               19, 20, 21, 22, 23,
               25, 26, 27, 28,
               30, 32, 33,
               34, 35, 36.5, 38, 41
               ]
    # めも: PJ18は怪しい

if (exdate == '002/20250427') and (target_moon == 'Ganymede'):
    exnum = ['021', '022', '023', '024', '025',
             '026', '027', '089', '090', '091',
             '092', '093', '094', '095', '096',
             '097', '098', '099', '100', '101',
             '102', '103', '106', '107', '108',
             '109', '110', '111'
             ]
    PJ_list = [3, 4, 6, 8, 9.5,
               11, 12, 13, 14, 15,
               16, 17, 18, 19, 20,
               21, 22, 25, 26, 27,
               30, 32, 33, 34, 35,
               37, 38, 42
               ]

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
    hems = [0, 0, 0, 0,
            0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0,
            -1, 1,
            -1, 1, -1, 1,]

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
             # '114', '116',
             # '147', '148', '149', '150', '151',
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
               # 18, 12,
               # 45, 46, 48, 58, 60,
               ]
    hems = [0, 0, 0, 0, 0,
            -1, 1, -1, 1, -1, 1,
            -1, 1,
            -1, 1, -1, 0, 0,
            -1, 0, 0, 0, 0,
            1, 1, 1, 0, -1,
            1, 1, 1, 1, 1,
            -1,
            -1, 1, 1, 1,
            1, 0, 0, 0, 0,
            0, 0, -1, 1,
            # 1, 0,
            # 1, 1, 1, 1, 1,
            ]

if (exdate == '003/20250516') and (target_moon == 'Ganymede'):
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
    hems = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, -1, 1, 0, 0,
            0, 0, 0, 0,
            ]

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


column_mass_1dN = np.loadtxt(
    'results/column_mass/'+exdate+'_'+target_moon+'/ftmc_mag_1dN.txt')
column_mass_1dS = np.loadtxt(
    'results/column_mass/'+exdate+'_'+target_moon+'/ftmc_mag_1dS.txt')
column_mass_1d = column_mass_1dN+column_mass_1dS
column_mass_3d = column_mass_1d.reshape(ni_num, Ai_num, Ti_num)


# %% Threshold scale height
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
    column_mass_2d = column_mass_3d[:, min_idx_Ai, :]

    if thres_scaleheight:
        Ai_2d = Ai_2d[np.where(H_2d < H_thres)]
        ni_2d = ni_2d[np.where(H_2d < H_thres)]
        Ti_2d = Ti_2d[np.where(H_2d < H_thres)]
        column_mass_2d = column_mass_2d[np.where(H_2d < H_thres)]
        chi2_2d = chi2_2d[np.where(H_2d < H_thres)]
        H_2d = H_2d[np.where(H_2d < H_thres)]  # 一番最後に

    d_chi2 = chi2_2d-np.min(chi2_2d)
    Ai_2d = Ai_2d[np.where(d_chi2 < dchi_3s)]
    ni_2d = ni_2d[np.where(d_chi2 < dchi_3s)]
    H_2d = H_2d[np.where(d_chi2 < dchi_3s)]
    Ti_2d = Ti_2d[np.where(d_chi2 < dchi_3s)]
    column_mass_2d = column_mass_2d[np.where(d_chi2 < dchi_3s)]
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
    return column_mass_2d, chi2_R, moon_et, Ai_2d


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
# %% 横軸LT(48時間) 散布図 ベストフィット付き (これが一番大事)
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

if target_moon == 'Io':
    ymax = 15
    ticks = np.arange(0, 15+1, 5)
elif target_moon == 'Europa':
    ymax = 12
    ticks = np.arange(0, 10+1, 5)
elif target_moon == 'Ganymede':
    ymax = 3.5
    ticks = np.arange(0, 3, 1)

F.set_figparams(nrows=1, figsize=(7, 4), dpi='L')
F.initialize()

F.set_xaxis(label=target_moon+' local time [hour]',
            min=0, max=48,
            ticks=np.arange(0, 48+1, 3),
            ticklabels=np.arange(0, 48+1, 3),
            minor_num=3)
F.set_yaxis(ax_idx=0,
            label='$\eta_B$\n[10$^{-3}$ kg Wb$^{-1}$]',
            min=0, max=ymax,
            ticks=ticks,
            ticklabels=ticks,
            minor_num=5)
F.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

half_width = 0.4

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
        column_mass, chi2r, moon_et, _ = data_load(exname)     # [kg m-2]
        column_mass *= 1000.0  # [10^-3 kg Wb-1]

        # Local time
        lt_arr = np.zeros(moon_et.size)
        for k in range(moon_et.size):
            lt_arr[k] = local_time_moon(moon_et[k], target_moon)

        lt_center = ((lt_arr[0]+lt_arr[-1])/2)+24.0*ii
        lt_range = abs(lt_arr[0]-lt_arr[-1])

        q25, medians, q75 = np.percentile(
            column_mass, [25, 50, 75], axis=0)
        q05, medians, q95 = np.percentile(
            column_mass, [5, 50, 95], axis=0)

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
# perr = np.sqrt(np.diag(output.cov_beta)) # maybe wrong

# ベストフィット
nstd = 3  # 95% 信頼区間
x_fit = np.linspace(0.0, 48.0, 120)
y_fit = fit_func2(popt, x_fit)

# 予測区間の計算
y_fit_err2 = ((np.pi*popt[0]/12.0)*np.sin(np.pi*(x_fit-popt[1])/12.0)**2)*(
    perr[1]**2) + (np.cos(np.pi*(x_fit-popt[1])/12.0)*perr[0])**2 + perr[2]**2
y_fit_err = np.sqrt(y_fit_err2)
y_fit_up = y_fit + nstd * y_fit_err
y_fit_dw = y_fit - nstd * y_fit_err
"""
lmean, = F.ax.plot(x_fit, y_fit, zorder=0.1, label='1', color='k')
lsigma = F.ax.fill_between(x_fit, y_fit_up, y_fit_dw,
                           color=UC.gray, alpha=0.2,
                           label='2',
                           zorder=0.01)
"""

jj = 0
median_arr = np.zeros(len(exnum))
for ii in range(2):
    for i in range(len(exnum)):
        # %% Load the data
        exname = exdate+'_'+exnum[i]
        column_mass, chi2r, moon_et, _ = data_load(exname)     # [kg m-2]
        column_mass *= 1000.0  # [10^-3 kg Wb-1]

        # Local time
        lt_arr = np.zeros(moon_et.size)
        s3_arr = np.zeros(moon_et.size)
        for k in range(moon_et.size):
            lt_arr[k] = local_time_moon(moon_et[k], target_moon)
            _, _, _, _, _, _, s3_arr[k] = spice_moonS3(moon_et[k], target_moon)

        lt_center = ((lt_arr[0]+lt_arr[-1])/2)+24.0*ii
        lt_range = abs(lt_arr[0]-lt_arr[-1])

        q25, medians, q75 = np.percentile(
            column_mass, [25, 50, 75], axis=0)
        q05, medians, q95 = np.percentile(
            column_mass, [5, 50, 95], axis=0)
        F.ax.scatter(lt_center*np.ones(column_mass.size),
                     column_mass,
                     s=0.6, color=UC.blue, alpha=0.11)

        F.ax.scatter(lt_center, medians, s=15, marker='o',
                     edgecolor='k', facecolor='w', linewidth=0.5,
                     zorder=4)

        if abs(PJ_list[i]-round(PJ_list[i])) < 0.1:
            F.ax.errorbar(lt_center, medians,
                          xerr=np.array([[abs(lt_arr[0]+(24.0*ii)-lt_center)],
                                         [abs(lt_arr[-1]+(24.0*ii)-lt_center)]]),
                          color='k', linewidth=0.4, elinewidth=0.4,
                          capsize=8.0*half_width,
                          capthick=0.4,
                          zorder=5)

        if jj < len(exnum):
            jj += 1

        median_arr[i] = medians

# Dummy
sc = F.ax.scatter(-5, 1, s=2, color=UC.blue, label='3', zorder=4)

# LT=24.0
F.ax.axvline(x=24.0, color=UC.gray, linewidth=0.75)

# Repeated for clarity
F.ax.text(0.98, 0.03,
          '2400-4800LT is repeated for clarity.',
          color='k',
          horizontalalignment='right',
          verticalalignment='bottom',
          transform=F.ax.transAxes,
          fontsize=F.fontsize*0.5)

"""legend = F.legend(ax_idx=0,
                  handles=[sc, (lsigma, lmean)],
                  labels=['Estimated', r'Best fit + 3$\sigma$'],
                  bbox_to_anchor=(1.0, 1.02),
                  ncol=3, markerscale=3,
                  fontsize_scale=0.6, textcolor=True, handletextpad=0.2)
legend_shadow(fig=F.fig, ax=F.ax, legend=legend)"""

F.ax.set_title(r'Flux tube mass contents ('+target_moon+')',
               fontsize=F.fontsize, weight='bold')
save_dir = 'img/column_mass/'+exdate+'_'+target_moon+'/'
save_name = 'Wb_LT48'
if thres_scaleheight:
    save_name += '_H_thres'
save_name += '_sc'
save_name += '_ODRfit'
F.fig.savefig(save_dir+save_name+'.jpg', bbox_inches='tight')
F.fig.savefig(save_dir+save_name+'.pdf', bbox_inches='tight')

print('Median average [10^-3 kg Wb-1]:', np.average(median_arr))
