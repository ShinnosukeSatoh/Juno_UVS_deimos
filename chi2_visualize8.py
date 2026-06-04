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
import math
import datetime

import Leadangle_wave as Wave
from Leadangle_fit_JunoUVS import eqwlong_err
from Leadangle_fit_JunoUVS import TEB_transit
from Leadangle_fit_JunoUVS import create_argmesh
from column_mass import calc as column_calc
from UniversalColor import UniversalColor
from SharedX import ShareXaxis
from legend_shadow import legend_shadow

import JupiterMag as jm

jm.Internal.Config(Model='jrm33', CartesianIn=True, CartesianOut=True)
jm.Con2020.Config(equation_type='analytic')

UC = UniversalColor()
UC.set_palette()

F = ShareXaxis()
F.fontsize = 20
F.fontname = 'Liberation Sans Narrow'
F.set_default()


# %%
exdir = '1001/20260421'
TARGET_MOON = 'Ganymede'
target_fp = ['MAW', 'TEB']
PJ_num = [23]
hem = 'N'
Ai_num = 3
ni_num = 150
Ti_num = 1
Zi = 1.3                # Io: 1.3 / Eu: 1.4 / Ga: 1.3
Te = 300.0              # Io: 6.0 [eV]/ Eu: 20.0 / Ga: 300.0


# %% Footprint obs. list (Ganymede)
PJ_LIST = [3, 4, 5, 6,
           7,  # 8,
           8, 11, 12,
           12, 13, 14, 15,
           16, 17, 19, 20, 21,
           22, 23,
           ]
HEM_LIST = ['S', 'both', 'S', 'both',
            'S',  # 'N',
            'S', 'N', 'N',
            'S', 'both', 'S', 'N',
            'S', 'S', 'S', 'N', 'S',
            'N', 'both'
            ]
EXNAME_LIST = ['030', '031', '032', '033',
               '034',  # '035',
               '036', '037', '038',
               '039', '040', '042', '043',
               '044', '045', '048', '049', '050',
               '051', '052',
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
    xticks = np.array([1, 10, 100, 1000])


# %% 経度0度(y=0)平面のx-z対応テーブル (900km高度)
extradius = np.loadtxt('data/Alt_900km/rthetaphi.txt')
r_e = extradius[0, :]        # [RJ]
theta_e = np.radians(extradius[1, :])    # [rad]
phi_e = np.radians(extradius[2, :])      # [rad]


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


# %% Load the retrival results
ni_best = np.zeros(len(PJ_LIST))
ni_err_0 = np.zeros(len(PJ_LIST))
ni_err_1 = np.zeros(len(PJ_LIST))
Hp = np.zeros(len(PJ_LIST))
D_thick = np.zeros(len(PJ_LIST))
mu_i_Con2020 = np.zeros(len(PJ_LIST))
rho_Con2020 = np.zeros(len(PJ_LIST))
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
    sigma_obs = sigma_total
    print('chi2_1d.shape:', chi2_1d.shape)
    print('eqlead_est.shape:', eqlead_est.shape)

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

    # Fing the matched pj number
    select_pj = np.where(con20_pj_idx == PJ_LIST[i])
    print('Con2020 PJ', con20_pj_idx[select_pj][0])

    # Read the current constant
    mui_coef = np.loadtxt('results/azimuthal_current_fit/' +
                          TARGET_MOON[0:2]+'_coef_0.txt')
    mui_coef_err = np.loadtxt('results/azimuthal_current_fit/' +
                              TARGET_MOON[0:2]+'_coef_1.txt')

    # Read the magnetodisk thickness coefficient
    D_coef = np.loadtxt('results/magdisk_thickness_fit/' +
                        TARGET_MOON[0:2]+'_coef_0.txt')
    D_coef_err = np.loadtxt('results/magdisk_thickness_fit/' +
                            TARGET_MOON[0:2]+'_coef_1.txt')

    D_disk = 3.6*RJ                                      # [m]
    Hp[i] = (2/np.sqrt(np.pi))*D_disk*D_coef[select_pj]  # [m]
    D_thick[i] = D_disk*D_coef[select_pj]                # [m]

    # Best fit ion density
    d_chi2 = d_chi2_3d[:, 1, 0]
    print(d_chi2.shape)
    print(np.where(d_chi2 <= 9.00))
    Ai_best = Ai_3d[0, 1, 0]
    ni_best[i] = ni_3d[:, 1, 0][np.argmin(d_chi2)]
    ni_err_0[i] = ni_best[i]-np.min(ni_3d[:, 1, 0][np.where(d_chi2 <= 9.00)])
    ni_err_1[i] = np.max(ni_3d[:, 1, 0][np.where(d_chi2 <= 9.00)])-ni_best[i]
    print('ni_best:', ni_best[i])
    print('ni_err:', ni_err_0[i], ni_err_1[i])

    # Connerney+2020の結果から質量密度を類推する
    mu_i_default = 139.6    # default: 139.6 [nT]
    d_rj_default = 3.6      # default: 3.6 [RJ]
    jm.Con2020.Config(mu_i=mu_i_default*mui_coef[select_pj],
                      d=d_rj_default*D_coef[select_pj],
                      equation_type='analytic')
    Bx0, By0, Bz0 = jm.Internal.Field(r_moon/RJ, 0.0, 0.0)  # [nT]
    Bx1, By1, Bz1 = jm.Con2020.Field(r_moon/RJ, 0.0, 0.0)   # [nT]
    Bx = (Bx0+Bx1)*1E-9     # [T]
    By = (By0+By1)*1E-9     # [T]
    Bz = (Bz0+Bz1)*1E-9     # [T]
    rho_Con2020[i] = - con20_mu_i_tot[select_pj] * \
        (1E-9)*(2*(Bz0)*(1E-9)/(MU0*(OMGJ*r_moon)**2))
    rho_Con2020[i] *= (1/AMU2KG)*1E-6      # [AMU cm-3]
    mu_i_Con2020[i] = mu_i_default*mui_coef[select_pj]  # [nT]
    print(rho_Con2020[i], '[AMU cm-3]')


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


# %% 横軸を時間でプロットする
# ======================
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
            label=r'$n_{\rm i}$ [cm$^{-3}$]',
            min=0, max=30,
            ticks=np.linspace(0, 30, 7),
            ticklabels=np.linspace(0, 30, 7),
            minor_num=2)

# Magnetodisk thickness [RJ]
Rax = F.ax.twinx()
Rax.set_ylabel(r'Disk thickness [$R_{\rm J}$]')
Rax.set_ylim(0, 5)
Rax.set_yticks(np.linspace(0, 5, 6))
Rax.yaxis.set_minor_locator(AutoMinorLocator(2))  # minor ticks

for i in range(len(PJ_LIST)):
    x = JUNO_PJ_TIMES[int(PJ_LIST[i]-1)]
    y = ni_best[i]
    F.ax.scatter(x, y, marker='s', s=5.0, c=UC.red)
    F.ax.errorbar(x=x, y=y,
                  yerr=[[ni_err_0[i]], [ni_err_1[i]]],
                  elinewidth=1.1, linewidth=0., markersize=0,
                  color=UC.red)

    # F.ax.scatter(x, rho_Con2020[i]/Ai_best,
    #              marker='s', s=5.0, c=UC.orange)

    Rax.scatter(x, D_thick[i]/RJ, marker='s', s=5.0, c=UC.blue)

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

F.fig.savefig('img/ftmc/'+TARGET_MOON[0:2]+'/' + exdir + '/ni_timeseries.jpg',
              bbox_inches='tight')
F.close()


# %% 横軸: n_i / 縦軸: Current constant
# ======================
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(5.0, 5.0), dpi='L')
F.initialize()
# F.panelname = [' a. Io ', ' b. Europa ', ' c. Ganymede ']

F.set_xaxis(label=r'$n_{\rm i}$ [cm$^{-3}$]',
            min=0, max=25,
            ticks=np.linspace(0, 25, 6),
            ticklabels=np.linspace(0, 25, 6),
            minor_num=5)
F.set_yaxis(ax_idx=0,
            label='Current constant [nT]',
            min=0, max=200,
            ticks=np.linspace(0, 200, 5),
            ticklabels=np.linspace(0, 200, 5),
            minor_num=5)

for i in range(len(PJ_LIST)):
    x = ni_best[i]
    y = mu_i_Con2020[i]
    F.ax.scatter(x, y, marker='s', s=5.0, c=UC.red)
    F.ax.errorbar(x=x, y=y,
                  xerr=[[ni_err_0[i]], [ni_err_1[i]]],
                  elinewidth=1.1, linewidth=0., markersize=0,
                  color=UC.red)


F.fig.savefig('img/ftmc/'+TARGET_MOON[0:2]+'/' + exdir + '/ni_vs_current.jpg',
              bbox_inches='tight')
F.close()


# %% 横軸: FTMC / 縦軸: Current constant
# ======================
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(5.0, 5.0), dpi='L')
F.initialize()
# F.panelname = [' a. Io ', ' b. Europa ', ' c. Ganymede ']

F.set_xaxis(label=r'FTMC [10$^{-9}$ kg m$^{-2}$]',
            min=0, max=0.2,
            ticks=np.linspace(0, 2, 5)/10,
            ticklabels=np.linspace(0, 2, 5)/10,
            minor_num=5)
F.set_yaxis(ax_idx=0,
            label='Current constant [nT]',
            min=0, max=200,
            ticks=np.linspace(0, 200, 5),
            ticklabels=np.linspace(0, 200, 5),
            minor_num=5)

for i in range(len(PJ_LIST)):
    x = Ai_best*AMU2KG*ni_best[i]*1E+6*Hp[i]*np.sqrt(np.pi)
    y = mu_i_Con2020[i]
    F.ax.scatter(x*1E+9, y, marker='s', s=5.0, c=UC.red)
    """F.ax.errorbar(x=x, y=y,
                  xerr=[[ni_err_0[i]], [ni_err_1[i]]],
                  elinewidth=1.1, linewidth=0., markersize=0,
                  color=UC.red)"""
    print(Hp[i]/RJ)

F.fig.savefig('img/ftmc/'+TARGET_MOON[0:2]+'/' + exdir + '/ftmc_vs_current.jpg',
              bbox_inches='tight')
F.close()
