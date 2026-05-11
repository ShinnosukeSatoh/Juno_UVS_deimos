import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mplcolors
import matplotlib.ticker as ptick
import matplotlib.colorbar as mplcolorbar
import matplotlib.cm as mplcm
from matplotlib.ticker import AutoMinorLocator
import math
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
exname = '1001/20260421_015'
TARGET_MOON = 'Ganymede'
target_fp = ['MAW', 'TEB']
PJ_num = [14]
hem = 'N'
Ai_num = 3
ni_num = 150
Ti_num = 1
Zi = 1.3                # Io: 1.3 / Eu: 1.4 / Ga: 1.3
Te = 300.0              # Io: 6.0 [eV]/ Eu: 20.0 / Ga: 300.0


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

sigma_obs = sigma_total

print('Parameter ranges:')
print('--- Ai:', np.min(Ai_3d), np.max(Ai_3d))
print('--- ni:', np.min(ni_3d), np.max(ni_3d))
print('--- Ti:', np.min(Ti_3d), np.max(Ti_3d))
print('--- Hi:', np.min(H_3d)/71492E+3, np.max(H_3d)/71492E+3)
print('Degree of freedom:', (eqlead_est.shape[0]-3))


# %% Constants
dchi_1s = 2.30     # デルタchi2の1シグマ区間
dchi_2s = 6.17     # デルタchi2の2シグマ区間
dchi_3s = 11.8     # デルタchi2の3シグマ区間

MU0 = 1.26E-6            # 真空中の透磁率
AMU2KG = 1.66E-27        # 原子質量をkgに変換するファクタ [kg]
RJ = 71492E+3            # JUPITER RADIUS [m]
MJ = 1.90E+27            # JUPITER MASS [kg]
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

sigma_x = eqwlong_err(Psyn, dt=22.5)*np.ones(sigma_obs.shape)
print('sigma_x [deg]:', sigma_x)


# %% GANYMEDE ONLY === read the current constant
def read_current_coef():
    f0 = np.loadtxt('results/azimuthal_current_fit/' +
                    TARGET_MOON[0:2]+'_coef_0.txt')
    f1 = np.loadtxt('results/azimuthal_current_fit/' +
                    TARGET_MOON[0:2]+'_coef_1.txt')

    con20_pj_idx = np.array([1, 3, 4, 5, 6,
                             7, 8, 9, 10, 11,
                             12, 13, 14, 15, 16,
                             17, 18, 19, 20, 21,
                             22, 23, 24])

    select_pj = np.where(con20_pj_idx == PJ_num[0])

    return f0[select_pj], f1[select_pj]


# %% GANYMEDE ONLY === read the magnetodisk thickness coefficient
def read_disk_thick_coef():
    f0 = np.loadtxt('results/magdisk_thickness_fit/' +
                    TARGET_MOON[0:2]+'_coef_0.txt')
    f1 = np.loadtxt('results/magdisk_thickness_fit/' +
                    TARGET_MOON[0:2]+'_coef_1.txt')
    con20_pj_idx = np.array([1, 3, 4, 5, 6,
                             7, 8, 9, 10, 11,
                             12, 13, 14, 15, 16,
                             17, 18, 19, 20, 21,
                             22, 23, 24], dtype=int)

    select_pj = np.where(con20_pj_idx == PJ_num[0])
    return f0[select_pj], f1[select_pj]


# %%
min_idx = np.where(chi2_3d == np.min(chi2_3d))
print('Min chi2:', np.min(chi2_3d), 'at', min_idx)

print('Scale height [RJ]:', H_3d[min_idx][0]/(71492*1E+3))
print('Estimated eqlead [deg]:', eqlead_est_3d[min_idx][0])
print('Observed eqlead [deg]:', eqlead_obs[1])


# 横軸 ni
fig, ax = plt.subplots(1, 3, figsize=(11, 4), dpi=150, layout='constrained')
for i in range(3):
    x_value = ni_1d[::3]
    y_value = chi2_3d[:, i, 0].T-np.min(chi2_3d[:, i, 0])

    ax[i].set_xlabel(r'$n_i$ [cm$^{-3}$]')
    ax[0].set_ylabel(r'$\Delta \chi^2$')
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')

    ax[i].set_ylim(1, 10000)

    ax[i].plot(x_value, y_value)

    ax[i].axhline(y=1.00, color=UC.gray)
    ax[i].axhline(y=4.00, color=UC.gray)
    ax[i].axhline(y=9.00, color=UC.gray)

fig.tight_layout()

filename = exname[:-4]+'/PJ'+str(PJ_num[0]).zfill(2)
if hem != 'both':
    filename += hem
plt.savefig('img/ftmc/'+TARGET_MOON[0:2]+'/' + filename+'.jpg')

print(x_value[np.where(y_value <= 9.00)])


# %% Moon position when the Alfven waves launched (Time: t0-tau_A)
r_A0_arr = r_moon*np.ones(56)
S3wlon_A0_arr = np.linspace(-50, 370, r_A0_arr.size)
Ai_best = Ai_3d[min_idx][0]
ni_best = ni_3d[min_idx][0]
Hp_best = H_3d[min_idx][0]

eqlead_best_MAW_N = np.zeros(r_A0_arr.size)
eqlead_best_MAW_S = np.zeros(r_A0_arr.size)
eqlead_best_TEB_N = np.zeros(r_A0_arr.size)
eqlead_best_TEB_S = np.zeros(r_A0_arr.size)
TEB_dt_arr = np.zeros(r_A0_arr.size)

current_coef, _ = read_current_coef()
D_coef, _ = read_disk_thick_coef()
D_disk = 3.6*RJ               # [m]
Hp = (2/np.pi)*D_disk*D_coef  # [m]
Wave.Awave().update_Con2020(current_coef=current_coef, thickness_coef=D_coef)
print('Hp [RJ]:', Hp/RJ)
for i in range(r_A0_arr.size):
    r_A0 = r_A0_arr[i]
    S3wlon_A0 = S3wlon_A0_arr[i]
    S_A0_r = Wave.Awave().tracefield(r_A0,
                                     np.radians(S3wlon_A0),
                                     0,
                                     )
    tau, _, _, _ = Wave.Awave().trace3(
        r_A0,
        np.radians(S3wlon_A0),
        0,
        S_A0_r,
        Ai_best,
        ni_best,
        Hp,
        -1,
    )

    eqlead_best_MAW_N[i] = tau*360/Psyn     # [deg]
    TEB_dt_arr[i] = TEB_transit(r_A0, S3wlon_A0, TARGET_MOON)
    eqlead_best_TEB_S[i] = (tau+TEB_dt_arr[i])*360/Psyn     # [deg]

    tau, _, _, _ = Wave.Awave().trace3(
        r_A0,
        np.radians(S3wlon_A0),
        0,
        S_A0_r,
        Ai_best,
        ni_best,
        Hp,
        1,
    )

    eqlead_best_MAW_S[i] = tau*360/Psyn     # [deg]
    eqlead_best_TEB_N[i] = (tau+TEB_dt_arr[i])*360/Psyn     # [deg]


# %%
F = ShareXaxis()
F.fontsize = 23
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(7, 4), dpi='M')
F.initialize()

ymax = 1
if TARGET_MOON == 'Io':
    ymax = 15
elif TARGET_MOON == 'Europa':
    ymax = 15
elif TARGET_MOON == 'Ganymede':
    ymax = 40

F.set_xaxis(label='Moon System III longitude [deg]',
            min=0, max=360,
            ticks=np.arange(0, 360+1, 45),
            ticklabels=np.arange(0, 360+1, 45),
            minor_num=3)
F.set_yaxis(ax_idx=0, label='Eq. lead angle [deg]',
            min=-1, max=ymax,
            ticks=np.arange(0, ymax+1, 5),
            ticklabels=np.arange(0, ymax+1, 5),
            minor_num=5)

north = np.where((hem_obs == -1))
F.ax.errorbar(moon_S3wlon_obs[north], eqlead_obs[north],
              xerr=np.array([np.abs(sigma_x[north]),
                             np.abs(sigma_x[north])]),
              yerr=np.array([np.abs(sigma_obs[north]),
                             np.abs(sigma_obs[north])]),
              linewidth=0., markersize=2,
              elinewidth=1.0, color=UC.red,
              label='N MAW',
              zorder=0.7)

south = np.where((hem_obs == 1))
F.ax.errorbar(moon_S3wlon_obs[south], eqlead_obs[south],
              xerr=np.array([np.abs(sigma_x[south]),
                             np.abs(sigma_x[south])]),
              yerr=np.array([np.abs(sigma_obs[south]),
                             np.abs(sigma_obs[south])]),
              linewidth=0., markersize=2,
              elinewidth=1.0, color=UC.blue,
              label='S MAW',
              zorder=0.7)

if len(target_fp) == 2:
    north = np.where((hem_obs == -101))
    F.ax.errorbar(moon_S3wlon_obs[north], eqlead_obs[north],
                  xerr=np.array([np.abs(sigma_x[north]),
                                 np.abs(sigma_x[north])]),
                  yerr=np.array([np.abs(sigma_obs[north]),
                                 np.abs(sigma_obs[north])]),
                  linewidth=0., markersize=2,
                  elinewidth=1.0, color=UC.orange,
                  label='N TEB',
                  zorder=0.7)
    south = np.where((hem_obs == 101))
    F.ax.errorbar(moon_S3wlon_obs[south], eqlead_obs[south],
                  xerr=np.array([np.abs(sigma_x[south]),
                                 np.abs(sigma_x[south])]),
                  yerr=np.array([np.abs(sigma_obs[south]),
                                 np.abs(sigma_obs[south])]),
                  linewidth=0., markersize=2,
                  elinewidth=1.0, color=UC.lightblue,
                  label='S TEB',
                  zorder=0.7)

F.ax.plot(S3wlon_A0_arr+eqlead_best_MAW_N,
          eqlead_best_MAW_N, color=UC.red, linewidth=0.7)
F.ax.plot(S3wlon_A0_arr+eqlead_best_MAW_S,
          eqlead_best_MAW_S, color=UC.blue, linewidth=0.7)

if len(target_fp) == 2:
    F.ax.plot(S3wlon_A0_arr+eqlead_best_TEB_N,
              eqlead_best_TEB_N, color=UC.orange,
              linestyle='--', linewidth=0.5)
    F.ax.plot(S3wlon_A0_arr+eqlead_best_TEB_S,
              eqlead_best_TEB_S, color=UC.lightblue,
              linestyle='--', linewidth=0.5)

fig_title = TARGET_MOON
for i in range(len(target_fp)):
    if i == 0:
        fig_title += ' '+target_fp[i]
    elif i > 0:
        fig_title += ' & '+target_fp[i]
fig_title += ' (PJ'
for i in range(len(PJ_num)):
    if i == 0:
        fig_title += str(PJ_num[i])
    elif i > 0:
        fig_title += ' & '+str(PJ_num[i])
if hem != 'both':
    fig_title += hem
fig_title += ')'
F.ax.set_title(fig_title, fontsize=F.fontsize, weight='bold')

legend = F.legend(ax_idx=0, loc='upper right', ncol=4, markerscale=4,
                  fontsize_scale=0.7, textcolor=False, handletextpad=0.2)
legend_shadow(fig=F.fig, ax=F.ax, legend=legend)

F.fig.savefig('img/ftmc/'+TARGET_MOON[0:2]+'/' + filename + '_leadangle.jpg',
              bbox_inches='tight')
