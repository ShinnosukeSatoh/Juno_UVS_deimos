import numpy as np
import math
import matplotlib.pyplot as plt
from SharedX import ShareXaxis
from UniversalColor import UniversalColor

import JupiterMag as jm

from Leadangle_fit_JunoUVS import Obsresults
from Leadangle_fit_JunoUVS import calc_eqlead
from Leadangle_fit_JunoUVS import moonS3wlon_arr
from Leadangle_fit_JunoUVS import read_disk_thick_coef

import os
from IPython.display import clear_output

UC = UniversalColor()
UC.set_palette()

F = ShareXaxis()
F.fontsize = 23
F.fontname = 'Liberation Sans Narrow'
F.set_default()


# %% Input about Juno observation
TARGET_MOON = 'Ganymede'
TARGET_FP = ['MAW', 'TEB']
PJ_LIST = [1, 3, 4, 5, 6,
           7, 8, 9, 10, 11,
           12, 13, 14, 15, 16,
           17, 18, 19, 20, 21,
           22, 23]      # Connerney+2020のテーブル


# %% Constants
MU0 = 1.26E-6            # 真空中の透磁率
AMU2KG = 1.66E-27        # 原子質量をkgに変換するファクタ [kg]
RJ = 71492.0E+3            # JUPITER RADIUS [m]
MJ = 1.90E+27            # JUPITER MASS [kg]
C = 2.99792E+8           # LIGHT SPEED [m/s]
G = 6.67E-11             # 万有引力定数  [m^3 kg^-1 s^-2]

Psyn_io = (12.89)*3600      # Moon's synodic period [sec]
Psyn_eu = (11.22)*3600      # Moon's synodic period [sec]
Psyn_ga = (10.53)*3600      # Moon's synodic period [sec]


# %% Jupiter's surface radius at a given latitude
def calc_r_surf(lat):
    """
    `lat` ... latitude [rad]
    """
    r0 = RJ
    r1 = (14.4/15.4)*RJ

    rs = r0*r1/np.sqrt((r0*np.sin(lat))**2+(r1*np.cos(lat))**2)  # [m]

    return rs


# %% Select moon synodic orbital period
if TARGET_MOON == 'Ganymede':
    Psyn = Psyn_ga
    r_moon = 15.0*RJ


# %% Data from Connerney+2020: PJ index
con20_pj_idx = np.array([1, 3, 4, 5, 6,
                         7, 8, 9, 10, 11,
                         12, 13, 14, 15, 16,
                         17, 18, 19, 20, 21,
                         22, 23, 24])


# %% Data from Connerney+2020: Azimuthal urrent constant [nT]
con20_mu_i_azi = np.array([150.1, 137.8, 127.2, 129.1, 130.1,
                           142.3, 140.1, 143.8, 137.0, 141.4,
                           124.2, 148.9, 145.3, 144.8, 149.9,
                           132.1, 133.5, 152.9, 138.5, 138.8,
                           156.1, 141.4, 146.3])

# %% Data from Connerney+2020: Current constant [nT]
con20_mu_i_rho = np.array([35.2, 14.6, 7.7, 11.5, 20.8,
                           20.2, 12.2, 21.1, 20.9, 10.7,
                           26.3, 16.4, 12.0, 19.6, 12.0,
                           13.6, 20.0, 12.8, 16.0, 17.3,
                           9.9, 16.1, 10.3])


# %% Data load
j = 0
for PJ in PJ_LIST:
    wlon_fp, err_wlon_fp, lat_fp, err_lat_fp, moon_S3wlon, et_fp, hem_fp, pj_fp = Obsresults(
        [PJ], TARGET_MOON, TARGET_FP, TARGET_HEM='both', FLIP=False
    )

    _, _, _, wlon_fp_eq = calc_eqlead(wlon_fp,
                                      err_wlon_fp,
                                      lat_fp,
                                      err_lat_fp,
                                      hem_fp,
                                      moon_S3wlon,
                                      TARGET_MOON)
    print('PJ', PJ)
    D_coef, _ = read_disk_thick_coef(TARGET_MOON,
                                     TARGET_HEM='both',
                                     PJ_LIST=[PJ])

    # Backtraced field line position on the equatorial plane
    rho_arr = np.zeros(wlon_fp.size)
    phi_arr = np.zeros(wlon_fp.size)
    z_arr = np.zeros(wlon_fp.size)

    mu_i_default = 139.6    # default: 139.6 [nT]
    i_rho_default = 16.7    # default: 16.7 [MA]
    d_rj_default = 3.6      # default: 3.6 [RJ]
    jm.Con2020.Config(mu_i=con20_mu_i_azi[j],
                      d=d_rj_default*D_coef,
                      i_rho=con20_mu_i_rho[j],
                      equation_type='analytic')
    for i in range(rho_arr.size):
        latitude = lat_fp[i]
        theta = np.radians(90.0-latitude)
        phi = np.radians(360.0-wlon_fp[i])

        # radius of surface
        rs = calc_r_surf(np.radians(latitude))
        r = rs/RJ+(900.0E+3/RJ)   # [RJ]

        x0 = r*np.sin(theta)*np.cos(phi)
        y0 = r*np.sin(theta)*np.sin(phi)
        z0 = r*np.cos(theta)

        # create trace objects, pass starting position(s) x0,y0,z0 in RJ
        T1 = jm.TraceField(x0, y0, z0, Verbose=True,
                           IntModel='jrm33', ExtModel='Con2020',
                           MaxLen=2500, ErrMax=0.00000001)

        x1 = T1.x[0][~np.isnan(T1.x[0])]
        y1 = T1.y[0][~np.isnan(T1.y[0])]
        z1 = T1.z[0][~np.isnan(T1.z[0])]
        rho = np.sqrt(x1**2 + y1**2 + z1**2)

        # Satellite orbital plane
        idx_z0 = np.argmin(np.abs(z1))

        phi_z0 = np.arctan2(y1[idx_z0], x1[idx_z0])        # in SIII RH [rad]
        phi_z0 = np.mod(360.0-np.degrees(phi_z0), 360.0)   # LH [rad]
        rho_arr[i] = rho[idx_z0]
        phi_arr[i] = phi_z0
        z_arr = T1.z[0][idx_z0]

        clear_output(wait=True)  # Trace informationを消し去る

    # %% Results
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(0.0, 360.0)
    ax.set_xlabel('Table value [deg]')
    ax.set_xticks(np.arange(0, 360+1, 45))
    ax.set_ylim(0.0, 360.0)
    ax.set_xlabel('Back-traced [deg]')
    ax.set_yticks(np.arange(0, 360+1, 45))
    ax.scatter(wlon_fp_eq, phi_arr, s=1.0, c=UC.orange)
    fig.tight_layout()
    fig.savefig('data/Backtraced/PJ'+str(PJ).zfill(2)+'/' +
                TARGET_MOON[0]+'FP.jpg', bbox_inches='tight')
    plt.close()

    savefile = np.array([rho_arr,
                         phi_arr,
                         et_fp,
                         hem_fp,
                         ])
    print(savefile.shape)  # -> (3, N)
    # savefile: this is not filtered by the viewing angle.
    # savefile[0,:] -> rho_arr [RJ] (equatorial radial distance)
    # savefile[1,:] -> phi_arr [deg] (equatorial west longitude)
    # savefile[2,:] -> et_fp [et]
    # savefile[3,:] -> hemisphere & type of footprints (+/-1, +/-101)

    np.savetxt('data/Backtraced/PJ'+str(PJ).zfill(2)+'/' +
               TARGET_MOON[0]+'FP_info_v900km_fixed.txt', savefile)

    j += 1


# %% ========================================================
# Comparison
# (a) Radial current is constant at 16.7 MA.
# (b) Temporal variation of the radial current is considered.
# ===========================================================
fig, ax = plt.subplots(dpi=150)
ax.set_xlim(0, 360)
ax.set_xlabel('SIII w. longitude [deg]')
ax.set_ylim(-5, 5)
ax.set_ylabel('Diff [deg]')
j = 0
for PJ in PJ_LIST:
    data_a = np.loadtxt('data/Backtraced/PJ'+str(PJ).zfill(2)+'/' +
                        TARGET_MOON[0]+'FP_info_v900km_fixed_origin.txt')

    data_b = np.loadtxt('data/Backtraced/PJ'+str(PJ).zfill(2)+'/' +
                        TARGET_MOON[0]+'FP_info_v900km_fixed.txt')

    rho_a, phi_a = data_a[0, :], data_a[1, :]
    et_fp_a, hem_a = data_a[2, :], data_a[3, :]

    rho_b, phi_b = data_b[0, :], data_b[1, :]
    et_fp_b, hem_b = data_b[2, :], data_b[3, :]

    _, _, _, _, _, _, moon_S3wlon0 = moonS3wlon_arr(
        et_fp=et_fp_a, moon=TARGET_MOON)

    ax.scatter(moon_S3wlon0, phi_a-phi_b, s=2.0)

fig.tight_layout()
fig.savefig(
    'img/'+TARGET_MOON[0:2]+'_eqwlon_comparison_radialcurrent.jpg', bbox_inches='tight')
plt.close()


# %% ========================================================
# SIII longitude vs equatorial lead angle
# (a) Radial current is constant at 16.7 MA.
# (b) Temporal variation of the radial current is considered.
# ===========================================================
F = ShareXaxis()
fig_id = 'SS260616.004a'
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(7.0, 4.0), dpi='M')
F.initialize()

F.set_xaxis(label=r"Moon SIII longitude $\lambda_{\rm III}$ [deg]",
            min=0, max=360,
            ticks=np.arange(0, 360+1, 45),
            ticklabels=np.arange(0, 360+1, 45, dtype=int),
            minor_num=3)
F.set_yaxis(ax_idx=0,
            label='Eq. lead angle [deg]',
            min=-5, max=30,
            ticks=np.arange(-5, 30+1, 5),
            ticklabels=np.arange(-5, 30+1, 5, dtype=int),
            minor_num=5)
for PJ in PJ_LIST:
    data_a = np.loadtxt('data/Backtraced/PJ'+str(PJ).zfill(2)+'/' +
                        TARGET_MOON[0]+'FP_info_v900km_fixed_origin.txt')

    data_b = np.loadtxt('data/Backtraced/PJ'+str(PJ).zfill(2)+'/' +
                        TARGET_MOON[0]+'FP_info_v900km_fixed.txt')

    rho_a, phi_a = data_a[0, :], data_a[1, :]
    et_fp_a, hem_a = data_a[2, :], data_a[3, :]

    rho_b, phi_b = data_b[0, :], data_b[1, :]
    et_fp_b, hem_b = data_b[2, :], data_b[3, :]

    _, _, _, _, _, _, moon_S3wlon0 = moonS3wlon_arr(
        et_fp=et_fp_b, moon=TARGET_MOON)

    for i in range(et_fp_b.size):
        if hem_b[i] == 1:
            color = UC.blue
        elif hem_b[i] == 101:
            color = UC.lightblue
        elif hem_b[i] == -1:
            color = UC.red
        elif hem_b[i] == -101:
            color = UC.pink
        F.ax.scatter(moon_S3wlon0[i], moon_S3wlon0[i]-phi_b[i], s=3.0, c=color)

F.manage(ax_idx=0, id=fig_id, color=UC.lightgray)
F.fig.savefig(
    'img/'+TARGET_MOON[0:2]+'_eqlead_radialcurrent.jpg', bbox_inches='tight')
F.close()


F = ShareXaxis()
fig_id = 'SS260616.004b'
F = ShareXaxis()
F.fontsize = 22
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(7.0, 4.0), dpi='M')
F.initialize()

F.set_xaxis(label=r"Moon SIII longitude $\lambda_{\rm III}$ [deg]",
            min=0, max=360,
            ticks=np.arange(0, 360+1, 45),
            ticklabels=np.arange(0, 360+1, 45, dtype=int),
            minor_num=3)
F.set_yaxis(ax_idx=0,
            label='Eq. lead angle [deg]',
            min=-5, max=30,
            ticks=np.arange(-5, 30+1, 5),
            ticklabels=np.arange(-5, 30+1, 5, dtype=int),
            minor_num=5)
for PJ in PJ_LIST:
    data_a = np.loadtxt('data/Backtraced/PJ'+str(PJ).zfill(2)+'/' +
                        TARGET_MOON[0]+'FP_info_v900km_fixed_origin.txt')

    data_b = np.loadtxt('data/Backtraced/PJ'+str(PJ).zfill(2)+'/' +
                        TARGET_MOON[0]+'FP_info_v900km_fixed.txt')

    rho_a, phi_a = data_a[0, :], data_a[1, :]
    et_fp_a, hem_a = data_a[2, :], data_a[3, :]

    rho_b, phi_b = data_b[0, :], data_b[1, :]
    et_fp_b, hem_b = data_b[2, :], data_b[3, :]

    _, _, _, _, _, _, moon_S3wlon0 = moonS3wlon_arr(
        et_fp=et_fp_a, moon=TARGET_MOON)

    for i in range(et_fp_a.size):
        if hem_a[i] == 1:
            color = UC.blue
        elif hem_a[i] == 101:
            color = UC.lightblue
        elif hem_a[i] == -1:
            color = UC.red
        elif hem_a[i] == -101:
            color = UC.pink
        F.ax.scatter(moon_S3wlon0[i], moon_S3wlon0[i]-phi_a[i], s=3.0, c=color)

F.manage(ax_idx=0, id=fig_id, color=UC.lightgray)
F.fig.savefig(
    'img/'+TARGET_MOON[0:2]+'_eqlead_fixed_origin.jpg', bbox_inches='tight')
F.close()
