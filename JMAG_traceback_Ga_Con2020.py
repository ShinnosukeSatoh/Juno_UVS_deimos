import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import time
import multiprocessing
from multiprocessing import Pool

from SharedX import ShareXaxis
from UniversalColor import UniversalColor

import JupiterMag as jm

from Leadangle_fit_JunoUVS import Obsresults
from Leadangle_fit_JunoUVS import calc_eqlead
from Leadangle_fit_JunoUVS import moonS3wlon_arr

import os
from IPython.display import clear_output

# SPICE KERNELS
import spiceypy as spice
spice.furnsh('kernel/cassMetaK.txt')
radii = spice.bodvrd("JUPITER", "RADII", 3)[1]
RJ_km = radii[0]
RJ_pole = radii[2]
f = (RJ_km - RJ_pole) / RJ_km

UC = UniversalColor()
UC.set_palette()

F = ShareXaxis()
F.fontsize = 23
F.fontname = 'Liberation Sans Narrow'
F.set_default()


# %% Input about Juno observation
TARGET_MOON = 'Ganymede'
TARGET_FP = ['MAW', 'TEB']
PJ_LIST = [1, 3]+np.arange(4, 68+1, 1).tolist()


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
if TARGET_MOON == 'Io':
    Psyn = Psyn_io
    PJ_LIST.pop(54-2)
    PJ_LIST.pop(55-3)
    PJ_LIST.pop(56-4)
    PJ_LIST.pop(57-5)
    PJ_LIST.pop(61-6)
    PJ_LIST.pop(63-7)
    PJ_LIST.pop(64-8)
    PJ_LIST.pop(65-9)
    PJ_LIST.pop(67-10)
    r_moon = 5.9*RJ
elif TARGET_MOON == 'Europa':
    Psyn = Psyn_eu
    PJ_LIST.pop(24-2)
    PJ_LIST.pop(43-3)
    PJ_LIST.pop(47-4)
    PJ_LIST.pop(49-5)
    PJ_LIST.pop(50-6)
    PJ_LIST.pop(51-7)
    PJ_LIST.pop(53-8)
    PJ_LIST.pop(55-9)
    PJ_LIST.pop(56-10)
    PJ_LIST.pop(60-11)
    PJ_LIST.pop(61-12)
    PJ_LIST.pop(63-13)
    PJ_LIST.pop(64-14)
    PJ_LIST.pop(65-15)
    PJ_LIST.pop(66-16)
    PJ_LIST.pop(67-17)
    PJ_LIST.pop(68-18)
    r_moon = 9.4*RJ
elif TARGET_MOON == 'Ganymede':
    Psyn = Psyn_ga
    PJ_LIST.pop(24-2)
    PJ_LIST.pop(31-3)
    PJ_LIST.pop(39-4)
    PJ_LIST.pop(43-5)
    PJ_LIST.pop(44-6)
    PJ_LIST.pop(45-7)
    PJ_LIST.pop(51-8)
    PJ_LIST.pop(52-9)
    PJ_LIST.pop(53-10)
    PJ_LIST.pop(54-11)
    PJ_LIST.pop(55-12)
    PJ_LIST.pop(56-13)
    PJ_LIST.pop(61-14)
    PJ_LIST.pop(62-15)
    PJ_LIST.pop(63-16)
    PJ_LIST.pop(64-17)
    PJ_LIST.pop(65-18)
    PJ_LIST.pop(66-19)
    PJ_LIST.pop(67-20)
    PJ_LIST.pop(68-21)
    r_moon = 15.0*RJ


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


# %% Data from Connerney+2020: Radial current constant [MA]
con20_mu_i_rho = np.array([35.2, 14.6, 7.7, 11.5, 20.8,
                           20.2, 12.2, 21.1, 20.9, 10.7,
                           26.26, 16.4, 12.0, 19.6, 12.0,
                           13.6, 20.0, 12.8, 16.0, 17.3,
                           9.9, 16.1, 10.3])


# %% Jupiter's surface radius at a given latitude
def calc_r_surf(lat):
    """
    `lat` ... latitude [rad]
    """
    r0 = RJ
    r1 = (14.4/15.4)*RJ

    rs = r0*r1/np.sqrt((r0*np.sin(lat))**2+(r1*np.cos(lat))**2)  # [m]

    return rs


# %% Derive the best fit magnetodisk thickness
def calc_thickness_loop(x0, y0, z0, r_moon_obs, moon_z0, mu_i_azi, mu_i_rho=16.7):
    start_loop = time.time()
    print('Loop started.')

    coef_arr = np.arange(0.62, 1.34, 0.01)

    # 中央値
    rho_eq_arr = np.zeros(coef_arr.size)
    phi_eq_arr = np.zeros(coef_arr.size)

    for i in range(coef_arr.size):
        coef = coef_arr[i]

        # 磁場モデルの設定
        d_rj_default = 3.6      # default: 3.6 [RJ]
        jm.Con2020.Config(mu_i=mu_i_azi,
                          i_rho=mu_i_rho,
                          d__cs_half_thickness_rj=d_rj_default*coef,
                          equation_type='analytic')

        # create trace objects, pass starting position(s) x0,y0,z0
        T1 = jm.TraceField(x0, y0, z0,
                           IntModel='jrm33', ExtModel='Con2020',
                           MaxStep=0.0003,
                           MaxLen=800000, ErrMax=0.000001)
        x1 = T1.x[0][~np.isnan(T1.x[0])]    # [RJ]
        y1 = T1.y[0][~np.isnan(T1.y[0])]    # [RJ]
        z1 = T1.z[0][~np.isnan(T1.z[0])]    # [RJ]
        rho = np.sqrt(x1**2 + y1**2 + z1**2)

        # Satellite orbital plane
        idx_z0 = np.argmin(np.abs(z1-moon_z0/RJ))

        phi_eq_arr[i] = np.arctan2(y1[idx_z0], x1[idx_z0])  # East long. [rad]
        rho_eq_arr[i] = rho[idx_z0]                         # Distance [RJ]
        # z_eq = T1.z[0][idx_z0]

    idx_rho_best = np.argmin(np.abs(rho_eq_arr-r_moon_obs/RJ))
    coef_best = coef_arr[idx_rho_best]
    rho_eq = rho_eq_arr[idx_rho_best]   # Distance [RJ]
    phi_eq = phi_eq_arr[idx_rho_best]   # East longitude [rad]

    print('------ Loop time [sec]:', round(time.time()-start_loop, 4))
    print('----------------- [RJ]:', round(rho_eq, 3))
    return np.array([coef_best, rho_eq, phi_eq])


# %% Derive the best fit magnetodisk thickness
def calc_azi_current_loop(x0, y0, z0, r_moon_obs, moon_z0, mu_i_azi=139.6, mu_i_rho=16.7):
    start_loop = time.time()
    print('Loop started.')

    mui_azi_arr = np.arange(50.0, 200.0+1.0, 2.0)   # [nT]

    # 中央値
    rho_eq_arr = np.zeros(mui_azi_arr.size)
    phi_eq_arr = np.zeros(mui_azi_arr.size)

    for i in range(mui_azi_arr.size):
        # 磁場モデルの設定
        d_rj_default = 3.6      # default: 3.6 [RJ]
        jm.Con2020.Config(mu_i=mui_azi_arr[i],
                          i_rho=mu_i_rho,
                          d__cs_half_thickness_rj=d_rj_default,
                          equation_type='analytic')

        # create trace objects, pass starting position(s) x0,y0,z0
        T1 = jm.TraceField(x0, y0, z0,
                           IntModel='jrm33', ExtModel='Con2020',
                           MaxStep=0.0003,
                           MaxLen=800000, ErrMax=0.000001)
        x1 = T1.x[0][~np.isnan(T1.x[0])]    # [RJ]
        y1 = T1.y[0][~np.isnan(T1.y[0])]    # [RJ]
        z1 = T1.z[0][~np.isnan(T1.z[0])]    # [RJ]
        rho = np.sqrt(x1**2 + y1**2 + z1**2)

        # Satellite orbital plane
        idx_z0 = np.argmin(np.abs(z1-moon_z0/RJ))

        phi_eq_arr[i] = np.arctan2(y1[idx_z0], x1[idx_z0])  # East long. [rad]
        rho_eq_arr[i] = rho[idx_z0]                         # Distance [RJ]
        # z_eq = T1.z[0][idx_z0]

    idx_rho_best = np.argmin(np.abs(rho_eq_arr-r_moon_obs/RJ))
    mui_azi_best = mui_azi_arr[idx_rho_best]
    rho_eq = rho_eq_arr[idx_rho_best]   # Distance [RJ]
    phi_eq = phi_eq_arr[idx_rho_best]   # East longitude [rad]

    print('-------- Loop time [sec]:', round(time.time()-start_loop, 4))
    print('- Residual distance [RJ]:', round(rho_eq-r_moon_obs/RJ, 3))
    print('---------- Best fit [nT]:', round(mui_azi_best, 3))
    return np.array([mui_azi_best, rho_eq, phi_eq])


# %% Select moon synodic orbital period
if TARGET_MOON == 'Ganymede':
    Psyn = con20_pj_idx
    r_moon = 15.0*RJ


# %%
def main():
    if FIT_TARGET == 'AZI_CURRENT':
        PJ_list = PJ_LIST
    if FIT_TARGET == 'THICKNESS':
        PJ_list = [1, 3]+np.arange(4, 24, 1).tolist()
    for PJ in PJ_list:
        if PJ < 24:
            print('Go to the next PJ.')
            continue
        j = 0
        wlon_fp, err_wlon_fp, lat_fp, err_lat_fp, moon_S3wlon, et_fp, hem_fp, pj_fp = Obsresults(
            [PJ], TARGET_MOON, TARGET_FP, TARGET_HEM='both', FLIP=False
        )

        # 観測時の衛星軌道動径距離
        _, _, moon_z0, r_moon_arr, _, _, _ = moonS3wlon_arr(et_fp, TARGET_MOON)

        # %% Backtraced field line position on the equatorial plane
        rho_eq = np.zeros(wlon_fp.size)
        phi_eq = np.zeros(wlon_fp.size)
        z_eq = np.zeros(wlon_fp.size)
        eq_lead_arr = np.zeros(wlon_fp.size)
        x0_fp = np.zeros(wlon_fp.size)
        y0_fp = np.zeros(wlon_fp.size)
        z0_fp = np.zeros(wlon_fp.size)

        mu_i_default = 139.6    # default: 139.6 [nT]
        i_rho_default = 16.7    # default: 16.7 [MA]
        d_rj_default = 3.6      # default: 3.6 [RJ]
        jm.Internal.Config(Model='jrm33', CartesianIn=True,
                           CartesianOut=True, Degree=18)
        for i in range(rho_eq.size):
            lat_c = math.radians(lat_fp[i])

            def calc_r_gr(r):
                """
                brentqで`r`を数値的に求めたいので引数は`r`だけだけ。
                - `r` [km]
                - `lat_c` [rad]
                - `target_alt` [km]
                """
                x = r*math.cos(lat_c)
                z = r*math.sin(lat_c)

                _, _, alt = spice.recpgr('Jupiter',
                                         np.array([x, 0.0, z]),
                                         RJ_km,
                                         f)
                return alt - 900.0      # [km]

            r_c = brentq(calc_r_gr, RJ_pole, RJ_km+5000.0)  # [km]

            theta = np.radians(90.0-lat_fp[i])
            phi = np.radians(360.0-wlon_fp[i])
            if error_num == 1:
                if lat_fp[i] >= 0.0:
                    theta = np.radians(90.0-(lat_fp[i]+err_lat_fp[i]))
                elif lat_fp[i] < 0.0:
                    theta = np.radians(90.0-(lat_fp[i]-err_lat_fp[i]))
            elif error_num == 2:
                if lat_fp[i] >= 0.0:
                    theta = np.radians(90.0-(lat_fp[i]-err_lat_fp[i]))
                elif lat_fp[i] < 0.0:
                    theta = np.radians(90.0-(lat_fp[i]+err_lat_fp[i]))
            x0_fp[i] = r_c*np.sin(theta)*np.cos(phi)/RJ_km   # [RJ]
            y0_fp[i] = r_c*np.sin(theta)*np.sin(phi)/RJ_km   # [RJ]
            z0_fp[i] = r_c*np.cos(theta)/RJ_km               # [RJ]

        if FIT_TARGET == 'THICKNESS':
            args = list(zip(x0_fp,
                            y0_fp,
                            z0_fp,
                            r_moon_arr,
                            moon_z0,
                            con20_mu_i_tot[j]*np.ones(x0_fp.size),
                            con20_mu_i_rho[j]*np.ones(x0_fp.size),))
            with Pool(processes=8) as pool:
                results_list = list(pool.starmap(calc_thickness_loop, args))

        elif FIT_TARGET == 'AZI_CURRENT':
            args = list(zip(x0_fp,
                            y0_fp,
                            z0_fp,
                            r_moon_arr,
                            moon_z0,
                            np.ones(x0_fp.size),
                            np.ones(x0_fp.size),))
            with Pool(processes=parallel) as pool:
                results_list = list(pool.starmap(calc_azi_current_loop, args))

        results_arr = np.array(results_list)
        # print(results_arr.shape)      # >>> (XXX, 3)
        bestfit_arr = results_arr[:, 0]      # 3.6*C [RJ]
        rho_eq_best_arr = results_arr[:, 1]          # [RJ]
        phi_eq_best_arr = np.mod(
            360.0-np.degrees(results_arr[:, 2]), 360.0)    # [deg]

        savefile = np.array([rho_eq_best_arr,
                            phi_eq_best_arr,
                            et_fp,
                            hem_fp,
                            eq_lead_arr,
                            bestfit_arr,
                             ])
        print(savefile.shape)  # -> (6, N)
        # savefile[0,:] -> rho_eq [RJ]
        # savefile[1,:] -> phi_eq [deg] (west longitude)
        # savefile[2,:] -> et_fp [et]
        # savefile[3,:] -> hemisphere & type of footprints (+/-1, +/-101)
        # savefile[4,:] -> equatorial lead angle [deg]
        # savefile[5,:] -> best-fit thickness coefficient or best-fit azimuthal current

        np.savetxt('data/Backtraced_'+FIT_TARGET+'/PJ'+str(PJ).zfill(2)+'/' +
                   TARGET_MOON[0]+'FP_info_v900km_'+str(error_num)+'.txt', savefile)

        j += 1


# %% EXECUTE
if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)

    FIT_TARGET = 'AZI_CURRENT'      # 'AZI_CURRENT' or 'THICKNESS'
    error_num = 1   # 0, 1, or 2
    parallel = 20

    main()
