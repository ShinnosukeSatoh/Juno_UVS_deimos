import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import brentq

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
# print(PJ_LIST)


# %% Data load
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

    # 観測時の衛星軌道動径距離
    _, _, moon_z0, r_moon_arr, _, _, _ = moonS3wlon_arr(et_fp, TARGET_MOON)

    # %% Backtraced field line position on the equatorial plane
    rho_arr = np.zeros(wlon_fp.size)
    phi_arr = np.zeros(wlon_fp.size)
    z_arr = np.zeros(wlon_fp.size)
    eq_lead_arr = np.zeros(wlon_fp.size)

    mu_i_default = 139.6    # default: 139.6 [nT]
    jm.Internal.Config(Model='jrm33', CartesianIn=True,
                       CartesianOut=True, Degree=18)
    jm.Con2020.Config(mu_i=mu_i_default*1.0, equation_type='analytic')
    for i in range(rho_arr.size):
        lat_c = math.radians(lat_fp[i])

        def calc_r_gr(r):
            """
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
        x0 = r_c*np.sin(theta)*np.cos(phi)/RJ_km
        y0 = r_c*np.sin(theta)*np.sin(phi)/RJ_km
        z0 = r_c*np.cos(theta)/RJ_km

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

        phi_z0 = np.arctan2(y1[idx_z0], x1[idx_z0])  # in SIII RH [rad]
        rho_arr[i] = rho[idx_z0]
        phi_arr[i] = np.mod(360.0-np.degrees(phi_z0), 360.0)
        z_arr = T1.z[0][idx_z0]

        # Equatorial lead angle
        eq_lead_arr[i] = np.mod(moon_S3wlon[i]-phi_arr[i], 360.0)

        clear_output(wait=True)  # Trace informationを消し去る

    savefile = np.array([rho_arr,
                         phi_arr,
                         et_fp,
                         hem_fp,
                         eq_lead_arr,
                         ])
    print(savefile.shape)  # -> (3, N)
    # savefile[0,:] -> rho_arr [RJ]
    # savefile[1,:] -> phi_arr [deg] (west longitude)
    # savefile[2,:] -> et_fp [et]
    # savefile[3,:] -> hemisphere & type of footprints (+/-1, +/-101)
    # savefile[4,:] -> equatorial lead angle [deg]

    print('np.average(rho_arr):', np.average(rho_arr))

    np.savetxt('data/Backtraced_2/PJ'+str(PJ).zfill(2)+'/' +
               TARGET_MOON[0]+'FP_info_v900km.txt', savefile)
