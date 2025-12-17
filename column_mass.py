""" column_mass.py

Created on Apr 28, 2025
@author: Shin Satoh

Description:
Using the lead angle values measured in one single Perijove of Juno,
this program iterates the Alfven wave tracing along the magnetic
field line and estimate the transit time of the Alfven wave from the
satellite to the auroral footprint.

Version
1.0.0 (Apr 28, 2025)

"""
# %% Import
import spiceypy as spice
from multiprocessing import Pool
import numpy as np
import math

from Leadangle_fit_JunoUVS import S3EQ
from Leadangle_fit_JunoUVS import create_argmesh
from Leadangle_fit_JunoUVS import eqwlong_err
from Leadangle_fit_JunoUVS import Alfven_launch_site
from Leadangle_fit_JunoUVS import scaleheight
from Leadangle_fit_JunoUVS import moonS3wlon_arr
from Leadangle_fit_JunoUVS import calc_eqlead
import Leadangle_wave as Wave
import time
# import os
import JupiterMag as jm

jm.Internal.Config(Model='jrm33', CartesianIn=True, CartesianOut=True)
jm.Con2020.Config(equation_type='analytic')


spice.furnsh('kernel/cassMetaK.txt')
savpath = 'data/Satellite_FP_JRM33.sav'


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


# %% Function to be in loop
def calc(Ai, ni, Hp, r_A0, S3wlon_A0, z_A0, hem, S_A0=0):
    # print('Calc loop in')
    # S_A0 = Wave.Awave().tracefield(r_A0,
    #                                np.radians(S3wlon_A0),
    #                                z_A0
    #                                )
    _, _, _, col_massdens = Wave.Awave().trace3(r_A0,
                                                np.radians(S3wlon_A0),
                                                z_A0,
                                                S_A0,
                                                Ai,
                                                ni,
                                                Hp,
                                                hem,
                                                )
    return col_massdens


# %% Main function
def main():
    # Select moon synodic orbital period
    if TARGET_MOON == 'Io':
        r_A0 = 5.9*RJ
        Zi = 1.3    # ION CHARGE [C]
        Te = 6.0    # ELECTRON TEMPERATURE [eV]
    elif TARGET_MOON == 'Europa':
        r_A0 = 9.4*RJ
        Zi = 1.4    # ION CHARGE [C]
        Te = 20.0   # ELECTRON TEMPERATURE [eV]
    elif TARGET_MOON == 'Ganymede':
        r_A0 = 15.0*RJ
        Zi = 1.3    # ION CHARGE [C]
        Te = 300.0  # ELECTRON TEMPERATURE [eV]

    # Satellite System III longitude [deg]
    S3wlon_A0 = 100.0

    # Tracing direction
    hem_fp = -1

    # パラメータ空間(meshgrid → 1d)の作成
    Ai_1d, ni_1d, Ti_1d, _, _, _ = create_argmesh(Ai_0, Ai_1, Ai_num, Ai_scale,
                                                  ni_0, ni_1, ni_num, ni_scale,
                                                  Ti_0, Ti_1, Ti_num, Ti_scale)
    H_1d = scaleheight(Ai=Ai_1d, Zi=Zi, Ti=Ti_1d, Te=Te)
    arg_size = Ai_1d.size

    # col_massdens_1d = np.zeros(arg_size)
    print('PJ number:', PJ_LIST)
    print('Target moon:', TARGET_MOON)
    print('Param space shape:', ni_num, Ai_num, Ti_num)
    start_all = time.time()

    S_A0 = Wave.Awave().tracefield(r_A0,
                                   np.radians(S3wlon_A0),
                                   0.0,
                                   )

    # Northward
    args = list(zip(
        Ai_1d,
        ni_1d,
        H_1d,
        r_A0*np.ones(arg_size),
        S3wlon_A0*np.ones(arg_size),
        0.0*np.ones(arg_size),
        hem_fp*np.ones(arg_size),
        S_A0*np.ones(arg_size)
    ))
    with Pool(processes=parallel) as pool:
        results_list_1 = list(pool.starmap(calc, args))
    col_massdens_1dN = np.array(results_list_1)[:, 0]    # [kg m-2]
    ftmc_mag_1dN = np.array(results_list_1)[:, 1]    # [kg Wb-1]
    # print('col_massdens_1dN.shape', col_massdens_1dN.shape)

    # Southward
    args = list(zip(
        Ai_1d,
        ni_1d,
        H_1d,
        r_A0*np.ones(arg_size),
        S3wlon_A0*np.ones(arg_size),
        0.0*np.ones(arg_size),
        -hem_fp*np.ones(arg_size),
        S_A0*np.ones(arg_size)
    ))
    with Pool(processes=parallel) as pool:
        results_list_2 = list(pool.starmap(calc, args))

    # Sum of northward and southward
    col_massdens_1dS = np.array(results_list_2)[:, 0]    # [kg m-2]
    ftmc_mag_1dS = np.array(results_list_2)[:, 1]    # [kg Wb-1]

    print('--- Total time [sec]:', round(time.time()-start_all, 4))

    np.savetxt('results/column_mass/'+exname+'/params_Ai.txt',
               np.array([Ai_0, Ai_1, Ai_num]))
    np.savetxt('results/column_mass/'+exname+'/params_ni.txt',
               np.array([ni_0, ni_1, ni_num]))
    np.savetxt('results/column_mass/'+exname+'/params_Ti.txt',
               np.array([Ti_0, Ti_1, Ti_num]))
    np.savetxt('results/column_mass/'+exname+'/col_massdens_1dN.txt',
               col_massdens_1dN)
    np.savetxt('results/column_mass/'+exname+'/col_massdens_1dS.txt',
               col_massdens_1dS)
    np.savetxt('results/column_mass/'+exname+'/ftmc_mag_1dN.txt',
               ftmc_mag_1dN)
    np.savetxt('results/column_mass/'+exname+'/ftmc_mag_1dS.txt',
               ftmc_mag_1dS)


# %% EXECUTE
if __name__ == '__main__':
    # Name of execution
    exname = '005/20250923_Ganymede'

    # Input about Juno observation
    TARGET_MOON = 'Ganymede'
    TARGET_FP = ['MAW']
    PJ_LIST = [3]

    # Input about the paremeter space
    Ai_0, Ai_1, Ai_num, Ai_scale = 12.0, 16.0, 3, 'linear'
    ni_0, ni_1, ni_num, ni_scale = 1.0, 100.0, 50, 'log'
    Ti_0, Ti_1, Ti_num, Ti_scale = 1.0, 200.0, 60, 'log'

    # Number of parallel processes
    parallel = 35
    main()
