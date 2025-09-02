""" Leadangle_fit_JunoUVS.py

Created on Apr 28, 2025
@author: Shin Satoh

Description:
Using the lead angle values measured in one single Perijove of Juno,
this program iterates the Alfven wave tracing along the magnetic
field line and estimate the transit time of the Alfven wave from the
satellite to the auroral footprint.

Interspot distance between TEB and MAW in one hemisphere depends
strongly on the density of ions.

Version
1.0.0 (Apr 28, 2025)

"""
# %% Import
import spiceypy as spice
from multiprocessing import Pool
# from numba import jit
import numpy as np
import math
# import copy
# import matplotlib.pyplot as plt
# from matplotlib.ticker import AutoMinorLocator
# from matplotlib.colors import LinearSegmentedColormap  # colormapをカスタマイズする
# import matplotlib.patheffects as pe

from Leadangle_fit_JunoUVS import TEB_transit
from Leadangle_fit_JunoUVS import create_argmesh
from Leadangle_fit_JunoUVS import eqwlong_err
from Leadangle_fit_JunoUVS import Alfven_launch_site
from Leadangle_fit_JunoUVS import scaleheight
from Leadangle_fit_JunoUVS import moonS3wlon_arr
from Leadangle_fit_JunoUVS import calc_eqlead
import Leadangle_wave as Wave
import datetime
import time
# import os
from scipy.io import readsav
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


# %% Read the savfile
def read1savfile(PJnum: int, target_moon: str, target_fp: str):
    """
    Args:
        `PJnum` (int): Perijove number \\
        `moon` (str): Name of the target (Io/Europa/Ganymede) \\
        `footprint` (str): `MAW` or `TEB` \\

    Returns:
        Tuple of \\
        `wlon_MAW` (ndarray) \\
        `wlon_TEB` (ndarray) \\
        `err_wlon_MAW` (ndarray) \\
        `err_wlon_TEB` (ndarray) \\
        `lat_MAW` (ndarray) \\
        `lat_TEB` (ndarray) \\
        `err_lat_MAW` (ndarray) \\
        `err_lat_TEB` (ndarray) \\
        `wlon_moon` (ndarray) \\
        `et` (ndarray) \\
        `hem` (ndarray) \\
    """
    # size of the data
    datasize = 0

    # Look for the file named
    # `IFP_info_v900km_fixed.sav` for Io footprint
    savpath = 'data/Output_v2_PaperHue2023_error2/' + 'PJ' + \
        str(PJnum).zfill(2)+'/'+target_moon[0]+'FP_info_v900km_fixed.sav'

    # Read
    savdata = readsav(savpath)

    var = savdata['fp_info']

    # 'MIDTIME_ET'を用いてスライス位置を決定する
    MIDTIME_ET = np.array(var['MIDTIME_ET'][0])
    idx = np.where(MIDTIME_ET > 0)

    wlon_MAW = np.array(var['LON_MAW'][0])[idx]
    err_wlon_MAW = np.array(var['LON_MAW_ERROR'][0])[idx]
    lat_MAW = np.array(var['LAT_MAW'][0])[idx]
    err_lat_MAW = np.array(var['LAT_MAW_ERROR'][0])[idx]

    wlon_TEB = np.array(var['LON_TEB'][0])[idx]
    err_wlon_TEB = np.array(var['LON_TEB_ERROR'][0])[idx]
    lat_TEB = np.array(var['LAT_TEB'][0])[idx]
    err_lat_TEB = np.array(var['LAT_TEB_ERROR'][0])[idx]

    wlon_moon = np.array(var['SIII_LON'][0])[idx]
    et = np.array(var['MIDTIME_ET'][0])[idx]
    hem = var['HEMISPHERE'][0][idx]

    # Extract MAWs (exclude values -999.)
    fpvalues = np.where((wlon_MAW > -100) & (wlon_TEB > -100))
    wlon_MAW = wlon_MAW[fpvalues]
    err_wlon_MAW = err_wlon_MAW[fpvalues]
    lat_MAW = lat_MAW[fpvalues]
    err_lat_MAW = err_lat_MAW[fpvalues]
    wlon_TEB = wlon_TEB[fpvalues]
    err_wlon_TEB = err_wlon_TEB[fpvalues]
    lat_TEB = lat_TEB[fpvalues]
    err_lat_TEB = err_lat_TEB[fpvalues]
    wlon_moon = wlon_moon[fpvalues]
    et = et[fpvalues]
    hem = hem[fpvalues]

    wlon_fp = np.array([wlon_MAW, wlon_TEB])
    err_wlon_fp = np.array([err_wlon_MAW, err_wlon_TEB])
    lat_fp = np.array([lat_MAW, lat_TEB])
    err_lat_fp = np.array([err_lat_MAW, err_lat_TEB])

    hem_N = np.where(hem == b'North')
    hem_S = np.where(hem == b'South')
    hem[hem_N] = -1
    hem[hem_S] = 1

    return wlon_fp, err_wlon_fp, lat_fp, err_lat_fp, wlon_moon, et, hem


# %% Function to be in loop
def calc(Ai, ni, Hp, r_A0, S3wlon_A0, z_A0, hem, S_A0=0):
    # print('Calc loop in')
    # S_A0 = Wave.Awave().tracefield(r_A0,
    #                                np.radians(S3wlon_A0),
    #                                z_A0
    #                                )
    tau, _, _, _ = Wave.Awave().trace3(r_A0,
                                       np.radians(S3wlon_A0),
                                       z_A0,
                                       S_A0,
                                       Ai,
                                       ni,
                                       Hp,
                                       hem
                                       )
    return tau


# %%
def Obsresults(PJ_LIST, TARGET_MOON, TARGET_FP):
    # 初期化
    wlon_MAW = np.zeros(3)
    err_wlon_MAW = np.zeros(3)
    lat_MAW = np.zeros(3)
    err_lat_MAW = np.zeros(3)

    wlon_TEB = np.zeros(3)
    err_wlon_TEB = np.zeros(3)
    lat_TEB = np.zeros(3)
    err_lat_TEB = np.zeros(3)

    wlon_moon_fp = np.zeros(3)
    et_fp = np.zeros(3)
    hem_fp = np.zeros(3)
    pj_fp = np.zeros(3)

    for i in PJ_LIST:
        for j in TARGET_FP:
            wlon_fp1, err_wlon_fp1, lat_fp1, err_lat_fp1, wlon_moon_fp1, et_fp1, hem_fp1 = read1savfile(
                PJnum=i, target_moon=TARGET_MOON, target_fp=j)

            wlon_MAW1 = wlon_fp1[0, :]
            err_wlon_MAW1 = err_wlon_fp1[0, :]
            lat_MAW1 = lat_fp1[0, :]
            err_lat_MAW1 = err_lat_fp1[0, :]

            wlon_TEB1 = wlon_fp1[1, :]
            err_wlon_TEB1 = err_wlon_fp1[1, :]
            lat_TEB1 = lat_fp1[1, :]
            err_lat_TEB1 = err_lat_fp1[1, :]

            wlon_MAW = np.append(wlon_MAW, wlon_MAW1)
            err_wlon_MAW = np.append(err_wlon_MAW, err_wlon_MAW1)
            lat_MAW = np.append(lat_MAW, lat_MAW1)
            err_lat_MAW = np.append(err_lat_MAW, err_lat_MAW1)

            wlon_TEB = np.append(wlon_TEB, wlon_TEB1)
            err_wlon_TEB = np.append(err_wlon_TEB, err_wlon_TEB1)
            lat_TEB = np.append(lat_TEB, lat_TEB1)
            err_lat_TEB = np.append(err_lat_TEB, err_lat_TEB1)

            wlon_moon_fp = np.append(wlon_moon_fp, wlon_moon_fp1)
            et_fp = np.append(et_fp, et_fp1)
            hem_fp = np.append(hem_fp, hem_fp1)
            pj_fp = np.append(pj_fp, i*np.ones(wlon_fp1.size))

    # 余計な部分を削除
    wlon_MAW = wlon_MAW[3:]
    err_wlon_MAW = err_wlon_MAW[3:]
    lat_MAW = lat_MAW[3:]
    err_lat_MAW = err_lat_MAW[3:]
    wlon_TEB = wlon_TEB[3:]
    err_wlon_TEB = err_wlon_TEB[3:]
    lat_TEB = lat_TEB[3:]
    err_lat_TEB = err_lat_TEB[3:]

    wlon_fp = np.array([wlon_MAW, wlon_TEB])
    err_wlon_fp = np.array([err_wlon_MAW, err_wlon_TEB])
    lat_fp = np.array([lat_MAW, lat_TEB])
    err_lat_fp = np.array([err_lat_MAW, err_lat_TEB])

    wlon_moon_fp = wlon_moon_fp[3:]
    et_fp = et_fp[3:]
    hem_fp = hem_fp[3:]
    pj_fp = pj_fp[3:]

    return wlon_fp, err_wlon_fp, lat_fp, err_lat_fp, wlon_moon_fp, et_fp, hem_fp, pj_fp


# %% Main function
def main():
    # Select moon synodic orbital period
    if TARGET_MOON == 'Io':
        Psyn = Psyn_io
    elif TARGET_MOON == 'Europa':
        Psyn = Psyn_eu
    elif TARGET_MOON == 'Ganymede':
        Psyn = Psyn_ga

    wlon_fp, err_wlon_fp, lat_fp, err_lat_fp, _, et_fp, hem_fp, _ = Obsresults(
        PJ_LIST, TARGET_MOON, ['MAW']
    )

    # Time: t0, the observation time
    _, _, _, _, _, _, moon_S3wlon = moonS3wlon_arr(et_fp, TARGET_MOON)

    # Equatorial values for MAW
    eqlead_MAW, eqlead_MAW_0, _, wlon_MAW_eq = calc_eqlead(wlon_fp[0, :],
                                                           err_wlon_fp[0, :],
                                                           lat_fp[0, :],
                                                           err_lat_fp[0, :],
                                                           hem_fp,
                                                           moon_S3wlon,
                                                           TARGET_MOON)
    _, _, z_A0, r_A0, _, _, S3wlon_A0 = Alfven_launch_site(et_fp,
                                                           eqlead_MAW,
                                                           TARGET_MOON)

    # Equatorial values for TEB
    eqlead_TEB, eqlead_TEB_0, _, wlon_TEB_eq = calc_eqlead(wlon_fp[1, :],
                                                           err_wlon_fp[1, :],
                                                           lat_fp[1, :],
                                                           err_lat_fp[1, :],
                                                           hem_fp,
                                                           moon_S3wlon,
                                                           TARGET_MOON)
    _, _, z_A1, r_A1, _, _, S3wlon_A1 = Alfven_launch_site(et_fp,
                                                           eqlead_TEB,
                                                           TARGET_MOON)

    # パラメータ空間(meshgrid → 1d)の作成
    Ai_1d, ni_1d, Ti_1d, _, _, _ = create_argmesh(Ai_0, Ai_1, Ai_num, Ai_scale,
                                                  ni_0, ni_1, ni_num, ni_scale,
                                                  Ti_0, Ti_1, Ti_num, Ti_scale)
    H_1d = scaleheight(Ai=Ai_1d, Zi=Zi, Ti=Ti_1d, Te=Te)
    arg_size = Ai_1d.size

    # 衛星本体の経度: Juno spin中の変位 (どのデータに対しても一定値)
    sigma_x = eqwlong_err(Psyn, dt=22.5)  # [deg]

    # 注意: iは観測データ点のインデックス
    i_size = et_fp.size
    y_obs = np.zeros((i_size, arg_size))
    sigma_total = np.zeros((i_size, arg_size))
    sigma_y_MAW = np.zeros(i_size)
    sigma_y_TEB = np.zeros(i_size)
    eqlead_est_MAW = np.zeros((i_size, arg_size))
    eqlead_est_TEB = np.zeros((i_size, arg_size))
    print('PJ number:', PJ_LIST)
    print('Target moon:', TARGET_MOON)
    print('Target fp:', 'Interspot distance')
    print('Number of data points used/total:', i_size, '/', et_fp.size)
    print('Param space shape:', ni_num, Ai_num, Ti_num)
    start_all = time.time()
    for i in range(i_size):
        # print('r_A0 [RJ]:', r_A0[i]/RJ)
        # print('S3wlon_A0 [deg]:', S3wlon_A0[i])

        start_1loop = time.time()
        # MAW fitting
        S_A0 = Wave.Awave().tracefield(r_A0[i],
                                       np.radians(S3wlon_A0[i]),
                                       z_A0[i]
                                       )
        args_0 = list(zip(
            Ai_1d,
            ni_1d,
            H_1d,
            r_A0[i]*np.ones(arg_size),
            S3wlon_A0[i]*np.ones(arg_size),
            z_A0[i]*np.ones(arg_size),
            hem_fp[i]*np.ones(arg_size),
            S_A0*np.ones(arg_size)
        ))

        # TEB fitting
        S_A1 = Wave.Awave().tracefield(r_A1[i],
                                       np.radians(S3wlon_A1[i]),
                                       z_A1[i]
                                       )
        args_1 = list(zip(
            Ai_1d,
            ni_1d,
            H_1d,
            r_A1[i]*np.ones(arg_size),
            S3wlon_A1[i]*np.ones(arg_size),
            z_A1[i]*np.ones(arg_size),
            -hem_fp[i]*np.ones(arg_size),
            S_A1*np.ones(arg_size)
        ))

        # MAW fitting
        with Pool(processes=parallel) as pool:
            results_list = list(pool.starmap(calc, args_0))
        tau_0 = np.array(results_list)    # [sec]

        # TEB fitting
        with Pool(processes=parallel) as pool:
            results_list = list(pool.starmap(calc, args_1))
        tau_1 = np.array(results_list) + \
            TEB_transit(r_A1[i], S3wlon_A1[i])    # [sec]

        print(str(i).zfill(2), '- Loop time [sec]:', round(
            time.time()-start_1loop, 4))

        y_obs[i, :] = (eqlead_MAW[i]-eqlead_TEB[i])*np.ones(arg_size)

        sigma_y_MAW[i] = eqlead_MAW_0[i]+sigma_x
        sigma_y_TEB[i] = eqlead_TEB_0[i]+sigma_x
        sigma_total[i, :] = (eqlead_MAW_0[i]+eqlead_TEB_0[i])*np.ones(arg_size)
        eqlead_est_MAW[i, :] = tau_0*360/Psyn
        eqlead_est_TEB[i, :] = tau_1*360/Psyn
        # print('Hemisphere', hem_fp[i])
        # print('TEB position:', wlon_fp[1, i])
        # print('MAW eq position:', wlon_MAW_eq[i], S3wlon_A0[i])
        # print('TEB eq position:', wlon_TEB_eq[i], S3wlon_A1[i])
        # print('MAW eq lead:', eqlead_MAW[i])
        # print('TEB eq lead:', eqlead_TEB[i])
        # print('Eq interspot obs:', y_obs[i, 0])
        # print('Sigma total:', sigma_total[i, 0])

    print('--- Total time [sec]:', round(time.time()-start_all, 4))

    # Chi square value (Interspot distance [deg])
    y_est = eqlead_est_MAW-eqlead_est_TEB
    chi2 = np.sum(((y_obs-y_est)/sigma_total)**2, axis=0)
    np.savetxt('results/fit/'+exname+'/params_chi2.txt',
               chi2)
    np.savetxt('results/fit/'+exname+'/params_Ai.txt',
               np.array([Ai_0, Ai_1, Ai_num]))
    np.savetxt('results/fit/'+exname+'/params_ni.txt',
               np.array([ni_0, ni_1, ni_num]))
    np.savetxt('results/fit/'+exname+'/params_Ti.txt',
               np.array([Ti_0, Ti_1, Ti_num]))
    np.savetxt('results/fit/'+exname+'/params_H.txt',
               H_1d)
    np.savetxt('results/fit/'+exname+'/eqlead_est_MAW.txt',
               eqlead_est_MAW)
    np.savetxt('results/fit/'+exname+'/eqlead_est_TEB.txt',
               eqlead_est_TEB)
    np.savetxt('results/fit/'+exname+'/eqlead_obs_MAW.txt',
               eqlead_MAW)
    np.savetxt('results/fit/'+exname+'/eqlead_obs_TEB.txt',
               eqlead_TEB)
    np.savetxt('results/fit/'+exname+'/sigma_y_MAW.txt',
               sigma_y_MAW)
    np.savetxt('results/fit/'+exname+'/sigma_y_TEB.txt',
               sigma_y_TEB)
    np.savetxt('results/fit/'+exname+'/hems_obs.txt',
               hem_fp)
    np.savetxt('results/fit/'+exname+'/moon_S3wlon_obs.txt',
               moon_S3wlon)
    np.savetxt('results/fit/'+exname+'/et_obs.txt',
               et_fp)


# %% EXECUTE
if __name__ == '__main__':
    # Name of execution
    exname = '1001/20250428_002'

    # Input about Juno observation
    TARGET_MOON = 'Io'
    PJ_LIST = [8]
    Zi = 1.3    # ION CHARGE [C] !!! CONSTANT !!!
    Te = 6      # ELECTRON TEMPERATURE [eV] !!! CONSTANT !!!

    # Input about the paremeter space
    Ai_0, Ai_1, Ai_num, Ai_scale = 20.0, 24.0, 3, 'linear'
    ni_0, ni_1, ni_num, ni_scale = 500.0, 5000.0, 50, 'log'
    Ti_0, Ti_1, Ti_num, Ti_scale = 10.0, 1000.0, 60, 'log'

    # Number of parallel processes
    parallel = 25
    main()


"""
for j in range(Ai_1d.size):
    Ai = Ai_1d[j]
    ni = ni_1d[j]
    Hp = H_1d[j]
    S_A0 = Wave.Awave().tracefield(r_A0[i],
                                   np.radians(S3wlon_A0[i]),
                                   z_A0[i]
                                   )
    tau, _, _, _ = Wave.Awave().trace3(r_A0[i],
                                       np.radians(S3wlon_A0[i]),
                                       z_A0[i],
                                       S_A0,
                                       Ai,
                                       ni,
                                       Hp,
                                       hem_MAW[i]
                                       )

    print('eqlead [deg]:', tau*360/Psyn_eu)
"""

# 作業メモ
# Juno/UVSデータ(フットプリントの緯度経度)を取り込む → done
# とりあえずMAWだけを使う方針で → done
# 衛星本体のz座標は必ずしも0ではない → Spiceで計算 done
# 各時刻の赤道リード角を計算する部分を追加 done
# トレースもほぼできた。伝播時間からリード角も計算OK。
# トレース終着点は観測されたフットプリントの経度と違っていいのか？
