""" Leadangle_fit.py

Created on Jul 21, 2023
@author: Shin Satoh

Description:


Version
1.0.0 (Jul 21, 2023)

"""
# %% Import
from multiprocessing import Pool
from numba import jit
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LinearSegmentedColormap  # colormapをカスタマイズする
import matplotlib.patheffects as pe
import B_JRM33 as BJRM
import B_equator as BEQ
import Leadangle_wave as LeadA
import time

# Color universal design
cud4 = ['#FF3300', '#FFF100', '#03AF7A', '#005AFF',
        '#4DC4FF', '#FF8082', '#F6AA00', '#990099', '#804000']
cud4bs = ['#FFCABF', '#FFFF80', '#D8F255', '#BFE4FF',
          '#FFCA80', '#77D9A8', '#C9ACE6', '#84919E']


# %% Switch
hem = 'North'       # 'North' or 'South'
year = 2022         # 2014, 2022, 202218509, 202231019, 202234923
exname = '2022_R4_12B'


# %% matplotlib フォント設定
fontname = 'Nimbus Sans'
plt.rcParams.update({'font.sans-serif': fontname,
                     'font.family': 'sans-serif',
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': fontname,
                     'mathtext.it': fontname+':italic',
                     # 'mathtext.bf': 'Nimbus Sans:italic:bold',
                     'mathtext.bf': fontname+':bold'
                     })
params = {
    # 'lines.markersize': 1,
    # 'lines.linewidth': 1,
    'axes.linewidth': 2,
    'xtick.major.size': 5,
    'xtick.minor.size': 3.5,
    'xtick.major.width': 2.0,
    'xtick.minor.width': 1.25,
    'ytick.major.size': 5,
    'ytick.minor.size': 3,
    'ytick.major.width': 2.0,
    'ytick.minor.width': 1.25,
}
plt.rcParams.update(params)


# %% 定数
MOON = 'Europa'
MU0 = 1.26E-6            # 真空中の透磁率
AMU = 1.66E-27           # [kg]
RJ = 71492E+3            # JUPITER RADIUS [m]
C = 2.99792E+8           # 光速 [m/s]
OMGJ = 1.75868E-4        # 木星の自転角速度 [rad/s]
satovalN = np.recfromtxt('data/JRM33/satellite_foot_N.txt', skip_header=3,
                         names=['wlon', 'amlat', 'amwlon', 'iolat', 'iowlon', 'eulat', 'euwlon', 'galat', 'gawlon'])
satovalS = np.recfromtxt('data/JRM33/satellite_foot_S.txt', skip_header=3,
                         names=['wlon', 'amlat', 'amwlon', 'iolat', 'iowlon', 'eulat', 'euwlon', 'galat', 'gawlon'])


# 定数
r_orbitM = 9.38*RJ   # ORBITAL RADIUS (average) [m] (Bagenal+2015)
r_orbitC = 9.30*RJ   # ORBITAL RADIUS (closest) [m] (Bagenal+2015)
r_orbitF = 9.47*RJ   # ORBITAL RADIUS (farthest) [m] (Bagenal+2015)
MOONRADI = 1.56E+6     # MOON RADIUS [m]
OMG_E = 2.0478E-5     # 衛星の公転角速度 [rad/s]
n0 = 110             # [cm-3] (Cassidy+2013)

ne_1 = 63            # 電子数密度 [cm-3] (Bagenal+2015)
ne_2 = 158           # 電子数密度 [cm-3] (Bagenal+2015)
ne_3 = 290           # 電子数密度 [cm-3] (Bagenal+2015)

Ai_1 = 18            # 平均イオン原子量 (Bagenal+2015)
# 平均イオン原子量 (Bagenal+2015) お試し ==============================
Ai_1 = 12

Hp0 = 0.64*RJ        # H0 [m] (Bagenal&Delamere2011)

wlonN = copy.copy(satovalN.wlon)
FwlonN = copy.copy(satovalN.euwlon)
FlatN = copy.copy(satovalN.eulat)
FwlonS = copy.copy(satovalS.euwlon)
FlatS = copy.copy(satovalS.eulat)

Psyn_eu = (11.22)*3600      # Moon's synodic period [sec]
OMGR = 2*np.pi/(Psyn_eu)    # Moon's synodic angular velocity [rad/sec]
print(OMGR)


# %% Colors
cpalette = ['#E7534B', '#EF8C46', '#F7D702', '#3C8867',
            '#649CDB', '#5341A5', '#A65FAC', '#A2AAAD']


# %% HST data
north_doy14 = ['14/006_v06', '14/013_v13', '14/016_v12']
north_doy22 = ['22/271_v18', '22/274_v17']
north_doy = north_doy14 + north_doy22
south_doy = ['22/185_v09', '22/310_v19', '22/349_v23']
south_doy = ['22/185_v09_MAW_']

nbkg_doy = ['14/001_v01', '14/002_v02', '14/005_v05', '14/006_v06', '14/013_v13', '14/016_v12',
            '22/185_v10', '22/228_v13', '22/271_v18', '22/274_v17', '22/309_v20', '22/349_v24']
sbkg_doy = ['22/140_v03', '22/185_v09', '22/186_v11',
            '22/229_v14', '22/310_v19', '22/311_v21', '22/349_v23']

refnum = 'N'
if hem == 'South':
    refnum = 'S'
satoval = np.recfromtxt('data/JRM33/satellite_foot_'+refnum+'.txt', skip_header=3,
                        names=['wlon', 'amlat', 'amwlon', 'iolat', 'iowlon', 'eulat', 'euwlon', 'galat', 'gawlon'])

if year == 2014:
    doy1422 = north_doy14

elif year == 2022:
    doy1422 = north_doy22

elif year == 202218509:
    doy1422 = ['22/185_v09R']

elif year == 2022185:
    doy1422 = ['22/185_v09_MAW_R']

elif year == 202231019:
    doy1422 = ['22/310_v19R']

elif year == 202234923:
    doy1422 = ['22/349_v23R']

else:
    raise ValueError


# %% Function to be in loop
def calc(RHO0: float, HP: float, MOONS3: float):
    """
    `RHO0`: plasma density at the equator [amu cm-3] \\
    `HP`: scale height of the plasma sheet [m] \\
    `moons3`: system iii longitude of the moon [deg]
    """

    S0 = LeadA.Awave().tracefield(r_orbitM, math.radians(MOONS3))
    tau, _, _ = LeadA.Awave().tracefield2(r_orbitM,
                                          math.radians(MOONS3),
                                          S0,
                                          RHO0,
                                          HP,
                                          refnum)
    # print('done')
    return tau


# %% Plot function
def LAplot(doy1422, estimations, RHO0: float, TI: float, chi2: float, II: int, JJ: int):
    """
    `doy1422`: doy list \\
    `estimations`: estimations of equatorial lead angle \\
    ---`estimations[0,:]`: satellite system iii longitude [deg] \\
    ---`estimations[1,:]`: equatorial lead angle of actual auroral spot [deg] \\
    ---`estimations[2,:]`: model estimation of equatorial lead angle of auroral spot [deg] \\
    `RHO0`: plasma density at the equator [amu cm-3] \\
    `TI`: plasma temperature [eV] \\
    `chi2`: chi square value \\
    `II`: index \\
    `JJ`: index
    """
    fontsize = 15
    fig, ax = plt.subplots(figsize=(4.6, 2.2), dpi=80)
    ax.set_title('('+str(II)+', '+str(JJ)+') Equatorial Lead Angle $\\delta$',
                 weight='bold', fontsize=fontsize)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 20)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xticklabels(np.arange(0, 361, 45), fontsize=fontsize)
    ax.xaxis.set_minor_locator(AutoMinorLocator(3))  # minor ticks
    ax.set_yticks(np.arange(0, 21, 5))
    ax.set_yticklabels(np.arange(0, 21, 5), fontsize=fontsize)
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # minor ticks
    ax.set_xlabel('Europa S3 longitude [deg]', fontsize=fontsize)
    ax.set_ylabel('$\\delta$ [deg]', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    for i in range(len(doy1422)):
        data_arr = np.loadtxt(
            'data/red3_leadangle/EUROPA/20'+doy1422[i]+'_eq.txt')
        moons30_arr = data_arr[0, :]    # moon s3 longitude [deg]
        eq_leadangle = data_arr[1, :]   # observed lead angle [deg]
        ax.scatter(moons30_arr, eq_leadangle, marker='x',
                   c=cpalette[i], linewidths=0.5, zorder=1)
        ax.scatter(estimations[0, :], estimations[2, :],
                   s=1, color='k', zorder=2)

    ax.text(0.98, 0.96, '$\\rho_{\,0}=$'+str(round(RHO0, 1))+' amu cm$^{-3}$', color='k',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=fontsize*0.72)
    ax.text(0.98, 0.79, '$\\langle T_i\, \\rangle=$'+str(round(TI, 1))+' eV', color='k',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=fontsize*0.72)
    ax.text(0.98, 0.58, '$\\chi^2=$'+str(round(chi2, 1)), color='k',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=fontsize*0.72)

    fig.tight_layout()
    plt.savefig('img/LeadangleFit/'+exname+'/img/fit_'+str(II)+'_' +
                str(JJ)+'.png', bbox_inches='tight')
    plt.close()
    return 0


# %% Main function
def main2(RHO0: float, Ti0: float, HP0: float, II: int, JJ: int):
    """
    `RHO0`: plasma density at the equator [amu cm-3] \\
    `Ti0`: plasma temperature [eV] \\
    `HP0`: scale height of the plasma sheet [m]
    """

    if year == 2014:
        DOY1422 = north_doy14
        img_cut = [[5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 3, 3, 3],
                   [5, 4, 4, 4, 4, 4, 4, 6, 6, 5, 5, 5, 5, 5],
                   [4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3]]
        cut_idx = [[0, 5, 10, 15, 20, 25, 30, 35, 40, 44, 48, 51, 54],
                   [0, 5, 9, 13, 17, 21, 25, 29, 35, 41, 46, 51, 56, 61],
                   [0, 4, 8, 11, 14, 17, 20, 23, 26, 29, 32]]
        ests = np.zeros((4, 38))
        la_err = np.array([1.0614052546455, 1.1145213015136, 0.8732066510795])

    elif year == 2022:
        DOY1422 = north_doy22
        img_cut = [[10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
                   [10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]]
        cut_idx = [[0, 10, 20, 30, 40, 50, 59, 68, 77, 86, 95, 104, 113, 122, 131, 140],
                   [0, 10, 20, 29, 38, 47, 56, 65, 74, 83, 92, 101, 110, 119, 128, 137, 146, 155, 164, 173, 182, 191, 200, 209]]
        ests = np.zeros((4, 40))
        la_err = np.array([1.0478863038047, 0.8994575700965])

    elif year == 202218509:
        DOY1422 = ['22/185_v09R']
        img_cut = [[7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]]
        cut_idx = [[0, 7, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74]]
        ests = np.zeros((4, 13))
        la_err = np.array([0.6379928352107])

    elif year == 2022185:
        DOY1422 = ['22/185_v09_MAW_R']
        img_cut = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        cut_idx = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                    10, 11, 12, 13, 14, 15, 16, 17, 18]]
        ests = np.zeros((4, 19))
        la_err = np.array([0.6379928352107])

    elif year == 202231019:
        DOY1422 = ['22/310_v19R']
        img_cut = [6, 6, 6, 6, 6, 5]
        cut_idx = [[0, 6, 12, 18, 24, 30]]
        ests = np.zeros((4, 6))
        la_err = np.array([0.56611511426765])

    elif year == 202234923:
        DOY1422 = ['22/349_v23R']
        img_cut = [6, 6, 6, 6, 6, 5]
        cut_idx = [[0, 6, 12, 18, 24, 30]]
        ests = np.zeros((4, 6))
        la_err = np.array([0.6325058969635])

    NN = 0     # end index of images of the doy
    for i in range(len(DOY1422)):
        data_arr = np.loadtxt(
            'data/red3_leadangle/EUROPA/20'+DOY1422[i]+'_eq.txt')
        params_arr = np.zeros((3, len(cut_idx[i])))
        for j in range(len(cut_idx[i])):
            moons30_arr = data_arr[0, cut_idx[i]
                                   [j]:cut_idx[i][j]+img_cut[i][j]]
            eq_leadangle = data_arr[1, cut_idx[i]
                                    [j]:cut_idx[i][j]+img_cut[i][j]]
            params_arr[0, j] = np.average(moons30_arr)      # 平均値 [deg]
            params_arr[1, j] = np.average(eq_leadangle)     # 平均値 [deg]
            params_arr[2, j] = la_err[i]                    # Wlong err [deg]
        # moon s3 longitude [deg]
        ests[0, NN:NN+params_arr.shape[1]] = params_arr[0, :]
        # observed lead angle [deg]
        ests[1, NN:NN+params_arr.shape[1]] = params_arr[1, :]
        # error [deg]
        ests[3, NN:NN+params_arr.shape[1]] = params_arr[2, :]

        NN += params_arr.shape[1]

    # 並列計算用 変数リスト(zip)
    # np.arrayは不可。ここが1次元なのでpoolの結果も1次元。
    args = list(zip(
        RHO0*np.ones(ests.shape[1]),
        HP0*np.ones(ests.shape[1]),
        ests[0, :]-ests[1, :]))

    start = time.time()
    with Pool(processes=19) as pool:
        result_list = list(pool.starmap(calc, args))
    tau = np.array(result_list)         # [sec]
    print(str(II)+' '+str(JJ)+' time', time.time()-start)

    leadangle_est = np.degrees(OMGR*tau)  # [deg]

    # estimated lead angle [deg]
    ests[2, :] = leadangle_est

    # CHI SQUARE VALUE
    chi2 = np.sum(((ests[1, :]-ests[2, :])/ests[3, :])**2)

    # PLOT
    LAplot(DOY1422, ests, RHO0, Ti0, chi2, II, JJ)

    return chi2


# %% EXECUTE
if __name__ == '__main__':
    # Parameters
    RHO0_len = 68
    Ti0_len = 54
    # RHO0_len = 30
    # Ti0_len = 20
    RHO0 = np.linspace(np.log(300), np.log(4000), RHO0_len)
    # RHO0 = np.linspace(np.log(500), np.log(9000), RHO0_len)
    RHO0 = np.exp(RHO0)

    """
    # Southern
    RHO0_len = 70
    Ti0_len = 60
    RHO0 = np.linspace(np.log(1200), np.log(9000), RHO0_len)
    RHO0 = np.exp(RHO0)
    """

    RHO0_len = RHO0.size
    print('RHO0_len', RHO0_len)

    Ti0 = np.linspace(np.log(20), np.log(700), Ti0_len)
    # Ti0 = np.linspace(np.log(50), np.log(800), Ti0_len)
    Ti0 = np.exp(Ti0)
    RHO0, Ti0 = np.meshgrid(RHO0, Ti0)

    # Plasma sheet scale height
    HP_arr = Hp0*np.sqrt(Ti0/Ai_1)     # [m] (Bagenal&Delamere2011)

    # Calculate the total number of images
    IMG_LEN = 0
    for i in range(len(doy1422)):
        data_arr = np.loadtxt(
            'data/red3_leadangle/EUROPA/20'+doy1422[i]+'_eq.txt')
        moons30_arr = data_arr[0, :]    # moon s3 longitude [deg]
        IMG_LEN += moons30_arr.size

    # chi2 data array
    chi2_arr = np.zeros((Ti0_len, RHO0_len))
    i = 0
    j = 0
    start = time.time()
    for i in range(Ti0_len):
        print('Hp [RJ]', HP_arr[i, 0]/RJ)
        # if HP_arr[i, 0] > 2.8*RJ:
        #     0
        # break
        for j in range(RHO0_len):
            chi2_arr[i, j] = main2(RHO0[i, j], Ti0[i, j],
                                   HP_arr[i, j],
                                   i, j)
    print('total time', time.time()-start)

    print(chi2_arr)

    np.savetxt('img/LeadangleFit/'+exname+'/params_RHO0.txt',
               RHO0)
    np.savetxt('img/LeadangleFit/'+exname+'/params_Ti0.txt',
               Ti0)
    np.savetxt('img/LeadangleFit/'+exname+'/params_chi2.txt',
               chi2_arr)
