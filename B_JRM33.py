""" B_JRM33.py

Created on Mar 18, 2023
@author: Shin Satoh

Description:


Version
1.0.0 (Apr 7, 2023)

"""


# %% LIBRARIES
from numba import jit
import numpy as np
import math
import pyshtools as pysh


# TOGGLES
JRM_NUM = 'JRM33'                           # 'JRM09' or 'JRM33'

# DATA LOADING FOR COEFFICIENTS AND POSITION OF SATELLITE FOOTPRINTS
jrm_coef = np.loadtxt(
    'data/'+JRM_NUM+'/coef.txt', skiprows=1, usecols=1
)

# DEGREE OF LEGENDRE FUNCTIONS
if JRM_NUM == 'JRM09':
    NN = 10
elif JRM_NUM == 'JRM33':
    NN = 30

# %% CONSTANTS
RJ = 71492E+3           # JUPITER RADIUS [m]

mu0 = 1.26E-6           # PERMEABILITY [N A^-2] = [kg m s^-2 A^-2]

me = 9.1E-31            # MASS OF ELECTRON [kg]
e = (1.6E-19)           # CHARGE OF ELECTRON [C]

R0 = 7.8*RJ      # [m] Connerney+2020
R1 = 51.4*RJ     # [m] Connerney+2020
AA = R0          # [m] Connerney+2020
BB = R1          # [m] Connerney+2020
CC = 139.6       # [nT] Connerney+2020 (A = mu0*I0/2)
DD = 3.6*RJ      # [m] Connerney+2020
# Disc normal from rotation axis Connerney+2020
THETA_D = math.radians(9.3)
# Azimuth angle of disc normal Connerney+2020
PHI_D = math.radians(204.2)


class B():
    def __init__(self):
        return None

    def JRM33_000(self, rs, theta, phi):
        """
        ### Parameters
        `rs` ... <float> radial distance [m] \\
        `theta` ... <float> colatitude of the point [rad] \\
        `phi` ... <float> eest-longitude of the point [rad] \\

        ### Returns
        <ndarray, shape (3,)> Magnetic field (B_r, B_theta, B_phi) [G]
        """

        # SCHMIDT QUASI-NORMALIZED LEGENDRE FUNCTIONS
        p_arr, dp_arr = pysh.legendre.PlmSchmidt_d1(NN, math.cos(theta))
        dp_arr *= -math.sin(theta)        # NECESSARY MULTIPLICATION

        p_arr = p_arr[1:]               # n <= 1
        dp_arr = dp_arr[1:]             # n <= 1

        # 磁場の計算
        # r成分
        dVdr = 0
        dVdr_n = np.zeros(NN)

        # theta成分
        dVdtheta = 0
        dVdtheta_n = np.zeros(NN)

        # phi成分
        dVdphi = 0
        dVdphi_n = np.zeros(NN)

        for i in range(NN):
            n = i+1                              # INDEX n
            m = np.arange(0, n+1, 1, dtype=int)  # INDEX m

            p_s = int((n-1)*(n+2)/2)             # LEGENDRE関数arrayの先頭
            p_e = p_s + n                        # LEGENDRE関数arrayの終端
            g_s = n**2 - 1                       # g_nm の先頭
            g_e = g_s + n                        # g_nm の終端
            h_s = g_e + 1                        # g_nm の先頭
            h_e = h_s + (n-1)                    # g_nm の終端
            # print(n, m, g_s+1, g_e+1, h_s+1, h_e+1)

            P_nm = p_arr[p_s:p_e+1]
            dP_nm = dp_arr[p_s:p_e+1]
            g_nm = jrm_coef[g_s:g_e+1]
            h_nm = np.zeros(g_nm.shape)          # m = 0のゼロを作る
            h_nm[1:] = jrm_coef[h_s:h_e+1]       # m >= 1に値を格納する

            # INDEX m方向に和をとる
            dVdr_n[i] = (-1-n)*(RJ/rs)**(n+2) * np.sum(
                P_nm*(g_nm*np.cos(m*phi) + h_nm*np.sin(m*phi))
            )

            # INDEX m方向に和をとる
            dVdtheta_n[i] = (RJ/rs)**(n+2) * np.sum(
                dP_nm*(g_nm*np.cos(m*phi) + h_nm*np.sin(m*phi))
            )

            # INDEX m方向に和をとる
            dVdphi_n[i] = (1/math.sin(theta))*((RJ/rs)**(n+2)) * np.sum(
                P_nm*m*(-g_nm*np.sin(m*phi) + h_nm*np.cos(m*phi))
            )

            # print(P_nm.shape)
            # print(g_nm.shape)
            # print(h_nm.shape)
            # print(h_nm)

        # INDEX n方向に和をとる
        dVdr = -np.sum(dVdr_n)
        dVdtheta = -np.sum(dVdtheta_n)
        dVdphi = -np.sum(dVdphi_n)

        # print(dVdr*1E-5)
        # print(dVdtheta*1E-5)
        # print(dVdphi*1E-5)
        # print(np.sqrt(dVdr**2 + dVdtheta**2 + dVdphi**2)*1E-5)

        return np.array([dVdr, dVdtheta, dVdphi])

    def JRM33(self, rs, theta, phi):
        """
        ### Parameters
        `rs` ... <float> radial distance [m] \\
        `theta` ... <float> colatitude of the point [rad] \\
        `phi` ... <float> eest-longitude of the point [rad] \\

        ### Returns
        <ndarray, shape (3,)> Magnetic field (B_r, B_theta, B_phi) [G]
        """

        # SCHMIDT QUASI-NORMALIZED LEGENDRE FUNCTIONS
        p_arr, dp_arr = pysh.legendre.PlmSchmidt_d1(NN, math.cos(theta))
        dp_arr *= -math.sin(theta)        # NECESSARY MULTIPLICATION

        p_arr = p_arr[1:]               # n <= 1
        dp_arr = dp_arr[1:]             # n <= 1

        return JRM33_calc(rs, theta, phi, p_arr, dp_arr)

    def BCS_000(self, X: float, Y: float, Z: float, phi: float):
        """
        Current sheet model by Connerney1981 & Connerney+2020\
        return ndarray(Bx, By, Bz) in [nT]
        """

        # ダイポール座標系に持っていく
        # S3RH で Z3軸 の(右ネジ)まわりに-65.8度回転
        phiRH0 = math.radians(-65.8)    # Connerney+2020
        rvec0 = np.array([
            X*math.cos(phiRH0) - Y*math.sin(phiRH0),
            X*math.sin(phiRH0) + Y*math.cos(phiRH0),
            Z
        ])

        # S3RH で X3軸 の(右ネジ)まわりに-7度回転
        rvec0 = np.array([
            rvec0[0],
            rvec0[1]*math.cos(THETA_D) - rvec0[2]*math.sin(THETA_D),
            rvec0[1]*math.sin(THETA_D) + rvec0[2]*math.cos(THETA_D)
        ])

        rho = math.sqrt(rvec0[0]**2 + rvec0[1]**2)
        z = rvec0[2]

        F1 = math.sqrt((z-DD)**2 + AA**2)
        F2 = math.sqrt((z+DD)**2 + AA**2)

        Brho = CC*rho*(1/F1 - 1/F2)
        Bphi = 0
        Bz = CC*(2*DD/math.sqrt(z**2+AA**2)
                 - 0.25*(rho**2)*((z-DD)/F1**3-(z+DD)/F2**3)
                 - 2*DD/math.sqrt(z**2+BB**2))

        Bx = Brho*math.cos(phi)
        By = Brho*math.sin(phi)

        # print(math.sqrt(Bx**2 + By**2 + Bz**2))

        return np.array([Bx, By, Bz])

    def BCS(self, X: float, Y: float, Z: float, phi: float):
        """
        Current sheet model by Connerney1981 & Connerney+2020\
        return ndarray(Bx, By, Bz) in [nT]
        """

        return BCS_calc(X, Y, Z, phi)


@jit(nopython=True, fastmath=True)
def JRM33_calc(rs: float, theta: float, phi: float, p_arr, dp_arr):
    """
    ### Parameters
    `rs` ... <float> radial distance [m] \\
    `theta` ... <float> colatitude of the point [rad] \\
    `phi` ... <float> eest-longitude of the point [rad] \\

    ### Returns
    <ndarray, shape (3,)> Magnetic field (B_r, B_theta, B_phi) [G]
    """

    # 磁場の計算
    # r成分
    dVdr = 0
    dVdr_n = np.zeros(NN)

    # theta成分
    dVdtheta = 0
    dVdtheta_n = np.zeros(NN)

    # phi成分
    dVdphi = 0
    dVdphi_n = np.zeros(NN)

    for i in range(NN):
        n = i+1                              # INDEX n
        m = np.arange(0, n+1, 1)             # INDEX m

        p_s = int((n-1)*(n+2)/2)             # LEGENDRE関数arrayの先頭
        p_e = p_s + n                        # LEGENDRE関数arrayの終端
        g_s = n**2 - 1                       # g_nm の先頭
        g_e = g_s + n                        # g_nm の終端
        h_s = g_e + 1                        # g_nm の先頭
        h_e = h_s + (n-1)                    # g_nm の終端
        # print(n, m, g_s+1, g_e+1, h_s+1, h_e+1)

        P_nm = p_arr[p_s:p_e+1]
        dP_nm = dp_arr[p_s:p_e+1]
        g_nm = jrm_coef[g_s:g_e+1]
        h_nm = np.zeros(g_nm.shape)          # m = 0のゼロを作る
        h_nm[1:] = jrm_coef[h_s:h_e+1]       # m >= 1に値を格納する

        # INDEX m方向に和をとる
        dVdr_n[i] = (-1-n)*(RJ/rs)**(n+2) * np.sum(
            P_nm*(g_nm*np.cos(m*phi) + h_nm*np.sin(m*phi))
        )

        # INDEX m方向に和をとる
        dVdtheta_n[i] = (RJ/rs)**(n+2) * np.sum(
            dP_nm*(g_nm*np.cos(m*phi) + h_nm*np.sin(m*phi))
        )

        # INDEX m方向に和をとる
        dVdphi_n[i] = (1/math.sin(theta))*((RJ/rs)**(n+2)) * np.sum(
            P_nm*m*(-g_nm*np.sin(m*phi) + h_nm*np.cos(m*phi))
        )

        # print(P_nm.shape)
        # print(g_nm.shape)
        # print(h_nm.shape)
        # print(h_nm)

    # INDEX n方向に和をとる
    dVdr = -np.sum(dVdr_n)
    dVdtheta = -np.sum(dVdtheta_n)
    dVdphi = -np.sum(dVdphi_n)

    # print(dVdr*1E-5)
    # print(dVdtheta*1E-5)
    # print(dVdphi*1E-5)
    # print(np.sqrt(dVdr**2 + dVdtheta**2 + dVdphi**2)*1E-5)

    return np.array([dVdr, dVdtheta, dVdphi])


@jit(nopython=True, fastmath=True)
def BCS_calc(X: float, Y: float, Z: float, Phi: float):
    """
    Current sheet model by Connerney1981 & Connerney+2020\
    return ndarray(Bx, By, Bz) in [nT]
    """

    """# ダイポール座標系に持っていく
    # S3RH で Z3軸 の(右ネジ)まわりに-65.8度回転
    phiRH0 = math.radians(-65.8)    # Connerney+2020
    rvec0 = np.array([
        X*math.cos(phiRH0) - Y*math.sin(phiRH0),
        X*math.sin(phiRH0) + Y*math.cos(phiRH0),
        Z
    ])

    # S3RH で X3軸 の(右ネジ)まわりに-9度回転
    rvec0 = np.array([
        rvec0[0],
        rvec0[1]*math.cos(THETA_D) - rvec0[2]*math.sin(THETA_D),
        rvec0[1]*math.sin(THETA_D) + rvec0[2]*math.cos(THETA_D)
    ])

    rho = math.sqrt(rvec0[0]**2 + rvec0[1]**2)
    phi = math.atan2(rvec0[1], rvec0[0])        # 磁場の円筒座標系における方位角
    z = rvec0[2]

    F1 = math.sqrt((z-DD)**2 + AA**2)
    F2 = math.sqrt((z+DD)**2 + AA**2)

    Brho = CC*rho*(1/F1 - 1/F2)
    Bphi = 0
    Bz = CC*(2*DD/math.sqrt(z**2+AA**2)
             - 0.25*(rho**2)*((z-DD)/F1**3-(z+DD)/F2**3)
             - 2*DD/math.sqrt(z**2+BB**2))

    Bx = Brho*math.cos(phi)
    By = Brho*math.sin(phi)

    Bvec = np.array([Bx, By, Bz])

    # 座標変換を戻す
    # S3RH で X3軸 の(右ネジ)まわりに9度回転
    Bvec = np.array([
        Bvec[0],
        Bvec[1]*math.cos(-THETA_D) - Bvec[2]*math.sin(-THETA_D),
        Bvec[1]*math.sin(-THETA_D) + Bvec[2]*math.cos(-THETA_D)
    ])

    # S3RH で Z3軸 の(右ネジ)まわりに65.8度回転
    Bvec = np.array([
        Bvec[0]*math.cos(-phiRH0) - Bvec[1]*math.sin(-phiRH0),
        Bvec[0]*math.sin(-phiRH0) + Bvec[1]*math.cos(-phiRH0),
        Bvec[2]
    ])"""

    # ダイポール座標系に持っていく
    # S3RH で Z3軸 の(右ネジ)まわりに-155.8度回転
    phiRH0 = math.radians(-155.8)    # Connerney+2020
    rvec0 = np.array([
        X*math.cos(phiRH0) - Y*math.sin(phiRH0),
        X*math.sin(phiRH0) + Y*math.cos(phiRH0),
        Z
    ])

    # S3RH で Y3軸 の(右ネジ)まわりに-9度回転
    THETA_D = math.radians(-9.3)
    rvec0 = np.array([
        rvec0[0]*math.cos(THETA_D) + rvec0[2]*math.sin(THETA_D),
        rvec0[1],
        -rvec0[0]*math.sin(THETA_D) + rvec0[2]*math.cos(THETA_D)
    ])

    rho = math.sqrt(rvec0[0]**2 + rvec0[1]**2)
    phi = math.atan2(rvec0[1], rvec0[0])        # 磁場の円筒座標系における方位角
    z = rvec0[2]

    F1 = math.sqrt((z-DD)**2 + AA**2)
    F2 = math.sqrt((z+DD)**2 + AA**2)

    Brho = CC*rho*(1/F1 - 1/F2)
    Bphi = 0
    Bz = CC*(2*DD/math.sqrt(z**2+AA**2)
             - 0.25*(rho**2)*((z-DD)/F1**3-(z+DD)/F2**3)
             - 2*DD/math.sqrt(z**2+BB**2))

    Bx = Brho*math.cos(phi)
    By = Brho*math.sin(phi)

    Bvec = np.array([Bx, By, Bz])

    # 座標変換を戻す
    Bvec = np.array([
        Bvec[0]*math.cos(-THETA_D) + Bvec[2]*math.sin(-THETA_D),
        Bvec[1],
        -Bvec[0]*math.sin(-THETA_D) + Bvec[2]*math.cos(-THETA_D)
    ])

    Bvec = np.array([
        Bvec[0]*math.cos(-phiRH0) - Bvec[1]*math.sin(-phiRH0),
        Bvec[0]*math.sin(-phiRH0) + Bvec[1]*math.cos(-phiRH0),
        Bvec[2]
    ])

    """# 位置ベクトルで確認
    rvec0 = np.array([
        rvec0[0]*math.cos(-THETA_D) + rvec0[2]*math.sin(-THETA_D),
        rvec0[1],
        -rvec0[0]*math.sin(-THETA_D) + rvec0[2]*math.cos(-THETA_D)
    ])

    rvec0 = np.array([
        rvec0[0]*math.cos(-phiRH0) - rvec0[1]*math.sin(-phiRH0),
        rvec0[0]*math.sin(-phiRH0) + rvec0[1]*math.cos(-phiRH0),
        rvec0[2]
    ])

    print(X-rvec0[0], Y-rvec0[1], Z-rvec0[2])"""

    return Bvec

# %%
