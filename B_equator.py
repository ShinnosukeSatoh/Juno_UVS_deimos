""" B_equator.py

Created on Apr 12, 2023
@author: Shin Satoh

Description:
This class is written specifically for calculating and locating
the magnetic equator of the Jovian magnetosphere.


Version
1.0.0 (Apr 12, 2023)

"""


# %% LIBRARIES
import numpy as np
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


class SUM():
    def __init__(self):
        return None

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
        p_arr, dp_arr = pysh.legendre.PlmSchmidt_d1(NN, np.cos(theta))
        dp_arr *= -np.sin(theta)        # NECESSARY MULTIPLICATION

        p_arr = p_arr[1:]               # n <= 1
        dp_arr = dp_arr[1:]             # n <= 1

        # 磁場の計算
        # r成分
        dVdr = 0
        dVdr_n = np.zeros(NN)

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
            dVdr_n[i] = np.sum(P_nm*(g_nm*np.cos(m*phi) + h_nm*np.sin(m*phi)))

        # INDEX n方向に和をとる
        dVdr = np.sum(dVdr_n)

        # print(dVdr*1E-5)
        # print(dVdtheta*1E-5)
        # print(dVdphi*1E-5)
        # print(np.sqrt(dVdr**2 + dVdtheta**2 + dVdphi**2)*1E-5)

        return dVdr
