""" Leadangle_wave.py

Created on Jul 3, 2023
@author: Shin Satoh

Description:


Version
1.0.0 (Jul 3, 2023)
2.0.0 (Apr 22, 2025)

"""

# %%
import numpy as np
import math
import copy

import JupiterMag as jm

# SPICE KERNELS
import spiceypy as spice
spice.furnsh('kernel/cassMetaK.txt')
radii = spice.bodvrd("JUPITER", "RADII", 3)[1]
a = radii[0]
c = radii[2]
f = (a - c) / a

# Jupiter magnetic field model initialization
jm.Internal.Config(Model='jrm33', CartesianIn=True,
                   CartesianOut=True, Degree=18)
jm.Con2020.Config(equation_type='analytic')

# 定数
MU0 = 1.26E-6            # 真空中の透磁率
AMU2KG = 1.66E-27        # [kg]
RJ = 71492E+3            # JUPITER RADIUS [m]
MJ = 1.90E+27            # JUPITER MASS [kg]
OMGJ = 1.75868E-4        # JUPITER SPIN ANGULAR VELOCITY [rad/s]
C = 2.99792E+8           # LIGHT SPEED [m/s]
G = 6.67E-11             # 万有引力定数  [m^3 kg^-1 s^-2]
phiRH0 = math.radians(-65.8)      # [rad]     Connerney+2020
TILT0 = math.radians(6.7)         # [rad]
Ai_H = 1.0               # 水素 [原子量]
Ai_O = 16.0              # 酸素 [原子量]
Ai_S = 32.0              # 硫黄 [原子量]


# %%
class Awave():
    def __init__(self) -> None:
        pass

    def update_Con2020(self,
                       current_coef=1.0,
                       thickness_coef=1.0,
                       i_rho=16.7,
                       equation_type='analytic'):

        # 磁場モデルの設定
        mu_i_default = 139.6    # default: 139.6 [nT]
        d_rj_default = 3.6      # default: 3.6 [RJ]
        jm.Con2020.Config(mu_i=mu_i_default*current_coef,
                          d=d_rj_default*thickness_coef,
                          i_rho=i_rho,
                          equation_type='analytic')

        return None

    def tracefield(self,
                   r_orbit: float,
                   S3wlong0: float,
                   z_orbit=0,
                   current_coef=1.0,
                   thickness_coef=1.0,):
        """
        `r_orbit` Europa公転距離 [m] \\
        `S3wlong0` EuropaのSystem III経度 [rad] \\
        `z_orbit` EuropaのSystem III Z座標 [m] \\
        Europaの位置から遠心力赤道まで磁力線をトレースし、沿磁力線の距離S0を計算する。
        """

        # 磁力線に沿ってトレースしたい
        rs = copy.copy(r_orbit)                # [m]
        theta = math.acos(z_orbit/r_orbit)     # [rad]
        phi = 2*np.pi-S3wlong0                 # [rad]
        x = rs*math.sin(theta)*math.cos(phi)    # [m]
        y = rs*math.sin(theta)*math.sin(phi)    # [m]
        z = copy.copy(z_orbit)                  # [m]

        # Centrifugal equator
        # a = 1.66                # [deg]
        # b = 0.131
        # c = 1.62
        # d = 7.76                # [deg]
        # e = math.radians(249)   # East longitude[deg]
        # Cent0 = (a*math.tanh(b*(rs/RJ)-c)+d)*math.sin(phi-e)  # [deg]
        # z_Cent0 = rs*math.cos(math.radians(90-Cent0))   # [m]

        Cent0, z_Cent0 = self.centri_eq(r_rj=rs/RJ,
                                        theta=theta,
                                        phi=phi)

        # 伝搬時間
        tau = 0     # [sec]

        # 線要素
        ds = 20000     # [m]

        # 衛星の沿磁力線座標を初期化
        S0 = 0.

        # 遠心力赤道までトレースして磁力線に沿った現在地座標を調べる
        # Launch siteが遠心力赤道の北側
        if z >= z_Cent0:
            lineNS = 1       # 元々: 南向きにトレースする
            # print(z/RJ, z_Cent0/RJ, 'Launch site is in North.')
        else:
            lineNS = -1     # 元々: 北向きにトレースする
            # print(z/RJ, z_Cent0/RJ, 'Launch site is in South.')

        for _ in range(500000):
            # Community codes
            Bx0, By0, Bz0 = jm.Internal.Field(x/RJ, y/RJ, z/RJ)  # [nT]
            Bx1, By1, Bz1 = jm.Con2020.Field(x/RJ, y/RJ, z/RJ)   # [nT]
            Bx = (Bx0+Bx1)*1E-9
            By = (By0+By1)*1E-9
            Bz = (Bz0+Bz1)*1E-9

            B0 = math.sqrt(Bx[0]**2+By[0]**2+Bz[0]**2)      # [T]

            # 座標更新 (x, y, z)
            x += ds*Bx[0]/B0*lineNS
            y += ds*By[0]/B0*lineNS
            z += ds*Bz[0]/B0*lineNS

            # 座標更新 (r, theta, phi)
            rs = math.sqrt(x**2 + y**2 + z**2)
            theta = math.acos(z/rs)
            phi = math.atan2(y, x)

            # 遠心力緯度
            Cent_lat = (90.0-math.degrees(theta))-Cent0

            # 座標更新 (沿磁力線: S0)
            S0 += ds*lineNS

            if abs(Cent_lat) < 0.005:
                # print('Find the centrifugal latitude.')
                # print('r =', math.sqrt(x**2 + y**2)/RJ, 'S0 =', S0/RJ)
                break

        return S0

    def tracefield2(self, r_orbit: float,
                    S3wlong0: float,
                    S0: float,
                    rho0: float,
                    Hp: float,
                    NS: str,
                    z_orbit=0, ):
        """
        `r_orbit` Europa公転距離 [m] \\
        `S3wlong0` EuropaのSystem III経度 [rad] \\
        `rho0` 遠心力赤道の質量密度 [amu cm-3] \\
        `Hp` プラズマシートのスケールハイト [m] \\
        `NS` トレースの向き `N` or `S` \\
        `z_orbit` 衛星のZ座標(IAU_JUPITER) [m] \\
        Europaの位置から遠心力赤道まで磁力線をトレースし、沿磁力線の距離S0を計算する。
        """
        Niter = int(600000)

        # 磁力線に沿ってトレースしたい
        rs = copy.copy(r_orbit)           # [m]
        theta = math.acos(z_orbit/rs)     # [rad]
        phi = 2*np.pi-S3wlong0            # [rad]
        x = rs*math.sin(theta)*math.cos(phi)    # [m]
        y = rs*math.sin(theta)*math.sin(phi)    # [m]
        z = copy.copy(z_orbit)                  # [m]

        # 伝搬時間
        tau = 0     # [sec]

        # 線要素
        ds = 75000     # [m]

        # Alfven速度を格納
        Va_arr = np.zeros(Niter)

        # Moon latitude (theta)を格納
        theta_arr = np.zeros(Niter)

        # Alfven wave front longitude (phi)を格納
        phi_arr = np.zeros(Niter)

        # Column mass density [kg m-2]
        mass_dens = 0

        # 電離圏の方角
        if NS == 'N':
            ns = -1     # 北向き
        else:
            ns = 1      # 南向き
        for i in range(Niter):
            # Community codes
            Bx0, By0, Bz0 = jm.Internal.Field(x/RJ, y/RJ, z/RJ)  # [nT]
            Bx1, By1, Bz1 = jm.Con2020.Field(x/RJ, y/RJ, z/RJ)   # [nT]
            Bx = (Bx0+Bx1)*1E-9     # [T]
            By = (By0+By1)*1E-9     # [T]
            Bz = (Bz0+Bz1)*1E-9     # [T]

            B0 = math.sqrt(Bx[0]**2+By[0]**2+Bz[0]**2)      # [T]

            # プラズマ質量密度 rho
            rho = rho0*AMU2KG*(1E+6)*np.exp(-(S0/Hp)**2)     # [kg m-3]

            # Alfven速度 Va
            Va = B0/math.sqrt(MU0*rho)    # [m/s]

            # 相対論効果の考慮
            if Va/C > 0.07:
                Va = Va/math.sqrt(1+(Va/C)**2)

            # 時間要素
            dt = ds/Va

            # 伝搬時間
            tau += dt   # [sec]

            # 座標更新 (x, y, z)
            x += (ds*Bx[0]/B0)*ns
            y += (ds*By[0]/B0)*ns
            z += (ds*Bz[0]/B0)*ns

            # 座標更新 (r, theta, phi)
            rs = math.sqrt(x**2 + y**2 + z**2)
            theta = math.acos(z/rs)
            phi = math.atan2(y, x)

            # 座標更新 (沿磁力線: S0)
            S0 += ds*(-ns)

            # 配列格納
            Va_arr[i] = Va          # [m/s]
            theta_arr[i] = theta    # [rad]
            mass_dens += rho*ds     # [kg m-2]

            # if Va/C > 0.2:
            #     print('       Too fast!', rs/RJ, rho/(AMU*1E+6))
            #     break

            # 電離圏の方角
            if (NS == 'N') and (Va/C > 0.3):    # VAで基準
                # if (NS == 'N') and (S0 > 2.1*Hp):
                # print('      N', i, rs/RJ, rho/(AMU*1E+6), S0/RJ, tau)
                # print('break')
                break
            if (NS == 'S') and (Va/C > 0.3):    # VAで基準
                # print('      S', i, rs/RJ, rho/(AMU*1E+6), S0/RJ, tau)
                # print('break')
                break

        # 値が格納されていない部分は削除
        Va_arr = Va_arr[:i]
        theta_arr = theta_arr[:i]

        return tau, theta_arr, Va_arr, mass_dens

    def trace3_reverse(self,
                       r_A0: float,
                       S3wlon_A0: float,
                       S3lat_A0: float,
                       S_A0: float,
                       Ai: float,
                       ni: float,
                       Hp: float,
                       NS: float):
        """
        `r_A0` Radial distance of the Alfven launch site [m] \\
        `S3wlon_A0` System III west longitude of the Alfven launch site [rad] \\
        `z_A0` z position of the Alfven launch site [m] \\
        `S_A0` Field line position of the Alfven launch site [m] \\
        `Ai` Ion mass of the plasma sheet [amu] \\
        `ni` Ion number density at the plasma sheet center [cm-3]
        `Hp` Scale height of the plasma sheet [m] \\
        `NS` Tracing direction North=-1 or South=1 \\
        """
        Niter = int(800000)

        # 磁力線に沿ってトレースしたい
        rs = copy.copy(r_A0)           # [m]
        theta = 0.5*np.pi-copy.copy(S3lat_A0)    # [rad]
        phi = 2*np.pi-copy.copy(S3wlon_A0)      # [rad]
        x = rs*math.sin(theta)*math.cos(phi)    # [m]
        y = rs*math.sin(theta)*math.sin(phi)    # [m]
        z = rs*math.cos(theta)                  # [m]

        # 伝搬時間
        tau = 0     # [sec]

        # 線要素
        ds = 2000     # [m]

        # Alfven速度を格納
        Va_arr = np.zeros(Niter)

        # Moon latitude (theta)を格納
        theta_arr = np.zeros(Niter)

        # Alfven wave front longitude (phi)を格納
        phi_arr = np.zeros(Niter)

        # Column mass density [kg m-2]
        col_massdens = 0

        # 沿磁力線座標
        s = copy.copy(S_A0)

        # Direction of tracing
        if NS == -1:    # North
            ns = -1     # 北向き
        elif NS == 1:   # South
            ns = 1      # 南向き
        for i in range(Niter):
            # Community codes
            Bx0, By0, Bz0 = jm.Internal.Field(x/RJ, y/RJ, z/RJ)  # [nT]
            Bx1, By1, Bz1 = jm.Con2020.Field(x/RJ, y/RJ, z/RJ)   # [nT]
            Bx = (Bx0+Bx1)*1E-9     # [T]
            By = (By0+By1)*1E-9     # [T]
            Bz = (Bz0+Bz1)*1E-9     # [T]

            B0 = math.sqrt(Bx[0]**2+By[0]**2+Bz[0]**2)      # [T]

            # プラズマ質量密度 rho
            # rho = Ai*AMU2KG*ni*(1E+6)*np.exp(-(s/Hp)**2)     # [kg m-3]

            # Alfven速度 Va
            # Va = B0/math.sqrt(MU0*rho)    # [m/s]

            # 相対論効果の考慮
            # if Va/C > 0.1:
            #     Va = Va/math.sqrt(1+(Va/C)**2)

            # 時間要素
            dt = 1

            # 伝搬時間
            tau += dt   # [sec]

            # 座標更新 (x, y, z)
            x += (ds*Bx[0]/B0)*ns
            y += (ds*By[0]/B0)*ns
            z += (ds*Bz[0]/B0)*ns

            # 座標更新 (r, theta, phi)
            rs = math.sqrt(x**2 + y**2 + z**2)
            theta = math.acos(z/rs)
            phi = math.atan2(y, x)

            # 座標更新 (沿磁力線: S0)
            s += ds*(-ns)

            # 配列格納
            # Va_arr[i] = Va          # [m/s]
            theta_arr[i] = theta    # [rad]
            # col_massdens += rho*ds     # [kg m-2]

            # 距離で基準
            if abs(z) < 0.001*RJ:
                print('RJ:', rs/RJ)
                break

        # 値が格納されていない部分は削除
        Va_arr = Va_arr[:i]
        theta_arr = theta_arr[:i]

        return tau, phi, Va_arr, theta_arr

    def trace3(self,
               r_A0: float,
               S3wlon_A0: float,
               z_A0: float,
               S_A0: float,
               Ai: float,
               ni: float,
               Hp: float,
               NS: float,
               limit=0,
               current_coef=1.0,
               thickness_coef=1.0,):
        """
        `r_A0` Radial distance of the Alfven launch site [m] \\
        `S3wlon_A0` System III west longitude of the Alfven launch site [rad] \\
        `z_A0` z position of the Alfven launch site [m] \\
        `S_A0` Field line position of the Alfven launch site [m] \\
        `Ai` Ion mass of the plasma sheet [amu] \\
        `ni` Ion number density at the plasma sheet center [cm-3]
        `Hp` Scale height of the plasma sheet [m] \\
        `NS` Tracing direction North=-1 or South=1 \\
        """
        Niter = int(600000)

        # 磁力線に沿ってトレースしたい
        rs = copy.copy(r_A0)           # [m]
        theta = math.acos(z_A0/rs)     # [rad]
        phi = 2*np.pi-copy.copy(S3wlon_A0)      # [rad]
        x = rs*math.sin(theta)*math.cos(phi)    # [m]
        y = rs*math.sin(theta)*math.sin(phi)    # [m]
        z = copy.copy(z_A0)                  # [m]

        # 伝搬時間
        tau = 0     # [sec]

        # 線要素
        ds = 100000     # [m]

        # Alfven速度を格納
        Va_arr = np.zeros(Niter)

        # Moon latitude (theta)を格納
        theta_arr = np.zeros(Niter)

        # Alfven wave front longitude (phi)を格納
        phi_arr = np.zeros(Niter)

        # Column mass density [kg m-2]
        col_massdens = np.zeros(2)

        # 沿磁力線座標
        s = copy.copy(S_A0)

        # Direction of tracing
        if NS == -1:       # Northern MAW
            ns = -1        # 北向き
        elif NS == 1:      # Southern MAW
            ns = 1         # 南向き
        elif NS == -101:   # Northern TEB
            ns = 1         # 南向き
        elif NS == 101:    # Southern TEB
            ns = -1        # 北向き
        for i in range(Niter):
            # Community codes
            Bx0, By0, Bz0 = jm.Internal.Field(x/RJ, y/RJ, z/RJ)  # [nT]
            Bx1, By1, Bz1 = jm.Con2020.Field(x/RJ, y/RJ, z/RJ)   # [nT]
            Bx = (Bx0+Bx1)*1E-9     # [T]
            By = (By0+By1)*1E-9     # [T]
            Bz = (Bz0+Bz1)*1E-9     # [T]

            B0 = math.sqrt(Bx[0]**2+By[0]**2+Bz[0]**2)      # [T]

            # プラズマ質量密度 rho
            rho = Ai*AMU2KG*ni*(1E+6)*np.exp(-(s/Hp)**2)     # [kg m-3]

            # Alfven速度 Va
            Va = B0/math.sqrt(MU0*rho)    # [m/s]

            # 相対論効果の考慮
            if Va/C > 0.1:
                Va = Va/math.sqrt(1+(Va/C)**2)

            # 時間要素
            dt = ds/Va

            # 伝搬時間
            tau += dt   # [sec]

            # 座標更新 (x, y, z)
            x += (ds*Bx[0]/B0)*ns
            y += (ds*By[0]/B0)*ns
            z += (ds*Bz[0]/B0)*ns

            # 座標更新 (r, theta, phi)
            rs = math.sqrt(x**2 + y**2 + z**2)
            theta = math.acos(z/rs)
            phi = math.atan2(y, x)

            # 座標更新 (沿磁力線: S0)
            s += ds*(-ns)

            # 配列格納
            Va_arr[i] = Va          # [m/s]
            theta_arr[i] = theta    # [rad]
            col_massdens[0] += rho*ds     # [kg m-2]
            col_massdens[1] += rho*ds/B0     # [kg Wb-1]

            # Vaで基準
            if limit == 0:
                if Va/C > 0.3:
                    break
            elif limit == 1:
                # 距離で基準
                if rs < 1.1*RJ:
                    break

        # 値が格納されていない部分は削除
        Va_arr = Va_arr[:i]
        theta_arr = theta_arr[:i]

        return tau, phi, Va_arr, col_massdens

    def trace3_reflect(self,
                       r_A0: float,
                       S3wlon_A0: float,
                       z_A0: float,
                       S_A0: float,
                       Ai: float,
                       ni: float,
                       Hp: float,
                       NS: float,):
        """
        Argument:
        - `r_A0` Radial distance of the Alfven launch site [m] 
        - `S3wlon_A0` SIII west longitude of the Alfven launch site [rad]
        - `z_A0` z position of the Alfven launch site [m] 
        - `S_A0` Field line position of the Alfven launch site [m]
        - `Ai` Ion mass of the plasma sheet [amu]
        - `ni` Ion number density at the plasma sheet center [cm-3]
        - `Hp` Scale height of the plasma sheet [m]
        - `NS` Tracing direction North=-1 or South=1

        Return:
        - `tau_arr` [s]
        - `rs` Radial distance from Jupiter [m]
        - `2*np.pi-phi_arr` SIII west longitude [rad]
        - `theta_arr` SIII colatitude [rad]
        - `s` Field line length
        - `altitude_arr` Altitude from the surface of Jupiter [km]
        """
        Niter = int(760000)

        # Initial reference table (the highest altitude)
        alt_flag = 0
        alt_ref = [1500.0, 1200.0, 1000.0, 900.0,
                   800.0, 700.0, 600.0, 500.0,
                   400.0, 300.0, 200.0, 100.0,
                   10.0, 5.0, 0.0]
        r_ref = []
        theta_ref = []
        phi_ref = []
        for i in range(len(alt_ref)):
            extradius = np.loadtxt(
                'data/Alt/Alt_'+str(int(alt_ref[i]))+'km/rthetaphi_2.txt')
            r_ref += [extradius[0, :]*RJ]   # [RJ]
            theta_ref += [extradius[1, :]]  # [rad]
            phi_ref += [extradius[2, :]]    # [rad]
        i = 0   # Safety

        # 高度のピンを立てる
        alt_pin_arr = np.zeros(Niter)

        # 磁力線に沿ってトレースしたい
        rs = copy.copy(r_A0)           # [m]
        theta = math.acos(z_A0/rs)     # [rad]
        phi = 2*np.pi-copy.copy(S3wlon_A0)      # [rad]
        x = rs*math.sin(theta)*math.cos(phi)    # [m]
        y = rs*math.sin(theta)*math.sin(phi)    # [m]
        z = copy.copy(z_A0)                  # [m]

        # 伝搬時間
        tau = 0     # [sec]
        tau_arr = np.zeros(Niter)   # Launchからの経過時間 [sec]

        # 線要素
        ds = 10000.0     # [m]

        # Alfven速度を格納
        Va_arr = np.zeros(Niter)

        # Magnetic latitudeとmagnetic east longitudeを格納
        theta_arr = np.zeros(Niter)
        phi_arr = np.zeros(Niter)

        # 沿磁力線座標
        s = copy.copy(S_A0)

        # 遠心力赤道 s=0 における質量密度
        rho_0 = Ai*AMU2KG*ni*(1E+6)

        # Direction of tracing
        if NS == -1:       # Northern MAW
            ns = -1        # 北向き
        elif NS == 1:      # Southern MAW
            ns = 1         # 南向き
        elif NS == -101:   # Northern TEB
            ns = 1         # 南向き
        elif NS == 101:    # Southern TEB
            ns = -1        # 北向き
        for i in range(Niter):
            # Community codes
            Bx0, By0, Bz0 = jm.Internal.Field(x/RJ, y/RJ, z/RJ)  # [nT]
            Bx1, By1, Bz1 = jm.Con2020.Field(x/RJ, y/RJ, z/RJ)   # [nT]
            Bx = (Bx0+Bx1)*1E-9     # [T]
            By = (By0+By1)*1E-9     # [T]
            Bz = (Bz0+Bz1)*1E-9     # [T]

            B0 = math.sqrt(Bx[0]**2+By[0]**2+Bz[0]**2)      # [T]

            # プラズマ質量密度 rho
            rho = rho_0*np.exp(-(s/Hp)**2)     # [kg m-3]

            # Alfven速度 Va / 大きくなりすぎないように調整
            if rho < rho_0/(math.e**8):
                rho = rho_0/(math.e**8)
            Va = B0/math.sqrt(MU0*rho)    # [m/s]

            # 相対論効果の考慮
            if Va/C > 0.1:
                Va = Va/math.sqrt(1.0+(Va/C)**2)

            # 光速の8割で速度はとめておく
            if Va/C > 0.85:
                Va = 0.85*C

            # 時間要素
            dt = ds/Va

            # 伝搬時間
            tau += dt   # [sec]

            # 座標更新 (x, y, z)
            x += (ds*Bx[0]/B0)*ns
            y += (ds*By[0]/B0)*ns
            z += (ds*Bz[0]/B0)*ns

            # 座標更新 (r, theta, phi)
            rs = math.sqrt(x**2 + y**2 + z**2)  # [m]
            theta = math.acos(z/rs)             # [rad]
            phi = math.atan2(y, x)              # [rad]
            theta_h = theta        # For savefile [rad]
            phi_h = phi            # For savefile [rad]

            # 座標更新 (沿磁力線: S0)
            s += ds*(-ns)

            # 高度h [km]の座標テーブルを参照
            if (i > 3500) and (rs < (1.0*RJ+2500.0E+3)):
                ds = 3000.0     # 線要素長を短く(5 kmより小さくする) [m]

                # ============================================
                # Altitude 1500 km -> 900 km -> 400 km -> 5 km
                # ============================================
                if alt_flag > len(theta_ref):
                    print('`alt_flag` is bigger than expected.')
                    print(alt_flag)

                # Jovigraphic (gr)
                lon_gr, lat_gr, alt_gr = spice.recpgr("JUPITER",
                                                      np.array([x/1000.0,
                                                                y/1000.0,
                                                                z/1000.0]),
                                                      a,
                                                      f)
                theta_h = 0.5*np.pi-lat_gr    # For savefile [rad]
                phi_h = 2*np.pi-lon_gr        # For savefile [rad]
                if abs(alt_gr-alt_ref[alt_flag])*1000.0 <= 0.5*ds:
                    alt_pin_arr[i] = alt_ref[alt_flag]
                    # print('Altitude [km]:', alt_ref[alt_flag], alt_gr)
                    if alt_flag == (len(theta_ref)-1):
                        print('End point [RJ]:', rs/RJ,
                              '// Altitude [km]:', alt_gr)
                        break
                    alt_flag += 1

            # 配列格納
            Va_arr[i] = Va            # [m/s]
            theta_arr[i] = theta_h    # SIII colatitude [rad]
            phi_arr[i] = phi_h        # SIII east longitude [rad]
            tau_arr[i] = tau          # [sec]

        # 値が格納されていない部分は削除
        Va_arr = Va_arr[:i]
        theta_arr = theta_arr[:i]
        phi_arr = phi_arr[:i]
        tau_arr = tau_arr[:i]
        alt_pin_arr = alt_pin_arr[:i]

        return tau_arr, rs, 2*np.pi-phi_arr, theta_arr, s, alt_pin_arr

    def centri_eq(self, r_rj, theta, phi, current_coef=1.0,
                  thickness_coef=1.0,):
        """_summary_

        Args:
            r_rj (_type_): _description_
            theta (_type_): _description_
            phi (_type_): _description_

        Returns:
            _type_: _description_
        """
        r = r_rj        # [RJ]

        x0 = r*math.sin(theta)*math.cos(phi)    # [RJ]
        y0 = r*math.sin(theta)*math.sin(phi)    # [RJ]
        z0 = r*math.cos(theta)                # [RJ]

        MaxLen = 70000
        if r > 14:
            MaxLen = 150000
        # Tracing grid: ds = 0.0002 RJ ~ 14 km
        T = jm.TraceField(x0, y0, z0, Verbose=True,
                          IntModel='jrm33',
                          ExtModel='Con2020',
                          MaxLen=MaxLen,
                          MaxStep=0.0002,
                          InitStep=0.00001,
                          MinStep=0.00001)

        # Distance from Jupiter's spin axis
        rho = np.sqrt(T.x**2 + T.y**2)

        # Index of the fartherst point
        idx_rhomax = np.where(rho == np.nanmax(rho))

        # Position of the farthest point
        x_max = T.x[idx_rhomax]
        y_max = T.y[idx_rhomax]
        z_max = T.z[idx_rhomax]
        r_max = math.sqrt(x_max**2+y_max**2+z_max**2)

        # Latitude of the centrifugal equator [deg]
        cent_lat = 90-math.degrees(math.acos(z_max/r_max))

        del T, rho

        return cent_lat, z_max*RJ

    def distance_from_h_km(self, x_in, y_in, z_in,
                           theta_in, phi_in, r_ref, theta_ref):
        # 高度h [km]上の球面でreferenceテーブルの緯度を参照
        dis = np.abs(theta_in-theta_ref)

        idx_0 = np.argmin(dis)  # 最も近い緯度グリッド
        idx_1 = idx_0 + 1       # 次に近い緯度グリッド
        if abs(theta_in-theta_ref[idx_0-1]) < abs(theta_in-theta_ref[idx_0+1]):
            idx_1 = idx_0 - 1

        # 高度h [km]上の球面で線形補間する
        X0 = theta_ref[idx_0]   # 最も近い緯度グリッド
        X1 = theta_ref[idx_1]   # 次に近い緯度グリッド
        Y0 = r_ref[idx_0]       # 最も近い動径距離グリッド
        Y1 = r_ref[idx_1]       # 次に近い動径距離グリッド
        r_h = (Y1-Y0)*(theta_in-X0)/(X1-X0) + Y0   # 線形補間した動径距離
        theta_h = theta_in      # 実際の緯度

        # 高度h [km]上のref点座標
        x_ref = r_h*np.sin(theta_h)*np.cos(phi_in)
        y_ref = r_h*np.sin(theta_h)*np.sin(phi_in)
        z_ref = r_h*np.cos(theta_h)

        # 高度h [km]上のref点座標とWavefrontの距離
        dis = math.sqrt((x_in-x_ref)**2 + (y_in-y_ref)**2 + (z_in-z_ref)**2)

        return r_h, theta_h, dis

    def use_SPICE(self):
        radii = spice.bodvrd("JUPITER", "RADII", 3)[1]
        a = radii[0]
        c = radii[2]
        f = (a - c) / a
        # print('a:', a, '// c:', c, '// f:', f)
        # print('Test altitude [km]:', alt_gr)

        return a, c, f
