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

# Jupiter magnetic field model initialization
jm.Internal.Config(Model='jrm33', CartesianIn=True, CartesianOut=True)
jm.Con2020.Config(equation_type='analytic')


# %%
class Awave():
    def __init__(self) -> None:
        pass

    def tracefield(self,
                   r_orbit: float,
                   S3wlong0: float,
                   z_orbit=0):
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
                                        phi=phi)          # [deg]
        # print('Cent [deg]:', Cent0)

        # 伝搬時間
        tau = 0     # [sec]

        # 線要素
        ds = 30000     # [m]

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

        for _ in range(300000):
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

    def trace3(self,
               r_A0: float,
               S3wlon_A0: float,
               z_A0: float,
               S_A0: float,
               Ai: float,
               ni: float,
               Hp: float,
               NS: float,
               limit=0):
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

    def trace3_magnetodisk(self,
                           r_A0: float,
                           S3wlon_A0: float,
                           z_A0: float,
                           S_A0: float,
                           Ai: float,
                           ni: float,
                           Hp: float,
                           NS: float,
                           limit=0,
                           current_coef=1.0,):
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
        # 磁場モデルの設定
        mu_i_default = 139.6    # default: 139.6 [nT]
        jm.Con2020.Config(mu_i=mu_i_default*current_coef,
                          equation_type='analytic')

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

    def centri_eq(self, r_rj, theta, phi):
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

        MaxLen = 50000
        if r > 14:
            MaxLen = 100000
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
