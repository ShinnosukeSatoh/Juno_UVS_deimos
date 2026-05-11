""" RAW_trace.py

Created on May 7, 2026
@author: Shin Satoh

Description:
This code calculates the location of the reflective Alfvén wave spot(s)
by tracing the Alfvén waves from a Galilean satellite, all the way along
the magnetic field line defined by the magnetic field models, including
reflection(s) near the planet. The reflection altitude is set at 900 km.

Version
1.0.0 (May 7, 2026)

"""
# %% Import
import spiceypy as spice
from multiprocessing import Pool
# from numba import jit
import numpy as np
import math

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
e = 1.60218E-19          # 素電荷 [J]
me = 9.10E-31            # 電子質量 [kg]

Psyn_io = (12.89)*3600      # Moon's synodic period [sec]
Psyn_eu = (11.22)*3600      # Moon's synodic period [sec]
Psyn_ga = (10.53)*3600      # Moon's synodic period [sec]

Fllen_io = 14.78*RJ     # Field line length [m]
Fllen_eu = 24.22*RJ     # Field line length [m]
Fllen_ga = 39.17*RJ     # Field line length [m]


# %% Calculate position of the target moon using Spiceypy
def spice_moonS3(et: float, MOON: str):
    """
    Args:
        et (float): Time
        MOON (str): Name of the target moon (IO/EUROPA/GANYMEDE)

    Returns:
        Tuple of \\
        `posx` (float) \\
        `posy` (float) \\
        `posz` (float) \\
        `posr` (float) \\
        `postheta` (float) \\
        `posphi` (float) \\
        `S3wlon` (float) \\
    """
    # Juno's position seen from Jupiter in IAU_JUPITER coordinate.
    _, lightTimes = spice.spkpos(
        targ='JUNO', et=et, ref='IAU_JUPITER', abcorr='LT+S', obs='JUPITER'
    )

    # Moon's position seen from Jupiter in IAU_JUPITER coordinate.
    pos, _ = spice.spkpos(
        targ=MOON, et=et, ref='IAU_JUPITER', abcorr='none', obs='JUPITER'
    )

    # S3 right-hand coordinate [m]
    posx, posy, posz = pos[0]*1E+3, pos[1]*1E+3, pos[2]*1E+3

    posr = np.sqrt(posx**2 + posy**2 + posz**2)
    postheta = np.arccos(posz/posr)
    posphi = np.arctan2(posy, posx)
    if posphi < 0:
        S3wlon = np.degrees(-posphi)
    else:
        S3wlon = np.degrees(2*np.pi - posphi)

    return posx, posy, posz, posr, postheta, posphi, S3wlon
