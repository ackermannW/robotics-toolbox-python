#!/usr/bin/env python
"""
@author: Mirza Hodžić
"""

from math import pi
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH, RevoluteMDH, PrismaticMDH

class Puma560akb(DHRobot):
    """
    Class that models a Puma 560 AKB manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``Puma560()`` is an object which models a Unimation Puma560 robot and
    describes its kinematic and dynamic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the x-direction
    - qn, arm is at a nominal non-singular configuration

    .. note::
        - SI units are used.
        - The model includes armature inertia and gear ratios.
        - The value of m1 is given as 0 here.  Armstrong found no value for it
          and it does not appear in the equation for tau1 after the
          substituion is made to inertia about link frame rather than COG
          frame.
        - Gravity load torque is the motor torque necessary to keep the joint
          static, and is thus -ve of the gravity caused torque.

    .. warning:: Compared to the MATLAB version of the Toolbox this model
        includes the pedestal, making the z-coordinates 26 inches larger.

    :references:
        - "A search for consensus among model parameters reported for the PUMA
          560 robot", P. Corke and B. Armstrong-Helouvry,
          Proc. IEEE Int. Conf. Robotics and Automation, (San Diego),
          pp. 1608-1613, May 1994. (for kinematic and dynamic parameters)
        - "A combined optimization method for solving the inverse kinematics
          problem", Wang & Chen, IEEE Trans. RA 7(4) 1991 pp 489-.
          (for joint angle limits)
        - https://github.com/4rtur1t0/ARTE/blob/master/robots/UNIMATE/puma560/parameters.m
        - "The Explicit Dynamic Model and Inertial Parameters of the Puma 560 Arm"
           Armstrong, Khatib and Burdick, 1986


    .. codeauthor:: Mirza Hodžić
    """  # noqa

    def __init__(self, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym

            zero = sym.zero()
            pi = sym.pi()
        else:
            from math import pi

            zero = 0.0

        deg = pi / 180
        inch = 0.0254

        base = 26.45 * inch
    
        L = [
            RevoluteDH(
                d=base,  # link length (Dennavit-Hartenberg notation)
                a=0,  # link offset (Dennavit-Hartenberg notation)
                alpha=0,  # link twist (Dennavit-Hartenberg notation)
                I=[0, 0, 0.35, 0, 0, 0],
                # inertia tensor of link with respect to
                # center of mass I = [L_xx, L_yy, L_zz,
                # L_xy, L_yz, L_xz]
                r=[0, 0, 0],
                # distance of ith origin to center of mass [x,y,z]
                # in link reference frame
                m=0,  # mass of link
                Jm=291e-6,  # actuator inertia
                G=-62.6111,  # gear ratio
                B=1.48e-3,  # actuator viscous friction coefficient (measured
                # at the motor)
                Tc=[0.395, -0.435],
                # actuator Coulomb friction coefficient for
                # direction [-,+] (measured at the motor)
                qlim=[-160 * deg, 160 * deg] # minimum and maximum joint angle
            ),
            RevoluteDH(
                d=0.2435,
                a=0,
                alpha=-pi / 2,
                I=[0.13, 0.524, 0.539, 0, 0, 0],
                r=[0.068, 0.006, -0.016],
                m=17.4,
                Jm=409e-6,
                G=107.815,
                B=0.817e-3,
                Tc=[0.126, -0.071],
                qlim=[-110 * deg, 110 * deg]  # qlim=[-45*deg, 225*deg]
            ),
            RevoluteDH(
                d=-0.0934,
                a=0.4318,
                alpha=zero,
                I=[0.066, 0.0125, 0.066, 0, 0, 0],
                r=[0, -0.070, 0.014 ],
                m=4.8,
                Jm=35e-6,
                G=-53.7063,
                B=1.38e-3,
                Tc=[0.132, -0.105],
                qlim=[-135 * deg, 135 * deg]  # qlim=[-225*deg, 45*deg]
            ),
            RevoluteDH(
                d=0.4331,
                a=-0.0203,
                alpha=pi / 2,
                I=[1.8e-3, 1.8e-3, 1.3e-3, 0, 0, 0],
                r=[0, 0, -0.019],
                m=0.82,
                Jm=35e-6,
                G=71.0364,
                B=71.2e-6,
                Tc=[11.2e-3, -16.9e-3],
                qlim=[-266 * deg, 266 * deg]  # qlim=[-110*deg, 170*deg]
            ),
            RevoluteDH(
                d=0,
                a=0,
                alpha=-pi / 2,
                I=[0.3e-3, 0.3e-3, 0.4e-3, 0, 0, 0],
                r=[0, 0, 0],
                m=0.34,
                Jm=35e-6,
                G=71.923,
                B=82.6e-6,
                Tc=[9.26e-3, -14.5e-3],
                qlim=[-100 * deg, 100 * deg]
            ),
            RevoluteDH(
                d=0,
                a=0,
                alpha=pi / 2,
                I=[0.15e-3, 0.15e-3, 0.04e-3, 0, 0, 0],
                r=[0, 0, 0.32],
                m=0.09,
                Jm=35e-6,
                G=76.686,
                B=36.7e-6,
                Tc=[3.96e-3, -10.5e-3],
                qlim=[-266 * deg, 266 * deg]
            )
        ]

        super().__init__(
            L,
            name="Puma 560-AKB",
            manufacturer="Unimation",
            keywords=("dynamics", "symbolic", "mesh"),
            symbolic=symbolic,
            meshdir="meshes/UNIMATE/puma560",
        )

        self.qr = np.array([0, pi / 2, -pi / 2, 0, 0, 0])
        self.qz = np.zeros(6)

        # nominal table top picking pose
        self.qn = np.array([0, pi / 4, pi, 0, pi / 4, 0])

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)
        self.addconfiguration("qn", self.qn)

        # straight and horizontal
        self.addconfiguration_attr("qs", np.array([0, 0, -pi / 2, 0, 0, 0]))


if __name__ == "__main__":  # pragma nocover

    puma_akb = Puma560akb(symbolic=False)
    print(puma_akb)
    print(puma_akb.dynamics())