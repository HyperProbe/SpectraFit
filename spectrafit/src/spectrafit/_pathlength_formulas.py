"""Pathlength formulas dPL = dA/dmu_a, generated from the sympy expressions in the
SpectraFit research code (data/pathlengths/carp.pickle and jacques.pickle).
Do not edit by hand.

delta-P1 (CARP) approximation: dA/dmu_a(mu_a, mu_s_red, g, k)
Jacques model:                  dA/dmu_a(mu_a, mu_s_red, m1, m2, m3)
mu_a and mu_s_red in cm^-1; result in cm.
"""
from numpy import sqrt, exp, log  # noqa: F401


def carp_dA_dmua(mu_a, mu_s_red, g, k):
    return -(3*mu_s_red*(1 - g**2)*(g/(g + 1) + 1)/((1 - g)*(3*mu_a*(mu_a + mu_s_red) - (mu_a + mu_s_red*(1 - g**2)/(1 - g))**2)) + 3*mu_s_red*(1 - g**2)*(-4*mu_a - 3*mu_s_red + 2*mu_s_red*(1 - g**2)/(1 - g))*(g*mu_a/(g + 1) + mu_a + mu_s_red*(1 - g**2)/(1 - g))/((1 - g)*(3*mu_a*(mu_a + mu_s_red) - (mu_a + mu_s_red*(1 - g**2)/(1 - g))**2)**2) + (18*g*k*mu_s_red*(1 - g**2)/((1 - g)*(g + 1)*(3*mu_a + 3*mu_s_red)**2) - 3*mu_s_red*(1 - g**2)*(g/(g + 1) + 1)*(2*k*(mu_a + mu_s_red*(1 - g**2)/(1 - g))/(3*mu_a + 3*mu_s_red) + 1)/((1 - g)*(3*mu_a*(mu_a + mu_s_red) - (mu_a + mu_s_red*(1 - g**2)/(1 - g))**2)) - 3*mu_s_red*(1 - g**2)*(-6*k*(mu_a + mu_s_red*(1 - g**2)/(1 - g))/(3*mu_a + 3*mu_s_red)**2 + 2*k/(3*mu_a + 3*mu_s_red))*(g*mu_a/(g + 1) + mu_a + mu_s_red*(1 - g**2)/(1 - g))/((1 - g)*(3*mu_a*(mu_a + mu_s_red) - (mu_a + mu_s_red*(1 - g**2)/(1 - g))**2)) - 3*mu_s_red*(1 - g**2)*(2*k*(mu_a + mu_s_red*(1 - g**2)/(1 - g))/(3*mu_a + 3*mu_s_red) + 1)*(-4*mu_a - 3*mu_s_red + 2*mu_s_red*(1 - g**2)/(1 - g))*(g*mu_a/(g + 1) + mu_a + mu_s_red*(1 - g**2)/(1 - g))/((1 - g)*(3*mu_a*(mu_a + mu_s_red) - (mu_a + mu_s_red*(1 - g**2)/(1 - g))**2)**2))/(2*sqrt(3)*k*sqrt(mu_a)*sqrt(mu_a + mu_s_red)/(3*mu_a + 3*mu_s_red) + 1) + (-6*g*k*mu_s_red*(1 - g**2)/((1 - g)*(g + 1)*(3*mu_a + 3*mu_s_red)) - 3*mu_s_red*(1 - g**2)*(2*k*(mu_a + mu_s_red*(1 - g**2)/(1 - g))/(3*mu_a + 3*mu_s_red) + 1)*(g*mu_a/(g + 1) + mu_a + mu_s_red*(1 - g**2)/(1 - g))/((1 - g)*(3*mu_a*(mu_a + mu_s_red) - (mu_a + mu_s_red*(1 - g**2)/(1 - g))**2)))*(6*sqrt(3)*k*sqrt(mu_a)*sqrt(mu_a + mu_s_red)/(3*mu_a + 3*mu_s_red)**2 - sqrt(3)*k*sqrt(mu_a)/(sqrt(mu_a + mu_s_red)*(3*mu_a + 3*mu_s_red)) - sqrt(3)*k*sqrt(mu_a + mu_s_red)/(sqrt(mu_a)*(3*mu_a + 3*mu_s_red)))/(2*sqrt(3)*k*sqrt(mu_a)*sqrt(mu_a + mu_s_red)/(3*mu_a + 3*mu_s_red) + 1)**2)/(3*mu_s_red*(1 - g**2)*(g*mu_a/(g + 1) + mu_a + mu_s_red*(1 - g**2)/(1 - g))/((1 - g)*(3*mu_a*(mu_a + mu_s_red) - (mu_a + mu_s_red*(1 - g**2)/(1 - g))**2)) + (-6*g*k*mu_s_red*(1 - g**2)/((1 - g)*(g + 1)*(3*mu_a + 3*mu_s_red)) - 3*mu_s_red*(1 - g**2)*(2*k*(mu_a + mu_s_red*(1 - g**2)/(1 - g))/(3*mu_a + 3*mu_s_red) + 1)*(g*mu_a/(g + 1) + mu_a + mu_s_red*(1 - g**2)/(1 - g))/((1 - g)*(3*mu_a*(mu_a + mu_s_red) - (mu_a + mu_s_red*(1 - g**2)/(1 - g))**2)))/(2*sqrt(3)*k*sqrt(mu_a)*sqrt(mu_a + mu_s_red)/(3*mu_a + 3*mu_s_red) + 1))


def jacques_dA_dmua(mu_a, mu_s_red, m1, m2, m3):
    return -1/3*sqrt(3)*m2*exp(log(mu_s_red/mu_a)/m3)/(m3*sqrt(mu_a)*sqrt(mu_a + mu_s_red)) - 1/6*sqrt(3)*sqrt(mu_a)*(m1 + m2*exp(log(mu_s_red/mu_a)/m3))/(mu_a + mu_s_red)**(3/2) + (1/6)*sqrt(3)*(m1 + m2*exp(log(mu_s_red/mu_a)/m3))/(sqrt(mu_a)*sqrt(mu_a + mu_s_red))


JACQUES_M_PARAMS = {
    'gray matter': (7.362351749434352, 0.07880996117891152, 1.8203649087961236),
    'blood vessel': (26327.239009030913, -26318.318943181792, 28718.23874621934),
    'tumor': (6.696295360508339, 0.7399469422388861, 4.4735924788862285),
    'general': (8.17504228539966, 0.010104473628491458, 1.4417049868829541),
}
