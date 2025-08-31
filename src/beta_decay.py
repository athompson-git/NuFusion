import numpy as np

# Constants
from .constants import *


# Functions for the beta decay spectrum
def beta_decay_dGamma_dEnu(Enu, mi, mf, zi):
    """
    mi : initial nuclear mass
    mf : final nuclear mass
    zi : initial nucleus proton number
    
    returns the differential decay width with respect to the neutrino
    energy per target isotope.
    """

    Qbeta = mi - mf - M_E

    # electron kinetic energy from neutrino energy
    Ee = M_E + Qbeta - Enu

    sigma_reduced = (G_F**2/2/np.pi) * VUD**2 * fermi_function(Ee, zi) \
            * (mf/mi) * Ee * np.sqrt(Ee**2 - M_E**2) \
            * (FERMI_MESQ + (1/1.2695)**2 * GT_MESQ)

    # endpoint energy and phase space factor - ignoring m_neutrino
    Eend = ((mi - M_E)**2 - mf**2)/(2*mi)

    y = Eend - Ee
    h = (1 - M_E**2 / (mi * Ee)) / (1 - 2*Ee/mi + M_E**2 / mi**2)**2 * y**2

    # ignoring m_neutrino so unitarity sum of PMNS --> 1
    return sigma_reduced * h / (np.pi**2)




def fermi_function(Ee, z):
    """
    Returns the Fermi function for nuclei of proton number z
    as a function of electron energy Ee
    """
    eta = z * ALPHA * Ee / np.sqrt(Ee**2 - M_E**2)

    return (2*np.pi*eta) / (1 - np.exp(-2*np.pi*eta))













