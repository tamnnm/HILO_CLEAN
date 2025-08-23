


# --------------- * Calculate the water vapour mass * -------------- #
def water_vapour_mass_relative(T, RH, P):
    """
    Calculate the water vapour mass in the air.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    RH : float
        Relative humidity in percentage.
    P : float
        Atmospheric pressure in Pa.

    Returns
    -------
    float
        Water vapour mass in the air in kg/m^3.

    """
    # Constants
    Mw = 18.01528e-3  # kg/mol
    R = 8.314462618  # J/(mol.K)

    # Calculate the saturation vapour pressure
    Tc = T - 273.15
        # This use like a linear approximation of the Clausius-Clapeyron equation
    Ps = 611.21 * np.exp((18.678 - Tc / 234.5) * (Tc / (257.14 + Tc)))

    # Calculate the water vapour mass
        # Eq: RH = Pv / Ps * 100
    Pv = RH / 100 * Ps
    rho_v = Pv * Mw / (R * T)
    return rho_v

def water_vapour_mass_absolute(SH, P):
    """
    Calculate the water vapour mass in the air.

    Parameters
    ----------
    SH : float
        Specific humidity in percentage.
    P : float
        Atmospheric pressure in Pa.

    Returns
    -------
    float
        Water vapour mass in the air in kg/m^3.

    """
    
    