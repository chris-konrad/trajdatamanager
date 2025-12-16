# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 18:55:06 2025

@author: Christoph M. Konrad
"""

import numpy as np

def limitAngle(theta):
    """Convert angle from [0,2*pi] to [-pi,pi]
    
    Function copied and modified from cyclistsocialforce.utils by Christoph 
    Konrad (MIT License). 
    """
    if isinstance(theta, np.ndarray):
        theta = np.floor(theta / (2 * np.pi)) * (-2 * np.pi) + theta

        theta[theta > np.pi] = (theta - 2 * np.pi)[theta > np.pi]
        theta[theta < -np.pi] = (theta + 2 * np.pi)[theta < -np.pi]
    else:
        theta = np.floor(theta / (2 * np.pi)) * (-2 * np.pi) + theta

        if theta > np.pi:
            theta = theta - 2 * np.pi
        elif theta < -np.pi:
            theta = theta + 2 * np.pi

    return theta


def cart2polar(x, y):
    """
    Transfrom cartesian coordinates into polar coordinates with the angle psi
    in the range [-pi, pi]
    
    Function copied and modified from cyclistsocialforce.utils by Christoph 
    Konrad (MIT License). 

    Parameters
    ----------
    x : array-like
    y : array-like

    Returns
    -------
    rho : array-like
    psi : array-like.

    """
    rho = np.sqrt(np.power(x, 2) + np.power(y, 2))

    psi = np.arccos(x / rho)
    if type(psi) is not np.ndarray:
        psi = np.array(psi)

    psi[y < 0] = -psi[y < 0]

    return rho, psi


def forward_fill_finite(data):
    """Fill non-finite (np.nan / np.inf) values with the 
    last finite value before that index.

    If the first value is not finite, it will be replaced
    with 0.

    Parameters
    ----------
    data : array-like
        1D data array with nan/inf.

    Returns
    -------
    _type_
        _description_
    """
    data = np.array(data)
    if data.ndim > 1:
        raise NotImplementedError("nd forward fill not implemented! Supply 1d arrays.")

    data = np.where(np.isfinite(data), d, 0)
    data = np.maximum.accumulate(data)

    return data