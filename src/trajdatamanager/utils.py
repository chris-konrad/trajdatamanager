# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 18:55:06 2025

@author: Christoph M. Konrad
"""

import numpy as np


def to_finite(data, test=None, axis=0, return_mask=False):
    """Remove non-finite entries (NaN, Inf) from a data array.

    Parameters
    ----------
    data : array-like
        The data array. Must be 1d or 2d. If 2d, all rows with one or more 
        non-finite entries will be removed.
    test : array-like, optional
        If provided, removes entries from data where test is non-finite.
        Default is None.
    axis : int, optional
        Specify the axis along which slices will be removed if non-finite 
        entries exist. The default is 0, which removes non-finite rows.
    return_mask : bool, optional
        Return the boolean mask used for filtering. Default is False.

    Returns
    -------
    data_finite : np.ndarray
        The data array without non-finite entries. 
    test_finite : np.ndarray
        The test array without non-finite entries. Only returned if test was 
        provided.
    mask : np.ndarray
        The boolean mask indicating non-finite entries. Only returned if 
        return_mask=True.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if test is not None:
        if not isinstance(data, np.ndarray):
            test = np.array(test)
        if not (test.shape == data.shape):
            raise ValueError((f"'data' and 'test' must have the same shape. Instead, the shapes were "
                              f"data.shape={data.shape} and test.shape={test.shape}"))
        return_test = True
    else:
        test = data
        return_test = False

    if data.ndim == 1:
        mask = np.isfinite(test)
        out = data[mask]
        if return_test:
            out_test = test[mask]
    elif data.ndim == 2:
        mask = np.all(np.isfinite(test), axis=0)
        if axis == 0:
            out = data[:,mask]
            if return_test:
                out_test = test[:,mask]
        elif axis == 1:
            out = data[mask,:]
            if return_test:
                out_test = test[mask,:]
        else:
            raise ValueError(f"'axis' must be 0 or 1, instead it was {axis}.")
    else:
        raise ValueError(f"'data' must be of dimension 1 or 2, not {data.ndim}")
    
    if return_test:
        if return_mask:
            return out, out_test, mask
        else:
            return out, out_test
    else:
        if return_mask:
            return out, mask
        else:
            return out
    
        

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
    with first following finite value.

    Parameters
    ----------
    data : array-like
        1D data array with nan/inf.

    Returns
    -------
    data_filled
        1D data array without nan/inf
    """
    data = np.array(data)
    if data.ndim > 1:
        raise NotImplementedError("nd forward fill not implemented! Supply 1d arrays.")

    idx_finite_last = np.where(np.isfinite(data), np.arange(len(data)), 0)
    idx_finite_last = np.maximum.accumulate(idx_finite_last)

    if np.any(idx_finite_last==0):
        if not np.isfinite(data[0]):
            idx_finite_last[idx_finite_last==0] = np.min(idx_finite_last[idx_finite_last>0])

    data_filled = data[idx_finite_last]

    return data_filled