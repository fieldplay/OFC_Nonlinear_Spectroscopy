#!/usr/bin/env python

"""
OFC_3rdOrderResponse.py:

Class containing C calls for 3rd order OFC response calculation.
Plots results obtained from C calls.
"""

__author__ = "Ayan Chattopadhyay"
__affiliation__ = "Princeton University"


# ---------------------------------------------------------------------------- #
#                           LOADING LIBRARY HEADERS                            #
# ---------------------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
from types import MethodType, FunctionType
from itertools import permutations, product, combinations_with_replacement
from collections import namedtuple
from ctypes import Structure, c_double, c_int, POINTER, Array
from OFC_wrapper import Molecule, Parameters, c_complex, CalculateResponse

############################################################################################
#                                                                                          #
#      Declare new types: ADict to access dictionary elements with a. rather than a[]      #
#                                                                                          #
############################################################################################


class ADict(dict):
    """
    Dictionary where you can access keys as attributes
    """
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            dict.__getattribute__(self, item)


class OFC_Response:
    """
    Main class initializing molecule and spectra calculation, Raman control
    optimization routines on the molecule.
    """

    def __init__(self, params, **kwargs):
        """
        __init__ function call to initialize variables from the
        parameters for the class instance provided in __main__ and
        add new variables for use in other functions in this class.
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            else:
                setattr(self, name, value)

        self.nonuniform_frequency_range_3(params)

        self.polarization = np.zeros_like(self.frequency, dtype=np.complex)
        self.polarization_mnv = np.zeros_like(self.polarization, dtype=np.complex)

    def nonuniform_frequency_range_3(self, params):
        """
        Generation of nonuniform frequency range taylored to the 3d order optical effects
        :param params:
        :return:
        """

        freq_points_pol3 = np.linspace(-params.N_comb * params.delta_freq, params.N_comb * params.delta_freq, params.N_comb + 1)[:, np.newaxis]
        freq_points_comb = np.linspace(-params.N_comb * params.delta_freq, params.N_comb * params.delta_freq, 2*params.N_comb + 1)[:, np.newaxis]
        resolution = np.linspace(-0.5 * params.delta_freq, 0.5 * params.delta_freq, params.N_res)

        frequency_12 = 2 * params.omega_M2 - params.omega_M1 + freq_points_pol3 + resolution
        frequency_21 = 2 * params.omega_M1 - params.omega_M2 + freq_points_pol3 + resolution

        field_freq1 = params.omega_M1 + freq_points_comb + resolution
        field_freq2 = params.omega_M2 + freq_points_comb + resolution

        self.frequency = np.sort(np.hstack([frequency_12.flatten(), frequency_21.flatten()]))
        self.freq_field1 = np.ascontiguousarray(field_freq1.flatten())
        self.freq_field2 = np.ascontiguousarray(field_freq2.flatten())

        print(self.freq_field1.size)

    def create_molecule(self, FourLevel):
        """
        Creates molecule instance
        :param FourLevel: molecule instance
        :param params:
        :return:
        """

        FourLevel.nDIM = self.energies.size
        FourLevel.energies = self.energies.ctypes.data_as(POINTER(c_double))
        FourLevel.gamma = self.gamma.ctypes.data_as(POINTER(c_double))
        FourLevel.mu = self.mu.ctypes.data_as(POINTER(c_complex))
        FourLevel.polarization_mnv = self.polarization_mnv.ctypes.data_as(POINTER(c_complex))

    def create_parameters(self, SysParams, params):
        """
        :param Parameters:
        :param params:
        :return:
        """
        SysParams.freq_unit = params.freq_unit
        SysParams.N_freq = len(self.frequency)
        SysParams.N_comb = params.N_comb
        SysParams.N_res = params.N_res
        SysParams.frequency = self.frequency.ctypes.data_as(POINTER(c_double))
        SysParams.field_freq1 = self.freq_field1.ctypes.data_as(POINTER(c_double))
        SysParams.field_freq2 = self.freq_field2.ctypes.data_as(POINTER(c_double))
        SysParams.gamma_comb = params.gamma_comb
        SysParams.delta_freq = params.delta_freq
        SysParams.N_terms = params.N_terms
        SysParams.index = np.ones(3).ctypes.data_as(POINTER(c_int))
        SysParams.modulations = np.zeros(3).ctypes.data_as(POINTER(c_double))

    def calculate_response(self, params):
        FourLevel = Molecule()
        self.create_molecule(FourLevel)
        SysParams = Parameters()
        self.create_parameters(SysParams, params)
        for i, modulations in enumerate(list(product(*(3 * [[params.omega_M1, params.omega_M2]])))):
            for m, n, v in permutations(range(1, len(self.energies)), 3):
                print(modulations, m, n, v)
                SysParams.index[0] = m
                SysParams.index[1] = n
                SysParams.index[2] = v
                SysParams.modulations = np.asarray(modulations).ctypes.data_as(POINTER(c_double))
                mu_product = self.mu[0, m] * self.mu[m, n] * self.mu[n, v] * self.mu[v, 0]
                self.polarization_mnv[:] = 0.
                CalculateResponse(FourLevel, SysParams)
                self.polarization_mnv *= mu_product
                self.polarization += self.polarization_mnv


if __name__ == '__main__':

    import time
    start = time.time()

    freq_unit = (0.024188 / 1000)
    np.set_printoptions(precision=4)

    gamma_decay = np.ones((4, 4)) * 1e-3 * freq_unit        # All population relaxation times equal 1 GHz (1 ns -1)
    np.fill_diagonal(gamma_decay, 0.0)                      # Diagonal elements zero; no decay to self
    gamma_decay = np.tril(gamma_decay)                      # Relaxation only to lower energy states

    gamma_electronic = 1e1 * freq_unit                      # All electronic dephasing rates are 10 THz (100 fs -1)
    gamma = np.ones((4, 4)) * gamma_electronic
    np.fill_diagonal(gamma, 0.0)

    gamma_vibrational = 0.59 * freq_unit                     # All vibrational dephasing rates are 0.59 THz (1.7 ps -1)
    gamma[0, 1] = gamma_vibrational
    gamma[1, 0] = gamma_vibrational
    gamma[2, 3] = gamma_vibrational
    gamma[3, 2] = gamma_vibrational

    # Net damping rates given by Boyd pg. 156, G_nm = (1/2) * \sum_i (g_decay_ni + g_decay_mi) + g_dephasing_nm

    for n in range(4):
        for m in range(4):
            for i in range(4):
                gamma[n, m] += 0.5 * (gamma_decay[n, i] + gamma_decay[m, i])

    energies = np.cumsum([0, 48, 500, 45]) * freq_unit

    mu_value = 2.
    mu = mu_value * np.ones_like(gamma, dtype=np.complex)
    np.fill_diagonal(mu, 0j)

    params = ADict(
        freq_unit=freq_unit,
        N_comb=100,
        N_res=51,
        omega_M1=6e-1 * freq_unit,
        omega_M2=2e-1 * freq_unit,
        gamma_comb=1e-12 * freq_unit,
        delta_freq=1e-0 * freq_unit,
        N_terms=5,
    )

    System = dict(
        energies=energies,
        gamma=gamma,
        mu=mu,
    )

    response = OFC_Response(params, **System)
    response.calculate_response(params)

    omega1 = response.freq_field1[:, np.newaxis]
    omega2 = response.freq_field2[:, np.newaxis]
    comb_omega = (params.delta_freq * np.arange(-params.N_comb, params.N_comb))[np.newaxis, :]
    field1 = (params.gamma_comb / ((omega1 - params.omega_M1 - comb_omega) ** 2 + params.gamma_comb ** 2)).sum(axis=1)
    field2 = (params.gamma_comb / ((omega2 - params.omega_M2 - comb_omega) ** 2 + params.gamma_comb ** 2)).sum(axis=1)

    print(field1.size)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    axes[0].plot(response.frequency / params.delta_freq, -response.polarization.real, 'r', linewidth=1.)
    axes[0].plot(response.freq_field1 / params.delta_freq, field1*response.polarization.real.max()/field1.max(), 'y', alpha=0.4)
    axes[0].plot(response.freq_field2 / params.delta_freq, field2*response.polarization.real.max()/field1.max(), 'g', alpha=0.4)
    axes[1].plot(response.frequency / params.delta_freq, -response.polarization.imag, 'r', linewidth=1., alpha=0.7)

    print(time.time() - start)
    plt.show()