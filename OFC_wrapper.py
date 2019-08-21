__doc__ = """
Python wrapper for eval_pol3_full.c

Note: You must compile the C shared library
       gcc -O3 -shared -o OFC_integrals.so OFC_integrals.c -lm -fopenmp -fPIC
"""

import os
import ctypes
from ctypes import c_double, c_int, POINTER, Structure


class c_complex(Structure):
    """
    Complex double ctypes
    """
    _fields_ = [('real', c_double), ('imag', c_double)]


class Parameters(Structure):
    """
    Parameters structure ctypes for comb-parameters and response
    """
    _fields_ = [
        ('freq_unit', c_double),
        ('N_freq', c_int),
        ('N_comb', c_int),
        ('N_res', c_int),
        ('frequency', POINTER(c_double)),
        ('field_freq1', POINTER(c_double)),
        ('field_freq2', POINTER(c_double)),
        ('gamma_comb', c_double),
        ('delta_freq', c_double),
        ('N_terms', c_int),
        ('index', POINTER(c_int)),
        ('modulations', POINTER(c_double))
    ]


class Molecule(Structure):
    """
    Parameters structure ctypes for molecular parameters
    """
    _fields_ = [
        ('nDIM', c_int),
        ('energies', POINTER(c_double)),
        ('gamma', POINTER(c_double)),
        ('mu', POINTER(c_complex)),
        ('polarization_mnv', POINTER(c_complex))
    ]


try:
    # Load the shared library assuming that it is in the same directory
    lib1 = ctypes.cdll.LoadLibrary(os.getcwd() + "/OFC_integrals.so")
except OSError:
    raise NotImplementedError(
        """
        The library is absent. You must compile the C shared library using the commands:
        gcc -O3 -shared -o OFC_integrals.so OFC_integrals.c -lm -lnlopt -fopenmp -fPIC
        """
    )

lib1.CalculateResponse.argtypes = (
    POINTER(Molecule),
    POINTER(Parameters),
)
lib1.CalculateResponse.restype = None


def CalculateResponse(molecule, params):
    return lib1.CalculateResponse(
        molecule,
        params
    )
