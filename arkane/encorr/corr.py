#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#                                                                             #
# RMG - Reaction Mechanism Generator                                          #
#                                                                             #
# Copyright (c) 2002-2019 Prof. William H. Green (whgreen@mit.edu),           #
# Prof. Richard H. West (r.west@neu.edu) and the RMG Team (rmg_dev@mit.edu)   #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the 'Software'),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
###############################################################################

"""
This module provides methods for applying energy and bond additivity
corrections.
"""

import rmgpy.constants as constants

import arkane.encorr.data as data
import arkane.encorr.pbac as pbac
import arkane.encorr.mbac as mbac


def get_energy_correction(model_chemistry, atoms, bonds, coords, nums, multiplicity=1,
                          atom_energies=None, apply_atom_corrections=True,
                          apply_bacs=False, bac_type='p'):
    """
    Calculate a correction to the electronic energy obtained from a
    quantum chemistry calculation at a given model chemistry such that
    it is consistent with the normal gas-phase reference states.
    Optionally, correct the energy using bond additivity corrections.
    """
    model_chemistry = model_chemistry.lower()

    corr = 0.0
    if apply_atom_corrections:
        corr += get_atom_correction(model_chemistry, atoms, atom_energies=atom_energies)
    if apply_bacs:
        corr += get_bac_correction(model_chemistry, bonds, coords, nums, bac_type=bac_type, multiplicity=multiplicity)

    return corr


def get_atom_correction(model_chemistry, atoms, atom_energies=None):
    """
    Calculate a correction to the electronic energy obtained from a
    quantum chemistry calculation at a given model chemistry such that
    it is consistent with the normal gas-phase reference states.

    `atoms` is a dictionary associating element symbols with the number
    of that element in the molecule. The atom energies are in Hartrees,
    which are from single atom calculations using corresponding model
    chemistries.

    The assumption for the multiplicity of each atom is:
    H doublet, C triplet, N quartet, O triplet, F doublet, Si triplet,
    P quartet, S triplet, Cl doublet, Br doublet, I doublet.
    """
    corr = 0.0

    # Step 1: Reference all energies to a model chemistry-independent
    # basis by subtracting out that model chemistry's atomic energies
    if atom_energies is None:
        try:
            atom_energies = data.atom_energies[model_chemistry]
        except KeyError:
            raise Exception('Missing atom energies for model chemistry {}'.format(model_chemistry))

    for symbol, count in atoms.items():
        if symbol in atom_energies:
            corr -= count * atom_energies[symbol] * 4.35974394e-18 * constants.Na
        else:
            raise Exception(
                'Unknown element "{}". Turn off atom corrections if only running a kinetics jobs '
                'or supply a dictionary of atom energies.'.format(symbol)
            )

    # Step 2: Atom energy corrections to reach gas-phase reference state
    atom_enthalpy_corrections = {symbol: data.atom_hf[symbol] - data.atom_thermal[symbol] for symbol in data.atom_hf}
    for symbol, count in atoms.items():
        if symbol in atom_enthalpy_corrections:
            corr += count * atom_enthalpy_corrections[symbol] * 4184.0
        else:
            raise Exception(
                'Element "{}" is not yet supported in Arkane.'
                ' To include it, add its experimental heat of formation'.format(symbol)
            )

    return corr


def get_bac_correction(model_chemistry, bonds, coords, nums, bac_type='p', multiplicity=1):
    """
    Calculate bond additivity correction.
    """
    if bac_type.lower() == 'p':  # Petersson-type BACs
        return pbac.get_bac_correction(model_chemistry, bonds)
    elif bac_type.lower() == 'm':  # Melius-type BACs
        # Return negative because the correction is subtracted in the Melius paper
        return -mbac.get_bac_correction(model_chemistry, coords, nums, multiplicity=multiplicity)
    else:
        raise Exception('BAC type {} is not available'.format(bac_type))
