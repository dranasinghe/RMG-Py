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
This module provides the :class:`ErrorCancelingScheme` and related classes for the automatic generation of error
canceling reactions (e.g. isodesmic reactions). This code is heavily based on algorithms and ideas found in the existing
literature, including the following:

Buerger, P., Akroyd, J., Mosbach, S., & Kraft, M. (2018). A systematic method to estimate and validate enthalpies of
formation using error-cancelling balanced reactions. Combustion and Flame (Vol. 187).
https://doi.org/10.1016/j.combustflame.2017.08.013

Dobek, F. J., Ranasinghe, D. S., Throssell, K., & Petersson, G. A. (2013). Evaluation of the heats of formation of
corannulene and C60 by means of inexpensive theoretical procedures. Journal of Physical Chemistry A, 117(22), 4726–4730.
https://doi.org/10.1021/jp404158v
"""

from __future__ import division

import numpy as np

from rmgpy.quantity import ScalarQuantity


class ErrorCancelingSpecies:
    """Class for target and known (benchmark) species participating in an error canceling reaction"""

    def __init__(self, molecule, low_level_hf298, high_level_hf298=None):
        """
        :param molecule: RMG Molecule object
        :param low_level_hf298: Tuple of (H_f(298 K), unit) evaluated using a lower level of theory (e.g. DFT)
        :param high_level_hf298: Tuple of (H_f(298 K), unit) evaluated using a high level of theory (e.g. expt. data)
        """
        self.molecule = molecule
        self.low_level_hf298 = ScalarQuantity(*low_level_hf298)

        # If the species is a benchmark species, then the high level data is already known
        if high_level_hf298:
            self.high_level_hf298 = ScalarQuantity(*high_level_hf298)


class ErrorCancelingReaction:
    """Class for representing an error canceling reaction, with the target species being an implicit reactant"""

    def __init__(self, target, species=None):
        """
        :param target: Species for which the user wants to estimate the high level H_f(298 K) (ErrorCancelingSpecies)
        :param species: {benchmark species (ErrorCancelingSpecies object): stoichiometric coefficient}
        """

        self.target = target

        # Does not include the target, which is handled separately.
        self.species = species or {}

    def calculate_target_thermo(self):
        """
        Estimate the high level thermochemistry for the target species using the error canceling scheme
        :return:
        """
        low_level_h_rxn = sum(map(lambda spec, v: spec.low_level_hf298.value_si*v, self.species)) - \
            self.target.low_level_hf298.value_si

        target_thermo = sum(map(lambda spec, v: spec.high_level_hf298.value_si*v, self.species)) - low_level_h_rxn
        return ScalarQuantity(target_thermo, 'J/mol')


class ConstraintMap:
    """A dictionary object that adds new keys to the dictionary with an incremented index as its value"""
    def __init__(self):
        self.mapping = {}

    def __setitem__(self, key, value):
        return self.mapping.__setitem__(key, value)

    def __getitem__(self, item):
        try:
            return self.mapping.__getitem__(item)
        except KeyError:
            self.mapping[item] = len(self.mapping)
            return self.mapping.__getitem__(item)

    def __len__(self):
        return len(self.mapping)


class SpeciesConstraints:
    """A class for defining and enumerating constraints to BenchmarkSpecies objects for error canceling reactions"""
    def __init__(self, allowed_atom_types, conserve_bonds=True, conserve_ring_size=True, ):
        """
        Define the constraints that will be enforced, and determine the mapping of indices in the constraint vector to
        individual constraints

        :param conserve_bonds: Enforce that the number of each bond type be conserved (boolean)
        :param conserve_ring_size: Enforce that the number of each ring size be conserved (boolean)
        :param allowed_atom_types: A list containing the atom types that occur in the target molecule (list)
        """

        self.conserve_bonds = conserve_bonds
        self.conserve_ring_size = conserve_ring_size
        self.allowed_atom_types = allowed_atom_types
        self.constraint_map = ConstraintMap()
        self.max_num_constraints = 3*(len(allowed_atom_types)**2)+10  # bond type constraints grow as N^2 at most

    def enumerate(self, molecule):
        """
        Determine the constraint vector for a species given the enforced constraints
        :param molecule: RMG Molecule object
        :return: constraint vector (np array)
        """
        constraint_vector = np.zeros(self.max_num_constraints)

        atoms = molecule.get_element_count()
        for atom_label, count in atoms.iteritems():
            constraint_vector[self.constraint_map[atom_label]] += count

        if self.conserve_bonds:
            bonds = molecule.enumerate_bonds()
            for bond_label, count in bonds.iteritems():
                constraint_vector[self.constraint_map[bond_label]] += count

        if self.conserve_ring_size:
            rings = molecule.getSmallestSetOfSmallestRings()
            if len(rings) > 0:
                for ring in rings:
                    constraint_vector[self.constraint_map['{0}_ring'.format(len(ring))]] += 1

        return constraint_vector


if __name__ == '__main__':
    pass
