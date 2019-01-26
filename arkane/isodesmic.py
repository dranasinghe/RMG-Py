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


if __name__ == '__main__':
    pass
