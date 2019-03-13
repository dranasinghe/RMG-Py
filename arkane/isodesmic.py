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

import signal
from collections import deque
from cPickle import load

from lpsolve55 import lpsolve, EQ, LE
import numpy as np
import pyomo.environ as pyo

from rmgpy.molecule import Molecule
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
        low_level_h_rxn = sum(map(lambda spec: spec[0].low_level_hf298.value_si*spec[1], self.species.items())) - \
            self.target.low_level_hf298.value_si

        target_thermo = sum(map(lambda spec: spec[0].high_level_hf298.value_si*spec[1], self.species.items())) - \
            low_level_h_rxn
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


class ErrorCancelingScheme(object):
    """A Base class for calculating target species thermochemisty using error canceling reactions"""

    def __init__(self, target, benchmark_set):
        """

        :param target: RMG molecule object for which H_f(298 K) will be calculated
        :param benchmark_set: List of benchmark species (ErrorCancelingSpecies objects) with well known and verified
                              high level thermochemistry
        """

        self.target = target
        allowed_elements = target.molecule.get_element_count().keys()
        self.constraints = SpeciesConstraints(allowed_elements)

        # Prune out species with non-allowable atom types
        self.benchmark_set = []
        for species in benchmark_set:
            elements = species.molecule.get_element_count().keys()
            for label in elements:
                if label not in allowed_elements:
                    break
            else:
                self.benchmark_set.append(species)

        self.target_constraint = None
        self.constraint_matrix = None

    def initialize(self):
        """Setup the remaining scheme attributes before running"""
        # Pre-compute the constraints for all species
        self.target_constraint,  self.constraint_matrix = self.calculate_constraints()

    def calculate_constraints(self):
        """Enumerate the constraints for the target and benchmark species"""
        target_constraints = self.constraints.enumerate(self.target.molecule)
        c_matrix = np.zeros((len(self.benchmark_set), self.constraints.max_num_constraints), dtype=int)
        for i, species in enumerate(self.benchmark_set):
            constraint_vector = self.constraints.enumerate(species.molecule)
            c_matrix[i, :] = constraint_vector

        cutoff_index = len(self.constraints.constraint_map)  # All columns past this index are all zeros and undefined

        c_matrix = c_matrix[:, :cutoff_index]
        target_constraints = target_constraints[:cutoff_index]

        return target_constraints, c_matrix

    def find_error_canceling_reaction(self, benchmark_subset, milp_software='lpsolve'):
        """
        Automatically find a valid error canceling reaction given a subset of the available benchmark species. This
        is done by solving a mixed integer linear programming (MILP) problem similiar to Buerger et al. See the Arkane
        documentation for the description of this MILP problem.

        :param benchmark_subset: A list of indices from the full benchmarking set that can participate in the reaction
        :param milp_software: 'pyomo' or 'lpsolve'
        :return: ErrorCancelingReaction (if a valid reaction is found, else `None`)
        """
        # Define the constraints based on the provided subset
        c_matrix = np.take(self.constraint_matrix, benchmark_subset, axis=0)
        c_matrix = np.tile(c_matrix, (2, 1))
        sum_constraints = np.sum(c_matrix, 1, dtype=int)
        targets = -1*self.target_constraint
        m = c_matrix.shape[0]
        n = c_matrix.shape[1]
        split = int(m/2)

        if milp_software == 'pyomo':
            # Setup the MILP problem using pyomo
            lp_model = pyo.ConcreteModel()
            lp_model.i = pyo.RangeSet(0, m - 1)
            lp_model.j = pyo.RangeSet(0, n - 1)
            lp_model.r = pyo.RangeSet(0, split-1)  # indices before the split correspond to reactants
            lp_model.p = pyo.RangeSet(split, m - 1)  # indices after the split correspond to products
            lp_model.v = pyo.Var(lp_model.i, domain=pyo.NonNegativeIntegers)  # The stoich. coef. we are solving for
            lp_model.c = pyo.Param(lp_model.i, lp_model.j, initialize=lambda _, i, j: c_matrix[i, j])
            lp_model.s = pyo.Param(lp_model.i, initialize=lambda _, i: sum_constraints[i])
            lp_model.t = pyo.Param(lp_model.j, initialize=lambda _, j: targets[j])

            def obj_expression(model):
                return pyo.summation(model.v, model.s, index=model.i)

            lp_model.obj = pyo.Objective(rule=obj_expression)

            def constraint_rule(model, j):
                return sum(model.v[i] * model.c[i, j] for i in model.r) - \
                       sum(model.v[i] * model.c[i, j] for i in model.p) == model.t[j]

            lp_model.constraints = pyo.Constraint(lp_model.j, rule=constraint_rule)

            # Solve the MILP problem using the CBC MILP solver (https://www.coin-or.org/Cbc/)
            opt = pyo.SolverFactory('glpk')
            results = opt.solve(lp_model)

            # Return None if a valid reaction is not found
            if results.solver.status != pyo.SolverStatus.ok:
                return None

            # Extract the solution and find the species with non-zero stoichiometric coefficients
            solution = lp_model.v.extract_values().values()

        elif milp_software == 'lpsolve':
            # Save the current signal handler
            sig = signal.getsignal(signal.SIGINT)

            # Setup the MILP problem using lpsolve
            lp = lpsolve('make_lp', 0, m)
            lpsolve('set_verbose', lp, 2)  # Reduce the logging from lpsolve
            lpsolve('set_obj_fn', lp, sum_constraints)
            lpsolve('set_minim', lp)

            for j in range(n):
                lpsolve('add_constraint', lp, np.concatenate((c_matrix[:split, j], -1*c_matrix[split:, j])), EQ,
                        targets[j])

            lpsolve('add_constraint', lp, np.ones(m), LE, 20)  # Use at most 20 species (including replicates)
            lpsolve('set_timeout', lp, 1)  # Move on if lpsolve can't find a solution quickly

            # Constrain v_i to be 4 or less
            for i in range(m):
                lpsolve('set_upbo', lp, i, 4)

            # All v_i must be integers
            lpsolve('set_int', lp, [True]*m)

            status = lpsolve('solve', lp)

            if status != 0:
                return None, None

            else:
                _, solution = lpsolve('get_solution', lp)[:2]

        reaction = ErrorCancelingReaction(self.target)
        subset_indices = []
        for index, v in enumerate(solution):
            if v > 0:
                subset_indices.append(index % split)
                if index < split:
                    reaction.species.update({self.benchmark_set[benchmark_subset[index]]: -v})
                else:
                    reaction.species.update({self.benchmark_set[benchmark_subset[index % split]]: v})

        return reaction, np.array(subset_indices)

    def multiple_error_canceling_reaction_search(self, n_reactions_max=20, milp_software='lpsolve'):
        subset_queue = deque()
        subset_queue.append(np.arange(0, len(self.benchmark_set)))
        reaction_list = []

        while (len(subset_queue) != 0) and (len(reaction_list) < n_reactions_max):
            subset = subset_queue.popleft()
            if len(subset) == 0:
                continue
            reaction, subset_indices = self.find_error_canceling_reaction(subset, milp_software=milp_software)
            if reaction is None:
                continue
            else:
                reaction_list.append(reaction)

                for index in subset_indices:
                    subset_queue.append(np.delete(subset, index))

        return reaction_list

    def calculate_target_enthalpy(self, n_reactions_max=20, milp_software='lpsolve'):
        reaction_list = self.multiple_error_canceling_reaction_search(n_reactions_max, milp_software)
        h298_list = np.zeros(len(reaction_list))

        for i, rxn in enumerate(reaction_list):
            h298_list[i] = rxn.calculate_target_thermo().value_si

        return ScalarQuantity(np.median(h298_list), 'J/mol')


class IsodesmicScheme(ErrorCancelingScheme):
    """An error canceling reaction where the number and type of both atoms and bonds are conserved"""
    def __init__(self, target, benchmark_set=None):
        super(IsodesmicScheme, self).__init__(target, benchmark_set)
        self.constraints.conserve_bonds = True
        self.initialize()


if __name__ == '__main__':
    pass
