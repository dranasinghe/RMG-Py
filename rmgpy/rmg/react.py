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
Contains functions for generating reactions.
"""
import itertools
import logging
import resource
import psutil

from rmgpy.data.rmg import getDB
from multiprocessing import Pool

def react(*spcTuples):
    """
    Generate reactions between the species in the 
    list of species tuples for all the reaction families available.

    For each tuple of one or more Species objects [(spc1,), (spc2, spc3), ...]
    the following is done:

    A list of tuples is created for each resonance isomer of the species.
    Each tuple consists of (Molecule, index) with the index the species index of the Species object.

    Possible combinations between the first spc in the tuple, and the second species in the tuple
    is obtained by taking the combinatorial product of the two generated [(Molecule, index)] lists.

    Returns a flat generator object containing the generated Reaction objects.
    """

    from rmgpy.rmg.main import maxproc
    
    # Available RAM memory (kB)
    mem = psutil.virtual_memory()
    usermem = mem.available
    
    tmp = divmod(usermem, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    tmp2 = min(maxproc, tmp[0])
    procnum = max(1, tmp2)
    logging.info('For reaction generation {0} processes are used.'.format(procnum))

    # Execute multiprocessing map. It blocks until the result is ready.
    # This method chops the iterable into a number of chunks which it 
    # submits to the process pool as separate tasks. 
    p = Pool(processes=procnum)
    
    reactions = p.map(
                reactSpecies,
                spcTuples)
    p.close()
    p.join()

    return itertools.chain.from_iterable(reactions)


def reactSpecies(speciesTuple):
    """
    Given a tuple of Species objects, generates all possible reactions
    from the loaded reaction families and combines degenerate reactions.

    The generated reactions are deflated.
    """
    speciesTuple = tuple([spc.copy(deep=True) for spc in speciesTuple])

    reactions = getDB('kinetics').generate_reactions_from_families(speciesTuple)

    return reactions


def reactAll(coreSpcList, numOldCoreSpecies, unimolecularReact, bimolecularReact, trimolecularReact=None):
    """
    Reacts the core species list via uni-, bi-, and trimolecular
    reactions.
    """

    # Select reactive species that can undergo unimolecular reactions:
    spcTuples = [(coreSpcList[i],)
     for i in xrange(numOldCoreSpecies) if (unimolecularReact[i] and coreSpcList[i].reactive)]

    for i in xrange(numOldCoreSpecies):
        for j in xrange(i, numOldCoreSpecies):
            # Find reactions involving the species that are bimolecular
            # This includes a species reacting with itself (if its own concentration is high enough)
            if bimolecularReact[i,j]:
                if coreSpcList[i].reactive and coreSpcList[j].reactive:
                    spcTuples.append((coreSpcList[i], coreSpcList[j]))

    if trimolecularReact is not None:
        for i in xrange(numOldCoreSpecies):
            for j in xrange(i, numOldCoreSpecies):
                for k in xrange(j, numOldCoreSpecies):
                    # Find reactions involving the species that are trimolecular
                    if trimolecularReact[i,j,k]:
                        if coreSpcList[i].reactive and coreSpcList[j].reactive and coreSpcList[k].reactive:
                            spcTuples.append((coreSpcList[i], coreSpcList[j], coreSpcList[k]))

    rxns = list(react(*spcTuples))
    return rxns
