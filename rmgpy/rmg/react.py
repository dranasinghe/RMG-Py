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


def reactSpecies(speciesTupleTmp):
    """
    Given a tuple of Species objects, generates all possible reactions
    from the loaded reaction families and combines degenerate reactions.

    speciesTuple = speciesTupleTmp[0:-1]
    own_families = speciesTupleTmp[-1]

    speciesTuple = tuple([spc.copy(deep=True) for spc in speciesTuple])

    reactions = getDB('kinetics').generate_reactions_from_families(speciesTuple, only_families=own_families)

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

    # Load kineticsFamilies to be added to reactant tuple to allow for improved load balancing 
    # in parallel jobs
    splitListOrig = []
    splitListTmp = []
    for key, value in getDB('kinetics').families.iteritems() :
        splitListOrig.append(key)
        splitListTmp.append(key)
 
    # Identify and split families that are prone to generate many reactions into sublists
    splitList = []
    for (i, v) in enumerate (splitListTmp):
        if splitListTmp[i] == 'H_Abstraction':
            splitListTmp[i] = []
            splitList.append(['H_Abstraction'])
        elif splitListTmp[i] == 'R_Recombination':
            splitListTmp[i] = []
            splitList.append(['R_Recombination'])
        elif splitListTmp[i] == 'Intra_Disproportionation':
            splitListTmp[i] = []
            splitList.append(['Intra_Disproportionation'])
        elif splitListTmp[i] == 'Intra_RH_Add_Endocyclic':
            splitListTmp[i] = []
            splitList.append(['Intra_RH_Add_Endocyclic'])
        elif splitListTmp[i] == 'Singlet_Carbene_Intra_Disproportionation':
            splitListTmp[i] = []
            splitList.append(['Singlet_Carbene_Intra_Disproportionation'])
        elif splitListTmp[i] == 'Intra_ene_reaction':
            splitListTmp[i] = []
            splitList.append(['Intra_ene_reaction'])
        elif splitListTmp[i] == 'Disproportionation':
            splitListTmp[i] = []
            splitList.append(['Disproportionation'])
        elif splitListTmp[i] == '1,4_Linear_birad_scission':
            splitListTmp[i] = []
            splitList.append(['1,4_Linear_birad_scission'])
        elif splitListTmp[i] == 'R_Addition_MultipleBond':
            splitListTmp[i] = []
            splitList.append(['R_Addition_MultipleBond'])
        elif splitListTmp[i] == '2+2_cycloaddition_Cd':
            splitListTmp[i] = []
            splitList.append(['2+2_cycloaddition_Cd'])
        elif splitListTmp[i] == 'Diels_alder_addition':
            splitListTmp[i] = []
            splitList.append(['Diels_alder_addition'])
        elif splitListTmp[i] == 'Intra_RH_Add_Exocyclic':
            splitListTmp[i] = []
            splitList.append(['Intra_RH_Add_Exocyclic'])
        elif splitListTmp[i] == 'Intra_Retro_Diels_alder_bicyclic':
            splitListTmp[i] = []
            splitList.append(['Intra_Retro_Diels_alder_bicyclic'])
        elif splitListTmp[i] == 'Intra_2+2_cycloaddition_Cd':
            splitListTmp[i] = []
            splitList.append(['Intra_2+2_cycloaddition_Cd'])
        elif splitListTmp[i] == 'Birad_recombination':
            splitListTmp[i] = []
            splitList.append(['Birad_recombination'])
        elif splitListTmp[i] == 'Intra_Diels_alder_monocyclic':
            splitListTmp[i] = []
            splitList.append(['Intra_Diels_alder_monocyclic'])
        elif splitListTmp[i] == '1,4_Cyclic_birad_scission':
            splitListTmp[i] = []
            splitList.append(['1,4_Cyclic_birad_scission'])
        elif splitListTmp[i] == '1,2_Insertion_carbene':
            splitListTmp[i] = []
            splitList.append(['1,2_Insertion_carbene'])

    splitList.append(splitListTmp)

    # Only employ family splitting for reactants that have a larger number of atoms than nAFS
    nAFS = 10
    #print ("splitList {0}").format(splitList)
    #print ("Length spcTuples {0}").format(len(spcTuples))
    spcTuplestmp = []
    # Append reaction families to reactant tuple
    for tmpj in spcTuples:
        if len(tmpj) == 1:
            if len(str(tmpj[0])) > nAFS:
                for tmpl in splitList: 
                    tmpk = list(tmpj)
                    tmpk.append(tmpl)
                    spcTuplestmp.append(tuple(tmpk))
            else:
                tmpk = list(tmpj)
                tmpk.append(splitListOrig)
                spcTuplestmp.append(tuple(tmpk))
        elif len(tmpj) == 2:
            if (len(str(tmpj[0])) > nAFS
               ) or (len(str(tmpj[1])) > nAFS):
                for tmpl in splitList:
                    tmpk = list(tmpj)
                    tmpk.append(tmpl)
                    spcTuplestmp.append(tuple(tmpk))
            else:
                tmpk = list(tmpj)
                tmpk.append(splitListOrig)
                spcTuplestmp.append(tuple(tmpk))
        elif len(tmpk) == 3:
            if (len(str(tmpk[0])) > nAFS
               ) or (len(str(tmpk[1])) > nAFS
                    ) or (len(str(tmpk[2])) > nAFS):
                for tmpl in splitList:
                    tmpk = list(tmpj)
                    tmpk.append(tmpl)
                    spcTuplestmp.append(tuple(tmpk))
            else:
                tmpk = list(tmpj)
                tmpk.append(splitListOrig)
                spcTuplestmp.append(tuple(tmpk))

#    print spcTuplestmp
#    print ("Length spcTuplestmp {0}").format(len(spcTuplestmp))

    rxns = list(react(*spcTuplestmp))

    return rxns

