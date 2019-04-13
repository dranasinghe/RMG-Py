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
RMG is an automatic chemical mechanism generator. It is awesomely awesome.
ARC is an automatic electronic structure computation scheduler. It is magnificently magnificent.
Here we iteratively execute RMG and ARC to generate and refine a chemical kinetic model.

run RMG and ARC in tandem by typing
  python $rmgpy/rmg.py input.py -a arc.yml
where input.py is an RMG input file, and arc.yml is an ARC input file (without specifying species nor reactions)
the ARC input file should have some of the following additional optional directives:

SA species: 10                  The number of top sensitive species to consider
SA reactions: 10                The number of top sensitive reactions from which reactants and products are considered
SA pdep: True                   Whether to consider species from a p-dep network of a top sensitive reaction
all core species: False         Whether to consider all core species
SA observables: True            Whether to consider SA observables
collision violators: True       Whether to consider species participating in collision violating reactions
RMG walltime: 01:00:00:00       A wall time by which to limit RMG runs
tightest RMG tolerance: 0.001   The lowest tolerance to use (highest is the tolerance set in the input file)
max iterations: 10              Maximal times of RMG-ARC iterations
max num exceptions allowed: 10  Maximal number of RMG error exceptions allowed
"""

import os
import shutil
import time
import datetime
import pandas as pd

from rmgpy.rmg.main import RMG
from rmgpy.chemkin import loadChemkinFile
from rmgpy import settings
from rmgpy.species import Species
from rmgpy.data.thermo import ThermoLibrary
from rmgpy.thermo import NASAPolynomial, NASA, ThermoData, Wilhoit
from rmgpy.exceptions import ChemicallySignificantEigenvaluesError, CollisionError, CoreError,\
    ILPSolutionError, InputError, KineticsError, ModifiedStrongCollisionError, NetworkError, PressureDependenceError,\
    ReactionError, ReservoirStateError, StatmechError, StatmechFitError, InvalidMicrocanonicalRateError
from pydas.daspk import DASPKError

from arc.main import ARC, read_file, time_lapse
from arc.arc_exceptions import InputError

################################################################################


t0 = None
log_file = None
species_labels = dict()  # a dictionary of ARC labels (keys) and original Chemkin labels (values)
rmg_exception_counter = 0
rmg_thermo_lib_base_path = os.path.join(settings['database.directory'], 'thermo', 'libraries')


def main(args, kwargs):
    """
    Execute the tandem RMG and ARC model generation and refinement module
    """
    rmg_input_file = args.file
    initialize_log(output_directory=args.output_directory)

    unconverged_species = list()
    all_species = list()  # store species_to_calc_in_iteration, includes unconverged species
    species_dict = dict()  # keys are labels, values are dicts of {'spc': RMG Species objects, 'reason': ``str``}

    arguments, arc_input_dict = parse_arc_input_file(input_file_path=args.arc)

    additional_calcs_required = True
    max_iterations = arguments['max iterations']  # default is 10
    i = 0  # iteration counter
    thermo_library = None  # `None` upon fist call to add_rmg_libraries()

    while additional_calcs_required and i != max_iterations:
        log('\n\nStarting RMG-ARC iteration {0}\n'.format(i))
        run_directory = os.path.join(args.output_directory, 'iteration_{0}'.format(i))
        if not os.path.exists(run_directory):
            os.mkdir(run_directory)

        run_rmg(input_file=rmg_input_file, output_directory=run_directory, kwargs=kwargs, arguments=arguments,
                thermo_library=thermo_library)

        species_to_calc_in_iteration, species_dict_in_iteration, additional_calcs_required =\
            determine_species_to_calculate(run_directory, arguments, unconverged_species, all_species)
        all_species.extend(species_to_calc_in_iteration)
        species_dict.update(species_dict_in_iteration)
        if additional_calcs_required:
            run_arc(arc_input_dict, run_directory, species_to_calc_in_iteration)
            # add the calculated RMG libraries to the database and input file
            unconverged_species_in_iteration = get_unconverged_species(run_directory, all_species)
            unconverged_species.extend(unconverged_species_in_iteration)
            if len(species_to_calc_in_iteration) > len(unconverged_species_in_iteration):
                # we calculated something, add to library
                thermo_library = add_rmg_libraries(run_directory, thermo_library)
            else:
                additional_calcs_required = False
        i += 1
    log_species_summary(species_dict, unconverged_species)
    log_footer()


def run_rmg(input_file, output_directory, kwargs, arguments, thermo_library=None):
    """
    Run RMG.
    `input_file` is the path to the legacy RMG input file
    `output_directory` is the output directory for the current iteration
    `kwargs` are key word arguments to be passed when initializing the RMG object
    `arguments` contain various arguments for the RMG-ARC module
    `thermo_library` is a name for the thermo library and is consistent between iterations
      (it is None in iteration 0 where it is set)
    """
    global rmg_exception_counter
    global rmg_thermo_lib_base_path
    log('Running RMG...')
    max_num_exceptions_allowed = arguments['max num exceptions allowed']
    walltime = arguments['walltime'] if 'RMG walltime' in arguments else '00:00:00:00'
    tightest_tolerance = arguments['tightest RMG tolerance']
    tic = time.time()
    rmg = RMG(inputFile=input_file, outputDirectory=output_directory)
    if thermo_library is not None:
        rmg.thermoLibraries = [thermo_library]
    rmg.initialize(**kwargs)
    rmg.wallTime = walltime
    try:
        rmg.execute(initialize=False)
    except (ChemicallySignificantEigenvaluesError, CollisionError, CoreError, DASPKError, ILPSolutionError,
            InvalidMicrocanonicalRateError, KineticsError, ModifiedStrongCollisionError, NetworkError,
            PressureDependenceError, ReactionError, ReservoirStateError, StatmechError, StatmechFitError) as e:
        log('RMG Errored with {0}. Got the following message:\n{1}'.format(e.__class__, e.message), level='error')
        if rmg_exception_counter > max_num_exceptions_allowed:
            log('This is the {0} exception raised by RMG. Not allowing additional exceptions. '
                'Terminating the process'.format(rmg_exception_counter), level='error')
            raise
        else:
            log('This is the {0} exception raised by RMG (maximum number of allowed exceptions is {1})'.format(
                rmg_exception_counter, max_num_exceptions_allowed), level='warning')
        rmg_exception_counter += 1
    except InputError:
        log('Something seems to be wrong with the RMG input file.', level='error')
        raise
    elapsed_time = time_lapse(tic)
    log('RMG terminated. Overall execution time: {0}'.format(elapsed_time))


def run_arc(input_dict, run_directory, species_to_calc):
    """
    Run ARC.
    `input_dict` is a dictionary containing the directives to execute ARC
    `species_to_calc` is a list of RMG Species objects for which ARC will try to calculate thermodynamic properties
    `run_directory` is the RMG-ARC iteration directory
    """
    log('\nRunning ARC...')
    input_dict['project_directory'] = os.path.join(run_directory, 'ARC')
    if not os.path.exists(input_dict['project_directory']):
        os.mkdir(input_dict['project_directory'])
    input_dict['arc_species_list'] = species_to_calc
    if 'project' not in input_dict:
        input_dict['project'] = 'rmg_arc'
    tic = time.time()
    arc0 = ARC(**input_dict)
    try:
        arc0.execute()
    except Exception as e:
        log('ARC crushed with {0}. Got the following message:\n{1}'.format(e.__class__, e.message), level='error')
        raise
    elapsed_time = time_lapse(tic)
    log('ARC terminated. Overall execution time: {0}'.format(elapsed_time))


def parse_arc_input_file(input_file_path):
    """
    Extract relevant arguments for the RMG-ARC module from the ARC input file located in `input_file_path`
    These arguments could be:
        SA species: 10,
        SA reactions: 10,
        SA pdep: True,
        all core species: False,
        SA observables: True
        collision violators: True
        RMG walltime: 01:00:00:00
        tightest RMG tolerance: 0.001
        max iterations: 10
        max num exceptions allowed: 10
    The above arguments default to `False` or 0 if not given.
    This function also checks the ARC input file
    Returns the `arguments` for the RMG-ARC project and the ARC `input_dict`
    """
    default_max_iterations = 10
    input_dict = read_file(input_file_path[0])
    # check ARC's input file
    if 'species' in input_dict:
        raise InputError("The 'species' dictionary cannot be passed in the ARC input file when running in tandem with"
                         " RMG. Correct ARC's input file and run again.")
    if 'reactions' in input_dict:
        raise InputError("The 'reactions' dictionary cannot be passed in the ARC input file when running in tandem with"
                         " RMG. Correct ARC's input file and run again.")
    arguments = dict()
    arguments['SA species'] = input_dict['SA species'] if 'SA species' in input_dict else 0
    arguments['SA reactions'] = input_dict['SA reactions'] if 'SA reactions' in input_dict else 0
    arguments['SA pdep'] = input_dict['SA pdep'] if 'SA pdep' in input_dict else False
    arguments['all core species'] = input_dict['all core species'] if 'all core species' in input_dict else False
    arguments['SA observables'] = input_dict['SA observables'] if 'SA observables' in input_dict else False
    arguments['collision violators'] = input_dict['collision violators'] if 'collision violators' in input_dict\
        else False
    arguments['walltime'] = input_dict['RMG walltime'] if 'RMG walltime' in input_dict else '00:00:00:00'
    arguments['tightest RMG tolerance'] = input_dict['tightest RMG tolerance']\
        if 'tightest RMG tolerance' in input_dict else None
    arguments['max iterations'] = input_dict['max iterations'] if 'max iterations' in input_dict\
        else default_max_iterations
    arguments['max num exceptions allowed'] = input_dict['max num exceptions allowed']\
        if 'max num exceptions allowed' in input_dict else 10
    for argument in arguments.iterkeys():
        if argument in input_dict:
            del input_dict[argument]
    return arguments, input_dict


def set_legal_species_labels(species_to_calc, all_species):
    """
    ARC uses the species label as folder names on the servers and the local machine.
    Make sure each species has a legal and unique label (which consists of the molecular formula, underscore, and index)
    Store the new labels (keys) and original Chemkin labels (values) in the `species_labels` dictionary
    `species_to_calc` is a list of RMG Species object for which thermo should be calculated in the iteration
    `all_species` is a list of RMG Species object of previously considered species, used here to determine unique labels
    """
    global species_labels
    for i, _ in enumerate(species_to_calc):
        # we're changing arguments (labels) within the `species_to_calc` list, so iterate by index
        formula = species_to_calc[i].molecule[0].getFormula()
        existing_indices = []
        for spc in all_species + species_to_calc:
            # note that `all_species` does not include `species_to_calc` at this point
            if '_' in spc.label:
                split = spc.label.split('_')
                if len(split) == 2 and split[0] == formula and all([char.isdigit() for char in split[1]]):
                    existing_indices.append(int(split[1]))
        index = max(existing_indices) + 1 if existing_indices else 0
        species_labels[formula + '_' + str(index)] = species_to_calc[i].toChemkin()  # store the original Chemkin label
        species_to_calc[i].label = formula + '_' + str(index)  # reset the label
    return species_to_calc


def species_not_in_list(species, species_list):
    """
    Check whether `species` (either an RMG Species or a Chemkin label thereof) is NOT in `species_list`,
    which is a list of RMG SPecies objects.
    Use the unique Chemkin species labels for the comparison
    Note that the labels in `species_list` were changed, therefore the `species_labels` dict is used
    """
    global species_labels
    if isinstance(species, (str, unicode)):
        label = species
    elif isinstance(species, Species):
        label = species.toChemkin()
    else:
        raise TypeError('species can be either an RMG Species object or a Chemkin label thereof.\n'
                        'Got {0} which is a {1}'.format(species, type(species)))
    for spc in species_list:
        if spc.label in species_labels and label == species_labels[spc.label]:
            return False
        elif label == spc.label:
            return False
    return True


def get_species_by_label(label, species_list):
    """
    `label` is the species.label
    `species_list` a list of RMG Species objects
    Returns the RMG Species from `species_list` corresponding to `label`
    """
    for spc in species_list:
        if spc.toChemkin() == label:
            return spc
    return None


def get_reaction_by_index(index, reaction_list):
    """
    `index` is the reaction.index
    `reaction_list` a list of RMG Reaction objects
    Returns the RMG Reaction from `reaction_list` corresponding to `index`
    """
    for rxn in reaction_list:
        if rxn.index == index:
            return rxn
    return None


def should_species_be_calculated(species, unconverged_species, species_to_calc):
    """
    Determine whether a species should be calculated
    `species` is an RMG Species object for which the query is performed
    `unconverged_species` is a list of RMG Species that did not converge in ARC throughout the iterations
    `species_to_calc` is a dictionary of RMG Species to calculate thermo for in the current iteration
      keys are labels, values are dicts of {'spc': RMG Species objects, 'reason': ``str``}
    """
    print('*should_species_be_calculated*')
    print(species)
    print(species.label)
    if calc_based_on_thermo_comment(species)\
            and species_not_in_list(species, unconverged_species)\
            and species_not_in_list(species, [spc['spc'] for spc in species_to_calc.itervalues()]):
        return True
    return False


def determine_species_to_calculate(run_directory, arc_arguments, unconverged_species, all_species):
    """
    Determine which species in the executed RMG job located in the `run_directory` path
    should be calculated by ARC using the specified criteria in the `arc_arguments` dictionary
    parsed from the arc() section of the input file.
    `run_directory` is the RMG-ARC iteration directory
    `arc_arguments` is a dictionary of conditions to consider when determining which specie sto calculate
    `unconverged_species` is a list of RMG Species objects which previously did not converge, not to be calculated
    `all_species` is a list of RMG Species objects identified for calculation for far
    returns a list of species to calculate, a dictionary of species to calculate,
      and whether additional calculations are required
    """
    global species_labels
    species_to_calc = dict()  # keys are labels, values are dicts of {'spc': RMG Species objects, 'reason': ``str``}
    pdep_rxns_to_explore = list()  # contains RMG Reaction objects of pressure dependent reactions

    chemkin_path = os.path.join(run_directory, 'chemkin', 'chem_annotated.inp')
    dictionary_path = os.path.join(run_directory, 'chemkin', 'species_dictionary.txt')
    rmg_species, rmg_reactions = loadChemkinFile(chemkin_path, dictionary_path)
    log('This RMG model has {0} species and {1} reactions in its core.'.format(len(rmg_species), len(rmg_reactions)))

    if 'all core species' in arc_arguments and arc_arguments['all core species']:
        # we'll calculate all core species, if needed based on their thermo comment
        # don't bother checking sensitivities, collision rate violators, etc.
        for species in rmg_species:
            if should_species_be_calculated(species, unconverged_species, species_to_calc):
                species_to_calc[species.toChemkin()] = {'spc': species, 'reason': 'All core species'}
    else:
        if 'SA species' in arc_arguments or 'SA reactions' in arc_arguments\
                or 'SA pdep' in arc_arguments or 'SA observables' in arc_arguments:
            sa_path = os.path.join(run_directory, 'solver')
            if not os.path.exists(sa_path):
                log("Could not find path to RMG's sensitivity analysis output. Not executing "
                    "calculations in ARC based on sensitivity analysis!", level='error')
            else:
                sa_files = list()
                for file1 in os.listdir(sa_path):
                    if file1.endswith(".csv"):
                        sa_files.append(file1)
                for sa_file in sa_files:
                    # iterate through all SA .csv files in the solver folder
                    df = pd.read_csv(os.path.join(sa_path, sa_file))
                    sa_dict = dict()
                    sa_dict['rxn'], sa_dict['spc'] = dict(), dict()
                    for header in df.columns:
                        # iterate through all headers in the SA .csv file, but skip the `Time (s)` column
                        sa_type = None
                        if 'dlnk' in header and 'SA reactions' in arc_arguments and arc_arguments['SA reactions']:
                            sa_type = 'rxn'
                        elif 'dG' in header and 'SA species' in arc_arguments and arc_arguments['SA species']:
                            sa_type = 'spc'
                        if sa_type is not None:
                            # proceed only if we care about this column
                            entry = dict()
                            observable_label = header.split('[')[1].split(']')[0]
                            observable = get_species_by_label(observable_label, rmg_species)
                            if 'SA observables' in arc_arguments and arc_arguments['SA observables']\
                                    and should_species_be_calculated(observable, unconverged_species, species_to_calc):
                                species_to_calc[observable.toChemkin()] = {'spc': observable, 'reason': 'observable'}
                            if observable.toChemkin() not in sa_dict[sa_type]:
                                sa_dict[sa_type][observable] = list()
                            # parameter extraction examples:
                            # for species get `C2H4(8)` from `dln[ethane(1)]/dG[C2H4(8)]`
                            # for reaction, get int::8 from `dln[ethane(1)]/dln[k8]: H(6)+ethane(1)=H2(12)+C2H5(5)`
                            parameter = header.split('[')[2].split(']')[0]
                            if sa_type == 'rxn':
                                parameter = int(header.split('[')[2].split(']')[0][1:])
                            entry['parameter'] = parameter  # rxn number or spc label
                            entry['max_sa'] = max(df[header].max(), abs(df[header].min()))
                            sa_dict[sa_type][observable].append(entry)
                    num_rxn = arc_arguments['SA reactions'] if 'SA reactions' in arc_arguments else 0
                    num_spc = arc_arguments['SA species'] if 'SA species' in arc_arguments else 0
                    for observable, sa_list in sa_dict['rxn'].iteritems():
                        sa_list_sorted = sorted(sa_list, key=lambda item: item['max_sa'], reverse=True)
                        for i in xrange(min(num_rxn, len(sa_list_sorted))):
                            reaction = get_reaction_by_index(sa_list_sorted[i]['parameter'], rmg_reactions)
                            for species in reaction.reactants + reaction.products:
                                if should_species_be_calculated(species, unconverged_species, species_to_calc):
                                    ordinal = get_ordinal_indicator(i)
                                    reason = 'species participates in the {i}{ordinal} most sensitive reaction ' \
                                             'for observable {observable}: {reaction}'.format(
                                              i=i, ordinal=ordinal, observable=observable, reaction=reaction)
                                    species_to_calc[species.toChemkin()] = {'spc': species, 'reason': reason}
                            if reaction.kinetics.isPressureDependent()\
                                    and reaction not in [rxn_tup[0] for rxn_tup in pdep_rxns_to_explore]\
                                    and 'SA pdep' in arc_arguments and arc_arguments['SA pdep']:
                                pdep_rxns_to_explore.append((reaction, i, observable))
                    for observable, sa_list in sa_dict['spc'].iteritems():
                        sa_list_sorted = sorted(sa_list, key=lambda item: item['max_sa'], reverse=True)
                        for i in xrange(min(num_spc, len(sa_list_sorted))):
                            species = get_species_by_label(sa_list_sorted[i]['parameter'], rmg_species)
                            if should_species_be_calculated(species, unconverged_species, species_to_calc):
                                ordinal = get_ordinal_indicator(i)
                                reason = 'the {i}{ordinal} most sensitive species for observable {observable}'.format(
                                    i=i, ordinal=ordinal, observable=observable)
                                species_to_calc[species.toChemkin()] = {'spc': species, 'reason': reason}

        for reaction_tuple in pdep_rxns_to_explore:
            for species in reaction_tuple[0].network.reactants + reaction_tuple[0].network.isomers\
                           + reaction_tuple[0].network.products:
                if should_species_be_calculated(species, unconverged_species, species_to_calc):
                    ordinal = get_ordinal_indicator(reaction_tuple[1])
                    reason = 'species participates in pressure dependent network {network} from which reaction ' \
                             '{reaction} was derived, the {i}{ordinal} most sensitive reaction for observable ' \
                             '{observable}'.format(network=reaction_tuple[0].network, reaction=reaction_tuple[0],
                                                   i=reaction_tuple[1], ordinal=ordinal, observable=reaction_tuple[2])
                    species_to_calc[species.toChemkin()] = {'spc': species, 'reason': reason}

        if 'collision violators' in arc_arguments and arc_arguments['collision violators']:
            coll_violators_path = os.path.join(run_directory, 'collision_rate_violators.log')
            with open(coll_violators_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if '=' in line and '!' not in line:
                    labels = line.split()[0].split('=')
                    if '+' in labels[0]:
                        reactants = labels[0].split('+')
                    else:
                        reactants = [labels[0]]
                    if '+' in labels[1]:
                        products = labels[1].split('+')
                    else:
                        products = [labels[1]]
                    labels = reactants + products
                    for label in labels:
                        species = get_species_by_label(label, rmg_species)
                        print('*collision violators*')
                        print(species)
                        print(species.label)
                        if should_species_be_calculated(species, unconverged_species, species_to_calc):
                            reason = 'species participates in a collision rate violating reaction, {0}'.format(line)
                            species_to_calc[species.toChemkin()] = {'spc': species, 'reason': reason}

    species_list_to_calc = [spc['spc'] for spc in species_to_calc.itervalues()]
    additional_calcs_required = bool(len(species_list_to_calc))
    log('Additional calculations required: {0}\n'.format(additional_calcs_required))
    if additional_calcs_required:
        log_species_to_calculate(species_to_calc)
        set_legal_species_labels(species_list_to_calc, all_species)
    return species_list_to_calc, species_to_calc, additional_calcs_required


def calc_based_on_thermo_comment(spc):
    """
    A helper function for reading the Species `spc` thermo comment and determining whether to calculate it in ARC
    """
    print('*calc_based_on_thermo_comment*')
    print(spc)
    print(spc.label)
    if 'group additivity' in spc.thermo.comment or '+ radical(' in spc.thermo.comment:
        return True
    return False


def get_ordinal_indicator(number):
    """Returns the ordinal indicator for the integer `number`"""
    ordinal_dict = {1: 'st', 2: 'nd', 3: 'rd'}
    if number > 13 :
        number %= 10
    if number in ordinal_dict.iterkeys():
        return ordinal_dict[number]
    return 'th'


def log_species_to_calculate(species_dict):
    """
    Report the species to be calculated in the next iteration.
    `species_dict` is a dictionary of RMG Species containing the reason for calculating them
    """
    log('Species to calculate thermodynamic data for (label, SMILES, and reason):')
    for label, spc_dict in species_dict.iteritems():
        log('{0}, {1}, {2}'.format(label, spc_dict['spc'].molecule[0].toSMILES(), spc_dict['reason']))


def log_species_summary(species_dict, unconverged_species):
    """
    Report species summary
    The report will be saved as `RMG_ARC_species.log` in `run_directory`, the RMG run folder
    `species_dict` is a dictionary of RMG Species containing the reason for calculating them
    `unconverged_species` is a list of RMG Species for which thermodynamic calculations in ARC failed
    """
    global species_labels
    log('SPECIES SUMMARY')
    log('Species for which thermodynamic data was calculate by ARC (label, SMILES, and reason):')
    print(species_dict)  # debug, delete
    for label, spc_dict in species_dict.iteritems():
        if all([label != species_labels(spc.label) for spc in unconverged_species]):
            log('{0}, {1}, {2}'.format(label, spc_dict['spc'].molecule[0].toSMILES(), spc_dict['reason']))
    if unconverged_species:
        log('Species for which thermodynamic data did not converge (label, SMILES, and reason):')
        for uc_spc in unconverged_species:
            for label, spc_dict in species_dict.iteritems():
                if label == species_labels(uc_spc.label):
                    log('(FAILED) {0}, {1}, {2}'.format(label, spc_dict['spc'].molecule[0].toSMILES(), spc_dict['reason']))


def log_unconverged_species(unconverged_species):
    """
    Report unconverged species
    `unconverged_species` cis a list or RMG Species for which thermodynamic calculations in ARC failed
    """
    global species_labels
    if unconverged_species:
        log('\nThermodynamic calculations for the following species DID NOT converge (label, SMILES):')
        for spc in unconverged_species:
            log('{0}, {1}'.format(species_labels[spc.label], spc.molecule[0].toSMILES()))
        log('\n')
    else:
        log('\nAll species thermodynamic calculations in this RMG-ARC iteration successfully converged.')


def add_rmg_libraries(run_directory, library_name=None):
    """
    (function name is in a plural form, we anticipate having a kinetics library soon)
    Creates a thermo library in the correct place in RMG-database, and append the entries from library generated by ARC
    (located in `run_directory`/ARC/output/RMG libraries/)
    `run_directory` is the directory of the recently terminated RMG-ARC iteration
    `library_name` is None in the first iteration (where it is created)
    Returns the (created) `library_name`
    """
    global rmg_thermo_lib_base_path
    arc_thermo_lib_path = os.path.join(run_directory, 'ARC', 'output', 'RMG libraries', 'thermo', 'rmg_arc.py')
    if library_name is None:
        # This is the first iteration, come up with a unique library_name
        unique_library_name = False
        j = 0
        library_name = 'arc_thermo'
        while not unique_library_name:
            # make sure the new library name is unique and used throughout the project
            rmg_thermo_lib_path = os.path.join(rmg_thermo_lib_base_path, '{0}.py'.format(library_name))
            if not os.path.isfile(rmg_thermo_lib_path):
                unique_library_name = True
            else:
                j += 1
                library_name = 'arc_thermo' + '_' + str(j)
    else:
        # This is not the first iteration, use the provided library_name
        rmg_thermo_lib_path = os.path.join(settings['database.directory'], 'thermo', 'libraries',
                                           '{0}.py'.format(library_name))
    local_context = {
        'ThermoData': ThermoData,
        'Wilhoit': Wilhoit,
        'NASAPolynomial': NASAPolynomial,
        'NASA': NASA,
    }
    if os.path.isfile(rmg_thermo_lib_path) and os.path.isfile(arc_thermo_lib_path):
        # the rmg_arc thermo library already exists in RMG-database. load it, append, and save.
        rmg_thermo_lib, arc_thermo_lib = ThermoLibrary(), ThermoLibrary()
        rmg_thermo_lib.load(path=rmg_thermo_lib_path, local_context=local_context, global_context=dict())
        arc_thermo_lib.load(path=arc_thermo_lib_path, local_context=local_context, global_context=dict())
        description = arc_thermo_lib.longDesc
        description_to_append = '\n'
        append = False
        for line in description.splitlines():
            if 'Considered the following' in line:
                append = True
            if append:
                description_to_append += line + '\n'
        rmg_thermo_lib.longDesc += description_to_append
        for entry in arc_thermo_lib.entries.itervalues():
            unique_species_name, j = False, 0
            label = entry.label
            while not unique_species_name:
                # make sure each entry has a unique label in the unified library
                for existing_entry in rmg_thermo_lib.entries.itervalues():
                    if label == existing_entry.label:
                        label = entry.labe + '_' + str(j)
                        j += 1
                        break
                else:
                    unique_species_name = True
            entry.label = label
            rmg_thermo_lib.entries[entry.label] = entry
        rmg_thermo_lib.save(path=rmg_thermo_lib_path)
    elif not os.path.isfile(rmg_thermo_lib_path) and os.path.isfile(arc_thermo_lib_path):
        # the rmg_arc thermo library doesn't exist in RMG-database. just copy the library generated by ARC.
        shutil.copy(arc_thermo_lib_path, rmg_thermo_lib_path)
    return library_name


def get_unconverged_species(run_directory, all_species):
    """
    Get the labels of unconverged species from the present ARC iteration
    `run_directory` is the directory of the present iteration
    `all_species` is a list of RMG Species objects sent to calculation so far
    returns a list of RMG Species objects where could not be calculated in the present iteration
    """
    unconverged_species_labels, unconverged_species = list(), list()
    info_path = os.path.join(run_directory, 'ARC', 'rmg_arc.info')
    if os.path.isfile(info_path):
        with open(info_path, 'r') as f:
            read = False
            for line in f:
                if read:
                    if line == '\n':
                        break
                    if 'Species' in line and '(Failed!)' in line:
                        unconverged_species_labels.append(line.split()[1])
                if 'Considered the following species' in line:
                    read = True
    for label in unconverged_species_labels:
        for spc in all_species:
            if label == spc.label:
                unconverged_species.append(spc)
    log_unconverged_species(unconverged_species)
    return unconverged_species


def log(message, level='info'):
    """
    RMG and ARC have loggers that will override a conventional logger used here imported from logging
    Hence we define our own simple logging tool here
    `message` is the message to be logged
    `level` controls the prefix and suffix to be added to the message.
    Allowed values for `level` are: 'info' (default), 'warning', and 'error'.
    """
    global log_file
    if level not in ['info', 'warning', 'error']:
        log('Got an illegal level argument "{0}"'.format(level), level='error')
        level = 'info'
    prefix = {'info': '', 'warning': '\nWARNING: ', 'error': '\n\n\nERROR: '}
    suffix = {'info': '', 'warning': '\n', 'error': '\n\n'}
    if isinstance(message, dict):
        message = dict_to_str(message)
    elif not isinstance(message, (str, unicode)):
        message = str(message)
    message = prefix[level] + message + suffix[level]
    # also print to stdout
    print(message)
    # log to file
    message += '\n'
    with open(log_file, 'a') as f:
        f.write(message)


def dict_to_str(dictionary, level=0):
    """
    A helper function to log dictionaries in a pretty way
    """
    message = ''
    for key, value in dictionary.items():
        if isinstance(value, dict):
            message += ' ' * level * 2 + str(key) + ':\n' + dict_to_str(value, level + 1)
        else:
            message += ' ' * level * 2 + str(key) + ': ' + str(value) + '\n'
    return message


def initialize_log(output_directory):
    """
    Set up the logger
    """
    global log_file
    log_file = os.path.join(output_directory, 'rmg_arc.log')
    if os.path.isfile(log_file):
        if not os.path.isdir(os.path.join(os.path.dirname(log_file), 'log_archive')):
            os.mkdir(os.path.join(os.path.dirname(log_file), 'log_archive'))
        local_time = datetime.datetime.now().strftime("%H%M%S_%b%d_%Y")
        log_backup_name = 'rmg_arc.' + local_time + '.log'
        shutil.copy(log_file, os.path.join(os.path.dirname(log_file), 'log_archive', log_backup_name))
        os.remove(log_file)
    log_header()


def log_header():
    """
    Output a header containing identifying information about the RMG-ARC feature to the log.
    """
    global t0
    t0 = time.time()
    log('Tandem RMG-ARC model generation and refinement\n'
        'Execution initiated on {0}'.format(time.asctime()))


def log_footer():
    """
    Output a footer to the log.
    """
    global t0
    execution_time = time_lapse(t0)
    log('\n\nTotal RMG-ARC execution time: {0}'.format(execution_time))
    log('RMG-ARC execution terminated on {0}'.format(time.asctime()))
