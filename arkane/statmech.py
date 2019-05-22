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
This module provides the :class:`StatMechJob` class, which represents a
statistical mechanics job used to compute and save the statistical mechanics
information for a single species or transition state.
"""

import os.path
import math
import numpy as np
import logging

from rdkit.Chem import GetPeriodicTable

import rmgpy.constants as constants
from rmgpy.species import TransitionState, Species
from rmgpy.statmech.translation import Translation, IdealGasTranslation
from rmgpy.statmech.rotation import Rotation, LinearRotor, NonlinearRotor, KRotor, SphericalTopRotor
from rmgpy.statmech.vibration import Vibration, HarmonicOscillator
from rmgpy.statmech.torsion import Torsion, HinderedRotor, FreeRotor
from rmgpy.statmech.conformer import Conformer
from rmgpy.exceptions import InputError, StatmechError
from rmgpy.quantity import Quantity
from rmgpy.molecule.molecule import Molecule

from arkane.output import prettify
from arkane.log import Log
from arkane.gaussian import GaussianLog
from arkane.molpro import MolproLog
from arkane.qchem import QChemLog
from arkane.common import symbol_by_number
from arkane.common import ArkaneSpecies
from arkane.encorr.corr import get_energy_correction


################################################################################


class ScanLog(object):
    """
    Represent a text file containing a table of angles and corresponding
    scan energies.
    """

    angleFactors = {
        'radians': 1.0,
        'rad': 1.0,
        'degrees': 180.0 / math.pi,
        'deg': 180.0 / math.pi,
    }
    energyFactors = {
        'J/mol': 1.0,
        'kJ/mol': 1.0 / 1000.,
        'cal/mol': 1.0 / 4.184,
        'kcal/mol': 1.0 / 4184.,
        'cm^-1': 1.0 / (constants.h * constants.c * 100. * constants.Na),
        'hartree': 1.0 / (constants.E_h * constants.Na),
    }

    def __init__(self, path):
        self.path = path

    def load(self):
        """
        Load the scan energies from the file. Returns arrays containing the
        angles (in radians) and energies (in J/mol).
        """
        angles, energies = [], []
        angleUnits, energyUnits, angleFactor, energyFactor = None, None, None, None

        with open(self.path, 'r') as stream:
            for line in stream:
                line = line.strip()
                if line == '':
                    continue

                tokens = line.split()
                if angleUnits is None or energyUnits is None:
                    angleUnits = tokens[1][1:-1]
                    energyUnits = tokens[3][1:-1]

                    try:
                        angleFactor = ScanLog.angleFactors[angleUnits]
                    except KeyError:
                        raise ValueError('Invalid angle units {0!r}.'.format(angleUnits))
                    try:
                        energyFactor = ScanLog.energyFactors[energyUnits]
                    except KeyError:
                        raise ValueError('Invalid energy units {0!r}.'.format(energyUnits))

                else:
                    angles.append(float(tokens[0]) / angleFactor)
                    energies.append(float(tokens[1]) / energyFactor)

        angles = np.array(angles)
        energies = np.array(energies)
        energies -= energies[0]

        return angles, energies

    def save(self, angles, energies, angleUnits='radians', energyUnits='kJ/mol'):
        """
        Save the scan energies to the file using the given `angles` in radians
        and corresponding energies `energies` in J/mol. The file is created to
        use the given `angleUnits` for angles and `energyUnits` for energies.
        """
        assert len(angles) == len(energies)

        try:
            angleFactor = ScanLog.angleFactors[angleUnits]
        except KeyError:
            raise ValueError('Invalid angle units {0!r}.'.format(angleUnits))
        try:
            energyFactor = ScanLog.energyFactors[energyUnits]
        except KeyError:
            raise ValueError('Invalid energy units {0!r}.'.format(energyUnits))

        with open(self.path, 'w') as stream:
            stream.write('{0:>24} {1:>24}\n'.format(
                'Angle ({0})'.format(angleUnits),
                'Energy ({0})'.format(energyUnits),
            ))
            for angle, energy in zip(angles, energies):
                stream.write('{0:23.10f} {1:23.10f}\n'.format(angle * angleFactor, energy * energyFactor))


################################################################################


def hinderedRotor(scanLog, pivots, top, symmetry=None, fit='best'):
    """Read a hindered rotor directive, and return the attributes in a list"""
    return [scanLog, pivots, top, symmetry, fit]


def freeRotor(pivots, top, symmetry):
    """Read a free rotor directive, and return the attributes in a list"""
    return [pivots, top, symmetry]


class StatMechJob(object):
    """
    A representation of a Arkane statistical mechanics job. This job is used
    to compute and save the statistical mechanics information for a single
    species or transition state.
    """

    def __init__(self, species, path):
        self.species = species
        self.path = path
        self.modelChemistry = ''
        self.frequencyScaleFactor = 1.0
        self.includeHinderedRotors = True
        self.applyAtomEnergyCorrections = True
        self.applyBondEnergyCorrections = True
        self.bondEnergyCorrectionType = 'p'
        self.atomEnergies = None
        self.supporting_info = [self.species.label]
        self.bonds = None
        self.arkane_species = ArkaneSpecies(species=species)

    def execute(self, outputFile=None, plot=False, pdep=False):
        """
        Execute the statistical mechanics job, saving the results to the
        given `outputFile` on disk.
        """
        self.load(pdep)
        if outputFile is not None:
            self.save(outputFile)
        logging.debug('Finished statmech job for species {0}.'.format(self.species))
        logging.debug(repr(self.species))

    def load(self, pdep=False):
        """
        Load the statistical mechanics parameters for each conformer from
        the associated files on disk. Creates :class:`Conformer` objects for
        each conformer and appends them to the list of conformers on the
        species object.
        """
        path = self.path
        is_ts = isinstance(self.species, TransitionState)
        _, file_extension = os.path.splitext(path)
        if file_extension in ['.yml', '.yaml']:
            self.arkane_species.load_yaml(path=path, species=self.species, pdep=pdep)
            self.species.conformer = self.arkane_species.conformer
            if is_ts:
                self.species.frequency = self.arkane_species.imaginary_frequency
            else:
                self.species.transportData = self.arkane_species.transport_data
                self.species.energyTransferModel = self.arkane_species.energy_transfer_model
                if self.arkane_species.adjacency_list is not None:
                    self.species.molecule = [Molecule().fromAdjacencyList(adjlist=self.arkane_species.adjacency_list)]
                elif self.arkane_species.inchi is not None:
                    self.species.molecule = [Molecule().fromInChI(inchistr=self.arkane_species.inchi)]
                elif self.arkane_species.smiles is not None:
                    self.species.molecule = [Molecule().fromSMILES(smilesstr=self.arkane_species.smiles)]
            return

        logging.info('Loading statistical mechanics parameters for {0}...'.format(self.species.label))

        global_context = {
            '__builtins__': None,
        }
        local_context = {
            '__builtins__': None,
            'True': True,
            'False': False,
            'HinderedRotor': hinderedRotor,
            'FreeRotor': freeRotor,
            # File formats
            'GaussianLog': GaussianLog,
            'QChemLog': QChemLog,
            'MolproLog': MolproLog,
            'ScanLog': ScanLog,
            'Log': Log
        }

        directory = os.path.abspath(os.path.dirname(path))

        with open(path, 'r') as f:
            try:
                exec f in global_context, local_context
            except (NameError, TypeError, SyntaxError):
                logging.error('The species file {0} was invalid:'.format(path))
                raise

        if self.bonds is None:
            try:
                self.bonds = local_context['bonds']
            except KeyError:
                self.bonds = {}

        try:
            linear = local_context['linear']
        except KeyError:
            linear = None

        try:
            externalSymmetry = local_context['externalSymmetry']
        except KeyError:
            externalSymmetry = None

        try:
            spinMultiplicity = local_context['spinMultiplicity']
        except KeyError:
            spinMultiplicity = 0

        try:
            opticalIsomers = local_context['opticalIsomers']
        except KeyError:
            logging.debug('No opticalIsomers provided, estimating them from the quantum file.')
            opticalIsomers = None

        try:
            energy = local_context['energy']
        except KeyError:
            raise InputError('Required attribute "energy" not found in species file {0!r}.'.format(path))
        if isinstance(energy, dict):
            energy = {k.lower(): v for k, v in energy.items()}  # Make model chemistries lower-case
            try:
                energy = energy[self.modelChemistry]
            except KeyError:
                raise InputError('Model chemistry {0!r} not found in from dictionary of energy values in species file '
                                 '{1!r}.'.format(self.modelChemistry, path))
        E0_withZPE, E0 = None, None
        energyLog = None
        if isinstance(energy, Log) and not isinstance(energy, (GaussianLog, QChemLog, MolproLog)):
            energyLog = determine_qm_software(os.path.join(directory, energy.path))
        elif isinstance(energy, (GaussianLog, QChemLog, MolproLog)):
            energyLog = energy
            energyLog.path = os.path.join(directory, energyLog.path)
        elif isinstance(energy, float):
            E0 = energy
        elif isinstance(energy, tuple) and len(energy) == 2:
            # this is likely meant to be a quantity object with ZPE already accounted for
            energy_temp = Quantity(energy)
            E0_withZPE = energy_temp.value_si  # in J/mol
        elif isinstance(energy, tuple) and len(energy) == 3:
            if energy[2] == 'E0':
                energy_temp = Quantity(energy[:2])
                E0 = energy_temp.value_si / constants.E_h / constants.Na  # convert J/mol to Hartree
            elif energy[2] == 'E0-ZPE':
                energy_temp = Quantity(energy[:2])
                E0_withZPE = energy_temp.value_si  # in J/mol
            else:
                raise InputError('The third argument for E0 energy value should be E0 (for energy w/o ZPE) or E0-ZPE. '
                                 'Value entered: {0}'.format(energy[2]))
        try:
            geomLog = local_context['geometry']
        except KeyError:
            raise InputError('Required attribute "geometry" not found in species file {0!r}.'.format(path))
        if isinstance(geomLog, Log) and not isinstance(energy, (GaussianLog, QChemLog, MolproLog)):
            geomLog = determine_qm_software(os.path.join(directory, geomLog.path))
        else:
            geomLog.path = os.path.join(directory, geomLog.path)

        try:
            statmechLog = local_context['frequencies']
        except KeyError:
            raise InputError('Required attribute "frequencies" not found in species file {0!r}.'.format(path))
        if isinstance(statmechLog, Log) and not isinstance(energy, (GaussianLog, QChemLog, MolproLog)):
            statmechLog = determine_qm_software(os.path.join(directory, statmechLog.path))
        else:
            statmechLog.path = os.path.join(directory, statmechLog.path)

        if 'frequencyScaleFactor' in local_context:
            logging.warning('Ignoring frequency scale factor in species file {0!r}.'.format(path))

        rotors = []
        if self.includeHinderedRotors:
            try:
                rotors = local_context['rotors']
            except KeyError:
                pass

        # If hindered/free rotors are included in Statmech job, ensure that the same (freq) log file is used for
        # both the species's optimized geometry and Hessian. This approach guarantees that the geometry and Hessian
        # will be defined in the same Cartesian coordinate system ("Input Orientation", as opposed to
        # "Standard Orientation", or something else). Otherwise, if the geometry and Hessian are read from different
        # log files, it is very easy for them to be defined in different coordinate systems, unless the user is very
        # careful. The current implementation only performs this check for Gaussian logs. If QChem logs are used, only
        # a warning is output reminding the user to ensure the geometry and Hessian are defined in consistent
        # coordinates.
        if len(rotors) > 0:
            if isinstance(statmechLog, GaussianLog):
                if statmechLog.path != geomLog.path:
                    raise InputError('For {0!r}, the geometry log, {1!r}, and frequency log, {2!r}, are not the same.'
                                     ' In order to ensure the geometry and Hessian of {0!r} are defined in consistent'
                                     ' coordinate systems for hindered/free rotor projection, either use the frequency'
                                     ' log for both geometry and frequency, or remove rotors.'.format(
                                      self.species.label, geomLog.path, statmechLog.path))
            elif isinstance(statmechLog, QChemLog):
                logging.warning('QChem log will be used for Hessian of {0!r}. Please verify that the geometry'
                                ' and Hessian of {0!r} are defined in the same coordinate system'.format(
                                 self.species.label))

        logging.debug('    Reading molecular degrees of freedom...')
        conformer, unscaled_frequencies = statmechLog.loadConformer(symmetry=externalSymmetry,
                                                                    spinMultiplicity=spinMultiplicity,
                                                                    opticalIsomers=opticalIsomers,
                                                                    label=self.species.label)
        for mode in conformer.modes:
            if isinstance(mode, (LinearRotor, NonlinearRotor)):
                self.supporting_info.append(mode)
                break
        if unscaled_frequencies:
            self.supporting_info.append(unscaled_frequencies)

        if conformer.spinMultiplicity == 0:
            raise ValueError("Could not read spin multiplicity from log file {0},\n"
                             "please specify the multiplicity in the input file.".format(self.path))

        logging.debug('    Reading optimized geometry...')
        coordinates, number, mass = geomLog.loadGeometry()

        # Infer atoms from geometry
        atoms = {}
        for atom_num in number:
            try:
                symbol = symbol_by_number[atom_num]
            except KeyError:
                raise Exception('Could not recognize element number {0}.'.format(atom_num))
            atoms[symbol] = atoms.get(symbol, 0) + 1

        # Save atoms for use in writing thermo output
        if isinstance(self.species, Species):
            self.species.props['elementCounts'] = atoms

        conformer.coordinates = (coordinates, "angstroms")
        conformer.number = number
        conformer.mass = (mass, "amu")

        logging.debug('    Reading energy...')
        if E0_withZPE is None:
            # The E0 that is read from the log file is without the ZPE and corresponds to E_elec
            if E0 is None:
                E0 = energyLog.loadEnergy(self.frequencyScaleFactor)
            else:
                E0 = E0 * constants.E_h * constants.Na  # Hartree/particle to J/mol
            if not self.applyAtomEnergyCorrections:
                logging.warning('Atom corrections are not being used. Do not trust energies and thermo.')

            E0 += get_energy_correction(
                self.modelChemistry, atoms, self.bonds, coordinates, number,
                multiplicity=conformer.spinMultiplicity, atom_energies=self.atomEnergies,
                apply_atom_corrections=self.applyAtomEnergyCorrections, apply_bacs=self.applyBondEnergyCorrections,
                bac_type=self.bondEnergyCorrectionType
            )

            if len(number) > 1:
                ZPE = statmechLog.loadZeroPointEnergy() * self.frequencyScaleFactor
            else:
                # Monoatomic species don't have frequencies
                ZPE = 0.0
            logging.debug('Corrected minimum energy is {0} J/mol'.format(E0))
            # The E0_withZPE at this stage contains the ZPE
            E0_withZPE = E0 + ZPE

            logging.debug('         Scaling factor used = {0:g}'.format(self.frequencyScaleFactor))
            logging.debug('         ZPE (0 K) = {0:g} kcal/mol'.format(ZPE / 4184.))
            logging.debug('         E0 (0 K) = {0:g} kcal/mol'.format(E0_withZPE / 4184.))

        conformer.E0 = (E0_withZPE * 0.001, "kJ/mol")

        # If loading a transition state, also read the imaginary frequency
        if is_ts:
            neg_freq = statmechLog.loadNegativeFrequency()
            self.species.frequency = (neg_freq * self.frequencyScaleFactor, "cm^-1")
            self.supporting_info.append(neg_freq)

        # Read and fit the 1D hindered rotors if applicable
        # If rotors are found, the vibrational frequencies are also
        # recomputed with the torsional modes removed

        F = statmechLog.loadForceConstantMatrix()

        if F is not None and len(mass) > 1 and len(rotors) > 0:

            logging.debug('    Fitting {0} hindered rotors...'.format(len(rotors)))
            rotorCount = 0
            for q in rotors:
                symmetry = None
                if len(q) == 3:
                    # No potential scan is given, this is a free rotor
                    pivots, top, symmetry = q
                    inertia = conformer.getInternalReducedMomentOfInertia(pivots, top) * constants.Na * 1e23
                    rotor = FreeRotor(inertia=(inertia, "amu*angstrom^2"), symmetry=symmetry)
                    conformer.modes.append(rotor)
                    rotorCount += 1
                elif len(q) in [4, 5]:
                    # This is a hindered rotor
                    if len(q) == 5:
                        scanLog, pivots, top, symmetry, fit = q
                    elif len(q) == 4:
                        # the symmetry number will be derived from the scan
                        scanLog, pivots, top, fit = q
                    # Load the hindered rotor scan energies
                    if isinstance(scanLog, Log) and not isinstance(energy, (GaussianLog, QChemLog, MolproLog)):
                        scanLog = determine_qm_software(os.path.join(directory, scanLog.path))
                    if isinstance(scanLog, GaussianLog):
                        scanLog.path = os.path.join(directory, scanLog.path)
                        v_list, angle = scanLog.loadScanEnergies()
                        scanLogOutput = ScanLog(os.path.join(directory, '{0}_rotor_{1}.txt'.format(
                            self.species.label, rotorCount + 1)))
                        scanLogOutput.save(angle, v_list)
                    elif isinstance(scanLog, QChemLog):
                        scanLog.path = os.path.join(directory, scanLog.path)
                        v_list, angle = scanLog.loadScanEnergies()
                        scanLogOutput = ScanLog(os.path.join(directory, '{0}_rotor_{1}.txt'.format(
                            self.species.label, rotorCount + 1)))
                        scanLogOutput.save(angle, v_list)
                    elif isinstance(scanLog, ScanLog):
                        scanLog.path = os.path.join(directory, scanLog.path)
                        angle, v_list = scanLog.load()
                    else:
                        raise InputError('Invalid log file type {0} for scan log.'.format(scanLog.__class__))

                    if symmetry is None:
                        symmetry = determine_rotor_symmetry(v_list, self.species.label, pivots)
                    inertia = conformer.getInternalReducedMomentOfInertia(pivots, top) * constants.Na * 1e23

                    cosineRotor = HinderedRotor(inertia=(inertia, "amu*angstrom^2"), symmetry=symmetry)
                    cosineRotor.fitCosinePotentialToData(angle, v_list)
                    fourierRotor = HinderedRotor(inertia=(inertia, "amu*angstrom^2"), symmetry=symmetry)
                    fourierRotor.fitFourierPotentialToData(angle, v_list)

                    Vlist_cosine = np.zeros_like(angle)
                    Vlist_fourier = np.zeros_like(angle)
                    for i in range(angle.shape[0]):
                        Vlist_cosine[i] = cosineRotor.getPotential(angle[i])
                        Vlist_fourier[i] = fourierRotor.getPotential(angle[i])

                    if fit == 'cosine':
                        rotor = cosineRotor
                        rotorCount += 1
                        conformer.modes.append(rotor)
                    elif fit == 'fourier':
                        rotor = fourierRotor
                        rotorCount += 1
                        conformer.modes.append(rotor)
                    elif fit == 'best':
                        rms_cosine = np.sqrt(np.sum((Vlist_cosine - v_list) * (Vlist_cosine - v_list)) /
                                                (len(v_list) - 1)) / 4184.
                        rms_fourier = np.sqrt(np.sum((Vlist_fourier - v_list) * (Vlist_fourier - v_list)) /
                                                 (len(v_list) - 1)) / 4184.

                        # Keep the rotor with the most accurate potential
                        rotor = cosineRotor if rms_cosine < rms_fourier else fourierRotor
                        # However, keep the cosine rotor if it is accurate enough, the
                        # fourier rotor is not significantly more accurate, and the cosine
                        # rotor has the correct symmetry
                        if rms_cosine < 0.05 and rms_cosine / rms_fourier < 2.0 and rms_cosine / rms_fourier < 4.0 \
                                and symmetry == cosineRotor.symmetry:
                            rotor = cosineRotor

                        conformer.modes.append(rotor)

                        self.plotHinderedRotor(angle, v_list, cosineRotor, fourierRotor, rotor, rotorCount, directory)

                        rotorCount += 1

            logging.debug('    Determining frequencies from reduced force constant matrix...')
            frequencies = np.array(projectRotors(conformer, F, rotors, linear, is_ts, label=self.species.label))

        elif len(conformer.modes) > 2:
            if len(rotors) > 0:
                logging.warn('Force Constant Matrix Missing Ignoring rotors, if running Gaussian if not already'
                             ' present you need to add the keyword iop(7/33=1) in your Gaussian frequency job for'
                             ' Gaussian to generate the force constant matrix, if running Molpro include keyword print,'
                             ' hessian')
            frequencies = conformer.modes[2].frequencies.value_si
            rotors = np.array([])
        else:
            if len(rotors) > 0:
                logging.warn('Force Constant Matrix Missing Ignoring rotors, if running Gaussian if not already'
                             ' present you need to add the keyword iop(7/33=1) in your Gaussian frequency job for'
                             ' Gaussian to generate the force constant matrix, if running Molpro include keyword print,'
                             ' hessian')
            frequencies = np.array([])
            rotors = np.array([])

        for mode in conformer.modes:
            if isinstance(mode, HarmonicOscillator):
                mode.frequencies = (frequencies * self.frequencyScaleFactor, "cm^-1")

        self.species.conformer = conformer

    def save(self, outputFile):
        """
        Save the results of the statistical mechanics job to the file located
        at `path` on disk.
        """

        logging.info('Saving statistical mechanics parameters for {0}...'.format(self.species.label))
        f = open(outputFile, 'a')

        conformer = self.species.conformer
        coordinates = conformer.coordinates.value_si * 1e10
        number = conformer.number.value_si

        f.write('# Coordinates for {0} in Input Orientation (angstroms):\n'.format(self.species.label))
        for i in range(coordinates.shape[0]):
            x = coordinates[i, 0]
            y = coordinates[i, 1]
            z = coordinates[i, 2]
            f.write('#   {0} {1:9.4f} {2:9.4f} {3:9.4f}\n'.format(symbol_by_number[number[i]], x, y, z))

        result = 'conformer(label={0!r}, E0={1!r}, modes={2!r}, spinMultiplicity={3:d}, opticalIsomers={4:d}'.format(
            self.species.label,
            conformer.E0,
            conformer.modes,
            conformer.spinMultiplicity,
            conformer.opticalIsomers,
        )
        try:
            result += ', frequency={0!r}'.format(self.species.frequency)
        except AttributeError:
            pass
        result += ')'
        f.write('{0}\n\n'.format(prettify(result)))
        f.close()

    def plotHinderedRotor(self, angle, v_list, cosineRotor, fourierRotor, rotor, rotorIndex, directory):
        """
        Plot the potential for the rotor, along with its cosine and Fourier
        series potential fits. The plot is saved to a set of files of the form
        ``hindered_rotor_1.pdf``.
        """
        try:
            import pylab
        except ImportError:
            return

        phi = np.arange(0, 6.3, 0.02, np.float64)
        Vlist_cosine = np.zeros_like(phi)
        Vlist_fourier = np.zeros_like(phi)
        for i in range(phi.shape[0]):
            Vlist_cosine[i] = cosineRotor.getPotential(phi[i])
            Vlist_fourier[i] = fourierRotor.getPotential(phi[i])

        fig = pylab.figure(figsize=(6, 5))
        pylab.plot(angle, v_list / 4184., 'ok')
        linespec = '-r' if rotor is cosineRotor else '--r'
        pylab.plot(phi, Vlist_cosine / 4184., linespec)
        linespec = '-b' if rotor is fourierRotor else '--b'
        pylab.plot(phi, Vlist_fourier / 4184., linespec)
        pylab.legend(['scan', 'cosine', 'fourier'], loc=1)
        pylab.xlim(0, 2 * constants.pi)
        pylab.xlabel('Angle')
        pylab.ylabel('Potential (kcal/mol)')
        pylab.title('{0} hindered rotor #{1:d}'.format(self.species.label, rotorIndex + 1))

        axes = fig.get_axes()[0]
        axes.set_xticks([float(j * constants.pi / 4) for j in range(0, 9)])
        axes.set_xticks([float(j * constants.pi / 8) for j in range(0, 17)], minor=True)
        axes.set_xticklabels(
            ['$0$', '$\pi/4$', '$\pi/2$', '$3\pi/4$', '$\pi$', '$5\pi/4$', '$3\pi/2$', '$7\pi/4$', '$2\pi$'])

        pylab.savefig(os.path.join(directory, '{0}_rotor_{1:d}.pdf'.format(self.species.label, rotorIndex + 1)))
        pylab.close()


################################################################################


def determine_qm_software(fullpath):
    """
    Given a path to the log file of a QM software, determine whether it is Gaussian, Molpro, or QChem
    """
    with open(fullpath, 'r') as f:
        line = f.readline()
        software_log = None
        while line != '':
            if 'gaussian' in line.lower():
                f.close()
                software_log = GaussianLog(fullpath)
                break
            elif 'qchem' in line.lower():
                f.close()
                software_log = QChemLog(fullpath)
                break
            elif 'molpro' in line.lower():
                f.close()
                software_log = MolproLog(fullpath)
                break
            line = f.readline()
        else:
            raise InputError(
                "File at {0} could not be identified as a Gaussian, QChem or Molpro log file.".format(fullpath))
    return software_log


def is_linear(coordinates):
    """
    Determine whether or not the species is linear from its 3D coordinates
    First, try to reduce the problem into just two dimensions, use 3D if the problem cannot be reduced
    `coordinates` is a numpy.array of the species' xyz coordinates
    """
    # epsilon is in degrees
    # (from our experience, linear molecules have precisely 180.0 degrees between all atom triples)
    epsilon = 0.1

    number_of_atoms = len(coordinates)
    if number_of_atoms == 1:
        return False
    if number_of_atoms == 2:
        return True

    # A tensor containing all distance vectors in the molecule
    d = -np.array([c[:, np.newaxis] - c[np.newaxis, :] for c in coordinates.T])
    for i in range(2, len(coordinates)):
        u1 = d[:, 0, 1] / np.linalg.norm(d[:, 0, 1])  # unit vector between atoms 0 and 1
        u2 = d[:, 1, i] / np.linalg.norm(d[:, 1, i])  # unit vector between atoms 1 and i
        a = math.degrees(np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0)))  # angle between atoms 0, 1, i
        if abs(180 - a) > epsilon and abs(a) > epsilon:
            return False
    return True


def projectRotors(conformer, F, rotors, linear, is_ts, label):
    """
    For a given `conformer` with associated force constant matrix `F`, lists of
    rotor information `rotors`, `pivots`, and `top1`, and the linearity of the
    molecule `linear`, project out the nonvibrational modes from the force
    constant matrix and use this to determine the vibrational frequencies. The
    list of vibrational frequencies is returned in cm^-1.

    Refer to Gaussian whitepaper (http://gaussian.com/vib/) for procedure to calculate
    harmonic oscillator vibrational frequencies using the force constant matrix.
    """
    mass = conformer.mass.value_si
    coordinates = conformer.coordinates.getValue()
    if linear is None:
        linear = is_linear(coordinates)
        if linear:
            logging.info('Determined species {0} to be linear.'.format(label))
    Nrotors = len(rotors)
    Natoms = len(conformer.mass.value)
    Nvib = 3 * Natoms - (5 if linear else 6) - Nrotors - (1 if is_ts else 0)

    # Put origin in center of mass
    xm = 0.0
    ym = 0.0
    zm = 0.0
    totmass = 0.0
    for i in range(Natoms):
        xm += mass[i] * coordinates[i, 0]
        ym += mass[i] * coordinates[i, 1]
        zm += mass[i] * coordinates[i, 2]
        totmass += mass[i]

    xm /= totmass
    ym /= totmass
    zm /= totmass

    for i in range(Natoms):
        coordinates[i, 0] -= xm
        coordinates[i, 1] -= ym
        coordinates[i, 2] -= zm
    # Make vector with the root of the mass in amu for each atom
    amass = np.sqrt(mass / constants.amu)

    # Rotation matrix
    I = conformer.getMomentOfInertiaTensor()
    PMoI, Ixyz = np.linalg.eigh(I)

    external = 6
    if linear:
        external = 5

    D = np.zeros((Natoms * 3, external), np.float64)

    P = np.zeros((Natoms, 3), np.float64)

    # Transform the coordinates to the principal axes
    P = np.dot(coordinates, Ixyz)

    for i in range(Natoms):
        # Projection vectors for translation
        D[3 * i + 0, 0] = amass[i]
        D[3 * i + 1, 1] = amass[i]
        D[3 * i + 2, 2] = amass[i]

    # Construction of the projection vectors for external rotation
    for i in range(Natoms):
        D[3 * i, 3] = (P[i, 1] * Ixyz[0, 2] - P[i, 2] * Ixyz[0, 1]) * amass[i]
        D[3 * i + 1, 3] = (P[i, 1] * Ixyz[1, 2] - P[i, 2] * Ixyz[1, 1]) * amass[i]
        D[3 * i + 2, 3] = (P[i, 1] * Ixyz[2, 2] - P[i, 2] * Ixyz[2, 1]) * amass[i]
        D[3 * i, 4] = (P[i, 2] * Ixyz[0, 0] - P[i, 0] * Ixyz[0, 2]) * amass[i]
        D[3 * i + 1, 4] = (P[i, 2] * Ixyz[1, 0] - P[i, 0] * Ixyz[1, 2]) * amass[i]
        D[3 * i + 2, 4] = (P[i, 2] * Ixyz[2, 0] - P[i, 0] * Ixyz[2, 2]) * amass[i]
        if not linear:
            D[3 * i, 5] = (P[i, 0] * Ixyz[0, 1] - P[i, 1] * Ixyz[0, 0]) * amass[i]
            D[3 * i + 1, 5] = (P[i, 0] * Ixyz[1, 1] - P[i, 1] * Ixyz[1, 0]) * amass[i]
            D[3 * i + 2, 5] = (P[i, 0] * Ixyz[2, 1] - P[i, 1] * Ixyz[2, 0]) * amass[i]

    # Make sure projection matrix is orthonormal
    import scipy.linalg

    I = np.identity(Natoms * 3, np.float64)

    P = np.zeros((Natoms * 3, 3 * Natoms + external), np.float64)

    P[:, 0:external] = D[:, 0:external]
    P[:, external:external + 3 * Natoms] = I[:, 0:3 * Natoms]

    for i in range(3 * Natoms + external):
        norm = 0.0
        for j in range(3 * Natoms):
            norm += P[j, i] * P[j, i]
        for j in range(3 * Natoms):
            if norm > 1E-15:
                P[j, i] /= np.sqrt(norm)
            else:
                P[j, i] = 0.0
        for j in range(i + 1, 3 * Natoms + external):
            proj = 0.0
            for k in range(3 * Natoms):
                proj += P[k, i] * P[k, j]
            for k in range(3 * Natoms):
                P[k, j] -= proj * P[k, i]

    # Order D, there will be vectors that are 0.0
    i = 0
    while i < 3 * Natoms:
        norm = 0.0
        for j in range(3 * Natoms):
            norm += P[j, i] * P[j, i]
        if norm < 0.5:
            P[:, i:3 * Natoms + external - 1] = P[:, i + 1:3 * Natoms + external]
        else:
            i += 1

    # T is the transformation vector from cartesian to internal coordinates
    T = np.zeros((Natoms * 3, 3 * Natoms - external), np.float64)

    T[:, 0:3 * Natoms - external] = P[:, external:3 * Natoms]

    # Generate mass-weighted force constant matrix
    # This converts the axes to mass-weighted Cartesian axes
    # Units of Fm are J/m^2*kg = 1/s^2
    Fm = F.copy()
    for i in range(Natoms):
        for j in range(Natoms):
            for u in range(3):
                for v in range(3):
                    Fm[3 * i + u, 3 * j + v] /= math.sqrt(mass[i] * mass[j])

    Fint = np.dot(T.T, np.dot(Fm, T))

    # Get eigenvalues of internal force constant matrix, V = 3N-6 * 3N-6
    eig, V = np.linalg.eigh(Fint)

    logging.debug('Frequencies from internal Hessian')
    for i in range(3 * Natoms - external):
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered in sqrt')
            logging.debug(np.sqrt(eig[i]) / (2 * math.pi * constants.c * 100))

    # Now we can start thinking about projecting out the internal rotations
    Dint = np.zeros((3 * Natoms, Nrotors), np.float64)

    counter = 0
    for i, rotor in enumerate(rotors):
        if len(rotor) == 5:
            scanLog, pivots, top, symmetry, fit = rotor
        elif len(rotor) == 3:
            pivots, top, symmetry = rotor
        # Determine pivot atom
        if pivots[0] in top:
            pivot1 = pivots[0]
            pivot2 = pivots[1]
        elif pivots[1] in top:
            pivot1 = pivots[1]
            pivot2 = pivots[0]
        else:
            raise Exception('Could not determine pivot atom.')
        # Projection vectors for internal rotation
        e12 = coordinates[pivot1 - 1, :] - coordinates[pivot2 - 1, :]
        for j in range(Natoms):
            atom = j + 1
            if atom in top:
                e31 = coordinates[atom - 1, :] - coordinates[pivot1 - 1, :]
                Dint[3 * (atom - 1):3 * (atom - 1) + 3, counter] = np.cross(e31, e12) * amass[atom - 1]
            else:
                e31 = coordinates[atom - 1, :] - coordinates[pivot2 - 1, :]
                Dint[3 * (atom - 1):3 * (atom - 1) + 3, counter] = np.cross(e31, -e12) * amass[atom - 1]
        counter += 1

    # Normal modes in mass weighted cartesian coordinates
    Vmw = np.dot(T, V)
    eigM = np.zeros((3 * Natoms - external, 3 * Natoms - external), np.float64)

    for i in range(3 * Natoms - external):
        eigM[i, i] = eig[i]

    Fm = np.dot(Vmw, np.dot(eigM, Vmw.T))

    # Internal rotations are not normal modes => project them on the normal modes and orthogonalize
    # Dintproj =  (3N-6) x (3N) x (3N) x (Nrotors)
    Dintproj = np.dot(Vmw.T, Dint)

    # Reconstruct Dint
    for i in range(Nrotors):
        for j in range(3 * Natoms):
            Dint[j, i] = 0
            for k in range(3 * Natoms - external):
                Dint[j, i] += Dintproj[k, i] * Vmw[j, k]

    # Ortho normalize
    for i in range(Nrotors):
        norm = 0.0
        for j in range(3 * Natoms):
            norm += Dint[j, i] * Dint[j, i]
        for j in range(3 * Natoms):
            Dint[j, i] /= np.sqrt(norm)
        for j in range(i + 1, Nrotors):
            proj = 0.0
            for k in range(3 * Natoms):
                proj += Dint[k, i] * Dint[k, j]
            for k in range(3 * Natoms):
                Dint[k, j] -= proj * Dint[k, i]

    Dintproj = np.dot(Vmw.T, Dint)
    Proj = np.dot(Dint, Dint.T)
    I = np.identity(Natoms * 3, np.float64)
    Proj = I - Proj
    Fm = np.dot(Proj, np.dot(Fm, Proj))
    # Get eigenvalues of mass-weighted force constant matrix
    eig, V = np.linalg.eigh(Fm)
    eig.sort()

    # Convert eigenvalues to vibrational frequencies in cm^-1
    # Only keep the modes that don't correspond to translation, rotation, or internal rotation

    logging.debug('Frequencies from projected Hessian')
    for i in range(3 * Natoms):
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered in sqrt')
            logging.debug(np.sqrt(eig[i]) / (2 * math.pi * constants.c * 100))

    return np.sqrt(eig[-Nvib:]) / (2 * math.pi * constants.c * 100)


def assign_frequency_scale_factor(model_chemistry):
    """
    Assign the frequency scaling factor according to the model chemistry.
    Refer to https://comp.chem.umn.edu/freqscale/index.html for future updates of these factors
    """
    freq_dict = {'cbs-qb3': 0.99,  # J. Chem. Phys. 1999, 110, 2822–2827
                 'cbs-qb3-paraskevas': 0.99,
                 # 'g3': ,
                 'm08so/mg3s*': 0.983,  # DOI: 10.1021/ct100326h, taken as 'M08-SO/MG3S'
                 'm06-2x/cc-pvtz': 0.955,  # http://cccbdb.nist.gov/vibscalejust.asp
                 # 'klip_1': ,
                 # 'klip_2': ,
                 # 'klip_3': ,
                 # 'klip_2_cc': ,
                 # 'ccsd(t)-f12/cc-pvdz-f12_h-tz': ,
                 # 'ccsd(t)-f12/cc-pvdz-f12_h-qz': ,
                 'ccsd(t)-f12/cc-pvdz-f12': 0.979,
                 # http://cccbdb.nist.gov/vibscalejust.asp, taken as 'ccsd(t)/cc-pvdz'
                 'ccsd(t)-f12/cc-pvtz-f12': 0.984,
                 # Taken from https://comp.chem.umn.edu/freqscale/version3b2.htm as CCSD(T)-F12a/cc-pVTZ-F12
                 'ccsd(t)-f12/cc-pvqz-f12': 0.970,
                 # http://cccbdb.nist.gov/vibscalejust.asp, taken as 'ccsd(t)/cc-pvqz'
                 'ccsd(t)-f12/cc-pcvdz-f12': 0.971,
                 # http://cccbdb.nist.gov/vibscalejust.asp, taken as 'ccsd(t)/cc-pcvdz'
                 'ccsd(t)-f12/cc-pcvtz-f12': 0.966,
                 # 'ccsd(t)-f12/cc-pcvqz-f12': ,
                 # 'ccsd(t)-f12/cc-pvtz-f12(-pp)': ,
                 # 'ccsd(t)/aug-cc-pvtz(-pp)': ,
                 'ccsd(t)-f12/aug-cc-pvdz': 0.963,
                 # http://cccbdb.nist.gov/vibscalejust.asp, taken as 'ccsd(t)/aug-cc-pvdz'
                 'ccsd(t)-f12/aug-cc-pvtz': 0.970,
                 # http://cccbdb.nist.gov/vibscalejust.asp, taken as 'ccsd(t)/aug-cc-pvtz'
                 'ccsd(t)-f12/aug-cc-pvqz': 0.975,
                 # http://cccbdb.nist.gov/vibscalejust.asp, taken as 'ccsd(t)/aug-cc-pvqz'
                 # 'b-ccsd(t)-f12/cc-pvdz-f12': ,
                 # 'b-ccsd(t)-f12/cc-pvtz-f12': ,
                 # 'b-ccsd(t)-f12/cc-pvqz-f12': ,
                 # 'b-ccsd(t)-f12/cc-pcvdz-f12': ,
                 # 'b-ccsd(t)-f12/cc-pcvtz-f12': ,
                 # 'b-ccsd(t)-f12/cc-pcvqz-f12': ,
                 # 'b-ccsd(t)-f12/aug-cc-pvdz': ,
                 # 'b-ccsd(t)-f12/aug-cc-pvtz': ,
                 # 'b-ccsd(t)-f12/aug-cc-pvqz': ,
                 'mp2_rmp2_pvdz': 0.953,  # http://cccbdb.nist.gov/vibscalejust.asp, taken as ',p2/cc-pvdz'
                 'mp2_rmp2_pvtz': 0.950,  # http://cccbdb.nist.gov/vibscalejust.asp, taken as ',p2/cc-pvdz'
                 'mp2_rmp2_pvqz': 0.962,  # http://cccbdb.nist.gov/vibscalejust.asp, taken as ',p2/cc-pvdz'
                 'ccsd-f12/cc-pvdz-f12': 0.947,  # http://cccbdb.nist.gov/vibscalejust.asp, taken as ccsd/cc-pvdz
                 # 'ccsd(t)-f12/cc-pvdz-f12_noscale': ,
                 # 'g03_pbepbe_6-311++g_d_p': ,
                 # 'fci/cc-pvdz': ,
                 # 'fci/cc-pvtz': ,
                 # 'fci/cc-pvqz': ,
                 # 'bmk/cbsb7': ,
                 # 'bmk/6-311g(2d,d,p)': ,
                 'b3lyp/6-31g**': 0.961,  # http://cccbdb.nist.gov/vibscalejust.asp
                 'b3lyp/6-311+g(3df,2p)': 0.967,  # http://cccbdb.nist.gov/vibscalejust.asp
                 'wb97x-d/aug-cc-pvtz': 0.974,
                 # Taken from https://comp.chem.umn.edu/freqscale/version3b2.htm as ωB97X-D/maug-cc-pVTZ
                 }
    scale_factor = freq_dict.get(model_chemistry.lower(), 1)
    if scale_factor == 1:
        logging.warning('No frequency scale factor found for model chemistry {0}; assuming a value of unity.'.format(
            model_chemistry))
    else:
        logging.info('Assigned a frequency scale factor of {0} for model chemistry {1}'.format(
            scale_factor, model_chemistry))
    return scale_factor


def determine_rotor_symmetry(energies, label, pivots):
    """
    Determine the rotor symmetry number from the potential scan given in :list:`energies` in J/mol units
    Assumes the list represents a 360 degree scan
    str:`label` is the species name, used for logging and error messages
    list:`pivots` are the rotor's pivots, used for logging and error messages
    The *worst* resolution for each peak and valley is determined.
    The first criterion for a symmetric rotor is that the highest peak and the lowest peak must be within the
    worst peak resolution (and the same is checked for valleys).
    A second criterion for a symmetric rotor is that the highest and lowest peaks must be within 10% of
    the highest peak value. This is only applied if the highest peak is above 2 kJ/mol.
    """
    symmetry = None
    min_e = min(energies)
    max_e = max(energies)
    if max_e > 2000:
        tol = 0.10 * max_e  # tolerance for the second criterion
    else:
        tol = max_e
    peaks, valleys = list(), [energies[0]]  # the peaks and valleys of the scan
    worst_peak_resolution, worst_valley_resolution = 0, max(energies[1] - energies[0], energies[-2] - energies[-1])
    for i, e in enumerate(energies):
        # identify peaks and valleys, and determine worst resolutions in the scan
        if i != 0 and i != len(energies) - 1:
            last_point = energies[i - 1]
            next_point = energies[i + 1]
            # this is an intermediate point in the scan
            if e > last_point and e > next_point:
                # this is a local peak
                if any([diff > worst_peak_resolution for diff in [e - last_point, e - next_point]]):
                    worst_peak_resolution = max(e - last_point, e - next_point)
                peaks.append(e)
            elif e < last_point and e < next_point:
                # this is a local valley
                if any([diff > worst_valley_resolution for diff in [energies[i - 1] - e, next_point - e]]):
                    worst_valley_resolution = max(last_point - e, next_point - e)
                valleys.append(e)
    # The number of peaks and valley must always be the same (what goes up must come down), if it isn't then there's
    # something seriously wrong with the scan
    if len(peaks) != len(valleys):
        raise InputError('Rotor of species {0} between pivots {1} does not have the same number'
                         ' of peaks and valleys.'.format(label, pivots))
    min_peak = min(peaks)
    max_peak = max(peaks)
    min_valley = min(valleys)
    max_valley = max(valleys)
    # Criterion 1: worst resolution
    if max_peak - min_peak > worst_peak_resolution:
        # The rotor cannot be symmetric
        symmetry = 1
        reason = 'worst peak resolution criterion'
    elif max_valley - min_valley > worst_valley_resolution:
        # The rotor cannot be symmetric
        symmetry = 1
        reason = 'worst valley resolution criterion'
    # Criterion 2: 10% * max_peak
    elif max_peak - min_peak > tol:
        # The rotor cannot be symmetric
        symmetry = 1
        reason = '10% of the maximum peak criterion'
    else:
        # We declare this rotor as symmetric and the symmetry number in the number of peaks (and valleys)
        symmetry = len(peaks)
        reason = 'number of peaks and valleys, all within the determined resolution criteria'
    if symmetry not in [1, 2, 3]:
        logging.warn('Determined symmetry number {0} for rotor of species {1} between pivots {2};'
                     ' you should make sure this makes sense'.format(symmetry, label, pivots))
    else:
        logging.info('Determined a symmetry number of {0} for rotor of species {1} between pivots {2}'
                     ' based on the {3}.'.format(symmetry, label, pivots, reason))
    return symmetry
