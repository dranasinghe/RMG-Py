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
Contains the :class:`SimpleReactor` class, providing a reaction system
consisting of a homogeneous, isothermal, isobaric batch reactor.
"""

import numpy, logging
cimport numpy

import itertools
    
from base cimport ReactionSystem
cimport cython

import rmgpy.constants as constants
cimport rmgpy.constants as constants
from rmgpy.quantity import Quantity
from rmgpy.quantity cimport ScalarQuantity, ArrayQuantity
from rmgpy.kinetics.arrhenius import Arrhenius

###############################################################################
def get_filterlist_of_all_RMG_families():
    """
    List of available reaction families in RMG. Has to be updated once a new family is added.
    In addition, an Arrhenius fit based on the max. rates from this families training reations 
    has to be added to get_uni_bimolecular_threshold_rate_constant and the corresponding position.
    """
    all_families = [
            'Intra_R_Add_Exocyclic', 'Cyclopentadiene_scission', '2+2_cycloaddition_CO',
                'R_Addition_CSm', 'Disproportionation', '1,2-Birad_to_alkene',
                'Intra_R_Add_Exo_scission', 'H2_Loss', '1,3_Insertion_ROR',
                'Baeyer-Villiger_step1_cat', 'Intra_RH_Add_Endocyclic',
                'Baeyer-Villiger_step2_cat', 'Korcek_step2', 'Singlet_Val6_to_triplet',
                'Intra_Retro_Diels_alder_bicyclic', 'R_Addition_MultipleBond',
                'Concerted_Intra_Diels_alder_monocyclic_1,2_shiftH', 'Cyclic_Thioether_Formation',
                'Intra_R_Add_Endocyclic', '1,3_Insertion_CO2', '1+2_Cycloaddition',
                'Bimolec_Hydroperoxide_Decomposition', 'Intra_R_Add_ExoTetCyclic',
                'Peroxyl_Termination', 'CO_Disproportionation', 'Intra_Disproportionation',
                'SubstitutionS', 'Korcek_step1', 'intra_substitutionS_cyclization',
                'Korcek_step1_cat', '1,4_Linear_birad_scission', '1,2_Insertion_carbene',
                'H_Abstraction', 'Intra_5_membered_conjugated_C=C_C=C_addition',
                'Intra_ene_reaction', 'intra_H_migration', 'Baeyer-Villiger_step2',
                '1,2_Insertion_CO', 'Substitution_O', 'Intra_RH_Add_Exocyclic',
                'Cyclic_Ether_Formation', '1,2_shiftC', 'lone_electron_pair_bond',
                'HO2_Elimination_from_PeroxyRadical', 'Birad_recombination',
                'Diels_alder_addition', 'R_Addition_COm', 'intra_substitutionCS_cyclization',
                '2+2_cycloaddition_CS','1,2_shiftS', 'intra_OH_migration',
                'Birad_R_Recombination','Singlet_Carbene_Intra_Disproportionation',
                '6_membered_central_C-C_shift','intra_substitutionCS_isomerization',
                '2+2_cycloaddition_CCO','intra_substitutionS_isomerization',
                '2+2_cycloaddition_Cd','Intra_2+2_cycloaddition_Cd',
                'Intra_Diels_alder_monocyclic', 'R_Recombination',
                '1,3_Insertion_RSR', '1,4_Cyclic_birad_scission',
                'intra_NO2_ONO_conversion','ketoenol', 'Peroxyl_Disproportionation'
                ]
    return all_families


def get_uni_bi_trimolecular_threshold_rate_constant(T):
    """
    Get the bimolecular threshold rate constants for reaction filtering.
    Using current 66 RMG reaction families.
    """

    # List of unimolecular kinetics
    unimol_kinetics_list = [
        Arrhenius(A=(2.1548e-15,'s^-1'), n=8.65061, Ea=(-122.419,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 20.5283, dn = +|- 0.383077, 
            dEa = +|- 2.61431 kJ/mol"""),
        Arrhenius(A=(2.51056e+17,'s^-1'), n=-1.48728, Ea=(95.6898,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 2.11795e-15, 
            dEa = +|- 1.4454e-14 kJ/mol"""),
        Arrhenius(A=(4.19097e+11,'s^-1'), n=0.542031, Ea=(48.0397,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 2.67845e-15, 
            dEa = +|- 1.82791e-14 kJ/mol"""),
        Arrhenius(A=(3.06643e+07,'s^-1'), n=4.05506, Ea=(364.729,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 3.71715, dn = +|- 0.166445, 
            dEa = +|- 1.1359 kJ/mol"""),
        [],
        Arrhenius(A=(1e+10,'s^-1'), n=-6.64137e-15, Ea=(6.11426e-14,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 1.69925e-15, 
            dEa = +|- 1.15965e-14 kJ/mol"""),
        Arrhenius(A=(7.809e+07,'s^-1'), n=1.057, Ea=(63.0152,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 1.62696e-15, 
            dEa = +|- 1.11032e-14 kJ/mol"""),
        Arrhenius(A=(4.82588e+09,'s^-1'), n=0.803687, Ea=(72.0667,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1.15661, dn = +|- 0.0184441,
            dEa = +|- 0.125872 kJ/mol"""),
        Arrhenius(A=(4.49138e+06,'s^-1'), n=3.19054, Ea=(123.447,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 3.08198e-15, 
            dEa = +|- 2.1033e-14 kJ/mol"""),
        [],
        [],
        [],
        [],
        Arrhenius(A=(2.69922e+09,'s^-1'), n=2.23108, Ea=(-23.2466,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 5.18462, dn = +|- 0.208626, 
            dEa = +|- 1.42377 kJ/mol"""),
        [],
        Arrhenius(A=(8.36742e-37,'s^-1'), n=15.6692, Ea=(-79.9819,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 37.755, dn = +|- 0.46032, 
            dEa = +|- 3.14146 kJ/mol"""),
        Arrhenius(A=(3.60572e+16,'s^-1'), n=-0.490221, Ea=(118.723,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 3.56237e-15, 
            dEa = +|- 2.43114e-14 kJ/mol"""),
        [],
        Arrhenius(A=(31.1508,'s^-1'), n=4.20415, Ea=(-67.1521,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 4.40734, dn = +|- 0.188036,
            dEa = +|- 1.28325 kJ/mol"""),
        Arrhenius(A=(22135.6,'s^-1'), n=3.10576, Ea=(287.824,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 4.16477, dn = +|- 0.180859,
            dEa = +|- 1.23428 kJ/mol"""),
        Arrhenius(A=(1.25061e+25,'s^-1'), n=-2.6694, Ea=(362.294,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1.10756, dn = +|- 0.0129508,
            dEa = +|- 0.088383 kJ/mol"""),
        [],
        [],
        [],
        [],
        Arrhenius(A=(1.949e+11,'s^-1'), n=0.486, Ea=(22.8614,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 3.78153e-15,
            dEa = +|- 2.58071e-14 kJ/mol"""),
        [],
        [],
        Arrhenius(A=(7.15311,'s^-1'), n=3.63158, Ea=(32.7853,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 51.476, dn = +|- 0.499619,
            dEa = +|- 3.40965 kJ/mol"""),
        [],
        [],
        Arrhenius(A=(9.09546e+17,'s^-1'), n=-0.747035, Ea=(447.041,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 1.42527, dn = +|- 0.0449229,
            dEa = +|- 0.306577 kJ/mol"""),
        [],
        Arrhenius(A=(9.86304e+10,'s^-1'), n=0.836047, Ea=(79.2187,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 1.74493e-15,
            dEa = +|- 1.19083e-14 kJ/mol"""),
        Arrhenius(A=(27.788,'s^-1'), n=3.56981, Ea=(17.5282,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 2.95262, dn = +|- 0.137254,
            dEa = +|- 0.936691 kJ/mol"""),
        Arrhenius(A=(32804.1,'s^-1'), n=2.27586, Ea=(-3.34049,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 5.22479, dn = +|- 0.209605,
            dEa = +|- 1.43045 kJ/mol"""),
        Arrhenius(A=(2.8594e+09,'s^-1'), n=1.09585, Ea=(90.8931,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 2.42346, dn = +|- 0.112217,
            dEa = +|- 0.765828 kJ/mol"""),
        Arrhenius(A=(3.33623e+08,'s^-1'), n=1.58566, Ea=(274.448,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 4.34887e-15,
            dEa = +|- 2.96789e-14 kJ/mol"""),
        [],
        [],
        Arrhenius(A=(0.0211122,'s^-1'), n=4.09341, Ea=(4.27861,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 33.1388, dn = +|- 0.443787,
            dEa = +|- 3.02863 kJ/mol"""),
        Arrhenius(A=(8.66e+11,'s^-1'), n=0.438, Ea=(94.4747,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 1.63603e-15,
            dEa = +|- 1.11651e-14 kJ/mol"""),
        [],
        Arrhenius(A=(21583.8,'s^-1'), n=2.90923, Ea=(106.401,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 6.91976, dn = +|- 0.245223,
            dEa = +|- 1.67353 kJ/mol"""),
        Arrhenius(A=(2.18e+16,'s^-1'), n=-1.0059e-14, Ea=(2.9288,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 5.97248e-15,
            dEa = +|- 4.07593e-14 kJ/mol"""),
        Arrhenius(A=(1.17242e-21,'s^-1'), n=11.0661, Ea=(17.1225,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 4.46687e-15,
            dEa = +|- 3.04842e-14 kJ/mol"""),
        Arrhenius(A=(1.5378e+14,'s^-1'), n=0.264564, Ea=(20.2142,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 3.09632e-15,
            dEa = +|- 2.11309e-14 kJ/mol"""),
        [],
        [],
        [],
        Arrhenius(A=(181.219,'s^-1'), n=2.3668, Ea=(50.9862,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 5.67183, dn = +|- 0.220012,
            dEa = +|- 1.50148 kJ/mol"""),
        Arrhenius(A=(9.40883e+22,'s^-1'), n=-0.830214, Ea=(159.721,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 5.2588e-15,
            dEa = +|- 3.58887e-14 kJ/mol"""),
        Arrhenius(A=(1.454e+12,'s^-1'), n=0.178, Ea=(0.85772,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 3.373e-15,
            dEa = +|- 2.30191e-14 kJ/mol"""),
        Arrhenius(A=(3.53521e+20,'s^-1'), n=-2.14941, Ea=(84.4898,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 2.0056e-15, 
            dEa = +|- 1.36872e-14 kJ/mol"""),
        Arrhenius(A=(607.614,'s^-1'), n=2.9594, Ea=(180.721,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 6.44969, dn = +|- 0.236305,
            dEa = +|- 1.61267 kJ/mol"""),
        [],
        Arrhenius(A=(2228.54,'s^-1'), n=2.59523, Ea=(79.3728,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 9.24147, dn = +|- 0.281901, 
            dEa = +|- 1.92383 kJ/mol"""),
        Arrhenius(A=(3.47729e+15,'s^-1'), n=-0.0133205, Ea=(253.025,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 3.42822e-15, 
            dEa = +|- 2.33959e-14 kJ/mol"""),
        Arrhenius(A=(0.0435533,'s^-1'), n=4.08745, Ea=(103.277,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 15.3826, dn = +|- 0.346495, 
            dEa = +|- 2.36466 kJ/mol"""),
        Arrhenius(A=(1.4544e+12,'s^-1'), n=0.301801, Ea=(-1.2548,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 1.47428e-15, 
            dEa = +|- 1.00612e-14 kJ/mol"""),
        Arrhenius(A=(6.85921e-07,'s^-1'), n=5.65681, Ea=(-64.6036,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 21.0499, dn = +|- 0.386257, 
            dEa = +|- 2.63602 kJ/mol"""),
        Arrhenius(A=(6.13062e+10,'s^-1'), n=2.21279, Ea=(128.085,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 2.304e-15, 
            dEa = +|- 1.57237e-14 kJ/mol"""),
        Arrhenius(A=(1.67986e+13,'s^-1'), n=0.420292, Ea=(21.9887,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 2.88394e-15, 
            dEa = +|- 1.96815e-14 kJ/mol"""),
        Arrhenius(A=(722095,'s^-1'), n=2.22801, Ea=(238.399,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 1.94083, dn = +|- 0.0840635, 
            dEa = +|- 0.573692 kJ/mol"""),
        Arrhenius(A=(104,'s^-1'), n=3.21, Ea=(82.0482,'kJ/mol'),
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'),
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 7.65184e-15, 
            dEa = +|- 5.22201e-14 kJ/mol"""),
        []
    ]
    
    # List of bimolecular kinetics
    bimol_kinetics_list = [
        [],
        [],
        Arrhenius(A=(2.319e-07,'m^3/(mol*s)'), n=3.416, Ea=(322.616,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 5.81451e-15, 
            dEa = +|- 3.96812e-14 kJ/mol"""),
        Arrhenius(A=(1.2e+07,'m^3/(mol*s)'), n=2.11, Ea=(10.2926,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 1.66533e-15, 
            dEa = +|- 1.13651e-14 kJ/mol"""),
        Arrhenius(A=(3.71358e-08,'m^3/(mol*s)'), n=4.90833, Ea=(-21.5849,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 8.35058, dn = +|- 0.26905, 
            dEa = +|- 1.83613 kJ/mol"""),
        [],
        [],
        Arrhenius(A=(477137,'m^3/(mol*s)'), n=2.9449, Ea=(-96.2703,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 3.48379e-15, 
            dEa = +|- 2.37752e-14 kJ/mol"""),
        Arrhenius(A=(4.86e-07,'m^3/(mol*s)'), n=3.55, Ea=(101.797,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 2.24939e-15, 
            dEa = +|- 1.5351e-14 kJ/mol"""),
        Arrhenius(A=(3.46333e+06,'m^3/(mol*s)'), n=-0.929161, Ea=(42.315,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 9.48759e-16, 
            dEa = +|- 6.47482e-15 kJ/mol"""),
        [],
        Arrhenius(A=(4.74858e-09,'m^3/(mol*s)'), n=4.24247, Ea=(83.3223,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 2.32974e-15, 
            dEa = +|- 1.58993e-14 kJ/mol"""),
        [],
        [],
        [],
        Arrhenius(A=(3.92165e-11,'m^3/(mol*s)'), n=5.41917, Ea=(-92.6718,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 17.0778, dn = +|- 0.359748, 
            dEa = +|- 2.4551 kJ/mol"""),
        [],
        [],
        [],
        Arrhenius(A=(1.33741,'m^3/(mol*s)'), n=2.15746, Ea=(304.387,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 4.07671, dn = +|- 0.17815, 
            dEa = +|- 1.21579 kJ/mol"""),
        Arrhenius(A=(8.19836e+06,'m^3/(mol*s)'), n=0.145041, Ea=(-3.96815,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1.50944, dn = +|- 0.0521967, 
            dEa = +|- 0.356217 kJ/mol"""),
        Arrhenius(A=(127965,'m^3/(mol*s)'), n=0.933342, Ea=(111.519,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 6.41962, dn = +|- 0.235713, 
            dEa = +|- 1.60862 kJ/mol"""),
        [],
        Arrhenius(A=(120000,'m^3/(mol*s)'), n=-1.31735e-14, Ea=(-4.184,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 1.94625e-15, 
            dEa = +|- 1.32822e-14 kJ/mol"""),
        Arrhenius(A=(1.2e+08,'m^3/(mol*s)'), n=-1.20033e-14, Ea=(1.11491e-13,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 3.09852e-15, 
            dEa = +|- 2.11459e-14 kJ/mol"""),
        [],
        Arrhenius(A=(0.000160472,'m^3/(mol*s)'), n=3.8024, Ea=(-10.9764,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 15.0279, dn = +|- 0.343538, 
            dEa = +|- 2.34448 kJ/mol"""),
        [],
        Arrhenius(A=(17580.9,'m^3/(mol*s)'), n=1.98548, Ea=(-106.532,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 4.80061e-15, 
            dEa = +|- 3.27618e-14 kJ/mol"""),
        [],
        [],
        Arrhenius(A=(51795.3,'m^3/(mol*s)'), n=0.681821, Ea=(-9.92113,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 2.01108, dn = +|- 0.0885713, 
            dEa = +|- 0.604456 kJ/mol"""),
        Arrhenius(A=(1.46107e-12,'m^3/(mol*s)'), n=4.73222, Ea=(-278.229,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 16576.3, dn = +|- 1.23167, 
            dEa = +|- 8.40556 kJ/mol"""),
        [],
        [],
        [],
        Arrhenius(A=(2.03275e-17,'m^3/(mol*s)'), n=6.6413, Ea=(205.734,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 6.88308e-15, 
            dEa = +|- 4.69737e-14 kJ/mol"""),
        Arrhenius(A=(1.27e-07,'m^3/(mol*s)'), n=3.7, Ea=(223.258,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 2.66187e-15, 
            dEa = +|- 1.81659e-14 kJ/mol"""),
        Arrhenius(A=(1.34785e-07,'m^3/(mol*s)'), n=4.7731, Ea=(-85.8887,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 2.54359e-15, 
            dEa = +|- 1.73587e-14 kJ/mol"""),
        [],
        Arrhenius(A=(8.18963e-31,'m^3/(mol*s)'), n=11.428, Ea=(-13.2083,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 140.706, dn = +|- 0.627094, 
            dEa = +|- 4.27961 kJ/mol"""),
        [],
        [],
        Arrhenius(A=(7.37936e-08,'m^3/(mol*s)'), n=3.90078, Ea=(20.1194,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 3.75901, dn = +|- 0.167864, 
            dEa = +|- 1.14559 kJ/mol"""),
        [],
        Arrhenius(A=(8.35312e-24,'m^3/(mol*s)'), n=9.17707, Ea=(-1.16914,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 29.3773, dn = +|- 0.428514, 
            dEa = +|- 2.9244 kJ/mol"""),
        Arrhenius(A=(0.0202143,'m^3/(mol*s)'), n=2.45362, Ea=(2.95866,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 3.00278, dn = +|- 0.13939, 
            dEa = +|- 0.951266 kJ/mol"""),
        [],
        [],
        [],
        [],
        Arrhenius(A=(1.29008e+06,'m^3/(mol*s)'), n=0.806257, Ea=(-5.31319,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 2.22228, dn = +|- 0.101231, 
            dEa = +|- 0.690853 kJ/mol"""),
        [],
        [],
        [],
        [],
        [],
        Arrhenius(A=(4.66,'m^3/(mol*s)'), n=1.65, Ea=(226.564,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 4.19081e-15, 
            dEa = +|- 2.86002e-14 kJ/mol"""),
        [],
        [],
        Arrhenius(A=(5.6874e+10,'m^3/(mol*s)'), n=-0.0206709, Ea=(-191.106,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 3.38094e-15, 
            dEa = +|- 2.30733e-14 kJ/mol"""),
        Arrhenius(A=(3.80823e-13,'m^3/(mol*s)'), n=5.9792, Ea=(141.605,'kJ/mol'), 
            T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
            comment="""Fitted to 30 data points; dA = *|/ 5.14419, dn = +|- 0.207634, 
            dEa = +|- 1.417 kJ/mol"""),
        [],
        [],
        [],
        Arrhenius(A=(1.1e+06,'m^3/(mol*s)'), n=-1.05506e-14, Ea=(-4.184,'kJ/mol'), 
             T0=(1,'K'), Tmin=(300,'K'), Tmax=(2500,'K'), 
             comment="""Fitted to 30 data points; dA = *|/ 1, dn = +|- 1.19083e-15, 
             dEa = +|- 8.12686e-15 kJ/mol""")
    ]
    
    # Evaluate kinetics at user defined temperature
    kvals_uni = []
    for k, unimol_kinetics in enumerate(unimol_kinetics_list):
        if unimol_kinetics:
            kvals_uni.append(unimol_kinetics.getRateCoefficient(T))
        else:
            kvals_uni.append(1e8)
    
    kvals_bi = []
    for k, bimol_kinetics in enumerate(bimol_kinetics_list):
        if bimol_kinetics:
            kvals_bi.append(bimol_kinetics.getRateCoefficient(T))
        else:
            kvals_bi.append(1e8)

    # List of all current RMG families
    all_families = get_filterlist_of_all_RMG_families() 

    # Generate dictionary with reaction families as keys and kinetics as values
    unimolecular_threshold_rate_constant = {
        key: value for key, value in zip(all_families, kvals_uni)
    }

    bimolecular_threshold_rate_constant = {
        key: value for key, value in zip(all_families, kvals_bi)
    }

    # Maximum trimolecular rate constants are approximately three
    # orders of magnitude smaller (accounting for the unit
    # conversion from m^3/mol/s to m^6/mol^2/s) based on
    # extending the Smoluchowski equation to three molecules
    trimolecular_threshold_rate_constant = {
        key: 1e-3*value for key, value in zip(all_families, kvals_bi)
    }

    return (unimolecular_threshold_rate_constant, 
            bimolecular_threshold_rate_constant,
            trimolecular_threshold_rate_constant
            )

###############################################################################
cdef class SimpleReactor(ReactionSystem):
    """
    A reaction system consisting of a homogeneous, isothermal, isobaric batch
    reactor. These assumptions allow for a number of optimizations that enable
    this solver to complete very rapidly, even for large kinetic models.
    """

    cdef public ScalarQuantity T
    cdef public ScalarQuantity P
    cdef public double V
    cdef public bint constantVolume
    cdef public dict initialMoleFractions

    # collider variables

    """
    pdepColliderKinetics:
    an array that contains a reference to the kinetics object of the reaction
    that has pressure dependent kinetics.
    """
    cdef public list pdepColliderKinetics

    """
    colliderEfficiencies:
    an array consisting of array elements, each element corresponding to a reaction.
    Each element is an array with each position in the array corresponding to the collider efficiency
    of the core species. The collider efficiency is set to 1 if the species was not found in the list
    of colliders.
    """
    cdef public numpy.ndarray colliderEfficiencies

    """
    pdepColliderReactionIndices: 
    array that contains the indices of those reactions that 
    have pressure dependent kinetics. E.g. [4, 10, 2, 123]
    """
    cdef public numpy.ndarray pdepColliderReactionIndices

    """
    pdepSpecificColliderKinetics:
    an array that contains a reference to the kinetics object of the reaction
    that has pressure dependent kinetics with a specific species as a third body collider.
    """
    cdef public list pdepSpecificColliderKinetics

    """
    specificColliderSpecies:
    a list that contains object references to species which are specific third body colliders
    in the respective reactions in pdepSpecificColliderReactionIndices.
    """
    cdef public list specificColliderSpecies

    """
    pdepSpecificColliderReactionIndices:
    an array that contains the indices of reactions that have
    a specifcCollider attribyte. E.g. [16, 155, 90]
    """
    cdef public numpy.ndarray pdepSpecificColliderReactionIndices
    
    cdef public dict sensConditions
    
    cdef public list Trange
    cdef public list Prange
    cdef public int nSims

    def __init__(self, T, P, initialMoleFractions, nSims=1, termination=None, sensitiveSpecies=None, sensitivityThreshold=1e-3,sensConditions=None):
        ReactionSystem.__init__(self, termination, sensitiveSpecies, sensitivityThreshold)
        
        
        if type(T) != list:
            self.T = Quantity(T)
        else:
            self.Trange = [Quantity(t) for t in T]
            
        if type(P) != list:
            self.P = Quantity(P)
        else:
            self.Prange = [Quantity(p) for p in P]
        
        self.initialMoleFractions = initialMoleFractions

        self.V = 0 # will be set in initializeModel
        self.constantVolume = False

        self.pdepColliderReactionIndices = None
        self.pdepColliderKinetics = None
        self.colliderEfficiencies = None
        self.pdepSpecificColliderReactionIndices = None
        self.pdepSpecificColliderKinetics = None
        self.specificColliderSpecies = None
        self.sensConditions = sensConditions
        self.nSims = nSims

    def __reduce__(self):
        """
        A helper function used when pickling an object.
        """
        return (self.__class__, 
            (self.T, self.P, self.initialMoleFractions, self.nSims, self.termination))


    def convertInitialKeysToSpeciesObjects(self, speciesDict):
        """
        Convert the initialMoleFractions dictionary from species names into species objects,
        using the given dictionary of species.
        """
        initialMoleFractions = {}
        for label, moleFrac in self.initialMoleFractions.iteritems():
            initialMoleFractions[speciesDict[label]] = moleFrac
        self.initialMoleFractions = initialMoleFractions

    cpdef initializeModel(self, list coreSpecies, list coreReactions, list edgeSpecies, list edgeReactions,
                          list surfaceSpecies=None, list surfaceReactions=None, list pdepNetworks=None,
                          atol=1e-16, rtol=1e-8, sensitivity=False, sens_atol=1e-6, sens_rtol=1e-4,
                          filterReactions=False, dict conditions=None):
        """
        Initialize a simulation of the simple reactor using the provided kinetic
        model.
        """
        
        if surfaceSpecies is None:
            surfaceSpecies = []
        if surfaceReactions is None:
            surfaceReactions = []
        
        
        # First call the base class version of the method
        # This initializes the attributes declared in the base class
        ReactionSystem.initializeModel(self, coreSpecies=coreSpecies, coreReactions=coreReactions, edgeSpecies=edgeSpecies,
                                       edgeReactions=edgeReactions, surfaceSpecies=surfaceSpecies, surfaceReactions=surfaceReactions,
                                       pdepNetworks=pdepNetworks, atol=atol, rtol=rtol, sensitivity=sensitivity, sens_atol=sens_atol,
                                       sens_rtol=sens_rtol, filterReactions=filterReactions, conditions=conditions)

        # Set initial conditions
        self.set_initial_conditions()

        # Compute reaction thresholds if reaction filtering is turned on
        if filterReactions:
            ReactionSystem.set_initial_reaction_thresholds(self)
        
        self.set_colliders(coreReactions, edgeReactions, coreSpecies)
        
        ReactionSystem.compute_network_variables(self, pdepNetworks)

        # Generate forward and reverse rate coefficients k(T,P)
        self.generate_rate_coefficients(coreReactions, edgeReactions)
        
        ReactionSystem.set_initial_derivative(self)
        # Initialize the model
        ReactionSystem.initialize_solver(self)

    def calculate_effective_pressure(self, rxn):
        """
        Computes the effective pressure for a reaction as:

        .. math:: P_{eff} = P * \\sum_i \\frac{y_i * eff_i}{\\sum_j y_j}

        with:
            - P the pressure of the reactor,
            - y the array of initial moles of the core species

        or as:

        .. math:: P_{eff} = \\frac{P * y_{specificCollider}}{\\sum_j y_j}

        if a specificCollider is mentioned.
        """

        y0_coreSpecies = self.y0[:self.numCoreSpecies]
        sum_core_species = numpy.sum(y0_coreSpecies)
        
        j = self.reactionIndex[rxn]
        for i in xrange(self.pdepColliderReactionIndices.shape[0]):
            if j == self.pdepColliderReactionIndices[i]:
                # Calculate effective pressure
                if rxn.specificCollider is None:
                    Peff = self.P.value_si * numpy.sum(self.colliderEfficiencies[i]*y0_coreSpecies / sum_core_species)
                else:
                    logging.debug("Calculating Peff using {0} as a specificCollider".format(rxn.specificCollider))
                    Peff = self.P.value_si * self.y0[self.speciesIndex[rxn.specificCollider]] / sum_core_species
                return Peff
        return self.P.value_si

    def generate_rate_coefficients(self, coreReactions, edgeReactions):
        """
        Populates the forward rate coefficients (kf), reverse rate coefficients (kb)
        and equilibrium constants (Keq) arrays with the values computed at the temperature
        and (effective) pressure of the reaction system.
        """

        for rxn in itertools.chain(coreReactions, edgeReactions):
            j = self.reactionIndex[rxn]
            Peff = self.calculate_effective_pressure(rxn)
            self.kf[j] = rxn.getRateCoefficient(self.T.value_si, Peff)

            if rxn.reversible:
                self.Keq[j] = rxn.getEquilibriumConstant(self.T.value_si)
                self.kb[j] = self.kf[j] / self.Keq[j]
                
    def get_threshold_rate_constants(self):
        """
        Get the threshold rate constants for reaction filtering.
        """
        # Set the maximum uni-/bi-/trimolecular rate by using custom rate constant thresholds for
        # each reaction family
        (unimolecular_threshold_rate_constant, 
                bimolecular_threshold_rate_constant,
                trimolecular_threshold_rate_constant) = get_uni_bi_trimolecular_threshold_rate_constant(self.T.value_si) 

        return (unimolecular_threshold_rate_constant,
                bimolecular_threshold_rate_constant,
                trimolecular_threshold_rate_constant)

    def set_colliders(self, coreReactions, edgeReactions, coreSpecies):
        """
        Store collider efficiencies and reaction indices for pdep reactions that have collider efficiencies,
        and store specific collider indices
        """
        pdepColliderReactionIndices = []
        self.pdepColliderKinetics = []
        colliderEfficiencies = []
        pdepSpecificColliderReactionIndices = []
        self.pdepSpecificColliderKinetics = []
        self.specificColliderSpecies = []

        for rxn in itertools.chain(coreReactions, edgeReactions):
            if rxn.kinetics.isPressureDependent():
                if rxn.kinetics.efficiencies:
                    j = self.reactionIndex[rxn]
                    pdepColliderReactionIndices.append(j)
                    self.pdepColliderKinetics.append(rxn.kinetics)
                    colliderEfficiencies.append(rxn.kinetics.getEffectiveColliderEfficiencies(coreSpecies))
                if rxn.specificCollider:
                    pdepSpecificColliderReactionIndices.append(self.reactionIndex[rxn])
                    self.pdepSpecificColliderKinetics.append(rxn.kinetics)
                    self.specificColliderSpecies.append(rxn.specificCollider)

        self.pdepColliderReactionIndices = numpy.array(pdepColliderReactionIndices, numpy.int)
        self.colliderEfficiencies = numpy.array(colliderEfficiencies, numpy.float64)
        self.pdepSpecificColliderReactionIndices = numpy.array(pdepSpecificColliderReactionIndices, numpy.int)


    def set_initial_conditions(self):
        """
        Sets the initial conditions of the rate equations that represent the 
        current reactor model.

        The volume is set to the value derived from the ideal gas law, using the 
        user-defined pressure, temperature, and the number of moles of initial species.

        The species moles array (y0) is set to the values stored in the
        initial mole fractions dictionary.

        The initial species concentration is computed and stored in the
        coreSpeciesConcentrations array.

        """

        ReactionSystem.set_initial_conditions(self)

        for spec, moleFrac in self.initialMoleFractions.iteritems():
            i = self.get_species_index(spec)
            self.y0[i] = moleFrac
        
        # Use ideal gas law to compute volume
        self.V = constants.R * self.T.value_si * numpy.sum(self.y0[:self.numCoreSpecies]) / self.P.value_si# volume in m^3
        for j in xrange(self.numCoreSpecies):
            self.coreSpeciesConcentrations[j] = self.y0[j] / self.V

    @cython.boundscheck(False)
    def residual(self, double t, numpy.ndarray[numpy.float64_t, ndim=1] y, numpy.ndarray[numpy.float64_t, ndim=1] dydt, numpy.ndarray[numpy.float64_t, ndim=1] senpar = numpy.zeros(1, numpy.float64)):

        """
        Return the residual function for the governing DAE system for the
        simple reaction system.
        """
        cdef numpy.ndarray[numpy.int_t, ndim=2] ir, ip, inet
        cdef numpy.ndarray[numpy.float64_t, ndim=1] res, kf, kr, knet, delta, equilibriumConstants
        cdef int numCoreSpecies, numCoreReactions, numEdgeSpecies, numEdgeReactions, numPdepNetworks
        cdef int i, j, z, first, second, third
        cdef double k, V, reactionRate, revReactionRate, T, P, Peff
        cdef numpy.ndarray[numpy.float64_t, ndim=1] coreSpeciesConcentrations, coreSpeciesRates, coreReactionRates, edgeSpeciesRates, edgeReactionRates, networkLeakRates, coreSpeciesConsumptionRates, coreSpeciesProductionRates
        cdef numpy.ndarray[numpy.float64_t, ndim=1] C, y_coreSpecies
        cdef numpy.ndarray[numpy.float64_t, ndim=2] jacobian, dgdk, colliderEfficiencies
        cdef numpy.ndarray[numpy.int_t, ndim=1] pdepColliderReactionIndices, pdepSpecificColliderReactionIndices
        cdef list pdepColliderKinetics, pdepSpecificColliderKinetics

        ir = self.reactantIndices
        ip = self.productIndices
        
        numCoreSpecies = len(self.coreSpeciesRates)
        numCoreReactions = len(self.coreReactionRates)
        numEdgeSpecies = len(self.edgeSpeciesRates)
        numEdgeReactions = len(self.edgeReactionRates)
        numPdepNetworks = len(self.networkLeakRates)
        kf = self.kf
        kr = self.kb
        
        y_coreSpecies = y[:numCoreSpecies]
        
        # Recalculate any forward and reverse rate coefficients that involve pdep collision efficiencies
        if self.pdepColliderReactionIndices.shape[0] != 0:
            T = self.T.value_si
            P = self.P.value_si
            equilibriumConstants = self.Keq
            pdepColliderReactionIndices = self.pdepColliderReactionIndices
            pdepColliderKinetics = self.pdepColliderKinetics
            colliderEfficiencies = self.colliderEfficiencies
            for i in xrange(pdepColliderReactionIndices.shape[0]):
                # Calculate effective pressure
                Peff = P*numpy.sum(colliderEfficiencies[i]*y_coreSpecies / numpy.sum(y_coreSpecies))
                j = pdepColliderReactionIndices[i]
                kf[j] = pdepColliderKinetics[i].getRateCoefficient(T, Peff)
                kr[j] = kf[j] / equilibriumConstants[j]
        if self.pdepSpecificColliderReactionIndices.shape[0] != 0:
            T = self.T.value_si
            P = self.P.value_si
            equilibriumConstants = self.Keq
            pdepSpecificColliderReactionIndices = self.pdepSpecificColliderReactionIndices
            pdepSpecificColliderKinetics = self.pdepSpecificColliderKinetics
            specificColliderSpecies = self.specificColliderSpecies
            for i in xrange(pdepSpecificColliderReactionIndices.shape[0]):
                # Calculate effective pressure
                Peff = P * y[self.speciesIndex[specificColliderSpecies[i]]] / numpy.sum(y_coreSpecies)
                j = pdepSpecificColliderReactionIndices[i]
                kf[j] = pdepSpecificColliderKinetics[i].getRateCoefficient(T, Peff)
                kr[j] = kf[j] / equilibriumConstants[j]
            
        inet = self.networkIndices
        knet = self.networkLeakCoefficients
        
        
        res = numpy.zeros(numCoreSpecies, numpy.float64)

        coreSpeciesConcentrations = numpy.zeros_like(self.coreSpeciesConcentrations)
        coreSpeciesRates = numpy.zeros_like(self.coreSpeciesRates)
        coreReactionRates = numpy.zeros_like(self.coreReactionRates)
        coreSpeciesConsumptionRates = numpy.zeros_like(self.coreSpeciesConsumptionRates)
        coreSpeciesProductionRates = numpy.zeros_like(self.coreSpeciesProductionRates)
        edgeSpeciesRates = numpy.zeros_like(self.edgeSpeciesRates)
        edgeReactionRates = numpy.zeros_like(self.edgeReactionRates)
        networkLeakRates = numpy.zeros_like(self.networkLeakRates)

        C = numpy.zeros_like(self.coreSpeciesConcentrations)
        
        # Use ideal gas law to compute volume
        V = constants.R * self.T.value_si * numpy.sum(y_coreSpecies) / self.P.value_si
        self.V = V

        for j in xrange(numCoreSpecies):
            C[j] = y[j] / V
            coreSpeciesConcentrations[j] = C[j]
        
        for j in xrange(ir.shape[0]):
            k = kf[j]
            if ir[j,0] >= numCoreSpecies or ir[j,1] >= numCoreSpecies or ir[j,2] >= numCoreSpecies:
                fReactionRate = 0.0
            elif ir[j,1] == -1: # only one reactant
                fReactionRate = k * C[ir[j,0]]
            elif ir[j,2] == -1: # only two reactants
                fReactionRate = k * C[ir[j,0]] * C[ir[j,1]]
            else: # three reactants!! (really?)
                fReactionRate = k * C[ir[j,0]] * C[ir[j,1]] * C[ir[j,2]]
            k = kr[j]
            if ip[j,0] >= numCoreSpecies or ip[j,1] >= numCoreSpecies or ip[j,2] >= numCoreSpecies:
                revReactionRate = 0.0
            elif ip[j,1] == -1: # only one reactant
                revReactionRate = k * C[ip[j,0]]
            elif ip[j,2] == -1: # only two reactants
                revReactionRate = k * C[ip[j,0]] * C[ip[j,1]]
            else: # three reactants!! (really?)
                revReactionRate = k * C[ip[j,0]] * C[ip[j,1]] * C[ip[j,2]]
                
            reactionRate = fReactionRate-revReactionRate
            
            # Set the reaction and species rates
            if j < numCoreReactions:
                # The reaction is a core reaction
                coreReactionRates[j] = reactionRate

                # Add/substract the total reaction rate from each species rate
                # Since it's a core reaction we know that all of its reactants
                # and products are core species
                first = ir[j,0]
                coreSpeciesRates[first] -= reactionRate
                coreSpeciesConsumptionRates[first] += fReactionRate
                coreSpeciesProductionRates[first] += revReactionRate
                second = ir[j,1]
                if second != -1:
                    coreSpeciesRates[second] -= reactionRate
                    coreSpeciesConsumptionRates[second] += fReactionRate
                    coreSpeciesProductionRates[second] += revReactionRate
                    third = ir[j,2]
                    if third != -1:
                        coreSpeciesRates[third] -= reactionRate
                        coreSpeciesConsumptionRates[third] += fReactionRate
                        coreSpeciesProductionRates[third] += revReactionRate
                first = ip[j,0]
                coreSpeciesRates[first] += reactionRate
                coreSpeciesProductionRates[first] += fReactionRate
                coreSpeciesConsumptionRates[first] += revReactionRate
                second = ip[j,1]
                if second != -1:
                    coreSpeciesRates[second] += reactionRate
                    coreSpeciesProductionRates[second] += fReactionRate
                    coreSpeciesConsumptionRates[second] += revReactionRate
                    third = ip[j,2]
                    if third != -1:
                        coreSpeciesRates[third] += reactionRate
                        coreSpeciesProductionRates[third] += fReactionRate
                        coreSpeciesConsumptionRates[third] += revReactionRate

            else:
                # The reaction is an edge reaction
                edgeReactionRates[j-numCoreReactions] = reactionRate

                # Add/substract the total reaction rate from each species rate
                # Since it's an edge reaction its reactants and products could
                # be either core or edge species
                # We're only interested in the edge species
                first = ir[j,0]
                if first >= numCoreSpecies: edgeSpeciesRates[first-numCoreSpecies] -= reactionRate
                second = ir[j,1]
                if second != -1:
                    if second >= numCoreSpecies: edgeSpeciesRates[second-numCoreSpecies] -= reactionRate
                    third = ir[j,2]
                    if third != -1:
                        if third >= numCoreSpecies: edgeSpeciesRates[third-numCoreSpecies] -= reactionRate
                first = ip[j,0]
                if first >= numCoreSpecies: edgeSpeciesRates[first-numCoreSpecies] += reactionRate
                second = ip[j,1]
                if second != -1:
                    if second >= numCoreSpecies: edgeSpeciesRates[second-numCoreSpecies] += reactionRate
                    third = ip[j,2]
                    if third != -1:
                        if third >= numCoreSpecies: edgeSpeciesRates[third-numCoreSpecies] += reactionRate

        for j in xrange(inet.shape[0]):
            k = knet[j]
            if inet[j,1] == -1: # only one reactant
                reactionRate = k * C[inet[j,0]]
            elif inet[j,2] == -1: # only two reactants
                reactionRate = k * C[inet[j,0]] * C[inet[j,1]]
            else: # three reactants!! (really?)
                reactionRate = k * C[inet[j,0]] * C[inet[j,1]] * C[inet[j,2]]
            networkLeakRates[j] = reactionRate

        self.coreSpeciesConcentrations = coreSpeciesConcentrations
        self.coreSpeciesRates = coreSpeciesRates
        self.coreSpeciesProductionRates = coreSpeciesProductionRates
        self.coreSpeciesConsumptionRates = coreSpeciesConsumptionRates
        self.coreReactionRates = coreReactionRates
        self.edgeSpeciesRates = edgeSpeciesRates
        self.edgeReactionRates = edgeReactionRates
        self.networkLeakRates = networkLeakRates

        res = coreSpeciesRates * V 
        
        
        if self.sensitivity:
            delta = numpy.zeros(len(y), numpy.float64)
            delta[:numCoreSpecies] = res
            if self.jacobianMatrix is None:
                jacobian = self.jacobian(t,y,dydt,0,senpar)
            else:
                jacobian = self.jacobianMatrix
            dgdk = ReactionSystem.computeRateDerivative(self)
            for j in xrange(numCoreReactions+numCoreSpecies):
                for i in xrange(numCoreSpecies):
                    for z in xrange(numCoreSpecies):
                        delta[(j+1)*numCoreSpecies + i] += jacobian[i,z]*y[(j+1)*numCoreSpecies + z] 
                    delta[(j+1)*numCoreSpecies + i] += dgdk[i,j]

        else:
            delta = res
        delta = delta - dydt
        
        # Return DELTA, IRES.  IRES is set to 1 in order to tell DASPK to evaluate the sensitivity residuals
        return delta, 1
    
    @cython.boundscheck(False)
    def jacobian(self, double t, numpy.ndarray[numpy.float64_t, ndim=1] y, numpy.ndarray[numpy.float64_t, ndim=1] dydt, double cj, numpy.ndarray[numpy.float64_t, ndim=1] senpar = numpy.zeros(1, numpy.float64)):
        """
        Return the analytical Jacobian for the reaction system.
        """
        cdef numpy.ndarray[numpy.int_t, ndim=2] ir, ip
        cdef numpy.ndarray[numpy.float64_t, ndim=1] kf, kr, C
        cdef numpy.ndarray[numpy.float64_t, ndim=2] pd
        cdef int numCoreReactions, numCoreSpecies, i, j
        cdef double k, V, Ctot, deriv, corr
        
        ir = self.reactantIndices
        ip = self.productIndices

        kf = self.kf
        kr = self.kb
        numCoreReactions = len(self.coreReactionRates)
        numCoreSpecies = len(self.coreSpeciesConcentrations)
        
        pd = -cj * numpy.identity(numCoreSpecies, numpy.float64)
        
        V = constants.R * self.T.value_si * numpy.sum(y[:numCoreSpecies]) / self.P.value_si
        
        Ctot = self.P.value_si /(constants.R * self.T.value_si)

        C = numpy.zeros_like(self.coreSpeciesConcentrations)
        for j in xrange(numCoreSpecies):
            C[j] = y[j] / V

        for j in xrange(numCoreReactions):
           
            k = kf[j]
            if ir[j,1] == -1: # only one reactant
                deriv = k
                pd[ir[j,0], ir[j,0]] -= deriv
                
                pd[ip[j,0], ir[j,0]] += deriv                
                if ip[j,1] != -1:
                    pd[ip[j,1], ir[j,0]] += deriv
                    if ip[j,2] != -1:
                        pd[ip[j,2], ir[j,0]] += deriv
                
                                
            elif ir[j,2] == -1: # only two reactants
                corr = - k * C[ir[j,0]] * C[ir[j,1]] / Ctot
                if ir[j,0] == ir[j,1]:  # reactants are the same
                    deriv = 2 * k * C[ir[j,0]]
                    pd[ir[j,0], ir[j,0]] -= 2 * deriv
                    for i in xrange(numCoreSpecies):
                        pd[ir[j,0], i] -= 2 * corr
                    
                    pd[ip[j,0], ir[j,0]] += deriv                       
                    for i in xrange(numCoreSpecies):
                        pd[ip[j,0], i] += corr    
                    if ip[j,1] != -1:
                        pd[ip[j,1], ir[j,0]] += deriv                                               
                        for i in xrange(numCoreSpecies):
                            pd[ip[j,1], i] += corr    
                        if ip[j,2] != -1:
                            pd[ip[j,2], ir[j,0]] += deriv                                          
                            for i in xrange(numCoreSpecies):
                                pd[ip[j,2], i] += corr    
                    
                else:
                    # Derivative with respect to reactant 1
                    deriv = k * C[ir[j, 1]]
                    pd[ir[j,0], ir[j,0]] -= deriv                    
                    pd[ir[j,1], ir[j,0]] -= deriv                        
                    
                    pd[ip[j,0], ir[j,0]] += deriv       
                    if ip[j,1] != -1:
                        pd[ip[j,1], ir[j,0]] += deriv
                        if ip[j,2] != -1:
                            pd[ip[j,2], ir[j,0]] += deriv
                    
                    # Derivative with respect to reactant 2
                    deriv = k * C[ir[j, 0]] 
                    pd[ir[j,0], ir[j,1]] -= deriv                    
                    pd[ir[j,1], ir[j,1]] -= deriv                                           
                    for i in xrange(numCoreSpecies):
                        pd[ir[j,0], i] -= corr
                        pd[ir[j,1], i] -= corr     
                            
                    pd[ip[j,0], ir[j,1]] += deriv                       
                    for i in xrange(numCoreSpecies):
                        pd[ip[j,0], i] += corr    
                    if ip[j,1] != -1:
                        pd[ip[j,1], ir[j,1]] += deriv                                               
                        for i in xrange(numCoreSpecies):
                            pd[ip[j,1], i] += corr    
                        if ip[j,2] != -1:
                            pd[ip[j,2], ir[j,1]] += deriv                                          
                            for i in xrange(numCoreSpecies):
                                pd[ip[j,2], i] += corr               
                    
                    
            else: # three reactants!! (really?)
                corr = - 2* k * C[ir[j,0]] * C[ir[j,1]] * C[ir[j,2]] / Ctot
                if (ir[j,0] == ir[j,1] & ir[j,0] == ir[j,2]):
                    deriv = 3 * k * C[ir[j,0]] * C[ir[j,0]] 
                    pd[ir[j,0], ir[j,0]] -= 3 * deriv                                                           
                    for i in xrange(numCoreSpecies):
                        pd[ir[j,0], i] -= 3 * corr
                    
                    pd[ip[j,0], ir[j,0]] += deriv                       
                    for i in xrange(numCoreSpecies):
                        pd[ip[j,0], i] += corr    
                    if ip[j,1] != -1:
                        pd[ip[j,1], ir[j,0]] += deriv                                               
                        for i in xrange(numCoreSpecies):
                            pd[ip[j,1], i] += corr    
                        if ip[j,2] != -1:
                            pd[ip[j,2], ir[j,0]] += deriv                                          
                            for i in xrange(numCoreSpecies):
                                pd[ip[j,2], i] += corr        
                    
                elif ir[j,0] == ir[j,1]:
                    # derivative with respect to reactant 1
                    deriv = 2 * k * C[ir[j,0]] * C[ir[j,2]]
                    pd[ir[j,0], ir[j,0]] -= 2 * deriv                  
                    pd[ir[j,2], ir[j,0]] -= deriv    
                    
                    pd[ip[j,0], ir[j,0]] += deriv       
                    if ip[j,1] != -1:
                        pd[ip[j,1], ir[j,0]] += deriv
                        if ip[j,2] != -1:
                            pd[ip[j,2], ir[j,0]] += deriv
                    
                    # derivative with respect to reactant 3
                    deriv = k * C[ir[j,0]] * C[ir[j,0]] 
                    pd[ir[j,0], ir[j,2]] -= 2 * deriv                  
                    pd[ir[j,2], ir[j,2]] -= deriv                                                                                           
                    for i in xrange(numCoreSpecies):
                        pd[ir[j,0], i] -= 2 * corr
                        pd[ir[j,2], i] -= corr
                        
                    pd[ip[j,0], ir[j,2]] += deriv                       
                    for i in xrange(numCoreSpecies):
                        pd[ip[j,0], i] += corr    
                    if ip[j,1] != -1:
                        pd[ip[j,1], ir[j,2]] += deriv                                               
                        for i in xrange(numCoreSpecies):
                            pd[ip[j,1], i] += corr    
                        if ip[j,2] != -1:
                            pd[ip[j,2], ir[j,2]] += deriv                                          
                            for i in xrange(numCoreSpecies):
                                pd[ip[j,2], i] += corr    
                    
                    
                elif ir[j,1] == ir[j,2]:                    
                    # derivative with respect to reactant 1
                    deriv = k * C[ir[j,1]] * C[ir[j,1]] 
                    pd[ir[j,0], ir[j,0]] -= deriv                    
                    pd[ir[j,1], ir[j,0]] -= 2 * deriv
                    
                    pd[ip[j,0], ir[j,0]] += deriv       
                    if ip[j,1] != -1:
                        pd[ip[j,1], ir[j,0]] += deriv
                        if ip[j,2] != -1:
                            pd[ip[j,2], ir[j,0]] += deriv  
                    # derivative with respect to reactant 2
                    deriv = 2 * k * C[ir[j,0]] * C[ir[j,1]]
                    pd[ir[j,0], ir[j,1]] -= deriv                    
                    pd[ir[j,1], ir[j,1]] -= 2 * deriv                                                                                                         
                    for i in xrange(numCoreSpecies):
                        pd[ir[j,0], i] -= corr
                        pd[ir[j,1], i] -= 2 * corr

                    pd[ip[j,0], ir[j,1]] += deriv                       
                    for i in xrange(numCoreSpecies):
                        pd[ip[j,0], i] += corr    
                    if ip[j,1] != -1:
                        pd[ip[j,1], ir[j,1]] += deriv                                               
                        for i in xrange(numCoreSpecies):
                            pd[ip[j,1], i] += corr    
                        if ip[j,2] != -1:
                            pd[ip[j,2], ir[j,1]] += deriv                                          
                            for i in xrange(numCoreSpecies):
                                pd[ip[j,2], i] += corr     
                
                elif ir[j,0] == ir[j,2]:                    
                    # derivative with respect to reactant 1
                    deriv = 2 * k * C[ir[j,0]] * C[ir[j,1]]
                    pd[ir[j,0], ir[j,0]] -= 2 * deriv                  
                    pd[ir[j,1], ir[j,0]] -= deriv    
                    
                    pd[ip[j,0], ir[j,0]] += deriv       
                    if ip[j,1] != -1:
                        pd[ip[j,1], ir[j,0]] += deriv
                        if ip[j,2] != -1:
                            pd[ip[j,2], ir[j,0]] += deriv
                    # derivative with respect to reactant 2
                    deriv = k * C[ir[j,0]] * C[ir[j,0]] 
                    pd[ir[j,0], ir[j,1]] -= 2 * deriv                    
                    pd[ir[j,1], ir[j,1]] -= deriv                                                                                                         
                    for i in xrange(numCoreSpecies):
                        pd[ir[j,0], i] -= 2 * corr
                        pd[ir[j,1], i] -= corr

                    pd[ip[j,0], ir[j,1]] += deriv                       
                    for i in xrange(numCoreSpecies):
                        pd[ip[j,0], i] += corr    
                    if ip[j,1] != -1:
                        pd[ip[j,1], ir[j,1]] += deriv                                               
                        for i in xrange(numCoreSpecies):
                            pd[ip[j,1], i] += corr    
                        if ip[j,2] != -1:
                            pd[ip[j,2], ir[j,1]] += deriv                                          
                            for i in xrange(numCoreSpecies):
                                pd[ip[j,2], i] += corr     
                                
                else:
                    # derivative with respect to reactant 1
                    deriv = k * C[ir[j,1]] * C[ir[j,2]]
                    pd[ir[j,0], ir[j,0]] -= deriv                    
                    pd[ir[j,1], ir[j,0]] -= deriv
                    pd[ir[j,2], ir[j,0]] -= deriv
                    
                    pd[ip[j,0], ir[j,0]] += deriv       
                    if ip[j,1] != -1:
                        pd[ip[j,1], ir[j,0]] += deriv
                        if ip[j,2] != -1:
                            pd[ip[j,2], ir[j,0]] += deriv     
                                    
                    # derivative with respect to reactant 2
                    deriv = k * C[ir[j,0]] * C[ir[j,2]]
                    pd[ir[j,0], ir[j,1]] -= deriv                    
                    pd[ir[j,1], ir[j,1]] -= deriv   
                    pd[ir[j,2], ir[j,1]] -= deriv
                    
                    pd[ip[j,0], ir[j,1]] += deriv       
                    if ip[j,1] != -1:
                        pd[ip[j,1], ir[j,1]] += deriv
                        if ip[j,2] != -1:
                            pd[ip[j,2], ir[j,1]] += deriv 
                                 
                    # derivative with respect to reactant 3
                    deriv = k * C[ir[j,0]] * C[ir[j,1]]             
                    pd[ir[j,0], ir[j,2]] -= deriv                    
                    pd[ir[j,1], ir[j,2]] -= deriv   
                    pd[ir[j,2], ir[j,2]] -= deriv
                    for i in xrange(numCoreSpecies):
                        pd[ir[j,0], i] -= corr
                        pd[ir[j,1], i] -= corr
                        pd[ir[j,2], i] -= corr
                        
                    pd[ip[j,0], ir[j,2]] += deriv                       
                    for i in xrange(numCoreSpecies):
                        pd[ip[j,0], i] += corr    
                    if ip[j,1] != -1:
                        pd[ip[j,1], ir[j,2]] += deriv                                               
                        for i in xrange(numCoreSpecies):
                            pd[ip[j,1], i] += corr    
                        if ip[j,2] != -1:
                            pd[ip[j,2], ir[j,2]] += deriv                                          
                            for i in xrange(numCoreSpecies):
                                pd[ip[j,2], i] += corr     
                    
            
            
            k = kr[j]         
            if ip[j,1] == -1: # only one reactant
                deriv = k
                pd[ip[j,0], ip[j,0]] -= deriv
                
                pd[ir[j,0], ip[j,0]] += deriv                
                if ir[j,1] != -1:
                    pd[ir[j,1], ip[j,0]] += deriv
                    if ir[j,2] != -1:
                        pd[ir[j,2], ip[j,0]] += deriv
                
                                
            elif ip[j,2] == -1: # only two reactants
                corr = -k * C[ip[j,0]] * C[ip[j,1]] / Ctot
                if ip[j,0] == ip[j,1]:
                    deriv = 2 * k * C[ip[j,0]] 
                    pd[ip[j,0], ip[j,0]] -= 2 * deriv                 
                    for i in xrange(numCoreSpecies):
                        pd[ip[j,0], i] -= 2 * corr
                        
                    pd[ir[j,0], ip[j,0]] += deriv                
                    for i in xrange(numCoreSpecies):
                        pd[ir[j,0], i] += corr   
                    if ir[j,1] != -1:
                        pd[ir[j,1], ip[j,0]] += deriv          
                        for i in xrange(numCoreSpecies):
                            pd[ir[j,1], i] += corr   
                        if ir[j,2] != -1:
                            pd[ir[j,2], ip[j,0]] += deriv  
                            for i in xrange(numCoreSpecies):
                                pd[ir[j,2], i] += corr   
                    
                else:
                    # Derivative with respect to reactant 1
                    deriv = k * C[ip[j, 1]]
                    pd[ip[j,0], ip[j,0]] -= deriv                    
                    pd[ip[j,1], ip[j,0]] -= deriv
                    
                    pd[ir[j,0], ip[j,0]] += deriv       
                    if ir[j,1] != -1:
                        pd[ir[j,1], ip[j,0]] += deriv
                        if ir[j,2] != -1:
                            pd[ir[j,2], ip[j,0]] += deriv
                    
                    # Derivative with respect to reactant 2
                    deriv = k * C[ip[j, 0]] 
                    pd[ip[j,0], ip[j,1]] -= deriv                    
                    pd[ip[j,1], ip[j,1]] -= deriv              
                    for i in xrange(numCoreSpecies):
                        pd[ip[j,0], i] -= corr
                        pd[ip[j,1], i] -= corr
                      
                    pd[ir[j,0], ip[j,1]] += deriv                
                    for i in xrange(numCoreSpecies):
                         pd[ir[j,0], i] += corr   
                    if ir[j,1] != -1:
                        pd[ir[j,1], ip[j,1]] += deriv          
                        for i in xrange(numCoreSpecies):
                            pd[ir[j,1], i] += corr   
                        if ir[j,2] != -1:
                            pd[ir[j,2], ip[j,1]] += deriv  
                            for i in xrange(numCoreSpecies):
                                pd[ir[j,2], i] += corr              
                    
                    
            else: # three reactants!! (really?)
                corr = - 2 * k * C[ip[j,0]] * C[ip[j,1]] * C[ip[j,2]] / Ctot
                if (ip[j,0] == ip[j,1] & ip[j,0] == ip[j,2]):
                    deriv = 3 * k * C[ip[j,0]] * C[ip[j,0]] 
                    pd[ip[j,0], ip[j,0]] -= 3 * deriv          
                    for i in xrange(numCoreSpecies):
                        pd[ip[j,0], i] -= 3 * corr
                    
                    pd[ir[j,0], ip[j,0]] += deriv                
                    for i in xrange(numCoreSpecies):
                        pd[ir[j,0], i] += corr   
                    if ir[j,1] != -1:
                        pd[ir[j,1], ip[j,0]] += deriv          
                        for i in xrange(numCoreSpecies):
                            pd[ir[j,1], i] += corr   
                        if ir[j,2] != -1:
                            pd[ir[j,2], ip[j,0]] += deriv  
                            for i in xrange(numCoreSpecies):
                                pd[ir[j,2], i] += corr       
                    
                elif ip[j,0] == ip[j,1]:
                    # derivative with respect to reactant 1
                    deriv = 2 * k * C[ip[j,0]] * C[ip[j,2]] 
                    pd[ip[j,0], ip[j,0]] -= 2 * deriv                    
                    pd[ip[j,2], ip[j,0]] -= deriv
                    
                    pd[ir[j,0], ip[j,0]] += deriv       
                    if ir[j,1] != -1:
                        pd[ir[j,1], ip[j,0]] += deriv
                        if ir[j,2] != -1:
                            pd[ir[j,2], ip[j,0]] += deriv
                    # derivative with respect to reactant 3
                    deriv = k * C[ip[j,0]] * C[ip[j,0]] 
                    pd[ip[j,0], ip[j,2]] -= 2 * deriv                    
                    pd[ip[j,2], ip[j,2]] -= deriv                       
                    for i in xrange(numCoreSpecies):
                        pd[ip[j,0], i] -= 2 * corr
                        pd[ip[j,2], i] -= corr

                    pd[ir[j,0], ip[j,2]] += deriv                
                    for i in xrange(numCoreSpecies):
                        pd[ir[j,0], i] += corr   
                    if ir[j,1] != -1:
                        pd[ir[j,1], ip[j,2]] += deriv          
                        for i in xrange(numCoreSpecies):
                            pd[ir[j,1], i] += corr   
                        if ir[j,2] != -1:
                            pd[ir[j,2], ip[j,2]] += deriv  
                            for i in xrange(numCoreSpecies):
                                pd[ir[j,2], i] += corr     
                    
                    
                elif ip[j,1] == ip[j,2]:                    
                    # derivative with respect to reactant 1
                    deriv = k * C[ip[j,1]] * C[ip[j,1]] 
                    pd[ip[j,0], ip[j,0]] -= deriv                    
                    pd[ip[j,1], ip[j,0]] -= 2 * deriv
                    
                    pd[ir[j,0], ip[j,0]] += deriv       
                    if ir[j,1] != -1:
                        pd[ir[j,1], ip[j,0]] += deriv
                        if ir[j,2] != -1:
                            pd[ir[j,2], ip[j,0]] += deriv                 
                    
                    # derivative with respect to reactant 2
                    deriv = 2 * k * C[ip[j,0]] * C[ip[j,1]] 
                    pd[ip[j,0], ip[j,1]] -= deriv                    
                    pd[ip[j,1], ip[j,1]] -= 2 * deriv   
                    for i in xrange(numCoreSpecies):
                        pd[ip[j,0], i] -= corr
                        pd[ip[j,1], i] -= 2 * corr
                        
                    pd[ir[j,0], ip[j,1]] += deriv                
                    for i in xrange(numCoreSpecies):
                        pd[ir[j,0], i] += corr   
                    if ir[j,1] != -1:
                        pd[ir[j,1], ip[j,1]] += deriv          
                        for i in xrange(numCoreSpecies):
                            pd[ir[j,1], i] += corr   
                        if ir[j,2] != -1:
                            pd[ir[j,2], ip[j,1]] += deriv  
                            for i in xrange(numCoreSpecies):
                                pd[ir[j,2], i] += corr                    
                                
                elif ip[j,0] == ip[j,2]:                    
                    # derivative with respect to reactant 1
                    deriv = 2 * k * C[ip[j,0]] * C[ip[j,1]]
                    pd[ip[j,0], ip[j,0]] -= 2 * deriv                  
                    pd[ip[j,1], ip[j,0]] -= deriv    
                    
                    pd[ir[j,0], ip[j,0]] += deriv       
                    if ir[j,1] != -1:
                        pd[ir[j,1], ip[j,0]] += deriv
                        if ir[j,2] != -1:
                            pd[ir[j,2], ip[j,0]] += deriv
                    # derivative with respect to reactant 2
                    deriv = k * C[ip[j,0]] * C[ip[j,0]] 
                    pd[ip[j,0], ip[j,1]] -= 2 * deriv                    
                    pd[ip[j,1], ip[j,1]] -= deriv                                                                                                         
                    for i in xrange(numCoreSpecies):
                        pd[ip[j,0], i] -= 2 * corr
                        pd[ip[j,1], i] -= corr

                    pd[ir[j,0], ip[j,1]] += deriv                       
                    for i in xrange(numCoreSpecies):
                        pd[ir[j,0], i] += corr    
                    if ir[j,1] != -1:
                        pd[ir[j,1], ip[j,1]] += deriv                                               
                        for i in xrange(numCoreSpecies):
                            pd[ir[j,1], i] += corr    
                        if ir[j,2] != -1:
                            pd[ir[j,2], ip[j,1]] += deriv                                          
                            for i in xrange(numCoreSpecies):
                                pd[ir[j,2], i] += corr     
                                
                else:
                    # derivative with respect to reactant 1
                    deriv = k * C[ip[j,1]] * C[ip[j,2]] 
                    pd[ip[j,0], ip[j,0]] -= deriv                    
                    pd[ip[j,1], ip[j,0]] -= deriv
                    pd[ip[j,2], ip[j,0]] -= deriv
                    
                    pd[ir[j,0], ip[j,0]] += deriv       
                    if ir[j,1] != -1:
                        pd[ir[j,1], ip[j,0]] += deriv
                        if ir[j,2] != -1:
                            pd[ir[j,2], ip[j,0]] += deriv     
                                    
                    # derivative with respect to reactant 2
                    deriv = k * C[ip[j,0]] * C[ip[j,2]] 
                    pd[ip[j,0], ip[j,1]] -= deriv                    
                    pd[ip[j,1], ip[j,1]] -= deriv   
                    pd[ip[j,2], ip[j,1]] -= deriv
                    
                    pd[ir[j,0], ip[j,1]] += deriv       
                    if ir[j,1] != -1:
                        pd[ir[j,1], ip[j,1]] += deriv
                        if ir[j,2] != -1:
                            pd[ir[j,2], ip[j,1]] += deriv 
                                 
                    # derivative with respect to reactant 3
                    deriv = k * C[ip[j,0]] * C[ip[j,1]] 
                    pd[ip[j,0], ip[j,2]] -= deriv                    
                    pd[ip[j,1], ip[j,2]] -= deriv   
                    pd[ip[j,2], ip[j,2]] -= deriv 
                    for i in xrange(numCoreSpecies):
                        pd[ip[j,0], i] -= corr
                        pd[ip[j,1], i] -= corr
                        pd[ip[j,2], i] -= corr
                    
                    pd[ir[j,0], ip[j,2]] += deriv                
                    for i in xrange(numCoreSpecies):
                        pd[ir[j,0], i] += corr   
                    if ir[j,1] != -1:
                        pd[ir[j,1], ip[j,2]] += deriv          
                        for i in xrange(numCoreSpecies):
                            pd[ir[j,1], i] += corr   
                        if ir[j,2] != -1:
                            pd[ir[j,2], ip[j,2]] += deriv  
                            for i in xrange(numCoreSpecies):
                                pd[ir[j,2], i] += corr  

        self.jacobianMatrix = pd + cj * numpy.identity(numCoreSpecies, numpy.float64)
        return pd
