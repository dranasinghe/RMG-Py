from yaml import load, dump
from yaml import CLoader as Loader, CDumper as Dumper
from copy import deepcopy
import os

from rmgpy.molecule import Molecule
from rmgpy import settings


class BenchmarkEntry:
    """
    A class for storing info related to a molecule for QM benchmarking
    """
    def __init__(self, name=None, smiles=None, inchi=None, index=None, categories=None, multiplicity=None, charge=None,
                 rmg_symmetry=None, expt_sources=None, preferred_expt_source=None, ff_xyz=None, dft_xyzs=None,
                 dft_freqs=None, yaml_file=None, qm_files=None, dft_thermo=None):
        """

        :param name: Name of the species
        :param smiles: SMILES string of the species
        :param inchi: InChi string of molecule
        :param index: Species index in the benchmarking set
        :param categories: list of categories (e.g. aromatics) that the species belongs to
        :param multiplicity: Spin multiplicity (2S+1)
        :param charge: The total net charge on the molecule
        :param rmg_symmetry: The symmetry number of the molecule as calculated by rmg
        :param expt_sources: A list of sources with experimental source objects
        :param preferred_expt_source: The chosen experimental data used for benchmarking this species
        :param ff_xyz: The xyz coordinates of the lowest energy conformer from force fields (as a string)
        :param dft_xyzs: A dictionary with the DFT method string as keys and xyz geometry as values
        :param dft_freqs: A dictionary with the DFT method string as keys and frequencies as values
        :param yaml_file: The relative location of the yaml file (used to store the data of this object)
        :param qm_files: A dictionary for the mapping {str(file_description):str(file_path)}
        :param dft_thermo: A dictionary mapping Arkane model chemistry strings to rmgpy thermo objects
        """
        self.name = name
        self.smiles = smiles
        self.inchi = inchi
        self.index = index
        self.categories = categories or []
        self.multiplicity = multiplicity
        self.charge = charge
        self.rmg_symmetry = rmg_symmetry
        self.expt_sources = expt_sources or []
        self.preferred_expt_source = preferred_expt_source
        self.ff_xyz = ff_xyz
        self.dft_xyzs = dft_xyzs or {}
        self.dft_freqs = dft_freqs or {}
        self.yaml_file = yaml_file
        self.qm_files = qm_files or {}
        self.dft_thermo = dft_thermo or {}

    def save_to_yaml(self, path=None):
        """
        Save the benchmark entry to a YAML file
        :param path: The relative location of the YAML file (if not specified self.yaml_file is used)
        :return: None
        """
        if path is None:
            if self.yaml_file is None:
                raise ValueError('YAML path not specified for BenchmarkEntry {0}'.format(self.name))
            else:
                path = self.yaml_file

        # Make parent directories if they don't exist
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, 'w') as f:
            # Dump the species name at the top of the file
            f.write('name: {}\n'.format(self.name))

            # Dump the remaining attributes
            remaining_attributes = deepcopy(self.__dict__)
            _ = remaining_attributes.pop('name')
            dump(remaining_attributes, f, Dumper=Dumper)

    def load_from_yaml(self, path=None):
        if path is None:
            if self.yaml_file is None:
                raise ValueError('YAML path not specified for BenchmarkEntry {0}'.format(self.name))
            else:
                path = self.yaml_file

        with open(path, 'r') as f:
            attributes = load(f, Loader=Loader)
            attributes['yaml_file'] = path

        self.__init__(**attributes)


if __name__ == '__main__':
    pass
