from enum import Enum, IntEnum
from typing import List, Optional, Tuple
from numpy.typing import NDArray
import numpy as np
from pathlib import Path
from .constants import (
    HELICOID_WAVELENGTHS,
    LN10,
    SPECTRA_UCL_NIR_DATA_DICT,
    SPECTRA_CREATIS_DATA,
    WAVELENGTHS,
)
from dataclasses import dataclass
from .wavelength_selection.enums import SampleType


class MoleculeMode(Enum):
    ALL = 0
    OXCCO2_WATER = 1
    OXCCO2 = 2
    OXCCO1 = 3


REFERENCE_SPECTRUM_T1 = {
    MoleculeMode.ALL: [20.0, 0.08],
    MoleculeMode.OXCCO2_WATER: [66.66666667, 3.8],
}


class MoleculeIndex(IntEnum):
    HBO2 = 0
    HB = 1
    COXA = 2
    CREDA = 3
    C_OXY = 4
    C_RED = 5
    B_OXY = 6
    B_RED = 7
    WATER = 8
    FAT = 9


@dataclass
class HelicoidMolecules:
    hbo2_f: NDArray[np.float64]
    """Absorption coefficients from CREATIS data"""
    hb_f: NDArray[np.float64]
    """Absorption coefficients from CREATIS data"""
    coxa: NDArray[np.float64]
    """Absorption coefficients from CREATIS data"""
    creda: NDArray[np.float64]
    """Absorption coefficients from CREATIS data"""
    c_oxy: NDArray[np.float64]
    """Absorption coefficients from UCL NIR data"""
    c_red: NDArray[np.float64]
    """Absorption coefficients from UCL NIR data"""
    b_oxy: NDArray[np.float64]
    """Absorption coefficients from UCL NIR data"""
    b_red: NDArray[np.float64]
    """Absorption coefficients from UCL NIR data"""
    water: NDArray[np.float64]
    """Absorption coefficients from CREATIS data"""
    fat: NDArray[np.float64]
    """Absorption coefficients from CREATIS data"""


@dataclass
class DatasetWavelengths:
    ucl_nir_wl: NDArray[np.float64]
    """Wavelengths from the UCL NIR data"""
    creatis_wl: NDArray[np.float64]
    """Wavelengths from the CREATIS data"""


class Molecules:
    def __init__(
        self,
        left_cut=510,
        right_cut=900,
        molecule_mode: MoleculeMode = MoleculeMode.ALL,
        sample_type: SampleType = SampleType.BIOPSY2,
    ):
        self.molecule_mode = molecule_mode
        self.left_cut = left_cut
        self.right_cut = right_cut

        if sample_type == SampleType.HELICOID:
            wavelength_array = HELICOID_WAVELENGTHS
        else:
            wavelength_array = WAVELENGTHS

        self.left_cut_index = np.where(wavelength_array >= self.left_cut)[0][0]
        right_cut_indices = np.where(wavelength_array > self.right_cut)[0]

        # handle right cut differently since upper bound is exclusive while lower bound is inclusive
        if len(right_cut_indices) == 0:
            self.right_cut_index = len(wavelength_array)  # set to the end if no cut
        else:
            self.right_cut_index = right_cut_indices[0]
        self.cut_wavelengths = wavelength_array[self.left_cut_index : self.right_cut_index]

        self.molecules, self.dataset_wavelengths = self._load_molecule_data()
        self.M = self._construct_M()

    def _load_molecule_data(self) -> Tuple[HelicoidMolecules, DatasetWavelengths]:
        """Load molecule absorption data along with the corresponding wavelengths from the CREATIS and UCL NIR datasets

        Returns:
            Tuple[HelicoidMolecules, DatasetWavelengths]: molecule absorption data and wavelengths of the datasets
        """
        # read cyto b and c
        ucl_nir_mols, ucl_nir_wavelengths = self._read_molecules_cytochrome_cb(
            left_cut=self.left_cut,
            right_cut=self.right_cut,
            x_waves=self.cut_wavelengths,
        )
        (c_oxy, c_red, b_oxy, b_red) = ucl_nir_mols

        # read rest from creatis
        creatis_mols, creatis_wavelengths = self._read_molecules_creatis(
            left_cut=self.left_cut,
            right_cut=self.right_cut,
            x_waves=self.cut_wavelengths,
        )
        hb_f, hbo2_f, coxa, creda, fat, water = creatis_mols

        molecules = HelicoidMolecules(
            hbo2_f=np.asarray(hbo2_f),
            hb_f=np.asarray(hb_f),
            coxa=np.asarray(coxa),
            creda=np.asarray(creda),
            c_oxy=np.asarray(c_oxy),
            c_red=np.asarray(c_red),
            b_oxy=np.asarray(b_oxy),
            b_red=np.asarray(b_red),
            water=np.asarray(water),
            fat=np.asarray(fat),
        )
        wavelengths = DatasetWavelengths(
            ucl_nir_wl=ucl_nir_wavelengths,
            creatis_wl=creatis_wavelengths,
        )
        return molecules, wavelengths

    def _read_molecules_cytochrome_cb(
        self, left_cut: float, right_cut: float, x_waves: Optional[NDArray] = None
    ) -> Tuple[List[NDArray[np.float64]], NDArray[np.float64]]:
        """
        Read spectra for molecules Cytochrome-b & Cytochrome-c.
        Performs interpolation if x_waves is provided


        Args:
            left_cut (float): left cut for considered wavelengths
            right_cut (float): right cut for considered wavelengths
            x_waves ([type], optional): [Selected (x) wavelengths]. Defaults to None.

        Returns:
            Tuple[List[np.array], np.array]: Tuple of List of absorption coeffs per molecule (1) and wavelengths (2)
        """
        # read spectra for: cytochrome oxydised/reduced, oxyhemoglobin, hemoglobin, water, fat
        mol_list = ["cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red"]

        x, y = {}, {}
        for i in mol_list:
            x[i], y[i] = self._read_spectra(SPECTRA_UCL_NIR_DATA_DICT[i])

        # from extinction to absorption
        y_list = ["cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red"]

        for i in y_list:
            y[i] *= LN10

        # cutting all spectra to the range [left_cut, right_cut] nm
        x_new = x["cytoc_oxy"][
            (x["cytoc_oxy"] >= left_cut) & (x["cytoc_oxy"] <= right_cut)
        ]

        for i in mol_list:
            y[i] = self._cut_spectra(x[i], y[i], left_cut, right_cut)

        if x_waves is not None:
            x_new, y = self._wave_interpolation(x_new, y, mol_list, x_waves)

        return [y[i] for i in mol_list], x_new

    def _construct_M(self):
        """
        Construct absorption matrix M from molecule data depending on molecule mode
        """
        match self.molecule_mode:
            case MoleculeMode.ALL:
                return np.transpose(
                    np.vstack(
                        (
                            np.asarray(self.molecules.hbo2_f),
                            np.asarray(self.molecules.hb_f),
                            np.asarray(self.molecules.coxa),
                            np.asarray(self.molecules.creda),
                            np.asarray(self.molecules.c_oxy),
                            np.asarray(self.molecules.c_red),
                            np.asarray(self.molecules.b_oxy),
                            np.asarray(self.molecules.b_red),
                            np.asarray(self.molecules.water),
                            np.asarray(self.molecules.fat),
                        )
                    )
                )
            case MoleculeMode.OXCCO2_WATER:
                return np.transpose(
                    np.vstack(
                        (
                            np.asarray(self.molecules.hbo2_f),
                            np.asarray(self.molecules.hb_f),
                            np.asarray(self.molecules.coxa),
                            np.asarray(self.molecules.creda),
                            np.asarray(self.molecules.water),
                            np.asarray(self.molecules.fat),
                        )
                    )
                )
            case MoleculeMode.OXCCO2:
                return np.transpose(
                    np.vstack(
                        (
                            np.asarray(self.molecules.hbo2_f),
                            np.asarray(self.molecules.hb_f),
                            np.asarray(self.molecules.coxa),
                            np.asarray(self.molecules.creda),
                            np.asarray(self.molecules.fat),
                        )
                    )
                )
            case MoleculeMode.OXCCO1:
                return np.transpose(
                    np.vstack(
                        (
                            np.asarray(self.molecules.hbo2_f),
                            np.asarray(self.molecules.hb_f),
                            np.asarray(self.molecules.coxa),
                            np.asarray(self.molecules.creda),
                        )
                    )
                )

    def _read_spectra(self, file: Path) -> Tuple[np.array, np.array]:
        """Read UCL_NIR spectra for a molecule from file

        Args:
            file (Path): file that contains the data

        Returns:
            Tuple[np.array, np.array]: Tuple of wavelength and corresponding ?extinction? coefficients
        """
        with open(file, "r") as data:
            x, y = [], []
            for line in data:
                p = line.split()
                if not p[0] == "\x00":
                    x.append(float(p[0]))
                    y.append(float(p[1]))
        return np.array(x), np.array(y)

    def _cut_spectra(self, x, y, left_cut, right_cut):
        """
        cuts off spectrogram according to cut values left_cut and right_cut
        """
        # print(x)
        ix_left = np.where(x >= left_cut)[0][0]
        ix_right = np.where(x >= right_cut)[0][0]
        # print("ix_left", ix_left)
        # print("ix_right", ix_right)
        return y[ix_left : ix_right + 1]

    def _wave_interpolation(self, x, y, mol_list, x_waves):
        """interpolate spectrogram values according to x_waves

        Args:
            x (_type_): _description_
            y (_type_): _description_
            mol_list (_type_): _description_
            x_waves (_type_): _description_

        Returns:
            _type_: _description_
        """
        lower_bound, upper_bound = x[0], x[-1]
        new_x = np.asarray([i for i in x_waves if lower_bound <= i <= upper_bound])
        new_y = {}
        for i in mol_list:
            new_y[i] = np.interp(new_x, x, y[i])

        return new_x, new_y

    def _read_molecules_creatis(
        self, left_cut, right_cut, x_waves=None
    ) -> Tuple[List[NDArray[np.float64]], NDArray[np.float64]]:
        """
        Read molecules from CREATIS data (range 400 - 1000nm).
        Performs interpolation if x_waves is provided

        Args:
            left_cut (float): left cut for considered wavelengths
            right_cut (float): right cut for considered wavelengths
            x_waves ([type], optional): [Selected (x) wavelengths]. Defaults to None.

        Returns:
            Tuple[List[np.array], np.array]: Tuple of List of absorption coeffs [Hb, Hbo2, oxCCO, redCCO, fat, water] per molecule (1) and wavelengths (2)
        """
        mol_list = [
            "eps_Hb",
            "eps_HbO2",
            "eps_oxCCO",
            "eps_redCCO",
            "mua_Fat",
            "mua_H2O",
        ]
        wavelengths = self._read_spectra_creatis(SPECTRA_CREATIS_DATA / "lambda.txt")

        y = {}
        for mol in mol_list:
            y[mol] = self._read_spectra_creatis(SPECTRA_CREATIS_DATA / f"{mol}.txt")

        # convert from extinction to absorption
        extinction_list = ["eps_Hb", "eps_HbO2", "eps_oxCCO", "eps_redCCO"]

        for i in extinction_list:
            y[i] *= LN10
            y[i] /= 1000  # to cm^-1 / mM (millimole)

        # cutting all spectra to the range [left_cut, right_cut] nm
        x_new = wavelengths[(wavelengths >= left_cut) & (wavelengths <= right_cut)]
        # print(len(x_new))
        for i in mol_list:
            y[i] = y[i][(wavelengths >= left_cut) & (wavelengths <= right_cut)]
            y[i][y[i] < 0] = 0

        if x_waves is not None:
            x_new, y = self._wave_interpolation(x_new, y, mol_list, x_waves)
        # print(len(x_new))

        # return as list
        return [y[i] for i in mol_list], x_new

    def _read_spectra_creatis(self, file_name: Path):
        """Read CREATIS spectra for molecule from file

        Args:
            file_name (Path): filepath

        Returns:
            np.array: spectra
        """
        with open(file_name, "r") as file:
            return np.array(file.read().split(), dtype=float)
