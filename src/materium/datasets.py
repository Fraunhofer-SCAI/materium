from collections import Counter
from functools import partial
import json
from typing import Dict, List, Optional
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from dataclasses import dataclass
from pymatgen.core import Lattice, Structure, Element, Composition
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.analysis.local_env import CrystalNN
import os
import re
from tqdm.contrib.concurrent import process_map
from materium.tokenizer import CrystalTokenizer


class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean.to(values.device)) / (
            self.std.to(values.device) + self.epsilon
        )

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        return values * self.std.to(values.device) + self.mean.to(values.device)


_symbol_to_int_map: dict[str, int] = {}

for spg_data in SpaceGroup.SYMM_OPS:
    sg_number = spg_data["number"]

    symbols_to_add = []
    if "hermann_mauguin" in spg_data:
        symbols_to_add.append(re.sub(r"\s", "", spg_data["hermann_mauguin"]))
    if "universal_h_m" in spg_data:
        symbols_to_add.append(re.sub(r"\s", "", spg_data["universal_h_m"]))
    if "hermann_mauguin_u" in spg_data:
        symbols_to_add.append(re.sub(r"\s", "", spg_data["hermann_mauguin_u"]))
    if "short_h_m" in spg_data:
        symbols_to_add.append(re.sub(r"\s", "", spg_data["short_h_m"]))

    for symbol in symbols_to_add:
        if symbol:
            _symbol_to_int_map[symbol] = sg_number

for symbol, data in SpaceGroup.sg_encoding.items():
    cleaned_symbol = re.sub(r"\s", "", symbol)
    if cleaned_symbol:
        _symbol_to_int_map[cleaned_symbol] = data["int_number"]


def get_spacegroup_int_number_fast(symbol_string: str) -> int:
    """
    Gets the international spacegroup number (1-230) from a spacegroup symbol
    string using pre-computed internal pymatgen data for fast lookup.

    Args:
        symbol_string: The spacegroup symbol string (e.g., "Pm-3m", "Pmmm", "P 2/m").
                       Whitespace is ignored. Case-sensitive.

    Returns:
        The integer international spacegroup number (1-230).

    Raises:
        ValueError: If the symbol string is not found in the internal mappings.
    """
    cleaned_symbol = re.sub(r"\s", "", symbol_string)

    try:
        return _symbol_to_int_map[cleaned_symbol]
    except KeyError:
        print(f"Invalid or unrecognized spacegroup symbol: {symbol_string!r}")
        return _symbol_to_int_map["Pm-3m"]


class MatterGenDataset(Dataset):
    """
    Dataset class for loading crystal structure data stored in separate
    files as used in MatterGen or similar setups (e.g., CDVAE).

    Loads data from a directory containing:
    - pos.npy: Cartesian coordinates (N_total_atoms, 3)
    - cell.npy: Lattice vectors for each structure (N_structures, 3, 3)
    - atomic_numbers.npy: Atomic numbers for each atom (N_total_atoms,)
    - num_atoms.npy: Number of atoms in each structure (N_structures,)
    - structure_id.npy: Identifier for each structure (N_structures,) [Optional]
    - Other property files (e.g., dft_band_gap.json) (N_structures,) [Optional]
    """

    def __init__(
        self,
        data_path: str,
        cutoff_radius: float = 6.0,
        transform=None,
        cache_id: Optional[str] = None,
        recalculate_cache: bool = False,
        lattice_scaler: Optional[StandardScaler] = None,
    ):
        """
        Args:
            data_path (str): Path to the directory containing the data files
                             (e.g., 'alex_mp_20/train').
            cutoff_radius (float): Cutoff radius for neighbor finding (graph edges).
            transform (callable, optional): Optional transform to apply to samples.
            cache_id (str, optional): Identifier for caching graph data. If None,
                                      defaults to the basename of data_path.
        """
        self.data_path = data_path
        self.cutoff_radius = cutoff_radius
        self.transform = transform
        self.cache_id = (
            cache_id if cache_id is not None else os.path.basename(data_path)
        )
        if not self.cache_id:
            self.cache_id = os.path.basename(os.path.dirname(data_path))

        self.lattice_scaler = lattice_scaler
        print(f"Initializing MatterGenDataset from: {self.data_path}")
        print(f"Cache ID: {self.cache_id}")

        try:
            self.pos_all = np.load(os.path.join(self.data_path, "pos.npy"))
            self.cell_all = np.load(os.path.join(self.data_path, "cell.npy"))

            self.atomic_numbers_all = np.load(
                os.path.join(self.data_path, "atomic_numbers.npy")
            )
            self.num_atoms_per_structure = np.load(
                os.path.join(self.data_path, "num_atoms.npy")
            ).astype(int)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Core data file not found in {self.data_path}: {e}"
            )

        self.num_structures = len(self.num_atoms_per_structure)
        if not self.num_structures == self.cell_all.shape[0]:
            raise ValueError(
                "Mismatch between number of structures in num_atoms.npy and cell.npy"
            )
        if (
            not self.atomic_numbers_all.shape[0]
            == self.pos_all.shape[0]
            == self.num_atoms_per_structure.sum()
        ):
            raise ValueError(
                "Mismatch in total number of atoms across pos.npy, atomic_numbers.npy, and sum(num_atoms.npy)"
            )

        self.cumulative_atoms = np.concatenate(
            ([0], np.cumsum(self.num_atoms_per_structure[:-1]))
        )

        self.structure_ids = self._load_optional_npy("structure_id.npy")
        self.band_gaps, self.band_gaps_mask = self._load_optional_json(
            os.path.join(self.data_path, "dft_band_gap.json"), default_value=0.0
        )
        self.band_gaps = np.log(np.array(self.band_gaps) + 1e-4).tolist()
        self.space_groups, self.space_groups_mask = self._load_optional_json(
            os.path.join(self.data_path, "space_group.json"), default_value="P1"
        )

        self.hhi, _ = self._load_optional_json(
            os.path.join(self.data_path, "hhi_score.json"), default_value=-1
        )
        self.hhi = (np.array(self.hhi) / 1000.0).tolist()
        self.dft_bulk_modulus, self.dft_bulk_modulus_mask = self._load_optional_json(
            os.path.join(self.data_path, "dft_bulk_modulus.json"), default_value=0.0
        )
        self.dft_bulk_modulus = (np.array(self.dft_bulk_modulus) / 100.0).tolist()
        self.dft_mag_density, self.dft_mag_density_mask = self._load_optional_json(
            os.path.join(self.data_path, "dft_mag_density.json"), default_value=0.0
        )
        self.dft_mag_density = np.log(
            np.clip(self.dft_mag_density, 0, np.max(self.dft_mag_density)) + 1e-5
        )
        m = np.isnan(self.dft_mag_density)
        self.dft_mag_density[m] = np.log(1e-5)
        self.dft_mag_density = self.dft_mag_density.tolist()
        self.space_groups = [
            get_spacegroup_int_number_fast(g) for g in self.space_groups
        ]

        # bulk_modulus >= 4.0 array(['mp-999545', 'mp-49'], dtype='<U18')
        # self.dft_mag_density >= np.log(0.2 + 1e-5) 'mp-155', 'mp-19981', 'mp-570087','mp-1221857', 'mp-753618', 'mp-761563', 'mp-600576', 'mp-1224869'
        # bandgap >= log(2.0) 'mp-656097', 'mp-676', 'mp-557719', ..., 'mp-989531', 'mp-989532','mp-995225'
        # hhi <= 1.25 'mp-1016277', 'mp-1079382', 'mp-1023510'
        # Add more optional properties here if needed
        # self.energies_above_hull = self._load_optional_json('energy_above_hull.json', default_value=float('nan'))

        print(
            f"Dataset {self.cache_id} initialized with {self.num_structures} structures."
        )

    def _load_optional_npy(self, filename: str) -> Optional[np.ndarray]:
        """Loads an optional numpy file, returns None if not found."""
        filepath = os.path.join(self.data_path, filename)
        print(f"Loading {filepath}")
        if os.path.exists(filepath):
            try:
                data = np.load(filepath)
                if len(data) != self.num_structures:
                    print(
                        f"Warning: Length mismatch for optional file {filename} ({len(data)}) vs num_structures ({self.num_structures}). Ignoring."
                    )
                    return None
                return data
            except Exception as e:
                print(f"Warning: Error loading {filepath}: {e}. Ignoring.")
                return None
        return None

    def _load_optional_json(self, filename: str, default_value=None) -> Optional[List]:
        """Loads an optional json file, returns None if not found, fills missing with default."""
        filepath = os.path.join(self.data_path, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)["values"]

                if len(data) != self.num_structures:
                    print(
                        f"Warning: Length mismatch for optional file {filename} ({len(data)}) vs num_structures ({self.num_structures}). Ignoring."
                    )
                    return None

                processed_data = []
                processed_mask = []
                for item in data:
                    v = default_value
                    v_mask = True
                    if type(item) == str or (item is not None and not np.isnan(item)):
                        v = item
                        v_mask = False

                    processed_mask.append(v_mask)
                    processed_data.append(v)

                return processed_data, processed_mask
            except Exception as e:
                print(
                    f"Warning: Error loading or processing {filepath}: {e}. Ignoring."
                )
                return None
        return None

    def __len__(self) -> int:
        return self.num_structures

    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieves a single data sample as a dictionary of tensors.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if not 0 <= idx < self.num_structures:
            raise IndexError(
                f"Index {idx} out of bounds for dataset with size {self.num_structures}"
            )

        start_atom_idx = self.cumulative_atoms[idx]
        num_atoms_i = self.num_atoms_per_structure[idx]
        end_atom_idx = start_atom_idx + num_atoms_i

        pos = torch.from_numpy(self.pos_all[start_atom_idx:end_atom_idx]).float()
        atomic_numbers = torch.from_numpy(
            self.atomic_numbers_all[start_atom_idx:end_atom_idx]
        ).long()
        elements = [Element.from_Z(num).symbol for num in atomic_numbers]

        composition_dict = Counter(elements)

        composition = Composition(composition_dict)
        reduced_formula_dict = composition.reduced_composition.as_dict()
        composition_symbol_tokens = torch.tensor(
            list(Element(symbol).number for symbol in reduced_formula_dict.keys()),
            dtype=torch.long,
        )
        composition_num_atoms = torch.tensor(
            list(reduced_formula_dict.values()), dtype=torch.long
        )
        reduced_formula_dict = {
            "symbol_tokens": composition_symbol_tokens,
            "num_atoms": composition_num_atoms,
        }

        cell = torch.from_numpy(self.cell_all[idx]).float()

        material_id = (
            str(self.structure_ids[idx])
            if self.structure_ids is not None
            else f"{self.cache_id}_Index_{idx}"
        )

        band_gap_val = self.band_gaps[idx]
        band_gap = torch.tensor(
            band_gap_val if np.isfinite(band_gap_val) else -1.0, dtype=torch.float32
        )
        band_gap_mask = torch.tensor(self.band_gaps_mask[idx], dtype=torch.bool)

        space_group_val = self.space_groups[
            idx
        ]  # if self.space_groups is not None and idx < len(self.space_groups) else 0
        space_group = torch.tensor(space_group_val, dtype=torch.long)
        space_group_mask = torch.tensor(self.space_groups_mask[idx], dtype=torch.bool)

        sample_dict = {
            "material_id": material_id,
            "pos": pos,  # Cartesian coordinates (N_i, 3)
            "cell": cell,  # Lattice matrix (3, 3)
            "atomic_numbers": atomic_numbers,  # Atomic numbers (N_i,)
            "band_gap": band_gap,  # Example scalar property ()
            "band_gap_mask": band_gap_mask,
            "space_group": space_group,  # Example scalar property ()
            "space_group_mask": space_group_mask,
            "num_atoms": torch.tensor(
                num_atoms_i, dtype=torch.long
            ),  # Number of atoms in this structure
            "reduced_formula": reduced_formula_dict,
            # Conditions
            "hhi": torch.tensor(self.hhi[idx], dtype=torch.float32),
            "bulk_modulus": torch.tensor(
                self.dft_bulk_modulus[idx], dtype=torch.float32
            ),
            "bulk_modulus_mask": torch.tensor(
                self.dft_bulk_modulus_mask[idx], dtype=torch.bool
            ),
            "mag_density": torch.tensor(self.dft_mag_density[idx], dtype=torch.float32),
            "mag_density_mask": torch.tensor(
                self.dft_mag_density_mask[idx], dtype=torch.bool
            ),
        }

        if self.transform:
            sample_dict = self.transform(sample_dict)

        return sample_dict


def _tokenize_structure(idx, tokenizer=None, dataset=None):
    """
    Tokenizes a single crystal structure.

    Args:
        data_tuple (tuple): A tuple containing the data dictionary and the tokenizer.

    Returns:
        dict: A dictionary containing the tokenized structure.
    """
    data = dataset[idx]
    structure = Structure(
        lattice=Lattice(data["cell"]),
        species=data["atomic_numbers"],
        coords=data["pos"],
        coords_are_cartesian=False,
    )
    tokens = tokenizer.tokenize(structure)

    out_data = {
        "band_gap": data["band_gap"].numpy().tolist(),  # Example scalar property ()
        "band_gap_mask": data["band_gap_mask"].numpy().tolist(),
        "space_group": data["space_group"]
        .numpy()
        .tolist(),  # Example scalar property ()
        # "space_group_mask": data["space_group_mask"].numpy().tolist(),
        "num_atoms": data["num_atoms"].numpy().tolist(),
        "reduced_formula": {
            k: v.numpy().tolist() for k, v in data["reduced_formula"].items()
        },
        "tokens": tokens,
        "density": structure.density,
        "hhi": data["hhi"].item(),
        "bulk_modulus": data["bulk_modulus"].item(),
        "bulk_modulus_mask": data["bulk_modulus_mask"].item(),
        "mag_density": data["mag_density"].item(),
        "mag_density_mask": data["mag_density_mask"].item(),
    }

    # The dictionary structure allows for adding other conditional data in the future
    return out_data


import joblib
from tqdm.contrib.concurrent import process_map


class CrystalDataset(Dataset):
    """A PyTorch Dataset for our tokenized crystal structures."""

    def __init__(
        self,
        dataset: MatterGenDataset,
        tokenizer: CrystalTokenizer,
        cache_path: str = "crystal_token_cache.joblib",
        recalculate_cache=False,
    ):
        self.tokenizer = tokenizer

        self.dataset = dataset

        cache_path_dir = os.path.join(os.path.dirname(__file__), ".cache")
        os.makedirs(cache_path_dir, exist_ok=True)

        self.pad_token_id = tokenizer._special_to_id["[PAD]"]
        self.cache_path = os.path.join(cache_path_dir, cache_path)
        self.processed_data: List[Dict[str, Any]] = []

        if os.path.exists(self.cache_path) and not recalculate_cache:
            print(f"Loading preprocessed data from cache: {self.cache_path}")
            self.processed_data = joblib.load(self.cache_path)
        else:
            print("No cache found. Preprocessing and tokenizing data in parallel...")
            indices = np.arange(0, len(self.dataset))
            self.processed_data = process_map(
                partial(
                    _tokenize_structure, tokenizer=self.tokenizer, dataset=self.dataset
                ),
                indices,
                max_workers=24,
                chunksize=1024,
                desc="Tokenizing",
                total=len(self.dataset),
            )
            print(f"Saving preprocessed data to cache: {self.cache_path}")
            joblib.dump(self.processed_data, self.cache_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        data = self.processed_data[idx]
        tokens = data["tokens"]
        input_seq = torch.tensor(tokens, dtype=torch.long)

        out_batch = {
            "tokens": input_seq,
            "density": torch.tensor(data["density"]),
            "reduced_formula": {
                k: torch.tensor(v, dtype=torch.long)
                for k, v in data["reduced_formula"].items()
            },
            "space_group": torch.tensor(data["space_group"], dtype=torch.long),
            "band_gap": torch.tensor(data["band_gap"], dtype=torch.float32),
            "band_gap_mask": torch.tensor(data["band_gap_mask"], dtype=torch.bool),
            "hhi": torch.tensor(data["hhi"], dtype=torch.float32),
            "bulk_modulus": torch.tensor(data["bulk_modulus"], dtype=torch.float32),
            "bulk_modulus_mask": torch.tensor(
                data["bulk_modulus_mask"], dtype=torch.bool
            ),
            "mag_density": torch.tensor(data["mag_density"], dtype=torch.float32),
            "mag_density_mask": torch.tensor(
                data["mag_density_mask"], dtype=torch.bool
            ),
        }
        return out_batch


from torch.nn.utils.rnn import pad_sequence


def pad_collate_fn(batch: List[torch.Tensor], pad_token_id: int = 0):
    """
    A simple collate function that pads sequences to the maximum length in a batch.

    Args:
        batch (List[torch.Tensor]): A list of token tensors from the dataset.
        pad_token_id (int): The ID to use for padding.

    Returns:
        A tuple of (input_sequences, target_sequences).
    """

    # `pad_sequence` expects a list of tensors and pads them to the max length.
    # `batch_first=True` makes the output shape [batch_size, max_len_in_batch].
    # `padding_value` is the token ID for the '[PAD]' token.
    batch_tokens = [b["tokens"] for b in batch]
    padded_tokens = pad_sequence(
        batch_tokens, batch_first=True, padding_value=pad_token_id
    )

    # The model's input is everything but the last token
    # The model's target is everything but the first token (shifted)
    input_seq = padded_tokens[:, :-1].contiguous()
    target_seq = padded_tokens[:, 1:].contiguous()
    density = (
        torch.stack([b["density"] for b in batch]).unsqueeze(1).unsqueeze(1)
    )  # Shape: [B, 1, 1]
    space_group = torch.stack([b["space_group"] for b in batch]).unsqueeze(
        1
    )  # Shape: [B, 1]

    conditions = {
        "density": density,
        "space_group": space_group,
    }

    continuous_property_keys = ["band_gap", "hhi", "bulk_modulus", "mag_density"]
    mask_keys = ["band_gap_mask", "bulk_modulus_mask", "mag_density_mask"]

    for key in continuous_property_keys:
        conditions[key] = torch.stack([b[key] for b in batch]).unsqueeze(1).unsqueeze(1)

    for key in mask_keys:
        conditions[key] = torch.stack([b[key] for b in batch])

    formula_symbol_tokens_list = []
    formula_num_atoms_list = []
    num_atoms_per_sample_list = []

    for sample in batch:
        formula = sample["reduced_formula"]

        # Get the sequence of atom types (e.g., [Fe, O])
        symbol_tokens = formula["symbol_tokens"]
        formula_symbol_tokens_list.append(symbol_tokens)

        # Get the count for each atom type (e.g., [2, 3])
        formula_num_atoms_list.append(formula["num_atoms"])

        num_atoms_per_sample_list.append(len(symbol_tokens))

        collated_formula_dict = {
            "composition_symbol_tokens": torch.cat(formula_symbol_tokens_list, dim=0),
            "composition_num_atoms": torch.cat(formula_num_atoms_list, dim=0),
            "num_atoms_per_sample": torch.tensor(
                num_atoms_per_sample_list, dtype=torch.long
            ),
        }

    conditions["reduced_formula"] = collated_formula_dict
    final_batch = {"tokens": input_seq, "targets": target_seq, "conditions": conditions}

    return final_batch
