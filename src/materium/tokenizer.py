from enum import Enum
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pymatgen.core import Structure, Element, Lattice
import torch
from pymatgen.core.periodic_table import Specie
from tqdm import tqdm


class SortingOrder(Enum):
    SPECIES = "species"
    RANDOM = "random"
    REVERSE_SPECIES = "rev_species"
    XYZ = "xyzorder"
    ELECTRON_NEG_DOWN = "ele_neg_down"  # Most negative elements first


class SequenceOrder(Enum):
    """
    Determines the order of atom type and coordinate tokens within the atom sequence.
    - ATOMS_FIRST: [elem_1, x, y, z, elem_2, x, y, z, ...]
    - COORDS_FIRST: [x, y, z, elem_1, x, y, z, elem_2, ...]
    """

    ATOMS_FIRST = "atoms_first"
    COORDS_FIRST = "coords_first"


class CrystalTokenizer:
    """
    A tokenizer to convert pymatgen Structure objects into a sequence of integer
    tokens and back, with optional oxidation-state-aware element tokens.

    Token sequence:
    [SOS] [ATOMS] [elem|ox_1] [3 coord_tokens] [elem|ox_2] ... [LATTICE] [6 lattice tokens] [EOS]
    """

    def __init__(
        self,
        element_vocab: List[str],
        lattice_stats: Dict[str, Tuple[float, float]],
        num_quant_bins: int = 1024,
        sorting_order: SortingOrder = SortingOrder.SPECIES,
        sequence_order: SequenceOrder = SequenceOrder.ATOMS_FIRST,
        include_oxidation: bool = True,
        oxidation_mode: str = "guess",
        allowed_oxidation_states: Optional[Dict[str, List[int]]] = None,
        fallback_to_zero_on_unseen: bool = True,
    ):
        """
        Args:
            element_vocab (List[str]): Elements in the vocabulary (e.g., ['H', 'O', 'Fe']).
            lattice_stats (Dict[str, Tuple[float, float]]): Min/max for lattice params (a,b,c,alpha,beta,gamma).
            num_quant_bins (int): Number of bins for quantizing continuous values.
            include_oxidation (bool): If True, expand tokens to include oxidation state variants.
            oxidation_mode (str): 'guess' -> use pymatgen to guess oxidation states when absent;
                                  'from_structure' -> require Structure to already have oxidation states.
            allowed_oxidation_states (dict): Per-element allowed oxidation states. If None, uses
                                             Element.common_oxidation_states (or Element.oxidation_states if empty) plus 0.
            fallback_to_zero_on_unseen (bool): If True, use ox=0 if guessed ox not in allowed set; else raise.
        """
        self.num_quant_bins = num_quant_bins
        self.lattice_stats = lattice_stats
        self._lattice_param_keys = ["a", "b", "c", "alpha", "beta", "gamma"]
        self.sorting_order = sorting_order
        self.sequence_order = sequence_order

        self.include_oxidation = include_oxidation
        self.oxidation_mode = oxidation_mode
        self.fallback_to_zero_on_unseen = fallback_to_zero_on_unseen

        self.special_tokens = ["[PAD]", "[SOS]", "[EOS]", "[LATTICE]", "[ATOMS]"]

        self.element_vocab = sorted({Element(el).symbol for el in element_vocab})

        if allowed_oxidation_states is None:
            self.allowed_oxidation_states = {}
            for el in self.element_vocab:
                elem = Element(el)
                common = list(getattr(elem, "common_oxidation_states", []))
                if not common:
                    common = list(getattr(elem, "oxidation_states", []))
                allowed = sorted(set(common + [0]))
                self.allowed_oxidation_states[el] = allowed
        else:
            self.allowed_oxidation_states = {
                Element(el).symbol: sorted(set([int(o) for o in ox_list] + [0]))
                for el, ox_list in allowed_oxidation_states.items()
            }
            for el in self.element_vocab:
                if el not in self.allowed_oxidation_states:
                    self.allowed_oxidation_states[el] = [0]

        if self.include_oxidation:
            # species keys like "O|-2", "Fe|+3", "O|0"
            species_tokens = []
            for el in self.element_vocab:
                for ox in self.allowed_oxidation_states[el]:
                    species_tokens.append((el, int(ox)))
            
            species_tokens.sort(key=lambda t: (Element(t[0]).Z, t[1], t[0]))
            self.species_vocab = [f"{el}|{ox:+d}" for el, ox in species_tokens]
        else:
            self.species_vocab = list(self.element_vocab)  # no ox info

        self._special_to_id = {tok: i for i, tok in enumerate(self.special_tokens)}
        self._id_to_special = {i: tok for tok, i in self._special_to_id.items()}

        base = len(self.special_tokens)
        self._species_to_id = {
            tok: base + i for i, tok in enumerate(self.species_vocab)
        }
        self._id_to_species = {i: tok for tok, i in self._species_to_id.items()}

        self.quant_offset = len(self.special_tokens) + len(self.species_vocab)
        self.vocab_size = self.quant_offset + self.num_quant_bins

    def __len__(self) -> int:
        return self.vocab_size

    def to_dict(self):
        """
        Gives the tokenizer's configuration to in JSON format.
        """
        config = {
            "element_vocab": self.element_vocab,
            "lattice_stats": self.lattice_stats,
            "num_quant_bins": self.num_quant_bins,
            "sorting_order": self.sorting_order.name,
            "sequence_order": self.sequence_order.name,
            "include_oxidation": self.include_oxidation,
            "oxidation_mode": self.oxidation_mode,
            "allowed_oxidation_states": self.allowed_oxidation_states,
            "fallback_to_zero_on_unseen": self.fallback_to_zero_on_unseen,
        }
        return config

    @classmethod
    def from_dict(cls, json_data: Dict[str, any]) -> "CrystalTokenizer":
        """
        Loads a tokenizer's configuration from a JSON config and creates an instance.

        Args:
            json_data (Dict[str, any]): The tokenizer config file, gotten from to_json().

        Returns:
            CrystalTokenizer: An instance of the tokenizer with the loaded configuration.
        """
        json_data["sorting_order"] = SortingOrder[json_data["sorting_order"]]
        json_data["sequence_order"] = SequenceOrder[json_data["sequence_order"]]

        return cls(**json_data)

    def _normalize_lattice_param(self, value: float, param_key: str) -> float:
        """Scales a lattice parameter to the [0, 1] range."""
        min_val, max_val = self.lattice_stats[param_key]
        return (value - min_val) / (max_val - min_val)

    def _denormalize_lattice_param(self, norm_value: float, param_key: str) -> float:
        """Scales a normalized value back to its original lattice parameter range."""
        min_val, max_val = self.lattice_stats[param_key]
        return norm_value * (max_val - min_val) + min_val

    def _get_site_species_token_key(self, site) -> str:
        """
        Returns the species token key for a site:
        - If include_oxidation: "El|+n" or "El|-n" or "El|+0"
        - Else: "El"
        """
        el = site.specie.symbol
        if not self.include_oxidation:
            return el

        ox = None
        if hasattr(site.specie, "oxi_state"):
            try:
                ox = int(round(site.specie.oxi_state))
            except Exception:
                ox = None

        if ox is None:
            ox = 0

        if ox not in self.allowed_oxidation_states.get(el, [0]):
            if self.fallback_to_zero_on_unseen:
                ox = 0
            else:
                raise ValueError(
                    f"Oxidation state {ox} for element {el} not in allowed set {self.allowed_oxidation_states.get(el)}."
                )

        return f"{el}|{ox:+d}"

    def _assign_oxidation_states_if_needed(self, structure: Structure) -> Structure:
        """
        Returns a structure with oxidation states assigned if include_oxidation=True.
        - 'guess': attempt pymatgen guessing; if it fails, use 0 as fallback later.
        - 'from_structure': assume oxidation states are already present.
        """
        if not self.include_oxidation:
            return structure

        s = structure.copy()
        if self.oxidation_mode == "guess":
            try:
                s.add_oxidation_state_by_guess()
            except Exception as e:
                print("Failed to add oxidization state")
        elif self.oxidation_mode == "from_structure":
            pass
        else:
            raise ValueError(f"Unknown oxidation_mode: {self.oxidation_mode}")
        return s

    def tokenize(self, structure: Structure) -> List[int]:
        """
        Converts a pymatgen Structure into a list of integer tokens.

        Returns:
            List[int]: [SOS] [ATOMS] (elem|ox, 3*coords) ... [LATTICE] (6*lattice) [EOS]
        """
        tokens = [self._special_to_id["[SOS]"]]

        reduced_structure = structure.get_reduced_structure(reduction_algo="niggli")
        reduced_structure = self._assign_oxidation_states_if_needed(reduced_structure)

        tokens.append(self._special_to_id["[ATOMS]"])

        # Sorting
        if self.sorting_order == SortingOrder.SPECIES:
            sorted_sites = sorted(
                reduced_structure.sites,
                key=lambda site: (
                    Element(site.specie.symbol).Z,
                    getattr(site.specie, "oxi_state", 0),
                    site.frac_coords.tolist(),
                ),
            )
        elif self.sorting_order == SortingOrder.RANDOM:
            sites_list = list(reduced_structure.sites)
            random.shuffle(sites_list)
            sorted_sites = sites_list
        elif self.sorting_order == SortingOrder.REVERSE_SPECIES:
            sorted_sites = sorted(
                reduced_structure.sites,
                key=lambda site: (
                    Element(site.specie.symbol).Z,
                    getattr(site.specie, "oxi_state", 0),
                    site.frac_coords.tolist(),
                ),
            )[::-1]
        elif self.sorting_order == SortingOrder.XYZ:
            sorted_sites = sorted(
                reduced_structure.sites,
                key=lambda site: (
                    site.frac_coords.tolist(),
                    Element(site.specie.symbol).Z,
                ),
            )
        elif self.sorting_order == SortingOrder.ELECTRON_NEG_DOWN:

            def _en_key(site):
                x = Element(site.specie.symbol).X
                primary = -x if x is not None else float("inf")
                return (
                    primary,
                    Element(site.specie.symbol).Z,  # tie-breaker: atomic number
                    getattr(site.specie, "oxi_state", 0),  # then oxidation state
                    site.frac_coords.tolist(),  # then fractional coords
                )

            sorted_sites = sorted(reduced_structure.sites, key=_en_key)
        else:
            raise ValueError(f"Unknown sorting order: {self.sorting_order}")

        # Atom tokens
        for site in sorted_sites:
            species_key = self._get_site_species_token_key(site)
            if species_key not in self._species_to_id:
                # This can only happen if user provided a very tight allowed_oxidation_states without 0,
                # or include_oxidation=False but site has ox. Try fallback to 0 if enabled.
                if self.include_oxidation and self.fallback_to_zero_on_unseen:
                    fallback_key = f"{site.specie.symbol}|+0"
                    if fallback_key in self._species_to_id:
                        species_key = fallback_key
                    else:
                        raise ValueError(
                            f"Species token '{species_key}' not in vocabulary and no valid fallback found."
                        )
                else:
                    raise ValueError(
                        f"Species token '{species_key}' not in vocabulary."
                    )

            element_token = self._species_to_id[species_key]
            coord_tokens = [
                int(np.floor(np.clip(coord % 1.0, 0, 1) * (self.num_quant_bins - 1)))
                + self.quant_offset
                for coord in site.frac_coords
            ]

            if self.sequence_order == SequenceOrder.ATOMS_FIRST:
                tokens.append(element_token)
                tokens.extend(coord_tokens)
            elif self.sequence_order == SequenceOrder.COORDS_FIRST:
                tokens.extend(coord_tokens)
                tokens.append(element_token)
            else:
                raise ValueError(f"Unknown sequence order: {self.sequence_order}")

        # Lattice section
        tokens.append(self._special_to_id["[LATTICE]"])
        lattice_params = reduced_structure.lattice.parameters
        for i, key in enumerate(self._lattice_param_keys):
            norm_val = self._normalize_lattice_param(lattice_params[i], key)
            quant_val = int(
                np.floor(np.clip(norm_val, 0, 1) * (self.num_quant_bins - 1))
            )
            tokens.append(quant_val + self.quant_offset)

        tokens.append(self._special_to_id["[EOS]"])
        return tokens

    def detokenize(self, tokens: List[int]) -> Structure:
        """
        Converts a sequence of tokens back into a pymatgen Structure.
        Works regardless of the order of the [ATOMS] and [LATTICE] sections.
        """
        lattice_params = []
        species = []
        coords = []
        current_state = None
        atom_buffer = []

        for token in tokens:
            if token in {self._special_to_id["[SOS]"], self._special_to_id["[PAD]"]}:
                continue
            if token == self._special_to_id["[EOS]"]:
                break
            if token == self._special_to_id["[LATTICE]"]:
                current_state = "LATTICE"
                continue
            if token == self._special_to_id["[ATOMS]"]:
                current_state = "ATOMS"
                continue

            if current_state == "LATTICE":
                param_idx = len(lattice_params)
                if param_idx >= 6:
                    continue
                param_key = self._lattice_param_keys[param_idx]
                norm_val = (token - self.quant_offset + 0.5) / self.num_quant_bins
                denorm_val = self._denormalize_lattice_param(norm_val, param_key)
                lattice_params.append(denorm_val)

            elif current_state == "ATOMS":
                atom_buffer.append(token)
                if len(atom_buffer) == 4:  # 1 species + 3 coords (or reversed)
                    if self.sequence_order == SequenceOrder.ATOMS_FIRST:
                        elem_token = atom_buffer[0]
                        coord_tokens = atom_buffer[1:]
                    elif self.sequence_order == SequenceOrder.COORDS_FIRST:
                        elem_token = atom_buffer[3]
                        coord_tokens = atom_buffer[:3]
                    else:
                        raise ValueError(
                            f"Unknown sequence order: {self.sequence_order}"
                        )

                    # Decode species
                    species_key = self._id_to_species[elem_token]
                    if self.include_oxidation:
                        el, ox_str = species_key.split("|")
                        ox = int(ox_str)
                        specie_obj = Specie(el, ox)
                        species.append(specie_obj)
                    else:
                        species.append(species_key)  # plain element symbol

                    frac_coords = [
                        (ct - self.quant_offset + 0.5) / self.num_quant_bins
                        for ct in coord_tokens
                    ]
                    coords.append(frac_coords)
                    atom_buffer = []

        if len(lattice_params) != 6:
            raise ValueError(
                f"Invalid token sequence: Found {len(lattice_params)} lattice parameters, expected 6."
            )
        if len(species) != len(coords) or not species:
            raise ValueError(
                "Invalid token sequence: Mismatch between species and coordinates or no atoms found."
            )

        coords = np.array(coords) % 1
        lattice = Lattice.from_parameters(*lattice_params)
        structure = Structure(lattice, species, coords, coords_are_cartesian=False)
        return structure

    def get_loss_weights(self, atom_type_weight: float) -> Optional[torch.Tensor]:
        """
        Creates a weight tensor for the cross-entropy loss function.

        This tensor assigns a higher weight to atom type tokens, encouraging the
        model to prioritize their prediction.

        Args:
            atom_type_weight (float): The weight to apply to atom type tokens.
                                      If this is 1.0 or less, no reweighting is
                                      applied and None is returned.

        Returns:
            Optional[torch.Tensor]: A 1D tensor of shape (vocab_size,) with custom
                                    weights, or None if no reweighting is applied.
        """

        print(f"Creating loss weights with atom_type_weight = {atom_type_weight}")

        weights = torch.ones(self.vocab_size, dtype=torch.float32)

        start_id = len(self.special_tokens)
        end_id = start_id + len(self.element_vocab)

        # Set the higher weight for the slice corresponding to element tokens
        weights[start_id:end_id] = atom_type_weight

        return weights


def calculate_stats_from_dataset(
    dataset: "MatterGenDataset",
) -> Tuple[List[str], Dict[str, Tuple[float, float]]]:
    """Calculates the element vocabulary and min/max for lattice parameters from a dataset."""
    unique_elements = set()
    lattice_params_lists = {
        "a": [],
        "b": [],
        "c": [],
        "alpha": [],
        "beta": [],
        "gamma": [],
    }
    param_keys = list(lattice_params_lists.keys())

    print(f"Analyzing {len(dataset)} structures to calculate stats...")
    for i in tqdm(10_000):
        sample = dataset[i]
        for z in sample["atomic_numbers"]:
            unique_elements.add(Element.from_Z(z).symbol)
        try:
            # Use the Niggli-reduced cell for consistent statistics
            params = Lattice(sample["cell"]).get_niggli_reduced_lattice().parameters
            for j, key in enumerate(param_keys):
                lattice_params_lists[key].append(params[j])
        except Exception:
            continue

    element_vocab = sorted(list(unique_elements))
    lattice_stats = {
        key: (float(np.min(values)), float(np.max(values)))
        for key, values in lattice_params_lists.items()
    }
    return element_vocab, lattice_stats
