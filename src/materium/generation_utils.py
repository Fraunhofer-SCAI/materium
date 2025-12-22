from collections import Counter, defaultdict
import torch
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from typing import Iterable, List, Dict, Tuple, Any, Union
from typing import Dict, List, Optional
import numpy as np
import torch
from dataclasses import dataclass
from pymatgen.core import Lattice, Structure, Element, Composition
import os
import re
from tqdm import tqdm
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import matplotlib.pyplot as plt
import shutil
from materium.model import ConditionConfig, LLamaTransformer, ModelArgs
from mattergen.evaluation.utils.utils import compute_rmsd_angstrom

from pymatgen.io.pwscf import PWInput
import time
from pymatgen.analysis.hhi import HHIModel

import torch.nn.functional as F


from pymatgen.core import Composition, Element


def parse_formula_to_reduced_format(
    formula: str, num_samples: int, device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Parse a chemical formula and create a reduced format suitable for batch processing.

    Args:
        formula (str): The chemical formula to process.
        num_samples (int): The number of samples for batch processing.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing tensors corresponding to the parsed formula.
    """
    # Parse the formula using pymatgen to get the composition
    composition = Composition(formula)
    reduced_formula_dict = composition.reduced_composition.as_dict()

    # Convert element symbols to atomic numbers
    composition_symbol_tokens = torch.tensor(
        [Element(symbol).number for symbol in reduced_formula_dict.keys()],
        dtype=torch.long,
    )

    # Get the number of atoms
    composition_num_atoms = torch.tensor(
        list(reduced_formula_dict.values()), dtype=torch.long
    )

    # Create batch composition index for multiple samples
    batch_composition_idx = torch.tensor(
        [i for i in range(num_samples) for _ in range(len(composition_symbol_tokens))],
        dtype=torch.long,
    )

    # Replicate the reduced formula for num_samples
    composition_symbol_tokens = composition_symbol_tokens.repeat(num_samples)
    composition_num_atoms = composition_num_atoms.repeat(num_samples)

    # Create the reduced formula format
    reduced_formula = {
        "composition_symbol_tokens": composition_symbol_tokens.to(device),
        "composition_num_atoms": composition_num_atoms.to(device),
        "num_atoms_per_sample": torch.bincount(batch_composition_idx.to(device)),
    }

    return reduced_formula


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.9,
    penalize_token_id: Optional[List[int]] = None,
    penalty_weight: float = 1.5,
    squeeze: float = 0.0,  # [0,1] flatten around mean
    tail_bias: float = 0.0,  # >0 boosts tail
    uniform_mix: float = 0.0,  # [0,1] uniform on support
    typical_p: float = 0.0,  # [0,1] typical decoding; 0 disables
    logit_bias: Optional[torch.Tensor] = None,  # [vocab], additive bias
    forbid_mask: Optional[torch.Tensor] = None,  # [vocab] bool mask to forbid tokens
    return_entropy: bool = False,
) -> int:
    """
    Temperature + top-p with optional tail boosting, typical decoding, token-specific bias.
    Expects logits shape [1, V].
    """
    logits = logits.clone()

    V = logits.size(-1)
    assert logits.dim() == 2 and logits.size(0) == 1, "expects [1, vocab]"

    if logit_bias is not None:
        logits = logits + logit_bias.view(1, V)

    if penalize_token_id is not None and len(penalize_token_id) > 0:
        logits[:, penalize_token_id] -= penalty_weight

    if forbid_mask is not None:
        # -inf to forbidden tokens
        logits = logits.masked_fill(forbid_mask.view(1, V), float("-inf"))

    if top_p <= 0.0 or temperature <= 1e-5:
        return torch.argmax(logits, dim=-1).item()

    if squeeze != 0.0:
        mu = logits.mean(dim=-1, keepdim=True)
        logits = mu + (1.0 - squeeze) * (logits - mu)

    logits = logits / (temperature + 1e-5)

    if tail_bias != 0.0:
        sorted_idx = torch.argsort(logits, dim=-1, descending=True)
        inv_rank = torch.zeros_like(logits)
        src = torch.arange(V, device=logits.device, dtype=logits.dtype).expand_as(
            sorted_idx
        )
        inv_rank.scatter_(dim=-1, index=sorted_idx, src=src)
        denom = max(1, V - 1)
        rank_norm = inv_rank / denom
        logits = logits + tail_bias * (rank_norm - 0.5)

    probs = F.softmax(logits, dim=-1)

    # Typical decoding (optional, before top-p)
    if 0.0 < typical_p < 1.0:
        # closeness to entropy
        entropy = -(
            probs * (logits - torch.logsumexp(logits, dim=-1, keepdim=True))
        ).sum(dim=-1, keepdim=True)
        surprisal = torch.abs(
            -(logits - torch.logsumexp(logits, dim=-1, keepdim=True)) - entropy
        )
        sorted_surprisal, sorted_indices = torch.sort(surprisal, dim=-1)
        sorted_probs = probs.gather(-1, sorted_indices)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        to_remove = cumsum > typical_p
        to_remove[..., 1:] = to_remove[..., :-1].clone()
        to_remove[..., 0] = False
        mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(
            -1, sorted_indices, to_remove
        )
        probs = probs.masked_fill(mask, 0.0)

    # Top-p nucleus
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    probs = probs.masked_fill(indices_to_remove, 0.0)

    # Optional uniform mix on support
    if uniform_mix > 0.0:
        support_mask = probs > 0
        support_sizes = support_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        uniform = support_mask.float() / support_sizes.float()
        probs = (1.0 - uniform_mix) * probs + uniform_mix * uniform

    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    # probs = torch.softmax(probs, dim=-1, keepdim=True)
    # print("here", probs.max(dim=-1))

    next_token = torch.multinomial(probs, num_samples=1)
    eps = 1e-12
    entropy_bits = -(probs.clamp_min(eps) * probs.clamp_min(eps).log2()).sum(dim=-1)
    if return_entropy:
        return next_token.item(), entropy_bits

    return next_token.item()


def cfg_blend(logits_uncond, logits_cond, guidance_weight: float):
    # Standard CFG: uncond + w*(cond - uncond)
    return logits_uncond + guidance_weight * (logits_cond - logits_uncond)


def get_condition_dict(model, conditions: Dict[str, Any], device):
    processed_conditions = {}
    if conditions:
        print(f"Generating with conditions: {conditions}")
        for name, value in conditions.items():
            if name not in model.params.condition_config:
                print(
                    f"Warning: Condition '{name}' not recognized by the model and will be ignored."
                )
                continue

            if name.lower() in [
                "density",
                "band_gap",
                "hhi",
                "bulk_modulus",
                "mag_density",
            ]:
                # Expected shape for linear layer: [B, SeqLen=1, FeatDim=1]
                tensor = torch.tensor(
                    [[value]], dtype=torch.float32, device=device
                ).unsqueeze(1)
            elif name == "space_group":
                # Expected shape for embedding lookup: [B, SeqLen=1]
                tensor = torch.tensor(
                    [value], dtype=torch.long, device=device
                ).unsqueeze(1)
            elif name == "reduced_formula":
                tensor = parse_formula_to_reduced_format(value, 1, device)

            else:
                print(
                    f"Warning: Pre-processing for condition '{name}' not implemented, skipping."
                )
                continue

            processed_conditions[name] = tensor
    else:
        print("Generating without any conditions (unconditional).")

    return processed_conditions


def generate_structure(
    model,
    tokenizer,
    max_len=100,
    device="cpu",
    temperature=1.0,
    top_p=0.9,
    conditions=None,
    classifier_guidance_weight=0.0,
    min_atoms: int = 2,
    max_atoms: int = 64,
    use_typical: bool = False,
    oxygen_penalty: float = 0.0,
    charge_neutral_bias: float = 0.0,
):
    model.eval()
    sos = tokenizer._special_to_id["[SOS]"]
    eos = tokenizer._special_to_id["[EOS]"]
    atoms_tok = tokenizer._special_to_id["[ATOMS]"]
    lattice_tok = tokenizer._special_to_id["[LATTICE]"]

    generated = [sos, atoms_tok]

    if conditions is not None:
        conditions = get_condition_dict(model, conditions, device)

    state = make_debug_state()

    with torch.no_grad():
        while len(generated) < max_len:
            inp = torch.tensor([generated], dtype=torch.long, device=device)

            if classifier_guidance_weight > 0.0 and conditions:
                logits_uncond = model(inp, conditions={})
                logits_cond = model(inp, conditions=conditions)
                logits = cfg_blend(
                    logits_uncond, logits_cond, classifier_guidance_weight
                )
            else:
                logits = model(inp, conditions=conditions)

            next_logits = logits[:, -1, :]  # [1, V]

            samp_kwargs = dict(
                temperature=temperature,
                top_p=0.99,
                tail_bias=0.25,
                uniform_mix=0.0,
                squeeze=0.0,
                typical_p=0.0,
            )

            next_token, entropy = sample_next_token(
                logits=next_logits,
                forbid_mask=None,
                logit_bias=None,
                **samp_kwargs,
                return_entropy=True,
            )

            # state = debug_print_token(next_token, entropy, tokenizer, state)
            generated.append(next_token)

            if next_token == eos:
                break

    return generated


def _to_int(tok):
    if isinstance(tok, int):
        return tok
    if hasattr(tok, "item"):
        return int(tok.item())
    return int(tok)


def make_debug_state():
    return {
        "mode": None,  # None | "ATOMS" | "LATTICE"
        "coord_idx": 0,  # 0..2 within current atom
        "param_idx": 0,  # 0..5 within lattice
        "atom_idx": 0,  # atom counter
    }


def decode_frac_from_token(tok_int, tokenizer):
    # fractional coord in [0,1)
    return (tok_int - tokenizer.quant_offset + 0.5) / tokenizer.num_quant_bins


def decode_lattice_from_token(tok_int, tokenizer, param_key):
    norm = (tok_int - tokenizer.quant_offset + 0.5) / tokenizer.num_quant_bins
    return tokenizer._denormalize_lattice_param(norm, param_key)


def debug_print_token(tok, entropy, tokenizer, state):
    tok_int = _to_int(tok)
    qoff = tokenizer.quant_offset
    dims = ("x", "y", "z")

    # Special tokens
    if tok_int in tokenizer._id_to_special:
        label = tokenizer._id_to_special[tok_int]
        print(f"Token: {label} Entropy: {float(entropy):.3f}")
        if label == "[ATOMS]":
            state["mode"] = "ATOMS"
            state["coord_idx"] = 0
        elif label == "[LATTICE]":
            state["mode"] = "LATTICE"
            state["param_idx"] = 0
        elif label == "[EOS]":
            state["mode"] = None
        return state

    # Species tokens
    if tok_int in tokenizer._id_to_species:
        species = tokenizer._id_to_species[tok_int]
        print(f"Token: {species} Entropy: {float(entropy):.3f}")
        # We are in the ATOMS section; coordinates come either before or after,
        # depending on sequence_order. We just keep counting coords per atom.
        if state["mode"] != "ATOMS":
            state["mode"] = "ATOMS"
            state["coord_idx"] = 0
        # If ATOMS_FIRST, expect coords next; if COORDS_FIRST, we just finished coords.
        if tokenizer.sequence_order == SequenceOrder.COORDS_FIRST:
            state["atom_idx"] += 1  # coords must have come before the species
            state["coord_idx"] = 0
        return state

    # Quantized tokens (coords or lattice)
    if tok_int >= qoff:
        if state["mode"] == "ATOMS":
            d = dims[state["coord_idx"]]
            frac = decode_frac_from_token(tok_int, tokenizer)
            print(f"Token: coord_{d}={frac:.6f} Entropy: {float(entropy):.3f}")
            state["coord_idx"] = (state["coord_idx"] + 1) % 3
            if (
                state["coord_idx"] == 0
                and tokenizer.sequence_order == SequenceOrder.ATOMS_FIRST
            ):
                state["atom_idx"] += 1
        elif state["mode"] == "LATTICE":
            idx = state["param_idx"]
            if idx < 6:
                key = tokenizer._lattice_param_keys[idx]
                val = decode_lattice_from_token(tok_int, tokenizer, key)
                print(f"Token: {key}={val:.6f} Entropy: {float(entropy):.3f}")
                state["param_idx"] += 1
            else:
                # Extra quantized tokens after 6 lattice params
                print(
                    f"Token: [EXTRA_LATTICE_BIN {tok_int - qoff}] Entropy: {float(entropy):.3f}"
                )
        else:
            # Quantized token outside known sections
            print(f"Token: [QUANT_BIN {tok_int - qoff}] Entropy: {float(entropy):.3f}")
        return state

    print(f"Token: [UNKNOWN {tok_int}] Entropy: {float(entropy):.3f}")
    return state


def calc_hhi(
    structure: Structure,
    kind: str = "production",
) -> Union[float, Tuple[float, float]]:
    """
    Calculate the Herfindahl–Hirschman Index (HHI) for a structure using pymatgen's HHIModel.

    Args:
        structure: pymatgen Structure (relaxed or not). HHI depends only on composition.
        kind: 'production', 'reserve', or 'both'.

    Returns:
        - If kind == 'production': production HHI (float)
        - If kind == 'reserve': reserve HHI (float)
        - If kind == 'both': (production HHI, reserve HHI)
    """
    model = HHIModel()
    structure = structure.remove_oxidation_states()
    hhi_p, hhi_r = model.get_hhi(structure.composition)
    if hhi_p is None:
        raise ValueError(
            "Could not compute HHI; check that all elements exist in hhi_data.csv"
        )

    if kind == "production":
        return float(hhi_p)
    if kind == "reserve":
        return float(hhi_r)
    if kind == "both":
        return float(hhi_p), float(hhi_r)
    raise ValueError("kind must be 'production', 'reserve', or 'both'")


def hhi_designations(hhi_p: float, hhi_r: float) -> Tuple[str, str]:
    """
    Return DOJ designations ('low'/'medium'/'high') for production and reserve HHI.
    """
    return HHIModel.get_hhi_designation(hhi_p), HHIModel.get_hhi_designation(hhi_r)


def calc_hhi_for_structures(
    structures: Iterable[Structure],
    kind: str = "production",
) -> List[Union[float, Tuple[float, float]]]:
    """
    Convenience function for multiple structures.
    """
    return [calc_hhi(s, kind=kind) for s in structures]


def create_qe_magdensity_input(
    structure: Structure,
    output_dir: str,
    pseudo_dir: str,
    ecutwfc: int = 60,
    file_name: str = "qe_relax.in",
):
    """
    Creates a Quantum ESPRESSO input file with a robust, direct method for
    setting starting magnetization that avoids API abstractions.

    Args:
        structure (Structure): The generated pymatgen Structure object.
        output_dir (str): Directory where the QE input file will be saved.
        pseudo_dir (str): Path to your SSSP pseudopotential directory.
        ecutwfc (int): Plane-wave cutoff energy in Rydbergs.
    """
    print(f"Creating Quantum ESPRESSO input file in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    element_to_pseudo_map = {}
    known_elements = sorted([e.symbol for e in Element], key=len, reverse=True)
    for filename in os.listdir(pseudo_dir):
        for symbol in known_elements:
            if re.match(f"^{symbol}[._-]", filename, re.IGNORECASE):
                element_to_pseudo_map[symbol] = filename
                break
    pseudos_for_structure = {e: element_to_pseudo_map[e] for e in structure.symbol_set}

    control_params = {
        "calculation": "vc-relax",
        "restart_mode": "from_scratch",
        "outdir": output_dir,
        "pseudo_dir": pseudo_dir,
        "tprnfor": True,
        "verbosity": "high",
    }
    system_params = {
        "ecutwfc": ecutwfc,
        "ecutrho": ecutwfc * 8,
        "nspin": 2,
        "occupations": "smearing",
        "smearing": "methfessel-paxton",
        "degauss": 0.01,
        "input_dft": "PBE",
    }
    electrons_params = {
        "mixing_mode": "local-TF",
        "mixing_beta": 0.3,
        "conv_thr": 1.0e-6,
        "diagonalization": "cg",
    }

    # This dictionary defines the initial guess for each element type.
    default_magmoms = {
        "Fe": 0.8,
        "Cr": 0.8,
        "Mn": 0.8,
        "Co": 0.8,
        "Ni": 0.8,
        "Gd": 0.8,
        "default": 0.1,
    }

    # Get the unique species types in the structure, in the order pymatgen uses.
    # E.g., for Fe2O3, this would be [Element Fe, Element O]
    species_types = structure.types_of_species

    # Iterate through the species types and add the specific namelist card
    # for each one. The index (i+1) corresponds to the order in the ATOMIC_SPECIES card.
    for i, species in enumerate(species_types):
        moment = default_magmoms.get(species.symbol, default_magmoms["default"])
        # This creates the literal key like "starting_magnetization(1)"
        fortran_style_key = f"starting_magnetization({i+1})"
        system_params[fortran_style_key] = moment

    print(
        f"Manually setting magnetization in &SYSTEM namelist: { {k:v for k,v in system_params.items() if 'start' in k} }"
    )

    # --- 4. Create the PWInput object ---
    # The system_params dictionary now contains the exact keys QE needs.
    pw_input = PWInput(
        structure=structure,
        control=control_params,
        system=system_params,
        electrons=electrons_params,
        pseudo=pseudos_for_structure,
    )

    input_filepath = os.path.join(output_dir, file_name)
    pw_input.write_file(input_filepath)

    print(f"Successfully created QE input: {input_filepath}")


def create_qe_scf_from_mp_config(
    structure: Structure,
    output_dir: str,
    pseudo_dir: str,
    file_name: str = "qe_mp_scf.in",
    gen_idx: int = 0,
):
    """
    Creates a Quantum ESPRESSO SCF input file by translating key parameters
    from the Materials Project's VASP configuration, including specific MAGMOM values.

    Args:
        structure (Structure): The pymatgen Structure object for the calculation.
        output_dir (str): Directory where the QE input file will be saved.
        pseudo_dir (str): Path to your pseudopotential directory.
        file_name (str): The name for the output QE input file.
    """
    print(
        f"Creating QE SCF input based on Materials Project VASP config in: {output_dir}"
    )
    os.makedirs(output_dir, exist_ok=True)

    EV_TO_RY = 1 / 13.605693122994
    ecutwfc_ry = 60  # 110 #520 * EV_TO_RY  # Corresponds to ENCUT = 520 eV
    # degauss_ry = 0.01 * EV_TO_RY # Corresponds to SIGMA = 0.05 eV

    n_atoms = len(structure.sites)
    kpoint_density = int(max(1, round(1000 / n_atoms)))

    mp_hubbard_u_values = {
        "Co": 3.32,
        "Cr": 3.7,
        "Fe": 5.3,
        "Mn": 3.9,
        "Mo": 4.38,
        "Ni": 6.2,
        "V": 3.25,
        "W": 6.2,
    }

    mp_magmom_values = {
        "Ce": 5,
        "Co": 0.6,
        "Cr": 5,
        "Fe": 5,
        "Mn": 5,
        "Mo": 5,
        "Ni": 5,
        "V": 5,
        "W": 5,
        "Eu": 10,
        "Gd": 7,
        "La": 0.6,
        "Lu": 0.6,
        "Nd": 3,
        "Pr": 2,
        "Sm": 5,
        "Tb": 6,
    }

    element_to_pseudo_map = {}
    known_elements = sorted([e.symbol for e in Element], key=len, reverse=True)
    if not os.path.isdir(pseudo_dir):
        raise FileNotFoundError(f"Pseudopotential directory not found at: {pseudo_dir}")

    for filename in os.listdir(pseudo_dir):
        for symbol in known_elements:
            if re.match(f"^{symbol}[._-]", filename, re.IGNORECASE):
                element_to_pseudo_map[symbol] = filename
                break
    pseudos_for_structure = {
        e: element_to_pseudo_map.get(e) for e in structure.symbol_set
    }
    missing_elements = [el for el, ps in pseudos_for_structure.items() if ps is None]
    if missing_elements:
        raise ValueError(
            f"Could not find pseudopotentials for elements: {missing_elements}"
        )

    control_params = {
        "calculation": "vc-relax",
        "prefix": f"relax_{gen_idx}",
        "restart_mode": "from_scratch",
        "outdir": os.path.join(output_dir, f"relax_{gen_idx}"),
        "pseudo_dir": pseudo_dir,
        "verbosity": "high",
        "tstress": True,
        "tprnfor": True,
        "disk_io": "low",
        "nstep": 75,
        "forc_conv_thr": 1e-2,
    }

    system_params = {
        "ecutwfc": 80,  # ecutwfc_ry, #round(ecutwfc_ry, 2),
        "ecutrho": 880,  # 1320,
        "occupations": "smearing",
        "smearing": "gauss",
        "degauss": 0.01,  # round(degauss_ry, 6)
        # "input_dft": "PBEsol"
    }

    electrons_params = {
        "mixing_mode": "local-TF",
        "mixing_beta": 0.4,
        "conv_thr": 1.0e-4,
        "mixing_ndim": 12,
        "diagonalization": "david",
        "electron_maxstep": 125,
    }

    species_types = structure.types_of_species
    contains_magnetic_element = any(
        sp.symbol in mp_magmom_values or sp.symbol in mp_hubbard_u_values
        for sp in species_types
    )

    sga = SpacegroupAnalyzer(structure)
    symprec = 0.1

    system_params["ibrav"] = 0

    primitive_structure = sga.get_primitive_standard_structure()
    reciprocal_lengths = primitive_structure.lattice.reciprocal_lattice.abc
    kpts = np.ceil(
        np.array(reciprocal_lengths) * (kpoint_density / (4 * np.pi**3)) ** (1 / 3)
    ).astype(int)
    kpoints_grid = tuple(max(1, k) for k in kpts)

    ions_params = {
        "ion_dynamics": "bfgs",
    }
    cell_params = {"cell_dynamics": "bfgs", "cell_dofree": "all", "press_conv_thr": 0.5}

    pw_input = PWInput(
        structure=structure,
        control=control_params,
        system=system_params,
        electrons=electrons_params,
        ions=ions_params,
        cell=cell_params,
        pseudo=pseudos_for_structure,
        kpoints_grid=kpoints_grid,
    )

    input_filepath = os.path.join(output_dir, file_name)
    pw_input.write_file(input_filepath)

    print(f"\nSuccessfully created QE SCF input translating MP VASP settings:")
    print(f"  - K-points: {kpoints_grid} (generated from density: {kpoint_density})")
    print(f"  - Ecutwfc: {system_params['ecutwfc']} Ry")
    if "nspin" in system_params and system_params["nspin"] == 2:
        mag_info = {k: v for k, v in system_params.items() if "magnetization" in k}
        print(f"  - Magnetism: Enabled with initial moments: {mag_info}")
    if system_params.get("lda_plus_u"):
        hubbard_info = {k: v for k, v in system_params.items() if "Hubbard" in k}
        print(f"  - DFT+U: Enabled with parameters: {hubbard_info}")
    print(f"  - File Location: {input_filepath}")


def total_oxi_charge(structure):
    charge = 0.0
    for site in structure.sites:
        for sp, occu in site.species.items():
            oxi = getattr(sp, "oxi_state", None)
            if oxi is not None:
                charge += oxi * occu
            else:
                charge += 0.0
    return charge


from enum import Enum


class TrackingType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class EvaluationTracker:
    """
    A class to track, analyze, and visualize properties of generated crystal structures.
    """

    def __init__(
        self,
        tracking_info: List[Tuple[str, float, TrackingType]],
        output_dir: str = "tracker_results",
    ):
        """
        Initializes the EvaluationTracker.

        Args:
            tracking_info (List[Tuple[str, TrackingType]]): A list of tuples,
                where each tuple contains the property key (str) and its TrackingType.
            output_dir (str): Directory to save plots and results.
        """
        self.keys = [t[0] for t in tracking_info]
        self.targets = {t[0]: t[1] for t in tracking_info}

        self.types = {t[0]: t[2] for t in tracking_info}
        self.results: Dict[str, List[Any]] = {key: [] for key in self.keys}
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(
            f"EvaluationTracker initialized. Results will be saved to '{self.output_dir}'"
        )

    def add_result(self, struct: Structure):
        """
        Calculates and records the properties for a given structure.

        Args:
            struct (Structure): The pymatgen Structure object to analyze.
        """
        for key in self.keys:
            value = None
            if key == "density":
                value = struct.density
            elif key == "space_group":
                try:
                    sga = SpacegroupAnalyzer(struct, symprec=0.1)
                    value = sga.get_space_group_number()
                except Exception:
                    value = 1
            elif key == "reduced_formula":
                value = struct.composition.reduced_formula
            elif key == "hhi":
                value = calc_hhi(struct) / 1000.0
            if value is not None:
                self.results[key].append(value)

    def _get_stats(self, data: List[float]) -> Dict[str, float]:
        """Calculates descriptive statistics for a list of numbers."""
        if not data:
            return {}
        return {
            "min": np.min(data),
            "max": np.max(data),
            "mean": np.mean(data),
            "std": np.std(data),
        }

    def _plot_histogram(self, key: str, data: List[float]):
        """Generates and saves a violin plot for numeric data with target on x and preds on y."""
        if not data:
            return

        target = self.targets.get(key, None)
        pos = [target if target is not None else 0.0]

        plt.figure()

        if len(data) >= 2:
            parts = plt.violinplot(
                [data],
                positions=pos,
                widths=0.5,
                showmeans=True,
                showextrema=True,
                showmedians=False,
            )
            for pc in parts.get("bodies", []):
                pc.set_facecolor("#87CEEB")
                pc.set_edgecolor("black")
                pc.set_alpha(0.7)
            for k in ("cmeans", "cbars", "cmaxes", "cmins"):
                if k in parts:
                    parts[k].set_color("black")
        else:
            plt.scatter(pos, data, color="black", zorder=3, label="pred")
            plt.legend()

        plt.title(f'Pred distribution vs Target for {key.replace("_", " ").title()}')
        plt.xlabel("Target")
        plt.ylabel("Pred")
        plt.grid(axis="y", alpha=0.5)

        if target is not None:
            plt.xticks([target], [f"{target:.3g}"])
            pad = max(1e-6, abs(target) * 0.2)
            plt.xlim(target - pad, target + pad)
        else:
            plt.xticks(pos, ["N/A"])
            plt.xlim(pos[0] - 0.5, pos[0] + 0.5)

        ymin, ymax = min(data), max(data)
        yr = ymax - ymin if ymax > ymin else 1.0
        plt.ylim(ymin - 0.1 * yr, ymax + 0.1 * yr)

        plot_path = os.path.join(self.output_dir, f"{key}_violin.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close()
        print("*" * 25)
        print(f"Saved violin plot for '{key}' to {plot_path}")

    def _plot_barchart(self, key: str, counts: Counter):
        """Generates and saves a bar chart for categorical data."""
        if not counts:
            return
        labels, values = zip(*counts.most_common(15))
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color="lightcoral")
        plt.title(f'Counts of {key.replace("_", " ").title()}')
        plt.xlabel(key.replace("_", " ").title())
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"{key}_barchart.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved bar chart for '{key}' to {plot_path}")

    def summarize_results(self) -> Dict[str, Any]:
        """
        Analyzes the collected data, prints a summary, and saves plots.

        Returns:
            Dict[str, Any]: A dictionary containing statistics for numeric
                            properties and counts for categorical properties.
        """
        summary = {}
        print("\n" + "=" * 20 + " Evaluation Summary " + "=" * 20)
        for key in self.keys:
            data = self.results[key]
            if not data:
                print(f"\n--- {key.upper()} ---\nNo data collected.")
                continue

            print(f"\n--- {key.replace('_', ' ').upper()} ---")
            if self.types[key] == TrackingType.NUMERIC:
                stats = self._get_stats(data)
                summary[key] = stats
                print(f"  - Min:  {stats.get('min', 'N/A'):.4f}")
                print(f"  - Max:  {stats.get('max', 'N/A'):.4f}")
                print(f"  - Mean: {stats.get('mean', 'N/A'):.4f}")
                print(f"  - Std:  {stats.get('std', 'N/A'):.4f}")
                self._plot_histogram(key, data)

            elif self.types[key] == TrackingType.CATEGORICAL:
                counts = Counter(data)
                summary[key] = counts
                print("  - Counts:")
                for item, count in counts.most_common(10):
                    print(f"    - {item}: {count}")
                self._plot_barchart(key, counts)

        print("\n" + "=" * 58 + "\n")
        return summary


class MultiEvaluationTracker:
    """
    A wrapper that manages multiple EvaluationTracker instances and produces combined plots.
    """

    def __init__(
        self,
        trackers: Optional[List["EvaluationTracker"]] = None,
        output_dir: str = "multi_tracker_results",
    ):
        """
        Args:
            trackers: Optional list of existing EvaluationTracker instances.
            output_dir: Directory to save combined plots.
        """
        self.trackers: List["EvaluationTracker"] = trackers or []
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(
            f"MultiEvaluationTracker initialized. Combined plots will be saved to '{self.output_dir}'"
        )

    def add_tracker(self, tracker: "EvaluationTracker"):
        """Add a tracker to the collection."""
        self.trackers.append(tracker)

    def _numeric_keys(self) -> List[str]:
        """Return the union of all numeric keys across trackers."""
        keys = set()
        for t in self.trackers:
            for k, tp in t.types.items():
                if tp == TrackingType.NUMERIC:
                    keys.add(k)
        return sorted(keys)

    def plot_combined_violin(self, key: str):
        """
        1D plot: combined violin for a numeric key across all trackers.
        Adds horizontal lines at each target value to aid visual comparison.
        Saves as ..._1d.png
        """
        if not self.trackers:
            print("No trackers available.")
            return

        grouped: Dict[Optional[float], List[float]] = defaultdict(list)
        for t in self.trackers:
            if key not in t.types or t.types[key] != TrackingType.NUMERIC:
                continue
            data = t.results.get(key, [])
            if not data:
                continue
            target = t.targets.get(key, None)
            grouped[target].extend(data)

        if not grouped:
            print(f"No numeric data found for key '{key}' across trackers.")
            return

        targets = [x for x in grouped.keys() if x is not None]
        targets.sort()
        include_na = None in grouped

        positions = targets.copy()
        labels = [f"{x:.3g}" for x in targets]
        data_list = [grouped[x] for x in targets]

        if include_na:
            positions = [0.0] + positions
            labels = ["N/A"] + labels
            data_list = [grouped[None]] + data_list

        pos_targets = ([None] + targets) if include_na else targets

        if len(data_list) == 1 and len(data_list[0]) < 2:
            plt.figure()
            plt.scatter(
                [positions[0]], data_list[0], color="black", zorder=3, label="pred"
            )
            tgt = pos_targets[0] if isinstance(pos_targets, list) else pos_targets
            if tgt is not None:
                half_width = 0.3
                plt.hlines(
                    tgt,
                    positions[0] - half_width,
                    positions[0] + half_width,
                    colors="red",
                    linestyles="--",
                    linewidth=1.5,
                    label="target",
                )
            plt.title(
                f"Pred distribution vs Target for {key.replace('_', ' ').title()}"
            )
            plt.xlabel("Target")
            plt.ylabel("Pred")
            plt.xticks(positions, labels)
            plt.grid(axis="y", alpha=0.5)
            plt.legend()
        else:
            plt.figure()
            parts = plt.violinplot(
                data_list,
                positions=positions,
                widths=0.6,
                showmeans=True,
                showextrema=True,
                showmedians=False,
                vert=True,
            )
            for pc in parts.get("bodies", []):
                pc.set_facecolor("#87CEEB")
                pc.set_edgecolor("black")
                pc.set_alpha(0.7)
            for k in ("cmeans", "cbars", "cmaxes", "cmins"):
                if k in parts:
                    parts[k].set_color("black")

            added_label = False
            half_width = 0.3
            for pos, tgt in zip(positions, pos_targets):
                if tgt is None:
                    continue
                label = "target" if not added_label else "_nolegend_"
                plt.hlines(
                    tgt,
                    pos - half_width,
                    pos + half_width,
                    colors="red",
                    linestyles="--",
                    linewidth=1.5,
                    label=label,
                )
                added_label = True

            plt.title(
                f"Pred distribution vs Target for {key.replace('_', ' ').title()}"
            )
            plt.xlabel("Target")
            plt.ylabel("Pred")
            plt.grid(axis="y", alpha=0.5)
            plt.xticks(positions, labels)

            all_vals = np.array([v for lst in data_list for v in lst])
            ymin, ymax = np.min(all_vals), np.max(all_vals)
            yr = ymax - ymin if ymax > ymin else 1.0
            plt.ylim(ymin - 0.1 * yr, ymax + 0.1 * yr)

            if positions:
                xmin, xmax = min(positions), max(positions)
                xr = xmax - xmin if xmax > xmin else 1.0
                plt.xlim(xmin - 0.1 * xr, xmax + 0.1 * xr)

            if added_label:
                plt.legend()

        plot_path = os.path.join(self.output_dir, f"{key}_combined_violin_1d.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved 1D violin plot for '{key}' to {plot_path}")

    def plot_combined_2d(self, key_x: str, key_y: str):
        """
        2D plot: scatter of predictions for two numeric keys across all trackers.
        - X-axis: predictions for key_x
        - Y-axis: predictions for key_y
        Colored by (target_x, target_y) pair like the example image.
        Saves as ..._2d.png
        """
        if not self.trackers:
            print("No trackers available.")
            return

        any_numeric = False
        for t in self.trackers:
            if (
                key_x in t.types
                and t.types[key_x] == TrackingType.NUMERIC
                and key_y in t.types
                and t.types[key_y] == TrackingType.NUMERIC
            ):
                any_numeric = True
                break
        if not any_numeric:
            print(
                f"Keys '{key_x}' and/or '{key_y}' are not numeric in the provided trackers."
            )
            return

        groups: Dict[
            Tuple[Optional[float], Optional[float]], List[Tuple[float, float]]
        ] = defaultdict(list)

        for t in self.trackers:
            if key_x not in t.types or key_y not in t.types:
                continue
            if (
                t.types[key_x] != TrackingType.NUMERIC
                or t.types[key_y] != TrackingType.NUMERIC
            ):
                continue

            xs = t.results.get(key_x, [])
            ys = t.results.get(key_y, [])
            n = min(len(xs), len(ys))
            if n == 0:
                continue

            tx = t.targets.get(key_x, None)
            ty = t.targets.get(key_y, None)
            for xv, yv in zip(xs[:n], ys[:n]):
                groups[(tx, ty)].append((xv, yv))

        if not groups:
            print(
                f"No paired numeric data found for '{key_x}' and '{key_y}' across trackers."
            )
            return

        plt.figure(figsize=(8, 6))
        keys_sorted = sorted(
            groups.keys(),
            key=lambda p: (
                float("inf") if p[0] is None else p[0],
                float("inf") if p[1] is None else p[1],
            ),
        )
        for i, pair in enumerate(keys_sorted):
            pts = np.array(groups[pair])
            label = f"({('N/A' if pair[0] is None else f'{pair[0]:.3g}')}, {('N/A' if pair[1] is None else f'{pair[1]:.3g}')})"
            plt.scatter(
                pts[:, 0], pts[:, 1], s=14, alpha=0.7, edgecolors="none", label=label
            )

        plt.title(
            f"Multi-Property Distribution of Generated Samples\n{key_x.replace('_',' ').title()} vs {key_y.replace('_',' ').title()}"
        )
        plt.xlabel(key_x.replace("_", " ").title())
        plt.ylabel(key_y.replace("_", " ").title())
        plt.grid(alpha=0.3)
        plt.legend(
            title="Targets",
            fontsize="small",
            title_fontsize="small",
            loc="best",
            frameon=True,
        )

        all_x = np.concatenate([np.array([p[0] for p in v]) for v in groups.values()])
        all_y = np.concatenate([np.array([p[1] for p in v]) for v in groups.values()])
        xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
        ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
        xr = xmax - xmin if xmax > xmin else 1.0
        yr = ymax - ymin if ymax > ymin else 1.0
        plt.xlim(xmin - 0.08 * xr, xmax + 0.08 * xr)
        plt.ylim(ymin - 0.08 * yr, ymax + 0.08 * yr)

        plot_path = os.path.join(
            self.output_dir, f"{key_x}__{key_y}_combined_scatter_2d.png"
        )
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved 2D scatter plot for '{key_x}' vs '{key_y}' to {plot_path}")

    def plot_auto(
        self, key: Optional[str] = None, pair: Optional[Tuple[str, str]] = None
    ):
        """
        Auto plot:
        - If exactly one numeric key (or 'key' provided): 1D violin for that key.
        - If two numeric keys (or 'pair' provided): 2D scatter for that pair.
        Defaults to the first two sorted numeric keys if more than two exist and 'pair' not provided.
        """
        nkeys = self._numeric_keys()

        if pair is not None:
            kx, ky = pair
            self.plot_combined_2d(kx, ky)
            return

        if key is not None:
            self.plot_combined_violin(key)
            return

        if len(nkeys) == 0:
            print("No numeric keys found across trackers.")
        elif len(nkeys) == 1:
            self.plot_combined_violin(nkeys[0])
        else:
            self.plot_combined_2d(nkeys[0], nkeys[1])

    def plot_all_numeric_violins(self):
        """Create combined violin plots for all numeric keys across trackers (1D only)."""
        keys = self._numeric_keys()
        if not keys:
            print("No numeric keys found across trackers.")
            return
        for key in keys:
            self.plot_combined_violin(key)
