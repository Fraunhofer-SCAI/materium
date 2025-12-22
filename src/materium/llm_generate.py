from collections import defaultdict
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
from materium.train import *
from pymatgen.io.pwscf import PWInput
import time
from pymatgen.analysis.hhi import HHIModel
from materium.generation_utils import *

from mattergen.evaluation.evaluate import evaluate
from pymatgen.io.ase import AseAtomsAdaptor
from mattersim.forcefield.potential import MatterSimCalculator

from mattersim.applications.relax import Relaxer


def run_mattersim_relaxation(
    pseudo_potential_path: str,
    conditions: Dict[str, float],
    qe_save_folder: str,
    generated_materials_for_conditions: List[Structure],
    DEVICE: str = "cpu",
):
    rmsds = []
    relaxed_structures = []
    relaxed_energies = []
    for i, structure in generated_materials_for_conditions:
        try:
            q = total_oxi_charge(structure)
            print(f"Structure {i}: total oxidation-state charge = {q:+.6f} e")
            if abs(q) > 1e-6:
                print(f"Warning: Structure {i} has non-zero total charge ({q:+.6f} e)")

            structure = structure.remove_oxidation_states()
            final_structure, final_energy, final_energy_per_atom, final_forces = (
                relax_structure_mattersim(
                    structure,
                    fmax=0.1,
                    device="cpu",
                    constrain_symmetry=True,
                    optimizer="BFGS",
                )
            )
            relaxed_structures.append(final_structure)
            relaxed_energies.append(final_energy)

            cif_path = os.path.join(
                qe_save_folder, f"material_sample_gen_relaxed_{i}.cif"
            )
            final_structure.to(cif_path, "cif")

            if "band_gap" in conditions.keys():
                print("Creating Band_gap QE")
                create_qe_scf_from_mp_config(
                    final_structure,
                    qe_save_folder,
                    pseudo_potential_path,
                    file_name=f"qe_input_{i}.in",
                    gen_idx=i,
                )
            else:
                print("Creatin Mag Density")
                create_qe_magdensity_input(
                    final_structure,
                    qe_save_folder,
                    pseudo_potential_path,
                    file_name=f"qe_input_{i}.in",
                )

            print(f"The final energy is {final_energy_per_atom:.3f} eV per atom.")
            difference = abs(structure.lattice.volume / final_structure.lattice.volume)
            print(f"Difference Volume: {difference:.3f} Å")
            print(
                f"Structure {structure.get_space_group_info()} Final structure Space group: {final_structure.get_space_group_info()}"
            )
            print(f"Reduced formula {final_structure.composition.reduced_formula}")

            def print_lattice(s):
                print(
                    f"Lattice: a={s.lattice.a:.3f}, b={s.lattice.b:.3f}, c={s.lattice.c:.3f}, "
                    f"alpha={s.lattice.alpha:.2f}, beta={s.lattice.beta:.2f}, gamma={s.lattice.gamma:.2f}"
                )
                print(f"Volume: {s.volume:.3f}")

            print_lattice(structure)
            print_lattice(final_structure)
            rmsd, found_match = compute_rmsd_angstrom(structure, final_structure)
            print("RMSD Angström", rmsd, "Found match", found_match)
            rmsds.append(rmsd)

            print(
                "MSE CART:",
                np.sqrt(
                    (structure.cart_coords - final_structure.cart_coords) ** 2
                ).mean(),
            )

        except Exception as e:
            print("Creating QE Input error", e)
            continue

    print("Mean rmsd", np.mean(rmsds))
    eval_output = evaluate(
        relaxed_structures,
        energies=relaxed_energies,
        device=DEVICE,
        save_as=os.path.join(
            qe_save_folder, f"RESULT_MATTERGENEVALUATE_llm_new_{num_total_samples}"
        ),
        relax=False,
    )
    print("Mattergen eval:")
    print(eval_output)
    with open(os.path.join(qe_save_folder, "mattergen_results_new.json"), "w") as f:
        json.dump(eval_output, f)


def run_generation(
    temperature: float,
    conditions: Dict[str, float],
    tracking_info_reference: Dict[str, TrackingType],
    multi_eval_tracker: MultiEvaluationTracker,
    qe_save_folder: str,
    DEVICE: str = "cpu",
):
    save_struct_path = os.path.join(
        ckpt_path,
        "generated",
        "cond_"
        + "_".join(
            [
                f"{k}:{v:.1f}" if isinstance(v, float) else f"{k}:{v}"
                for k, v in conditions.items()
            ]
        ),
    )
    os.makedirs(save_struct_path, exist_ok=True)
    invalid_structures = 0
    tracking_info = [(k, v, tracking_info_reference[k]) for k, v in conditions.items()]
    tracker = EvaluationTracker(tracking_info, output_dir=qe_save_folder)
    generated_materials_for_conditions = []
    for gen_idx in range(num_total_samples):
        try:
            start_time = time.time()
            generated_tokens = generate_structure(
                model,
                tokenizer,
                device=DEVICE,
                temperature=temperature,
                top_p=1.0,
                conditions=conditions,
                classifier_guidance_weight=2.5,
            )
            generated_material: Structure = tokenizer.detokenize(generated_tokens)
            print(f"Time taken to generate {time.time() - start_time} sec")
            print(
                generated_material.composition.reduced_formula,
                "Generated material density",
                generated_material.density,
                "Charge",
                generated_material.charge,
                generated_material.composition.oxi_state_guesses(),
            )

            tracker.add_result(generated_material)

            cif_path = os.path.join(
                qe_save_folder, f"material_sample_gen_{gen_idx}.cif"
            )
            generated_material.to(cif_path, "cif")

            print("Saved structure to", cif_path)
            all_generated.append(generated_material)
            generated_materials_for_conditions.append((gen_idx, generated_material))
        except Exception as e:
            print("Invalid structure", e)
            invalid_structures += 1

    multi_eval_tracker.add_tracker(tracker)
    final_summary = tracker.summarize_results()
    with open(os.path.join(qe_save_folder, "results.json"), "w") as f:
        json.dump(final_summary, f)
    return generated_materials_for_conditions


def create_condition_dict(args):
    conditions = {}
    tracking_info_reference = {
        "density": TrackingType.NUMERIC,
        "space_group": TrackingType.CATEGORICAL,
        "reduced_formula": TrackingType.CATEGORICAL,
        "band_gap": TrackingType.NUMERIC,
        "mag_density": TrackingType.NUMERIC,
        "bulk_modulus": TrackingType.NUMERIC,
        "hhi": TrackingType.NUMERIC,
    }
    if args.density is not None:
        conditions["density"] = [float(d) for d in args.density]
    if args.formula is not None:
        conditions["reduced_formula"] = args.formula
    if args.space_group is not None:
        conditions["space_group"] = [int(d) for d in args.space_group]

    if args.band_gap is not None:
        conditions["band_gap"] = [np.log(bg + 1e-4) for bg in args.band_gap]
    if args.mag_density is not None:
        conditions["mag_density"] = [np.log(md + 1e-4) for md in args.mag_density]
    if args.bulk_modulus is not None:
        conditions["bulk_modulus"] = args.bulk_modulus
    if args.hhi is not None:
        conditions["hhi"] = args.hhi

    keys = conditions.keys()
    vals = [conditions[k] for k in keys]
    all_combs = product(*vals)
    return tracking_info_reference, keys, all_combs


def load_model(args):
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", args.ckpt_dir)
    model, ckpt = LLamaTransformer.load(
        os.path.join(ckpt_path, args.ckpt_file), strict=False
    )

    # ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "llm_v2_material_llama_adamw_condition_latticelast_dim384")
    # model,ckpt = LLamaTransformer.load(os.path.join(ckpt_path,"model_41_0.815.pt"),strict=False)
    ELEMENT_VOCAB = [Element.from_Z(i).symbol for i in range(1, 100)]
    LATTICE_STATS = {
        "a": (2.0, 10.0),
        "b": (2.0, 12.5),
        "c": (2.0, 20.0),
        "alpha": (60.0, 120.0),
        "beta": (60.0, 120.0),
        "gamma": (60.0, 120.0),
    }
    if ckpt.get("tokenizer") is None:
        tokenizer = CrystalTokenizer(
            element_vocab=ELEMENT_VOCAB,
            lattice_stats=LATTICE_STATS,
            num_quant_bins=1024,
            sorting_order=SortingOrder.SPECIES,
            sequence_order=SequenceOrder.ATOMS_FIRST,
        )
    else:
        print("Loading tokenizer")
        tokenizer = ckpt.get("tokenizer")
        tokenizer = CrystalTokenizer.from_dict(tokenizer)

        so = getattr(tokenizer, "sequence_order", None)
        if so is None:
            # Kind of a fix for older versions of the tokenizer
            tokenizer = CrystalTokenizer(
                element_vocab=ELEMENT_VOCAB,
                lattice_stats=LATTICE_STATS,
                num_quant_bins=1024,
                sorting_order=tokenizer.sorting_order,
                sequence_order=getattr(
                    tokenizer, "sequence_order", SequenceOrder.ATOMS_FIRST
                ),
            )

    model = model.to(DEVICE)
    print("ROPE MODE", model.params.rope_mode)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters {num_parameters:.3f}")
    return model, tokenizer, ckpt_path, DEVICE


if __name__ == "__main__":

    import argparse
    from itertools import product

    parser = argparse.ArgumentParser(description="Run LLaMa Transformer Evaluation")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="llm_v3_LARGERopeStand_labelsmoothing=0.0_sospecies_seqoatoms_first_material_llama_adamw_condition_latticelast_dim512",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--ckpt_file", type=str, default="model_43_0.878.pt", help="Checkpoint filename"
    )
    parser.add_argument(
        "--out_dir_prefix",
        type=str,
        default="",
        help="Name of the folder to store the quantum espresso inputs and results",
    )

    parser.add_argument(
        "--num_samples", type=int, default=2, help="Number of samples to generate"
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument(
        "--relax", action="store_true", help="Relax then generated materials"
    )

    parser.add_argument(
        "--density", nargs="+", default=None, help="Condition: Density of the Material"
    )
    parser.add_argument(
        "--formula",
        nargs="+",
        default=None,
        help="Condition: Reduced Formula of the Material",
    )
    parser.add_argument(
        "--space_group",
        nargs="+",
        default=None,
        help="Condition: Space Group of the Material",
    )
    parser.add_argument(
        "--band_gap",
        nargs="+",
        type=float,
        default=None,
        help="Condition: Band Gap of the Material (eV)",
    )
    parser.add_argument(
        "--mag_density",
        nargs="+",
        type=float,
        default=None,
        help="Condition: Magnetic Moment Density (A^-3)",
    )
    parser.add_argument(
        "--bulk_modulus",
        nargs="+",
        type=float,
        default=None,
        help="Condition: Bulk Modulus (GPa)",
    )
    parser.add_argument(
        "--hhi",
        nargs="+",
        type=float,
        default=None,
        help="Condition: Herfindahl-Hirschman Index (element scarcity)",
    )

    args = parser.parse_args()
    if args.out_dir_prefix == "":
        args.out_dir_prefix = args.ckpt_dir
    print("outdir", args.out_dir_prefix)

    pseudo_potential_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "pseudo_potentials_qe")
    )

    model, tokenizer, ckpt_path, DEVICE = load_model(args)
    print("Relaxing", args.relax)
    RELAX_GENERATED_MATERIALS = args.relax
    num_total_samples = args.num_samples
    all_generated = []
    invalid_structures = 0
    temperature = args.temperature

    tracking_info_reference, keys, all_combs = create_condition_dict(args)

    multi_eval_tracker = MultiEvaluationTracker(
        output_dir=os.path.join(
            os.path.dirname(__file__), f"qe_input", args.out_dir_prefix
        )
    )

    all_qe_folders = []
    for cond_comb in all_combs:
        conditions = {k: v for k, v in zip(keys, cond_comb)}
        print(conditions)
        postfix = "_".join(
            (
                f"{k}_{v}"
                if k != "band_gap" and k != "mag_density"
                else f"{k}_{np.exp(v):.1f}"
            )
            for k, v in conditions.items()
        )
        postfix = f"{postfix}_temp={temperature:.2f}"
        qe_save_folder = os.path.join(
            f"{args.out_dir_prefix}", f"{postfix}"
        )  # f"gen_bandgap_{args.band_gap}_neg_down"
        qe_path = os.path.join(os.path.dirname(__file__), f"qe_input")
        qe_prefix_path = os.path.join(qe_path, args.out_dir_prefix)
        qe_save_folder = os.path.join(qe_path, qe_save_folder)
        os.makedirs(qe_save_folder, exist_ok=True)
        all_qe_folders.append(qe_save_folder)

        print("QE Save folder", qe_save_folder)

        # https://github.com/materialsproject/pymatgen/blob/master/src/pymatgen/analysis/hhi.py
        generated_materials_for_conditions = run_generation(
            temperature,
            conditions,
            tracking_info_reference,
            multi_eval_tracker,
            qe_save_folder,
            DEVICE=DEVICE,
        )

        if RELAX_GENERATED_MATERIALS:
            run_mattersim_relaxation(
                pseudo_potential_path,
                conditions,
                qe_save_folder,
                generated_materials_for_conditions,
                DEVICE=DEVICE,
            )

    print("Saving vc_relax")

    multi_eval_tracker.plot_auto()
    folders_str = " ".join(
        [f'"{os.path.abspath(folder)}"' for folder in all_qe_folders]
    )
    with open(
        os.path.join(qe_prefix_path, f"run_vc_relax_{'_'.join(keys)}.sbatch"), "w"
    ) as vc_file:
        vc_file.write(
            f"""#!/bin/bash
#SBATCH --job-name=qe_array
#SBATCH --output=logs/qe_job_%A_%a.out
#SBATCH --error=logs/qe_job_%A_%a.err
#SBATCH --array=0-{args.num_samples-1}%16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --exclusive

# Set OpenMP threads
export OMP_NUM_THREADS=1 #1
export NP=32 #32
unset TMPDIR # Important such that you dont get errors with /scratch/
# Define input/output filenames based on array index
IDX=${{SLURM_ARRAY_TASK_ID}}
QE_SIF=({os.path.join(qe_path, 'qe.sif')})
FOLDERS=({folders_str})

# INPUT_FILE="qe_input_${{IDX}}.in"
# INPUT_FILE="./test_set_bandgap/qe_dset_bandgap_input_${{IDX}}.in"
# INPUT_FILE="./gen_bandgap_[1.5]/qe_input_${{IDX}}.in" 
# OUTPUT_FILE="${{INPUT_FILE}}.out"

# Optional: Print diagnostic info
# echo "Running QE calculation for index ${{IDX}} on node ${{SLURMD_NODENAME}}"
# echo "Input:  ${{INPUT_FILE}}"
# echo "Output: ${{OUTPUT_FILE}}"

# Run the calculation
# singularity run ../qe.sif -in ${{INPUT_FILE}} > ${{OUTPUT_FILE}}

for F in "${{FOLDERS[@]}}"; do
  cd "$F" || continue

  INPUT_FILE="qe_input_${{IDX}}.in"
  if [[ -f "$INPUT_FILE" ]]; then
    OUTPUT_FILE="${{INPUT_FILE}}.out"
    echo "Running QE in $(pwd) for index ${{IDX}}"
    singularity run "$QE_SIF" -in "$INPUT_FILE" > "$OUTPUT_FILE"
  else
    echo "Skipping $(pwd): ${{INPUT_FILE}} not found"
  fi

  cd ..
done

"""
        )

# For 64 {'avg_energy_above_hull_per_atom': 0.0840182275803305, 'avg_rmsd_from_relaxation': 0.0010025098361401232, 'frac_novel_unique_stable_structures': 0.4032258064516129, 'frac_stable_structures': 0.7903225806451613, 'frac_successful_jobs': 1.0, 'avg_comp_validity': 0.8870967741935484, 'avg_structure_comp_validity': 0.8870967741935484, 'avg_structure_validity': 1.0, 'frac_novel_structures': 0.5967741935483871, 'frac_novel_systems': 0.3387096774193548, 'frac_novel_unique_structures': 0.5967741935483871, 'frac_unique_structures': 1.0, 'frac_unique_systems': 1.0, 'precision': 0.4032258064516129, 'recall': 3.3097044079352525e-05}
# python llm_generate.py --ckpt_dir="llm_v3_LARGERopeStand_labelsmoothing=0.0_soele_neg_down_seqoatoms_first_material_llama_adamw_condition_latticelast_dim512" --ckpt_file="model_18_0.907.pt" --num_samples=64 --relax
