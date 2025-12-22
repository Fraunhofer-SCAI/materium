# Materium
This is the official repository for the paper ["Materium: An Autoregressive Approach for Material Generation"](https://doi.org/10.48550/arXiv.2512.07486)

# Instructions

```bash
# pip install uv

$ uv venv llm_env --python 3.10
$ source llm_env/bin/activate
$ uv pip install -e .
```
## Docs
A doc generated with lazy doc can be found in the `docs` folder, there all functions that can be used are listed in the corresponding files

## Training
To get the training data, please follow the instruction on the [MatterGen page](https://github.com/microsoft/mattergen), but here are the main steps for this project:

```bash
$ cd vendor/mattergen
# Make sure you have git lfs
$ # Download file from LFS
$ git lfs pull -I data-release/alex-mp/alex_mp_20.zip --exclude=""
$ unzip data-release/alex-mp/alex_mp_20.zip -d datasets
$ csv-to-dataset --csv-folder datasets/alex_mp_20/ --dataset-name alex_mp_20 --cache-folder datasets/cache
# Copy the datasets/cache to data
$ mkdir -p src/materium/data/alex_mp_20
$ cp -r datasets/cache/train src/materium/data/alex_mp_20
$ git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""
```

```bash
# To train the model you can run in
$(llm_env) python src/materium/train.py
# or use the sbatch script
$(llm_env) sbatch trainMaterialLLM.sh
```
However you need to change the SORTING_ORDER such for the model that you want to have, the default is the REVERSE_SPECIES (which is then the M_large model, or the one with the heaviest atoms first)
```python
tokenizer = CrystalTokenizer(
        element_vocab=ELEMENT_VOCAB,
        lattice_stats=LATTICE_STATS,
        num_quant_bins=1024,
        sorting_order=SortingOrder.REVERSE_SPECIES,# <--- CHANGE THIS to either SortingOrder.SPECIES,SortingOrder.XYZ,SortingOrder.RANDOM
        sequence_order=SequenceOrder.ATOMS_FIRST
)
```

## Generation
```bash
$(llm_env) python src/materium/llm_generate.py
```

The llm_generate.py script has different arguments:
### Model and Output:
- `--ckpt_dir`: Specifies the directory containing the pre-trained model checkpoint.
- `--ckpt_file`: The specific checkpoint file (.pt) to use for generation.
- `--out_dir_prefix`: A prefix for the output folder where generated materials are saved. (in the src/materium/qe_input)

### Generation Parameters:

- `--num_samples`: The total number of material samples to generate. (Default: 2)
- `--temperature`: Controls the randomness of the generation. Higher values (e.g., 1.0) lead to more diversity, while lower values (e.g., 0.2) produce more predictable outputs. (Default: 1.0)
- `--relax`: If included, this flag triggers a relaxation calculation for the generated materials using MatterSim.

### Conditional Generation:

You can guide the generation process by specifying desired material properties. The script accepts single or multiple values for each property.
- `--density`: Target density of the material. e.g. 2.0
- `--formula`: Target reduced chemical formula (e.g., "SiO2").
- `--space_group`: Target space group number.
- `--band_gap`: Target band gap in eV.
- `--mag_density`: Target magnetic moment density in A⁻³.
- `--hhi`: Target Herfindahl-Hirschman Index for element scarcity.

Example generation
```bash
CKPT_DIR="llm_v3_LARGERopeStand_labelsmoothing=0.0_sorev_species_seqoatoms_first_material_llama_adamw_condition_latticelast_dim512"
CKPT_FILE="model_47_0.843.pt"
NUM_SAMPLES=512

$ python llm_generate.py --ckpt_dir=$CKPT_DIR --ckpt_file=$CKPT_FILE --num_samples=$NUM_SAMPLES --hhi 1.25 2.5 3.75 5.0 --density 1.0 2.5 5.0 7.5
```
When you specify multiple values for the density etc. then it will start a seperate calculation of $NUM_SAMPLES for all specified values.
When specifiying multiple values with multiple conditions then it creates ALL combinations of the conditions i.e. hhi=1.0 density=1.0, hhi=1.0 density=2.5, .... hhi=3.75 density=7.5

### Generation for Unconditional and Conditonal Generations

For simple to evaluate properties such as the reduced formula, density and the space group
```bash
$ sbatch src/materium/generate_materials_comparison.sh
```

This generates the 1024 unconditional materials for all the 4 models (random, xyz, low, high), the single conditional values reduced formula, density, space group, band gap and mag density, aswell as the multi conditional values as described in the paper.
These are stored in src/materium/qe_input/[MODEL_NAME]/[FOLDER_NAME]

Here MODEL_NAME is:
- For `M_low MODEL_NAME="llm_v3_LARGERopeStand_labelsmoothing=0.0_sospecies_seqoatoms_first_material_llama_adamw_condition_latticelast_dim512"`
- For `M_high MODEL_NAME="llm_v3_LARGERopeStand_labelsmoothing=0.0_sorev_species_seqoatoms_first_material_llama_adamw_condition_latticelast_dim512"`
- For `M_xyz MODEL_NAME"llm_v3_LARGERopeStand_labelsmoothing=0.0_soxyzorder_seqoatoms_first_material_llama_adamw_condition_latticelast_dim512"`
- For `M_random MODEL_NAME"llm_v3_LARGERopeStand_labelsmoothing=0.0_sorandom_seqoatoms_first_material_llama_adamw_condition_latticelast_dim512"`

Examples for FOLDER_NAME:
- `FOLDER_NAME=band_gap_0.0_hhi_1.25_temp=1.00` which is the generation for bandgap of 0.0 eV and HHI Score of 1250 at a temperature of 1.0
- `FOLDER_NAME=density_1.0_temp=1.00` which is the density at 1.0


## Conditional Generation for Band Gap and Mag Density
To evaluate these we need to calculate them via quantum espresso (v.7.4.1.) here we build that as a qe.sif (singularity file) in the src/materium/qe_input/qe.sif path
The corresponding container / docker image can be found in the "quantum_espresso" folder, you can read more in the `quantum_espresso/instructions.md`

After running the `sbatch src/materium/generate_materials_comparison.sh` you have the folders in the qe_input, then you can run the run_vc_relax_{COND}.sbatch where COND is either `band_gap` or `mag_density`

Run the scripts via `sbatch run_vc_relax_{COND}.sbatch`

### Mag density
After the `sbatch run_vc_relax_mag_density.sbatch` command is done you can run the `src/materium/qe_input/analyse_mag_density.py` to then create the graphs for the paper.
However depending on which you want to generate single/multi you need to modify the script slightly  at the start of the `if __name__ == "__main__"` block.
The current experiments are still in the `NEW_gen_rev_species_47` folder instead of the llm_v3...

### Band Gap
After the `sbatch run_vc_relax_band_gap.sbatch` we still need to do a couple of more steps
1. Run the `python src/materium/qe_input/create_bandgap_files.py` (here you need to modify the script to have the single or multi conditional folders)
This script creates the `run_bands_array.sbatch` in each of the subfolders for the bandgaps e.g. `qe_input/NEW_gen_rev_species_47/band_gap_0.0/run_band_array.sbatch`, or for the other files.
After running the sbatch files FOR EACH band gap folder, then you can do the collection
2. To see the results go to the `qe_input/analyse_bandgap.py` this will create the graphs by reading from the folders here again at the start of the main you need to specify the `target_folders`, i.e. which folders should be read.
