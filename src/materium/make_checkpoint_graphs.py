import json
import os
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
import seaborn as sns

import pandas as pd

from mattergen.evaluation.utils.utils import compute_rmsd_angstrom


def plot_density_summary(model_dir, title, output_filename, font_size=32):
    """
    Creates a plot of mean generated density vs. target density with error bars.
    This is used because the results.json only contains summary statistics.

    Args:
        model_dir (str): The path to the model's 'generated' directory.
        title (str): The title for the plot.
        output_filename (str): The filename to save the plot.
    """
    print(f"--- Plotting Density Summary for: {title} ---")
    target_densities = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    generated_means = []
    generated_stds = []

    for density in target_densities:
        folder_path = os.path.join(model_dir, f"density_{density}_temp=1.00")
        results_file = os.path.join(folder_path, "results.json")

        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)

                # Extract mean and std from the summary dictionary
                density_stats = data.get("density", {})
                mean = density_stats.get("mean")
                std = density_stats.get("std")

                if mean is not None and std is not None:
                    generated_means.append(mean)
                    generated_stds.append(std)
                    print(
                        f"Target: {density:.1f} -> Found Mean: {mean:.2f}, Std: {std:.2f}"
                    )
                else:
                    # Append NaN if data is missing to keep arrays aligned
                    generated_means.append(float("nan"))
                    generated_stds.append(float("nan"))
                    print(
                        f"Warning: Missing 'mean' or 'std' for target density {density} in {results_file}"
                    )

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading or parsing {results_file}: {e}")
        else:
            print(
                f"Warning: results.json not found for target density {density} at {folder_path}"
            )

    if not generated_means:
        print("No density data found to plot.")
        return

    plt.figure(figsize=(10, 8))

    min_val = min(target_densities)
    max_val = max(target_densities)
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        label="Ideal (Target = Generated)",
    )

    plt.errorbar(
        target_densities,
        generated_means,
        yerr=generated_stds,
        fmt="o",
        capsize=5,
        label="Generated Mean Density",
        markersize=8,
        ecolor="gray",
    )

    plt.title(title)
    plt.xlabel("Target Density (g/cm³)")
    plt.ylabel("Generated Mean Density (g/cm³)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(target_densities)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), output_filename)
    plt.savefig(output_path)
    print(f"Saved density plot to {output_path}")
    plt.close()


def get_canonical_formula(formula: str) -> str:
    """
    Converts a chemical formula string to a canonical representation by sorting elements alphabetically.
    This ensures that, for example, 'MgAl2O4' and 'Al2MgO4' are treated as the same.

    Example: 'YBa2Cu3O7' -> 'Ba2Cu3O7Y'
             'O2Si'      -> 'O2Si'
             'Al2MgO4'   -> 'Al2MgO4'
    """
    pairs = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    if not pairs:
        return formula

    composition = defaultdict(int)
    for element, count in pairs:
        composition[element] += int(count) if count else 1

    canonical_formula = ""
    for element in sorted(composition.keys()):
        count = composition[element]
        canonical_formula += element
        if count > 1:
            canonical_formula += str(count)

    return canonical_formula


from pymatgen.core import Structure
from tqdm import tqdm


def plot_categorical_data(
    model_dir,
    property_name,
    target_values,
    title,
    output_filename,
    title_size=36,  # 36
    axis_label_size=28,  # 30
    tick_size=28,  # 30
    legend_size=18,
    pct_label_fontsize=14,
    inside_label_fontsize=17,
    fig_size=(18, 9),
    bar_width=0.30,
    loc="lower left",
):
    """
    Creates an intelligent grouped bar chart of the top 3 generated categories.
    - Text is only placed inside bars if they are tall enough to avoid being cut off.
    - Text color (black/white) is chosen automatically for best contrast against the bar color.

    Args:
        model_dir (str): The path to the model's 'generated' directory.
        property_name (str): The name of the property (e.g., "space_group").
        target_values (list): The list of target property values.
        title (str): The title for the plot.
        output_filename (str): The filename to save the plot.
    """
    print(f"\n=============================================================")
    print(f"--- Plotting {property_name.replace('_', ' ')} for: {title} ---")
    print(f"=============================================================")

    top_results_by_target = {}
    charge_percentages = []
    for value in target_values:
        folder_path = os.path.join(model_dir, f"{property_name}_{value}_temp=1.00")
        number_of_zero_charges = 0
        number_of_structures = 0
        all_files = os.listdir(folder_path)
        for filename in tqdm(all_files, total=len(all_files)):
            if not filename.endswith(".cif"):
                continue
            number_of_structures += 1
            file_path = os.path.join(folder_path, filename)
            try:
                structure = Structure.from_file(file_path)
                if structure.charge == 0:
                    number_of_zero_charges += 1
            except Exception as e:
                # print(f"Could not read file {filename}", e)
                pass
        print(
            f"Number of charges = 0 {number_of_zero_charges}/{number_of_structures} = {number_of_zero_charges/number_of_structures*100.0:.3f}"
        )
        charge_percentages.append(number_of_zero_charges / number_of_structures * 100.0)
        results_file = os.path.join(folder_path, "results.json")

        print(f"\n----- Analyzing Target: [{value}] -----")

        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                counts_dict = data.get(property_name, {})

                if counts_dict:
                    total_samples = sum(counts_dict.values())
                    print(f"Total Valid Samples Generated: {total_samples}")
                    sorted_counts = sorted(
                        counts_dict.items(), key=lambda item: item[1], reverse=True
                    )

                    top_3 = []
                    for i, (category, count) in enumerate(sorted_counts[:3]):
                        percentage = (count / total_samples) * 100
                        category = get_canonical_formula(category)
                        value = get_canonical_formula(str(value))
                        top_3.append((category, percentage))
                        is_correct_marker = (
                            " <--- (Correct)" if category == str(value) else ""
                        )
                        print(
                            f"  {i+1}. {category:<12}: {count:>4} samples ({percentage:>5.1f}%){is_correct_marker}"
                        )
                    top_results_by_target[str(value)] = top_3
                else:
                    print("Warning: No count data found.")
                    top_results_by_target[str(value)] = []
            except Exception as e:
                print(f"Error processing file: {e}")
                top_results_by_target[str(value)] = []
        else:
            print("Warning: results.json not found.")
            top_results_by_target[str(value)] = []

    categories = list(top_results_by_target.keys())
    top1_labels, top1_percents = [], []
    top2_labels, top2_percents = [], []
    top3_labels, top3_percents = [], []

    for cat in categories:
        results = top_results_by_target[cat]
        top1_labels.append(results[0][0] if len(results) > 0 else "")
        top1_percents.append(results[0][1] if len(results) > 0 else 0)
        top2_labels.append(results[1][0] if len(results) > 1 else "")
        top2_percents.append(results[1][1] if len(results) > 1 else 0)
        top3_labels.append(results[2][0] if len(results) > 2 else "")
        top3_percents.append(results[2][1] if len(results) > 2 else 0)

    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    rects1 = ax.bar(
        x - bar_width, top1_percents, bar_width, label="Top 1 Guess", color="C0"
    )
    rects2 = ax.bar(x, top2_percents, bar_width, label="Top 2 Guess", color="C1")
    rects3 = ax.bar(
        x + bar_width, top3_percents, bar_width, label="Top 3 Guess", color="C2"
    )

    def get_text_color(bg_color):
        r, g, b, _ = bg_color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "black" if luminance > 0.5 else "white"

    def add_smart_bar_labels(rects, labels):
        HEIGHT_THRESHOLD_FOR_INTERNAL_TEXT = 4
        for rect, label in zip(rects, labels):
            height = rect.get_height()

            if height > 0:
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=pct_label_fontsize,
                )

            if height > HEIGHT_THRESHOLD_FOR_INTERNAL_TEXT:
                text_color = get_text_color(rect.get_facecolor())
                # Scale inside-label font slightly with bar height
                fs = min(
                    inside_label_fontsize + int(height / 15), inside_label_fontsize + 6
                )
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    height / 2,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    rotation=90,
                    fontsize=fs,
                )

    add_smart_bar_labels(rects1, top1_labels)
    add_smart_bar_labels(rects2, top2_labels)
    add_smart_bar_labels(rects3, top3_labels)

    ax.set_title(title, pad=20, fontsize=title_size)
    ax.set_xlabel(
        f"Target {property_name.replace('_', ' ').title()}",
        labelpad=15,
        fontsize=axis_label_size,
    )
    ax.set_ylabel(
        "Percentage of \nGenerated Samples (%)", labelpad=15, fontsize=axis_label_size
    )
    ax.set_xticks(x, categories)
    ax.tick_params(axis="x", labelsize=tick_size, rotation=45)
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(fontsize=legend_size, ncol=3, loc=loc)

    fig.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), output_filename)
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved improved grouped bar plot to {output_path}")
    plt.close()


def analyze_experiment(model_dir: str, experiment_folder: str):
    """
    Parses a multi-condition folder, prints a detailed analysis, and
    RETURNS a dictionary of performance metrics for plotting.
    """
    print(f"\n{'='*25} ANALYSIS FOR: {experiment_folder} {'='*25}")

    performance_metrics = {}

    target_conditions = {}
    conditions = experiment_folder.replace("cond_", "").split("_")
    for cond in conditions:
        try:
            key, value = cond.split(":", 1)
            target_conditions[key] = value
        except ValueError:
            print(f"  - Could not parse condition: {cond}")
            continue
    print("--- Target Conditions ---")
    for key, value in target_conditions.items():
        print(f"  - {key.replace('_', ' ').title()}: {value}")
    print("-------------------------")

    results_file = os.path.join(model_dir, experiment_folder, "results.json")
    if not os.path.exists(results_file):
        print("\n[ERROR] results.json not found for this experiment.")
        return None
    with open(results_file, "r") as f:
        data = json.load(f)

    if "density" in target_conditions and "density" in data:
        target_density = float(target_conditions["density"])
        mean_gen_density = data["density"].get("mean", None)
        print(f"\n--- Density Performance ---")
        print(f"  - Target Density: {target_density:.2f}")
        if mean_gen_density is not None:
            std_gen_density = data["density"].get("std", 0)
            error = abs(mean_gen_density - target_density)
            score = max(0, 100 * (1 - error / target_density))
            performance_metrics["Density"] = score
            print(
                f"  - Generated Mean: {mean_gen_density:.2f} (Std: {std_gen_density:.2f})"
            )
            print(f"  - Score (0-100): {score:.1f}")
        else:
            performance_metrics["Density"] = 0
            print(f"  - Generated Mean: Not available")

    if "reduced_formula" in data and "formula" in target_conditions:
        print(f"\n--- Formula Generation Performance ---")
        counts_dict = data.get("reduced_formula", {})
        canonical_counts = defaultdict(int)
        if counts_dict:
            for formula, count in counts_dict.items():
                canonical_counts[get_canonical_formula(formula)] += count
            total_samples = sum(canonical_counts.values())
            sorted_counts = sorted(
                canonical_counts.items(), key=lambda item: item[1], reverse=True
            )
            target_canonical = get_canonical_formula(target_conditions["formula"])

            correct_count = canonical_counts.get(target_canonical, 0)
            score = (correct_count / total_samples) * 100 if total_samples > 0 else 0
            performance_metrics["Formula"] = score
            print(f"  - Correct Formula Generated: {score:.1f}% of the time")
            for i, (formula, count) in enumerate(sorted_counts[:3]):
                is_correct_marker = (
                    " <--- (Correct)" if formula == target_canonical else ""
                )
                print(
                    f"    {i+1}. {formula:<12}: {count:>4} samples ({(count/total_samples)*100:.1f}%){is_correct_marker}"
                )
        else:
            performance_metrics["Formula"] = 0
            print("  - No formula data generated.")

    if "space_group" in data and "group" in target_conditions:
        print(f"\n--- Space Group Generation Performance ---")
        counts_dict = data.get("space_group", {})
        if counts_dict:
            total_samples = sum(counts_dict.values())
            sorted_counts = sorted(
                counts_dict.items(), key=lambda item: item[1], reverse=True
            )
            target_sg = target_conditions["group"]

            correct_count = counts_dict.get(str(target_sg), 0)
            score = (correct_count / total_samples) * 100 if total_samples > 0 else 0
            performance_metrics["Space Group"] = score
            print(f"  - Correct Space Group Generated: {score:.1f}% of the time")
            for i, (sg, count) in enumerate(sorted_counts[:3]):
                is_correct_marker = (
                    " <--- (Correct)" if str(sg) == str(target_sg) else ""
                )
                print(
                    f"    {i+1}. SG {sg:<10}: {count:>4} samples ({(count/total_samples)*100:.1f}%){is_correct_marker}"
                )
        else:
            performance_metrics["Space Group"] = 0
            print("  - No space group data generated.")

    return experiment_folder, performance_metrics


def plot_multi_condition_radar_charts(results: dict, output_filename: str):
    """
    Creates a grid of radar charts to visualize multi-condition performance.

    Args:
        results (dict): A dictionary where keys are experiment names and values are
                        dictionaries of performance metrics.
        output_filename (str): The filename to save the plot.
    """
    labels = ["Density", "Formula", "Space Group"]
    num_vars = len(labels)

    num_experiments = len(results)
    cols = 3
    rows = (num_experiments + cols - 1) // cols

    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(5 * cols, 5 * rows),
        subplot_kw=dict(polar=True),
    )
    axes = np.atleast_2d(
        axes
    ).flatten()  # Ensure axes is always a 2D array and flatten it

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for idx, (title, metrics) in enumerate(results.items()):
        ax = axes[idx]
        values = [metrics.get(label, 0) for label in labels]

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        ax.set_rlabel_position(30)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_ylim(0, 100)

        plot_values = values + values[:1]  # Close the plot
        ax.plot(angles, plot_values, linewidth=1, linestyle="solid")
        ax.fill(angles, plot_values, alpha=0.4)

        formatted_title = title.replace("cond_", "").replace("_", "\n")
        ax.set_title(formatted_title, size=10, color="blue", pad=20)

    for idx in range(num_experiments, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Multi-Condition Generation Performance", size=16, y=0.98)
    plt.tight_layout(pad=3.0)

    output_path = os.path.join(os.path.dirname(__file__), output_filename)
    plt.savefig(output_path, dpi=150)
    print(f"\n\nSaved multi-condition radar plot to {output_path}")
    plt.close()


def check_charge_percentage(folder_path, name):
    all_files = os.listdir(folder_path)
    number_of_structures = 0
    number_of_zero_charges = 0
    for filename in tqdm(all_files, total=len(all_files)):
        if not filename.endswith(".cif"):
            continue
        file_path = os.path.join(folder_path, filename)
        number_of_structures += 1

        # print(f"Reading CIF file: {file_path}")
        try:
            structure = Structure.from_file(file_path)
            if structure.charge == 0:
                number_of_zero_charges += 1

        except Exception as e:
            pass
            # print(f"Could not read file {filename}", e)

    print(
        f"{name} : Number of charges = 0 {number_of_zero_charges}/{number_of_structures} = {number_of_zero_charges/number_of_structures*100.0:.3f}"
    )
    # result = number_of_zero_charges/number_of_structures*100.0


def main():
    """
    Main function to run all plotting operations.
    """

    font_size = 48
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size + 20,
            "xtick.labelsize": font_size + 20,
            "ytick.labelsize": font_size + 20,
            "legend.fontsize": font_size + 10,
            "figure.titlesize": font_size,
        }
    )

    space_groups = [1, 14, 62, 139, 194, 225, 227]
    formulas = ["Si", "GaAs", "MgAl2O4", "CaTiO3", "Y3Al5O12", "YBa2Cu3O7", "Cu2ZnSnS4"]

    print_unconditional_comparison()

    """
    Small First Mean rmsd 0.2641128436909392
    Largest First Mean rmsd 0.22015461880882572
    XYZ Mean rmsd 0.1678878768057043
    RANDOM Mean rmsd 0.4890378240399519
    """

    (
        MODEL_SMALL_ELEMENTS_FIRST_DIR,
        MODEL_LARGE_ELEMENTS_FIRST_DIR,
        MODEL_XYZ_FIRST_DIR,
    ) = print_rmsd()

    # --- Generate Density Plots ---
    plot_density_summary(
        MODEL_SMALL_ELEMENTS_FIRST_DIR,
        "Density Generation (Small Elements First)",
        "density_small_elements.png",
    )
    plot_density_summary(
        MODEL_LARGE_ELEMENTS_FIRST_DIR,
        "Density Generation (Large Elements First)",
        "density_large_elements.png",
    )
    plot_density_summary(
        MODEL_XYZ_FIRST_DIR, "Density Generation (XYZ First)", "density_xyz.png"
    )

    # # --- Generate Space Group Plots ---
    plot_categorical_data(
        MODEL_SMALL_ELEMENTS_FIRST_DIR,
        "space_group",
        space_groups,
        "Space Group Generation (Small Elements First)",
        "space_group_small_elements.png",
    )
    plot_categorical_data(
        MODEL_LARGE_ELEMENTS_FIRST_DIR,
        "space_group",
        space_groups,
        "Space Group Generation (Large Elements First)",
        "space_group_large_elements.png",
    )
    plot_categorical_data(
        MODEL_XYZ_FIRST_DIR,
        "space_group",
        space_groups,
        "Space Group Generation (XYZ First)",
        "space_group_xyz.png",
    )

    # # --- Generate Formula Plots ---
    plot_categorical_data(
        MODEL_SMALL_ELEMENTS_FIRST_DIR,
        "reduced_formula",
        formulas,
        "Formula Generation (Small Elements First)",
        "formula_small_elements.png",
        loc="upper right",
    )
    plot_categorical_data(
        MODEL_LARGE_ELEMENTS_FIRST_DIR,
        "reduced_formula",
        formulas,
        "Formula Generation (Large Elements First)",
        "formula_large_elements.png",
        loc="upper right",
    )
    plot_categorical_data(
        MODEL_XYZ_FIRST_DIR,
        "reduced_formula",
        formulas,
        "Formula Generation (XYZ First)",
        "formula_xyz.png",
        loc="upper right",
    )

    # check_charge_percentage(os.path.join(MODEL_SMALL_ELEMENTS_FIRST_DIR, "cond_"), "Small Element First")
    # check_charge_percentage(os.path.join(MODEL_LARGE_ELEMENTS_FIRST_DIR, "cond_"), "Largest Element First")
    # check_charge_percentage(os.path.join(MODEL_XYZ_FIRST_DIR, "cond_"), "XYZ First")
    # check_charge_percentage(os.path.join(MODEL_RANDOM_DIR, "cond_"), "Random First")

    #

    # run_multi_condition_analysis()
    print("\nAll plots have been generated and saved as PNG files.")


def print_rmsd():
    CWD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qe_input")
    MODEL_SMALL_ELEMENTS_FIRST_DIR = os.path.join(
        CWD,
        "llm_v3_LARGERopeStand_labelsmoothing=0.0_sospecies_seqoatoms_first_material_llama_adamw_condition_latticelast_dim512",
    )
    MODEL_LARGE_ELEMENTS_FIRST_DIR = os.path.join(
        CWD,
        "llm_v3_LARGERopeStand_labelsmoothing=0.0_sorev_species_seqoatoms_first_material_llama_adamw_condition_latticelast_dim512",
    )
    MODEL_XYZ_FIRST_DIR = os.path.join(
        CWD,
        "llm_v3_LARGERopeStand_labelsmoothing=0.0_soxyzorder_seqoatoms_first_material_llama_adamw_condition_latticelast_dim512",
    )
    MODEL_RANDOM_DIR = os.path.join(
        CWD,
        "llm_v3_LARGERopeStand_labelsmoothing=0.0_sorandom_seqoatoms_first_material_llama_adamw_condition_latticelast_dim512",
    )

    print(os.listdir(MODEL_LARGE_ELEMENTS_FIRST_DIR))
    unc_large_path = os.path.join(MODEL_SMALL_ELEMENTS_FIRST_DIR, "_temp=1.00")

    # for f in os.listdir(unc_large_path)
    all_rmsd = []
    all_match = []

    for i in range(1024):
        orig_mat = os.path.join(unc_large_path, f"material_sample_gen_{i}.cif")
        relaxed_mat = os.path.join(
            unc_large_path, f"material_sample_gen_relaxed_{i}.cif"
        )
        try:
            orig = Structure.from_file(orig_mat)
            orig = orig.remove_oxidation_states()
            relaxed = Structure.from_file(relaxed_mat)
            relaxed = relaxed.remove_oxidation_states()

            # print("----"*100)
            # print(orig)
            # print(relaxed)
            # print("----"*100)
            # if i > 3:
            #     quit()
            # import pdb
            # pdb.set_trace()
            if relaxed is None or orig is None:
                print("Could not load file", orig, relaxed)
                continue
            rmsd, found_match = compute_rmsd_angstrom(orig, relaxed)
            all_rmsd.append(rmsd)
            all_match.append(found_match)
            print(f"RMSD Angström {rmsd:.4f}", "Found match", found_match)
        except Exception as e:
            print(e)

    data = pd.DataFrame({"RMSD (Å)": all_rmsd, "Match Found": all_match})
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x="RMSD (Å)", kde=True, bins=30, color="skyblue")
    plt.title("Distribution of RMSD Values", fontsize=16)
    plt.xlabel("RMSD (Å)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True)
    print("Displaying RMSD distribution plot.")
    rmsd_path = os.path.join(unc_large_path, "RMSD_dist.png")
    plt.savefig(rmsd_path)
    print("Saved to", rmsd_path)

    # 2. Create the Match Distribution Plot
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x="Match Found", palette=["#ff6961", "#77dd77"])
    plt.title("Distribution of Matches", fontsize=16)
    plt.xlabel("Match Found", fontsize=12)
    plt.ylabel("Count", fontsize=12)

    # To display the count on top of the bars
    ax = plt.gca()
    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width() / 2.0,
            p.get_height(),
            f"{int(p.get_height())}",
            fontsize=12,
            color="black",
            ha="center",
            va="bottom",
        )

    print("Displaying Match distribution plot.")
    match_path = os.path.join(unc_large_path, "match_dist.png")
    plt.savefig(match_path)
    print("Saved to", match_path)
    return (
        MODEL_SMALL_ELEMENTS_FIRST_DIR,
        MODEL_LARGE_ELEMENTS_FIRST_DIR,
        MODEL_XYZ_FIRST_DIR,
    )


def print_unconditional_comparison():
    CWD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qe_input")

    MODEL_SMALL_ELEMENTS_FIRST_DIR = os.path.join(
        CWD,
        "llm_v3_LARGERopeStand_labelsmoothing=0.0_sospecies_seqoatoms_first_material_llama_adamw_condition_latticelast_dim512",
    )
    MODEL_LARGE_ELEMENTS_FIRST_DIR = os.path.join(
        CWD,
        "llm_v3_LARGERopeStand_labelsmoothing=0.0_sorev_species_seqoatoms_first_material_llama_adamw_condition_latticelast_dim512",
    )
    MODEL_XYZ_FIRST_DIR = os.path.join(
        CWD,
        "llm_v3_LARGERopeStand_labelsmoothing=0.0_soxyzorder_seqoatoms_first_material_llama_adamw_condition_latticelast_dim512",
    )
    MODEL_RANDOM_DIR = os.path.join(
        CWD,
        "llm_v3_LARGERopeStand_labelsmoothing=0.0_sorandom_seqoatoms_first_material_llama_adamw_condition_latticelast_dim512",
    )

    for model_dir, name in [
        (MODEL_SMALL_ELEMENTS_FIRST_DIR, "Smallest First"),
        (MODEL_LARGE_ELEMENTS_FIRST_DIR, "Largest First"),
        (MODEL_XYZ_FIRST_DIR, "XYZ First"),
        (MODEL_RANDOM_DIR, "Random"),
    ]:
        mattergen_res_path = os.path.join(
            model_dir, "_temp=1.00", "mattergen_results_new.json"
        )
        try:
            with open(mattergen_res_path, "r") as f:
                mattergen_res = json.load(f)
                print(mattergen_res_path)
                print(
                    f"{name} -  avg_energy_above_hull_per_atom {mattergen_res['avg_energy_above_hull_per_atom']:.3f}",
                    f"frac_novel_unique_stable_structures {mattergen_res['frac_novel_unique_stable_structures']:.3f}",
                )
                print(mattergen_res)
        except Exception as e:
            print("No file found for ", model_dir)


def run_multi_condition_analysis():
    """Runs the new detailed analysis for dual and triple-condition experiments."""
    print("\n\n" + "#" * 70)
    print("### PART 2: MULTI-CONDITION EXPERIMENT ANALYSIS (CONSOLE REPORT) ###")
    print("#" * 70)

    # Define the specific multi-condition folders you want to analyze
    # Copy these directly from your `ls` output
    experiments_to_analyze = [
        # --- Dual Conditions (Harmonious) ---
        "cond_density:3.5_reduced_formula:MgAl2O4",
        "cond_reduced_formula:MgAl2O4_space_group:227",
        # --- Dual Conditions (Challenging) ---
        "cond_density:6.5_reduced_formula:YBa2Cu3O7",  # Rescue attempt 1
        "cond_reduced_formula:Si_space_group:1",
        # --- Triple Conditions ---
        "cond_density:3.5_reduced_formula:MgAl2O4_space_group:227",  # Harmonious
        "cond_density:6.5_reduced_formula:YBa2Cu3O7_space_group:62",
        "cond_density:8.0_reduced_formula:MgAl2O4",
        "cond_density:1.0_reduced_formula:MgAl2O4",
        "cond_reduced_formula:MgAl2O4_space_group:12",
        "cond_reduced_formula:MgAl2O4_space_group:139",
    ]

    print("\n--- Analyzing LARGE ELEMENTS FIRST Model ---")
    all_results = {}
    for exp_folder in experiments_to_analyze:
        result = analyze_experiment(MODEL_LARGE_ELEMENTS_FIRST_DIR, exp_folder)
        if result:
            folder, metrics = result
            all_results[folder] = metrics

    if all_results:
        plot_multi_condition_radar_charts(
            all_results, "multi_condition_performance.png"
        )


if __name__ == "__main__":
    main()
