<!-- markdownlint-disable -->

<a href="../src/materium/make_checkpoint_graphs.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `make_checkpoint_graphs`





---

<a href="../src/materium/make_checkpoint_graphs.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_density_summary`

```python
plot_density_summary(model_dir, title, output_filename, font_size=32)
```

Creates a plot of mean generated density vs. target density with error bars. This is used because the results.json only contains summary statistics. 



**Args:**
 
 - <b>`model_dir`</b> (str):  The path to the model's 'generated' directory. 
 - <b>`title`</b> (str):  The title for the plot. 
 - <b>`output_filename`</b> (str):  The filename to save the plot. 


---

<a href="../src/materium/make_checkpoint_graphs.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_canonical_formula`

```python
get_canonical_formula(formula: str) → str
```

Converts a chemical formula string to a canonical representation by sorting elements alphabetically. This ensures that, for example, 'MgAl2O4' and 'Al2MgO4' are treated as the same. 

Example: 'YBa2Cu3O7' -> 'Ba2Cu3O7Y'  'O2Si'      -> 'O2Si'  'Al2MgO4'   -> 'Al2MgO4' 


---

<a href="../src/materium/make_checkpoint_graphs.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_categorical_data`

```python
plot_categorical_data(
    model_dir,
    property_name,
    target_values,
    title,
    output_filename,
    title_size=36,
    axis_label_size=28,
    tick_size=28,
    legend_size=18,
    pct_label_fontsize=14,
    inside_label_fontsize=17,
    fig_size=(18, 9),
    bar_width=0.3,
    loc='lower left'
)
```

Creates an intelligent grouped bar chart of the top 3 generated categories. 
- Text is only placed inside bars if they are tall enough to avoid being cut off. 
- Text color (black/white) is chosen automatically for best contrast against the bar color. 



**Args:**
 
 - <b>`model_dir`</b> (str):  The path to the model's 'generated' directory. 
 - <b>`property_name`</b> (str):  The name of the property (e.g., "space_group"). 
 - <b>`target_values`</b> (list):  The list of target property values. 
 - <b>`title`</b> (str):  The title for the plot. 
 - <b>`output_filename`</b> (str):  The filename to save the plot. 


---

<a href="../src/materium/make_checkpoint_graphs.py#L320"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `analyze_experiment`

```python
analyze_experiment(model_dir: str, experiment_folder: str)
```

Parses a multi-condition folder, prints a detailed analysis, and RETURNS a dictionary of performance metrics for plotting. 


---

<a href="../src/materium/make_checkpoint_graphs.py#L424"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_multi_condition_radar_charts`

```python
plot_multi_condition_radar_charts(results: dict, output_filename: str)
```

Creates a grid of radar charts to visualize multi-condition performance. 



**Args:**
 
 - <b>`results`</b> (dict):  A dictionary where keys are experiment names and values are  dictionaries of performance metrics. 
 - <b>`output_filename`</b> (str):  The filename to save the plot. 


---

<a href="../src/materium/make_checkpoint_graphs.py#L483"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_charge_percentage`

```python
check_charge_percentage(folder_path, name)
```






---

<a href="../src/materium/make_checkpoint_graphs.py#L509"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `main`

```python
main()
```

Main function to run all plotting operations. 


---

<a href="../src/materium/make_checkpoint_graphs.py#L620"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `print_rmsd`

```python
print_rmsd()
```






---

<a href="../src/materium/make_checkpoint_graphs.py#L720"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `print_unconditional_comparison`

```python
print_unconditional_comparison()
```






---

<a href="../src/materium/make_checkpoint_graphs.py#L762"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `run_multi_condition_analysis`

```python
run_multi_condition_analysis()
```

Runs the new detailed analysis for dual and triple-condition experiments. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
