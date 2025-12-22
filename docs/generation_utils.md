<!-- markdownlint-disable -->

<a href="../src/materium/generation_utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `generation_utils`





---

<a href="../src/materium/generation_utils.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `parse_formula_to_reduced_format`

```python
parse_formula_to_reduced_format(
    formula: str,
    num_samples: int,
    device: device
) → Dict[str, Tensor]
```

Parse a chemical formula and create a reduced format suitable for batch processing. 



**Args:**
 
 - <b>`formula`</b> (str):  The chemical formula to process. 
 - <b>`num_samples`</b> (int):  The number of samples for batch processing. 



**Returns:**
 
 - <b>`Dict[str, torch.Tensor]`</b>:  A dictionary containing tensors corresponding to the parsed formula. 


---

<a href="../src/materium/generation_utils.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sample_next_token`

```python
sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 0.9,
    penalize_token_id: Optional[List[int]] = None,
    penalty_weight: float = 1.5,
    squeeze: float = 0.0,
    tail_bias: float = 0.0,
    uniform_mix: float = 0.0,
    typical_p: float = 0.0,
    logit_bias: Optional[Tensor] = None,
    forbid_mask: Optional[Tensor] = None,
    return_entropy: bool = False
) → int
```

Temperature + top-p with optional tail boosting, typical decoding, token-specific bias. Expects logits shape [1, V]. 


---

<a href="../src/materium/generation_utils.py#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `cfg_blend`

```python
cfg_blend(logits_uncond, logits_cond, guidance_weight: float)
```






---

<a href="../src/materium/generation_utils.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_condition_dict`

```python
get_condition_dict(model, conditions: Dict[str, Any], device)
```






---

<a href="../src/materium/generation_utils.py#L233"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `generate_structure`

```python
generate_structure(
    model,
    tokenizer,
    max_len=100,
    device='cpu',
    temperature=1.0,
    top_p=0.9,
    conditions=None,
    classifier_guidance_weight=0.0,
    min_atoms: int = 2,
    max_atoms: int = 64,
    use_typical: bool = False,
    oxygen_penalty: float = 0.0,
    charge_neutral_bias: float = 0.0
)
```






---

<a href="../src/materium/generation_utils.py#L310"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_debug_state`

```python
make_debug_state()
```






---

<a href="../src/materium/generation_utils.py#L319"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `decode_frac_from_token`

```python
decode_frac_from_token(tok_int, tokenizer)
```






---

<a href="../src/materium/generation_utils.py#L324"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `decode_lattice_from_token`

```python
decode_lattice_from_token(tok_int, tokenizer, param_key)
```






---

<a href="../src/materium/generation_utils.py#L329"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `debug_print_token`

```python
debug_print_token(tok, entropy, tokenizer, state)
```






---

<a href="../src/materium/generation_utils.py#L396"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `calc_hhi`

```python
calc_hhi(
    structure: Structure,
    kind: str = 'production'
) → Union[float, Tuple[float, float]]
```

Calculate the Herfindahl–Hirschman Index (HHI) for a structure using pymatgen's HHIModel. 



**Args:**
 
 - <b>`structure`</b>:  pymatgen Structure (relaxed or not). HHI depends only on composition. 
 - <b>`kind`</b>:  'production', 'reserve', or 'both'. 



**Returns:**
 
    - If kind == 'production': production HHI (float) 
    - If kind == 'reserve': reserve HHI (float) 
    - If kind == 'both': (production HHI, reserve HHI) 


---

<a href="../src/materium/generation_utils.py#L429"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `hhi_designations`

```python
hhi_designations(hhi_p: float, hhi_r: float) → Tuple[str, str]
```

Return DOJ designations ('low'/'medium'/'high') for production and reserve HHI. 


---

<a href="../src/materium/generation_utils.py#L436"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `calc_hhi_for_structures`

```python
calc_hhi_for_structures(
    structures: Iterable[Structure],
    kind: str = 'production'
) → List[Union[float, Tuple[float, float]]]
```

Convenience function for multiple structures. 


---

<a href="../src/materium/generation_utils.py#L446"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_qe_magdensity_input`

```python
create_qe_magdensity_input(
    structure: Structure,
    output_dir: str,
    pseudo_dir: str,
    ecutwfc: int = 60,
    file_name: str = 'qe_relax.in'
)
```

Creates a Quantum ESPRESSO input file with a robust, direct method for setting starting magnetization that avoids API abstractions. 



**Args:**
 
 - <b>`structure`</b> (Structure):  The generated pymatgen Structure object. 
 - <b>`output_dir`</b> (str):  Directory where the QE input file will be saved. 
 - <b>`pseudo_dir`</b> (str):  Path to your SSSP pseudopotential directory. 
 - <b>`ecutwfc`</b> (int):  Plane-wave cutoff energy in Rydbergs. 


---

<a href="../src/materium/generation_utils.py#L542"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_qe_scf_from_mp_config`

```python
create_qe_scf_from_mp_config(
    structure: Structure,
    output_dir: str,
    pseudo_dir: str,
    file_name: str = 'qe_mp_scf.in',
    gen_idx: int = 0
)
```

Creates a Quantum ESPRESSO SCF input file by translating key parameters from the Materials Project's VASP configuration, including specific MAGMOM values. 



**Args:**
 
 - <b>`structure`</b> (Structure):  The pymatgen Structure object for the calculation. 
 - <b>`output_dir`</b> (str):  Directory where the QE input file will be saved. 
 - <b>`pseudo_dir`</b> (str):  Path to your pseudopotential directory. 
 - <b>`file_name`</b> (str):  The name for the output QE input file. 


---

<a href="../src/materium/generation_utils.py#L702"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `total_oxi_charge`

```python
total_oxi_charge(structure)
```






---

<a href="../src/materium/generation_utils.py#L717"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrackingType`
An enumeration. 





---

<a href="../src/materium/generation_utils.py#L722"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EvaluationTracker`
A class to track, analyze, and visualize properties of generated crystal structures. 

<a href="../src/materium/generation_utils.py#L727"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    tracking_info: List[Tuple[str, float, TrackingType]],
    output_dir: str = 'tracker_results'
)
```

Initializes the EvaluationTracker. 



**Args:**
 
 - <b>`tracking_info`</b> (List[Tuple[str, TrackingType]]):  A list of tuples,  where each tuple contains the property key (str) and its TrackingType. 
 - <b>`output_dir`</b> (str):  Directory to save plots and results. 




---

<a href="../src/materium/generation_utils.py#L751"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_result`

```python
add_result(struct: Structure)
```

Calculates and records the properties for a given structure. 



**Args:**
 
 - <b>`struct`</b> (Structure):  The pymatgen Structure object to analyze. 

---

<a href="../src/materium/generation_utils.py#L856"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `summarize_results`

```python
summarize_results() → Dict[str, Any]
```

Analyzes the collected data, prints a summary, and saves plots. 



**Returns:**
 
 - <b>`Dict[str, Any]`</b>:  A dictionary containing statistics for numeric  properties and counts for categorical properties. 


---

<a href="../src/materium/generation_utils.py#L894"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiEvaluationTracker`
A wrapper that manages multiple EvaluationTracker instances and produces combined plots. 

<a href="../src/materium/generation_utils.py#L899"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    trackers: Optional[List[ForwardRef('EvaluationTracker')]] = None,
    output_dir: str = 'multi_tracker_results'
)
```



**Args:**
 
 - <b>`trackers`</b>:  Optional list of existing EvaluationTracker instances. 
 - <b>`output_dir`</b>:  Directory to save combined plots. 




---

<a href="../src/materium/generation_utils.py#L916"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_tracker`

```python
add_tracker(tracker: 'EvaluationTracker')
```

Add a tracker to the collection. 

---

<a href="../src/materium/generation_utils.py#L1185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `plot_all_numeric_violins`

```python
plot_all_numeric_violins()
```

Create combined violin plots for all numeric keys across trackers (1D only). 

---

<a href="../src/materium/generation_utils.py#L1158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `plot_auto`

```python
plot_auto(key: Optional[str] = None, pair: Optional[Tuple[str, str]] = None)
```

Auto plot: 
- If exactly one numeric key (or 'key' provided): 1D violin for that key. 
- If two numeric keys (or 'pair' provided): 2D scatter for that pair. Defaults to the first two sorted numeric keys if more than two exist and 'pair' not provided. 

---

<a href="../src/materium/generation_utils.py#L1055"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `plot_combined_2d`

```python
plot_combined_2d(key_x: str, key_y: str)
```

2D plot: scatter of predictions for two numeric keys across all trackers. 
- X-axis: predictions for key_x 
- Y-axis: predictions for key_y Colored by (target_x, target_y) pair like the example image. Saves as ..._2d.png 

---

<a href="../src/materium/generation_utils.py#L929"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `plot_combined_violin`

```python
plot_combined_violin(key: str)
```

1D plot: combined violin for a numeric key across all trackers. Adds horizontal lines at each target value to aid visual comparison. Saves as ..._1d.png 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
