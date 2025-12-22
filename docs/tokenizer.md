<!-- markdownlint-disable -->

<a href="../src/materium/tokenizer.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tokenizer`





---

<a href="../src/materium/tokenizer.py#L445"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `calculate_stats_from_dataset`

```python
calculate_stats_from_dataset(
    dataset: 'MatterGenDataset'
) → Tuple[List[str], Dict[str, Tuple[float, float]]]
```

Calculates the element vocabulary and min/max for lattice parameters from a dataset. 


---

<a href="../src/materium/tokenizer.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SortingOrder`
An enumeration. 





---

<a href="../src/materium/tokenizer.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SequenceOrder`
Determines the order of atom type and coordinate tokens within the atom sequence. 
- ATOMS_FIRST: [elem_1, x, y, z, elem_2, x, y, z, ...] 
- COORDS_FIRST: [x, y, z, elem_1, x, y, z, elem_2, ...] 





---

<a href="../src/materium/tokenizer.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CrystalTokenizer`
A tokenizer to convert pymatgen Structure objects into a sequence of integer tokens and back, with optional oxidation-state-aware element tokens. 

Token sequence: [SOS] [ATOMS] [elem|ox_1] [3 coord_tokens] [elem|ox_2] ... [LATTICE] [6 lattice tokens] [EOS] 

<a href="../src/materium/tokenizer.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    element_vocab: List[str],
    lattice_stats: Dict[str, Tuple[float, float]],
    num_quant_bins: int = 1024,
    sorting_order: SortingOrder = <SortingOrder.SPECIES: 'species'>,
    sequence_order: SequenceOrder = <SequenceOrder.ATOMS_FIRST: 'atoms_first'>,
    include_oxidation: bool = True,
    oxidation_mode: str = 'guess',
    allowed_oxidation_states: Optional[Dict[str, List[int]]] = None,
    fallback_to_zero_on_unseen: bool = True
)
```



**Args:**
 
 - <b>`element_vocab`</b> (List[str]):  Elements in the vocabulary (e.g., ['H', 'O', 'Fe']). 
 - <b>`lattice_stats`</b> (Dict[str, Tuple[float, float]]):  Min/max for lattice params (a,b,c,alpha,beta,gamma). 
 - <b>`num_quant_bins`</b> (int):  Number of bins for quantizing continuous values. 
 - <b>`include_oxidation`</b> (bool):  If True, expand tokens to include oxidation state variants. 
 - <b>`oxidation_mode`</b> (str):  'guess' -> use pymatgen to guess oxidation states when absent;  'from_structure' -> require Structure to already have oxidation states. 
 - <b>`allowed_oxidation_states`</b> (dict):  Per-element allowed oxidation states. If None, uses  Element.common_oxidation_states (or Element.oxidation_states if empty) plus 0. 
 - <b>`fallback_to_zero_on_unseen`</b> (bool):  If True, use ox=0 if guessed ox not in allowed set; else raise. 




---

<a href="../src/materium/tokenizer.py#L338"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `detokenize`

```python
detokenize(tokens: List[int]) → Structure
```

Converts a sequence of tokens back into a pymatgen Structure. Works regardless of the order of the [ATOMS] and [LATTICE] sections. 

---

<a href="../src/materium/tokenizer.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_dict`

```python
from_dict(json_data: Dict[str, <built-in function any>]) → CrystalTokenizer
```

Loads a tokenizer's configuration from a JSON config and creates an instance. 



**Args:**
 
 - <b>`json_data`</b> (Dict[str, any]):  The tokenizer config file, gotten from to_json(). 



**Returns:**
 
 - <b>`CrystalTokenizer`</b>:  An instance of the tokenizer with the loaded configuration. 

---

<a href="../src/materium/tokenizer.py#L415"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_loss_weights`

```python
get_loss_weights(atom_type_weight: float) → Optional[Tensor]
```

Creates a weight tensor for the cross-entropy loss function. 

This tensor assigns a higher weight to atom type tokens, encouraging the model to prioritize their prediction. 



**Args:**
 
 - <b>`atom_type_weight`</b> (float):  The weight to apply to atom type tokens.  If this is 1.0 or less, no reweighting is  applied and None is returned. 



**Returns:**
 
 - <b>`Optional[torch.Tensor]`</b>:  A 1D tensor of shape (vocab_size,) with custom  weights, or None if no reweighting is applied. 

---

<a href="../src/materium/tokenizer.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict()
```

Gives the tokenizer's configuration to in JSON format. 

---

<a href="../src/materium/tokenizer.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `tokenize`

```python
tokenize(structure: Structure) → List[int]
```

Converts a pymatgen Structure into a list of integer tokens. 



**Returns:**
 
 - <b>`List[int]`</b>:  [SOS] [ATOMS] (elem|ox, 3*coords) ... [LATTICE] (6*lattice) [EOS] 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
