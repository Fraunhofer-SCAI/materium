<!-- markdownlint-disable -->

<a href="../src/materium/datasets.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `datasets`




**Global Variables**
---------------
- **spg_data**
- **sg_number**
- **symbols_to_add**
- **symbol**
- **data**
- **cleaned_symbol**

---

<a href="../src/materium/datasets.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_spacegroup_int_number_fast`

```python
get_spacegroup_int_number_fast(symbol_string: str) → int
```

Gets the international spacegroup number (1-230) from a spacegroup symbol string using pre-computed internal pymatgen data for fast lookup. 



**Args:**
 
 - <b>`symbol_string`</b>:  The spacegroup symbol string (e.g., "Pm-3m", "Pmmm", "P 2/m").  Whitespace is ignored. Case-sensitive. 



**Returns:**
 The integer international spacegroup number (1-230). 



**Raises:**
 
 - <b>`ValueError`</b>:  If the symbol string is not found in the internal mappings. 


---

<a href="../src/materium/datasets.py#L517"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `pad_collate_fn`

```python
pad_collate_fn(batch: List[Tensor], pad_token_id: int = 0)
```

A simple collate function that pads sequences to the maximum length in a batch. 



**Args:**
 
 - <b>`batch`</b> (List[torch.Tensor]):  A list of token tensors from the dataset. 
 - <b>`pad_token_id`</b> (int):  The ID to use for padding. 



**Returns:**
 A tuple of (input_sequences, target_sequences). 


---

<a href="../src/materium/datasets.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StandardScaler`




<a href="../src/materium/datasets.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(mean=None, std=None, epsilon=1e-07)
```

Standard Scaler. The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module will work fine. :param mean: The mean of the features. The property will be set after a call to fit. :param std: The standard deviation of the features. The property will be set after a call to fit. :param epsilon: Used to avoid a Division-By-Zero exception. 




---

<a href="../src/materium/datasets.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(values)
```





---

<a href="../src/materium/datasets.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_transform`

```python
fit_transform(values)
```





---

<a href="../src/materium/datasets.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inverse_transform`

```python
inverse_transform(values)
```





---

<a href="../src/materium/datasets.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transform`

```python
transform(values)
```






---

<a href="../src/materium/datasets.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MatterGenDataset`
Dataset class for loading crystal structure data stored in separate files as used in MatterGen or similar setups (e.g., CDVAE). 

Loads data from a directory containing: 
- pos.npy: Cartesian coordinates (N_total_atoms, 3) 
- cell.npy: Lattice vectors for each structure (N_structures, 3, 3) 
- atomic_numbers.npy: Atomic numbers for each atom (N_total_atoms,) 
- num_atoms.npy: Number of atoms in each structure (N_structures,) 
- structure_id.npy: Identifier for each structure (N_structures,) [Optional] 
- Other property files (e.g., dft_band_gap.json) (N_structures,) [Optional] 

<a href="../src/materium/datasets.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    data_path: str,
    cutoff_radius: float = 6.0,
    transform=None,
    cache_id: Optional[str] = None,
    recalculate_cache: bool = False,
    lattice_scaler: Optional[StandardScaler] = None
)
```



**Args:**
 
 - <b>`data_path`</b> (str):  Path to the directory containing the data files  (e.g., 'alex_mp_20/train'). 
 - <b>`cutoff_radius`</b> (float):  Cutoff radius for neighbor finding (graph edges). 
 - <b>`transform`</b> (callable, optional):  Optional transform to apply to samples. 
 - <b>`cache_id`</b> (str, optional):  Identifier for caching graph data. If None,  defaults to the basename of data_path. 





---

<a href="../src/materium/datasets.py#L431"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CrystalDataset`
A PyTorch Dataset for our tokenized crystal structures. 

<a href="../src/materium/datasets.py#L434"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    dataset: MatterGenDataset,
    tokenizer: CrystalTokenizer,
    cache_path: str = 'crystal_token_cache.joblib',
    recalculate_cache=False
)
```











---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
