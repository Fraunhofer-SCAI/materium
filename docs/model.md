<!-- markdownlint-disable -->

<a href="../src/materium/model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `model`





---

<a href="../src/materium/model.py#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `precompute_freqs_cis`

```python
precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0)
```






---

<a href="../src/materium/model.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `reshape_for_broadcast`

```python
reshape_for_broadcast(freqs_cis: Tensor, x: Tensor)
```






---

<a href="../src/materium/model.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `build_blockwise_position_ids`

```python
build_blockwise_position_ids(
    tokens: Tensor,
    cond_len: int,
    atoms_token_id: int,
    lattice_token_id: int,
    block_size: int = 4
) → Tensor
```

tokens: (B, S) original token ids (without prepended condition embeddings) cond_len: number of condition tokens prepended to the sequence (0 if none) 

Returns: (B, cond_len + S) position ids where positions strictly between  [ATOMS] and [LATTICE] repeat 0..block_size-1; outside remain  standard monotonic positions. 

Generation-safe: if [LATTICE] is not present yet, applies block-wise up to the current sequence end; once [LATTICE] appears, block-wise stops there. 


---

<a href="../src/materium/model.py#L196"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `apply_rotary_emb`

```python
apply_rotary_emb(
    xq: Tensor,
    xk: Tensor,
    freqs_cos: Tensor,
    freqs_sin: Tensor
) → Tuple[Tensor, Tensor]
```






---

<a href="../src/materium/model.py#L221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `repeat_kv`

```python
repeat_kv(x: Tensor, n_rep: int) → Tensor
```

torch.repeat_interleave(x, dim=2, repeats=n_rep) 


---

<a href="../src/materium/model.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConditionConfig`
Configuration for a single conditional input. 

<a href="../<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(proj_layer_type: str, input_dim: int, out_dim: int) → None
```









---

<a href="../src/materium/model.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RopeMode`
An enumeration. 





---

<a href="../src/materium/model.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ModelArgs`
ModelArgs(dim: int = 4096, n_layers: int = 32, n_heads: int = 32, n_kv_heads: Optional[int] = None, vocab_size: int = 32000, hidden_dim: Optional[int] = None, multiple_of: int = 256, norm_eps: float = 1e-05, max_seq_len: int = 2048, dropout: float = 0.1, pad_id: int = -1, condition_config: Optional[Dict[str, model.ConditionConfig]] = None, rope_mode: model.RopeMode = <RopeMode.STANDARD: 'standard'>, atoms_token_id: Optional[int] = None, lattice_token_id: Optional[int] = None, atom_block_size: int = 4) 

<a href="../<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    dim: int = 4096,
    n_layers: int = 32,
    n_heads: int = 32,
    n_kv_heads: Optional[int] = None,
    vocab_size: int = 32000,
    hidden_dim: Optional[int] = None,
    multiple_of: int = 256,
    norm_eps: float = 1e-05,
    max_seq_len: int = 2048,
    dropout: float = 0.1,
    pad_id: int = -1,
    condition_config: Optional[Dict[str, ConditionConfig]] = None,
    rope_mode: RopeMode = <RopeMode.STANDARD: 'standard'>,
    atoms_token_id: Optional[int] = None,
    lattice_token_id: Optional[int] = None,
    atom_block_size: int = 4
) → None
```








---

<a href="../src/materium/model.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_dict`

```python
from_dict(config_dict: Dict) → ModelArgs
```

Deserializes a dictionary into a ModelArgs instance. 

This method correctly reconstructs nested dataclasses and Enums from the dictionary's contents. 

---

<a href="../src/materium/model.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → Dict
```

Serializes the ModelArgs instance to a dictionary. 

This method handles nested dataclasses and Enums correctly, making the output dictionary suitable for JSON serialization. 


---

<a href="../src/materium/model.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RMSNorm`




<a href="../src/materium/model.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dim: int, eps: float)
```








---

<a href="../src/materium/model.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```






---

<a href="../src/materium/model.py#L233"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Attention`




<a href="../src/materium/model.py#L234"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(args: ModelArgs)
```








---

<a href="../src/materium/model.py#L261"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x: Tensor, freqs_cos: Tensor, freqs_sin: Tensor)
```






---

<a href="../src/materium/model.py#L317"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FeedForward`




<a href="../src/materium/model.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dim: int, hidden_dim: int, multiple_of: int, dropout: float)
```








---

<a href="../src/materium/model.py#L329"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```






---

<a href="../src/materium/model.py#L333"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TransformerBlock`




<a href="../src/materium/model.py#L334"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(layer_id: int, args: ModelArgs)
```








---

<a href="../src/materium/model.py#L350"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x, freqs_cos, freqs_sin)
```






---

<a href="../src/materium/model.py#L359"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FormulaEmbedder`




<a href="../src/materium/model.py#L361"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(hidden_size, atom_embedder: Module)
```








---

<a href="../src/materium/model.py#L373"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(reduced_formula_dict: Dict[str, Tensor])
```






---

<a href="../src/materium/model.py#L396"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LLamaTransformer`




<a href="../src/materium/model.py#L399"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(params: ModelArgs, loss_weights: Optional[Tensor] = None)
```








---

<a href="../src/materium/model.py#L553"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `append_condition`

```python
append_condition(tokens, conditions, _bsz, h)
```





---

<a href="../src/materium/model.py#L504"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(
    tokens: Tensor,
    targets: Optional[Tensor] = None,
    conditions: Optional[Dict[str, Tensor]] = {},
    return_hidden_states: bool = False
) → Tensor
```





---

<a href="../src/materium/model.py#L481"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_positional_encodings`

```python
get_positional_encodings(tokens: Tensor, cond_len: int, final_seq_len: int)
```





---

<a href="../src/materium/model.py#L602"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load`

```python
load(path: str, strict=True)
```





---

<a href="../src/materium/model.py#L596"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(path: str, **kwargs)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
