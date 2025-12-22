<!-- markdownlint-disable -->

<a href="../src/materium/train.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `train`





---

<a href="../src/materium/train.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `relax_structure_mattersim`

```python
relax_structure_mattersim(
    structure: Structure,
    fmax: float = 0.1,
    optimizer: str = 'FIRE',
    filter_type: str = 'ExpCellFilter',
    constrain_symmetry: bool = True,
    max_steps: int = 500,
    device='cpu'
)
```






---

<a href="../src/materium/train.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `put_dict_on_device`

```python
put_dict_on_device(d: Dict[str, Tensor], device)
```






---

<a href="../src/materium/train.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `run_epoch`

```python
run_epoch(
    model,
    optimizer,
    dataloader,
    device,
    is_train=True,
    ema_model=None,
    ema_decay=0.999
)
```






---

<a href="../src/materium/train.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `init_ema`

```python
init_ema(model, decay=0.999, device='cuda')
```






---

<a href="../src/materium/train.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_model`

```python
train_model(
    model: LLamaTransformer,
    tokenizer: CrystalTokenizer,
    dataloader_train: DataLoader,
    dataloader_test: DataLoader,
    optimizer: Optimizer,
    epochs,
    device,
    start_epoch=0,
    ckpt_path='ckpt'
)
```

The main training loop. 


---

<a href="../src/materium/train.py#L277"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `configure_optimizer`

```python
configure_optimizer(model: LLamaTransformer, type: str = 'adamw')
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
