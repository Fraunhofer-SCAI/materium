<!-- markdownlint-disable -->

<a href="../src/materium/llm_generate.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llm_generate`





---

<a href="../src/materium/llm_generate.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `run_mattersim_relaxation`

```python
run_mattersim_relaxation(
    pseudo_potential_path: str,
    conditions: Dict[str, float],
    qe_save_folder: str,
    generated_materials_for_conditions: List[Structure],
    DEVICE: str = 'cpu'
)
```






---

<a href="../src/materium/llm_generate.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `run_generation`

```python
run_generation(
    temperature: float,
    conditions: Dict[str, float],
    tracking_info_reference: Dict[str, TrackingType],
    multi_eval_tracker: MultiEvaluationTracker,
    qe_save_folder: str,
    DEVICE: str = 'cpu'
)
```






---

<a href="../src/materium/llm_generate.py#L202"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_condition_dict`

```python
create_condition_dict(args)
```






---

<a href="../src/materium/llm_generate.py#L235"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_model`

```python
load_model(args)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
