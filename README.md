# RegressComparer

A lightweight Python utility for evaluating and comparing multiple regression models side by side. Built on top of `scikit-learn`, `numpy`, and `pandas`, it provides a clean pipeline for computing common regression metrics and visualizing results in a unified table.

---

## Installation

No separate installation is required beyond the standard dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Then simply place `RegressComparer.py` in your project directory and import it:

```python
from RegressComparer import EvalPipe, demo
```

---

## Classes

### `EvalPipe`

The core evaluation pipeline. Accepts ground truth values and one or more sets of predictions, then computes regression metrics across all models.

#### Constructor

```python
EvalPipe(x, y, y_pred, first_model)
```

| Parameter | Type | Description |
|---|---|---|
| `x` | array-like | Feature data (stored for reference) |
| `y` | array-like | Ground truth target values |
| `y_pred` | array or list of arrays | Predictions from the first model, or a list of prediction arrays |
| `first_model` | str or list of str | Name(s) corresponding to the prediction array(s) |

---

#### Methods

**`.add(model_name, new_preds)`**

Add a new model's predictions to the pipeline.

```python
pipe.add("Ridge Regression", ridge_preds)
```

- `model_name` must be a string
- `new_preds` must be array-like and match the length of `y`
- Returns a confirmation string on success

---

**`.vision()`**

Returns a `pandas.DataFrame` summarising all metrics for every model added to the pipeline. Metrics are shown as rows, models as columns.

```python
pipe.vision()
```

| Metric | Description |
|---|---|
| `MSE` | Mean Squared Error |
| `MAE` | Mean Absolute Error |
| `RMSE` | Root Mean Squared Error |
| `R_SQRT` | R² (Coefficient of Determination) |

---

**`.from_demo(demo_obj)` *(classmethod)***

Instantiates an `EvalPipe` directly from a `demo` object — useful for quickly testing the pipeline without real data.

```python
pipe = EvalPipe.from_demo(demo())
```

---

### `demo`

A built-in helper class that generates a simple synthetic dataset for testing and demonstration purposes.

```python
d = demo()
x, y, pred_list, pred_titles = d.fit()
```

It provides:
- `x` and `y`: a trivial linear sequence (`[1, 2, 3, 4, 5]`)
- Two sample prediction arrays (`preds_0`, `preds_1`) with slight noise

---

## Usage Examples

### Basic Usage

```python
import numpy as np
from RegressComparer import EvalPipe

x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])

model_a_preds = np.array([2.0, 4.1, 6.0, 8.0, 10.0])

pipe = EvalPipe(x, y, model_a_preds, "Model A")
print(pipe.vision())
```

### Comparing Multiple Models

```python
model_b_preds = np.array([2.3, 3.7, 6.5, 7.5, 10.4])

pipe.add("Model B", model_b_preds)
print(pipe.vision())
```

### Using the Demo Mode

```python
from RegressComparer import EvalPipe, demo

pipe = EvalPipe.from_demo(demo())
print(pipe.vision())
```

---

## Metrics Reference

| Metric | Formula | Best Value |
|---|---|---|
| MSE | mean((y - ŷ)²) | 0 |
| MAE | mean(\|y - ŷ\|) | 0 |
| RMSE | √MSE | 0 |
| R² | 1 - SS_res / SS_tot | 1 |

---

## File Structure

```
RegressComparer.py     # Core module (EvalPipe + demo classes)
RegressorEvaluation.ipynb  # Example notebook demonstrating usage
README.md              # This file
```

---

## Notes

- `EvalPipe` stores predictions in the order they are added; `vision()` reflects that order.
- All prediction arrays must match the length of `y` or a `ValueError` will be raised.
- The `x` parameter is stored on the instance and can be used for custom plotting, but is not used internally by any built-in metric methods.
