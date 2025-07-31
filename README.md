# AutoML-with-PyCaret

This project demonstrates how to use PyCaret for a full low-code machine learning pipeline on the `juice` dataset. We compare models, tune them, apply ensemble techniques, interpret results, and deploy the best model.

---

## Installation
```bash
pip install pycaret
pip install mlflow
pip install catboost
pip install shap
pip install pycaret[analysis]
pip install boto3
```

---

## Dataset
```python
from pycaret.datasets import get_data
index = get_data('index')
data = get_data('juice')
```

---

## Setup Environment
```python
from pycaret.classification import *
clf1 = setup(data, target='Purchase', session_id=123, log_experiment=False, experiment_name='juice1')
```

---

## Model Training & Comparison
```python
best_model = compare_models()
```

---

## Create Individual Models
```python
lr = create_model('lr')
dt = create_model('dt')
rf = create_model('rf', fold=5)
```

---

## Explore Available Models
```python
models()
models(type='ensemble').index.tolist()
```

---

## Ensemble & Tuning
```python
ensembled_models = compare_models(include=models(type='ensemble').index.tolist(), fold=3)
tuned_lr = tune_model(lr)
tuned_rf = tune_model(rf)

bagged_dt = ensemble_model(dt)
boosted_dt = ensemble_model(dt, method='Boosting')
```

---

## Blend & Stack
```python
blender = blend_models(estimator_list=[boosted_dt, bagged_dt, tuned_rf], method='soft')
stacker = stack_models(estimator_list=[boosted_dt, bagged_dt, tuned_rf], meta_model=rf)
```

---

## Visualization
```python
plot_model(rf)
plot_model(rf, plot='confusion_matrix')
plot_model(rf, plot='boundary')
plot_model(rf, plot='feature')
plot_model(rf, plot='pr')
plot_model(rf, plot='class_report')
```

---

## Model Evaluation
```python
evaluate_model(rf)
```

---

## Advanced Models & Interpretation
```python
catboost = create_model('catboost', cross_validation=False)
interpret_model(catboost)
interpret_model(catboost, plot='reason', observation=12)
```

---

## AutoML
```python
best = automl(optimize='Recall')
```

---

## Predictions
```python
pred_holdouts = predict_model(lr)
pred_holdouts.head()

new_data = data.copy()
new_data.drop(['Purchase'], axis=1, inplace=True)
predict_new = predict_model(best, data=new_data)
predict_new.head()
```

---

## Save & Load Model
```python
save_model(best, model_name='best-model')
loaded_bestmodel = load_model('best-model')
print(loaded_bestmodel)
```

---

## Sklearn Config View
```python
from sklearn import set_config
set_config(display='diagram')
loaded_bestmodel[0]

set_config(display='text')
```

---

## AWS Deployment (Optional)
```python
deploy_model(best, model_name='best-aws', authentication={'bucket': 'pycaret-test'})
```

---

## Config Utilities
```python
X_train = get_config('X_train')
X_train.head()

get_config('seed')

from pycaret.classification import set_config
set_config('seed', 999)
get_config('seed')
```

---

## Output
Trained models, performance plots, predictions, interpretability insights, and a deployable model.
