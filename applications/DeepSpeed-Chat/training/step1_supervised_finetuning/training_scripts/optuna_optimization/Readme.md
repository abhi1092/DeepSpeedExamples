# optuna optimization

After finding the best throughput parameters with DeepSpeed autotuner (step0) you can use optuna optimization to distributedly optimize different hyperparameters.

`optuna_optimization.py` performs such optimization over weight decay and learning rate. Then it tries getting the best development accuracy over a maximum period of 30 minutes, assuming the first 30 minutes is a good proxy for the final result, and then it saves the best hyperpamarameters in some file.

you can run this file from each node you want to use for optimization. And as long as the database file is the same, the different trials will be synchronized.

In the `optuna_optimization.py` file, change the parameters that should be given to the `main.py` file. Including the model and the dataset, the number of trials you want to execute in each node, the names of the study and the common file that should be used for the database. The database file should be in shared storage accross all nodes executing the optimization.

Then you can run the file with:

```bash
python -m training_scripts.optuna_optimization.optuna_optimization\
 --n_trials 100 \
 --study_name granite_13b\
 --storage sqlite:////new_data/granite_13b.db
```