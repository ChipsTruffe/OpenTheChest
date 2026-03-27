# OpenTheChest
Agent solutions on the openTheChest environment for ENSTA CSC_5IA05.

### Description

#### trainings.ipynb
-> Implements the training for every environment with predetermined algorithms and models. It is seeded for reproducibility, but the algorithms should converge most of the time with other seeds.

NOUVELLE META REVOLUTIONNAIRE : pretraining d'un modele sur v1, post training sur v2 : résultats de baisé

#### eval.py
-> Evaluates the best model for each environment with a given number of episodes.

### Main things to know
- A reward shaping is done during training to encourage solving the environment fast, by applying a small negative reward. It is not used during evaluation to get standardized results.

- register_envs.py has been modified to use non-discrete (multiBinary) action spaces for env v1 and v2.

