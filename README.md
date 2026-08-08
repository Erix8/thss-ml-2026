# Machine Learning Algorithms

Solutions to the programming assignments of **Machine Learning course (2026)**. Each problem includes source code, a detailed README with problem description and solution analysis.

## Homeworks

| HW# | Problem | Description | Score |
|-----|---------|-------------|-------|
| 1 | [Linear & SVM] | Linear Model & Support Vector Machines | 100/100 |
| 2 | [Tree & Boosting] | Decision Trees & Gradient Boosting Machines | 100/100 |
| 3 | [Reinforcement Learning] | Value-Based & Policy Gradient Methods | 100/100 |

[Linear & SVM]: ./LinearModels&SVM/README.md
[Tree & Boosting]: ./DecisionTree&Boosting/README.md
[Reinforcement Learning]: ./ReinforcementLearning/README.md

## Usage

### HW1: Linear Models & SVM

```bash
cd LinearModels&SVM
pip install -r requirements.txt

# Linear regression + basis expansion + gradient descent
cd linear
python start_code.py --basis_reg      # basis expansion + regularization (2.1-2.3)
python start_code.py --grad_check     # gradient checker
python start_code.py --gd_search      # full-batch GD hyperparameter search
python start_code.py --sgd            # SGD with batch size sweep
python start_code.py --classification # multiclass softmax (2.4)

# SVM text sentiment classification
cd ../svm
python start_code.py --linear-svm     # linear SVM
python start_code.py --kernel-svm     # kernel SVM (RBF)
```

### HW2: Decision Tree & Boosting

```bash
cd DecisionTreeBoosting
pip install -r requirements.txt

python tree.py       # train classification & regression trees (depth 1-6)
python boosting.py   # train GBM with L2/logistic loss (n_estimators 1-100)
```

### HW3: Reinforcement Learning

```bash
cd ReinforcementLearning
pip install -r requirements.txt

# Value-based: SARSA & Q-Learning on FrozenLake
cd sarsa_Q_learning
python main.py

# Policy gradient: REINFORCE & TD Actor-Critic on CartPole
cd ../policy_gradient
python policy_gradient.py  # set AC=True/False to switch algorithms
```