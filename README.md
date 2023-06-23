# LJ-ML-Learner
Utilizing neural networks to learn LJ potential 


### Installation
To create a local environment with [conda](https://docs.conda.io/en/latest/miniconda.html), run:
```bash
conda env create -f environment.yml
conda activate LJ-ML
```

### Running the code

1- Create a [Weights&Biases](https://wandb.ai/site) account and login by following [these instructions](https://wandb.ai/quickstart):

```bash
wandb login
```

2- In `flow` directory run:

```bash
python init.py
python project.py submit
```
