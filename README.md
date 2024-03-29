# ml-playground

This repository contains a collection of small python projects implementing various ML models and applications. Each directory corresponds to an independent project with its own README with further information.

These include the following topics:
- Large Language Models (LLMs): Variety of examples and mini-projects, primarily using HuggingFace
- Diffusion Models: Coming soon...
- Graph Neural Networks (GNNs): Examples using PyG (PyTorch) and jraph (JAX)
- Reinforcement Learning (RL): Coming soon...
- Multi-layer perceptron (MLP) and logistic regression: Implementation of backpropagation from scratch

A variety of ML libraries are used across the different projects:
- PyTorch
- TensorFlow
- JAX

You can also find my code from my published research on [ML applications to high-energy physics](https://www.jamesdmulligan.com/#projects) at the following:
- [arXiv 2305.08979](https://arxiv.org/abs/2305.08979) :  https://github.com/jdmulligan/JFN
- [arXiv 2210.06450](https://arxiv.org/abs/2210.06450) :  https://github.com/jdmulligan/ml-eic-flavor
- [arXiv 2111.14589](https://arxiv.org/abs/2111.14589) :  https://github.com/matplo/pyjetty/tree/master/pyjetty/alice_analysis/process/user/ml
- [arXiv 2102.11337](https://arxiv.org/abs/2102.11337) :  https://github.com/jdmulligan/bayesian-inference
  
And ongoing projects at:
- Conditional diffusion models: https://github.com/jdmulligan/QCD-Diffusion
- Graph neural networks: https://github.com/jdmulligan/Subjet-GNN

## To set up a new project

The environment installation instructions for each project can be found in its corresponding README.
(for organizational simplicity I elected to use separate environment/package management for each project within this git repo, since the projects will not be tagged/released).

In general, I use one of two different python package management tools, depending on the project: venv/pip or poetry.

### venv / pip
<details>
  <summary>Click for details</summary>
<br/> 

To initialize a new project:
```
mkdir MyProject
cd MyProject
mkdir myproject
python -m venv venv
source venv/bin/activate
pip install torch==2.0.0
...
```

And you can export the requirements with (or can write a setup.py):
```
pip freeze > requirements.txt
```

To install an existing project:
```
cd MyProject
python -m venv venv
source venv/bin/activate
```

To re-enter an existing environment, you can simply do:
```
cd MyProject
source venv/bin/activate
```

</details>

### poetry
<details>
  <summary>Click for details</summary>
<br/> 

To initialize a new project:
```
poetry new myproject
cd myproject
poetry shell
poetry add numpy
...
poetry install 
```

And you can export the requirements with:
```
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

To install an existing project:
```
cd myproject
poetry install
poetry shell
```

To re-enter an existing environment, you can simply do:
```
cd myproject
poetry shell
```

</details>
