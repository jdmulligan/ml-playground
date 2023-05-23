# ml-playground

This repository contains a collection of small python projects implementing various ML models and applications. Each directory corresponds to an independent project with its own README (for organizational simplicity I elected to use a separate projects / package management within this git repo, since the projects will not be tagged/released).

## To set up a new project

I use one of two different python package management tools, depending on the project: venv/pip or poetry. Note that for some packages, poetry seems to have some trouble interacting correctly with pyenv – better to use the venv/pip approach for those projects. For most other projects, I use poetry.

### venv / pip approach
```
mkdir MyProject
cd MyProject
mkdir myproject
python -m venv venv
source venv/bin/activate
pip install torch==2.0.0
...
```

And you can export the requirements with:
```
pip freeze > requirements.txt
```
(or can write a setup.py)

### poetry

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
