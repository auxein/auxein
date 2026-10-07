# auxein [![license](https://img.shields.io/hexpm/l/plug.svg?maxAge=2592000)](https://github.com/auxein/auxein/blob/master/LICENSE) [![CI](https://github.com/auxein/auxein/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/auxein/auxein/actions/workflows/ci.yml)

Auxein is a Python framework for evolutionary algorithms, currently a work in progress.

<p align="center">
  <img src="https://github.com/auxein/auxein/blob/master/docs/landscape.gif?raw=true" alt="Fitness landscape"/>
<p align="center">Evolution of a population while finding a <a href="https://en.wikipedia.org/wiki/Rastrigin_function">Rastrigin</a> global minimum.</p>
</p>

## Welcome to Auxein!

Auxein is an evolutionary algorithm framework, written in Python. It's currently a work in progress far away from a first 1.0 release. 

The main idea behind Auxein is to provide an efficient tool to *experiment with a variety of evolutionary algorithms implementing reusable fitness functions, genotypes mutations and recombinations.*

## Development with uv

Auxein is managed with [uv](https://docs.astral.sh/uv/).
Here how to do a bunch of things:

1. To install the project: `uv sync`
2. To run unit tests: `uv run pytest`
3. To run the linter: `uv run ruff check`
4. To run typecheck (Pyright): `uv run pyright`

------------------

## Auxein docs:

At the moment, the documentation is a bunch of Jupyter notebooks and some Python Docstrings:

- [Finding Rastrigin function global minimum with Auxein](https://nbviewer.jupyter.org/github/auxein/auxein/blob/master/notebooks/rastrigin.ipynb)
- [How to use Auxein to solve a simple linear regression problem](https://nbviewer.jupyter.org/github/auxein/auxein/blob/master/notebooks/linear_regression.ipynb)
- [How to use Auxein to solve a simple logistic regression problem](https://nbviewer.jupyter.org/github/auxein/auxein/blob/master/notebooks/logistic_regression.ipynb)

------------------

## Why this name, Auxein?

[Auxein](https://en.wikipedia.org/wiki/Auxin) (αυξειν) means _to grow_ in Greek.
