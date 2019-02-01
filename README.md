<p align="center">
<b><a href="#setup">Setup</a></b>
|
<b><a href="#code-organization">Code Organization</a></b>
|
<b><a href="#data-sources">Data Sources</a></b>
|
<b><a href="#models">Models</a></b>
|
<b><a href="#key-results">Key Results</a></b>
|
<b><a href="#acknowledgements">Acknowledgements</a></b>
</p>

This repository accompanies our research work, *"Mapping Philippine Poverty
using Machine Learning, Satellite Imagery, and Crowd-sourced Geospatial
Information"*, currently published in our
[website](https://stories.thinkingmachin.es/philippines-most-vulnerable-communities/).
In this work, **we developed five wealth prediction models** using
state-of-the-art methods and various geospatial data sources.

![pampanga map](./assets/pampanga-map.jpg)


# Setup

In order to run the notebooks, all dependencies must be installed. We provided
a `Makefile` to accomplish this task:

```s
make venv
make build
```

This creates a virtual environment, `venv`, and installs all dependencies found
in `requirements.txt`. In order to run the notebooks inside `venv`, execute the
following command:

```s
ipython kernel install --user --name=venv
```

# Code Organization 

This repository is divided into three main parts:
- `./notebooks`: contains all Jupyter notebooks for the different wealth
    prediction models.
- `./utils`: contains utility methods for loading datasets, building model, and
   performing training routines. 
