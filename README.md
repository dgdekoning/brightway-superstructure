# Brightway2 database superstructure

The code in this repository has been put together specifically to give users
a method for merging multiple [`brightway2`](https://github.com/brightway-lca/brightway2)
sqlite-type databases together. The code also allows for the storage of differences
between the databases in a file that allows for a quick overview as well as a
way to add, remove and edit these values.

## Building a superstructure

The `Builder` class in `src/superstructure.py` is used as a way to prepare
and construct the 'superstructure'. The [`build_superstructure`](./build_superstructure.ipynb)
notebook contains an example of the flow used to put together the superstructure
database and the related 'scenario' or 'difference' file.

An example [scenario file](./Scenario_difference_file_template.xlsx) is also
present with a very well done explanation by [`bsteubing`](https://github.com/bsteubing)
of each column and how to make or customize one without being dependent on the
code to build it.

## Requirements

The suggested requirements for using the notebook as intended are as follows:

```console
brightway2 >=2.1.2
jupyter
numpy
pandas >=0.24.1
python >=3.7
xlrd
```

The code was written and used through packages installed with the [`conda`](https://github.com/conda/conda)
tools. Setting up the required environment can be done like so:

```console
conda create -n structure -c defaults -c conda-forge -c cmutel -c haasad "brightway2>=2.1.2" python=3.7 jupyter numpy "pandas>=0.24.1" xlrd
```
