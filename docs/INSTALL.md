# Installation

Though it is not generally required we strongly encourage you to use a conda environment for setting up this software, especially as it is currently under active development and dependencies might change rapidly which could conflict with other installed python packages.

To do this follow the guide over on the conda homepage [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) (a miniconda installation will be sufficient).
After you installed conda you can create a conda environment like follows (we currently test our software against Python 3.13; 3.11 is the minimum):

```
conda create --name hmsm python=3.13
```

And activate it with:

```
conda activate hmsm
```

Now you can install our package, either directly from Github via pip by running the following:

```
pip install git+https://github.com/digital-organology/hmsm.git
```

Alternativelys you could downlaod the repository as a zip archive from the Github page and then executing the following commands (assuming you are in the directory where the downloaded zip file lives):

```
unzip hmsm-main.zip
cd hmsm-main
pip install .
```

Pip should automatically take care of installing all dependencies for you and you should now be set up to start using the software.
To test it you can run:

```
hmsm --help
```

This should list the available subcommands. Each of them also has its own help:

```
hmsm disc2midi --help
```