# glzip

Graph compression for graph learning systems

# Building from source/Developers

## Building the environment

Conda commands to recreate the virtual enviroment I am using

    conda env create -f ${CUDA}/environment.yml
    conda activate glzip
    pip install -r ${CUDA}/requirements.txt

where `${CUDA}` should be replaced by `cpu`, `cu102` or `cu113` depending
on your device.

## Updating the environment

If you need to make changes to the environemnt. First clear the environment:

    conda deactivate
    conda env remove -n glzip

then rebuild it.

## Building glzip

With the environment active run `maturin develop`. That will install
the glzip module into the environment.

## Updating glzip or making changes to the source.

Just run `maturin develop` again after making the changes. 
