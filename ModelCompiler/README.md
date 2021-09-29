# Generating E2e Model to IREE

This IREE Samples subdirectory is meant for e2e model compiling from Tensorflow to either generating MLIR that can be compiled through iree translate (i.e std,linalg,iree,util dialect land) or to be compiled to run straight away.

We have three different types of python scripts:

- python scripts that are meant to generate MLIR will be prefixed by \<model\_name\>\_**gen**.py 
- python scripts that are meant to compile and RUN will be prefixed by \<model\_name\>\_**run**.py
- python scripts that are meant to RUN with regular tf will be prefixed by \<model\_name\>\_**tf**.py

## Initial/First Time Setups
### VirtualEnv Setup
```bash
cd ~
pip install virtualenv
virtualenv -p /usr/bin/python3.9 ModelCompilerEnv
source ~/ModelCompilerEnv/bin/activate
```

### Installing Miscallaneous Deps
```bash
pip install tensorflow
pip install gin-config
```
### Installing IREE-Python
```bash
python -m pip install \
  iree-compiler-snapshot \
  iree-runtime-snapshot \
  iree-tools-tf-snapshot \
  iree-tools-tflite-snapshot \
  iree-tools-xla-snapshot \
  --find-links https://github.com/google/iree/releases
```

### Getting ModelCompiler
```bash
git clone https://github.com/google/iree-samples 
git submodule update --init --recursive 
```

## Generating IREE Model (After the first time, this is all you need)
```bash
source ~/ModelCompilerEnv/bin/activate
cd ModelCompiler
export PYTHONPATH=$PYTHONPATH:$PWD/tf_models
cd nlp_models # cd into subdir where category of model live
python bert_small_gen.py # Basically any <model_name>_gen.py should generate it to /tmp/model.mlir
```
