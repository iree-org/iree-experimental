# Generating E2e Model to IREE

This IREE Samples subdirectory is meant for e2e model compiling from Tensorflow to either generating MLIR that can be compiled through iree translate (i.e std,linalg,iree,util dialect land) or to be compiled to run straight away.

We have three different types of python scripts:

- python scripts that are meant to generate MLIR will be prefixed by \<model\_name\>\_**gen**.py 
- python scripts that are meant to compile and RUN will be prefixed by \<model\_name\>\_**run**.py
- python scripts that are meant to RUN with regular tf will be prefixed by \<model\_name\>\_**tf**.py

### Getting ModelCompiler tf_models
## Initial/First Time Setups
### VirtualEnv Setup
```bash
./setup_venv.sh
```

## Generating IREE Model (After the first time, this is all you need)
```bash
source iree-samples.venv/bin/activate
cd ModelCompiler/nlp_models
python bert_small_gen.py # Basically any <model_name>_gen.py should generate it to model.mlir
```
