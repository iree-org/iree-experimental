# IREE Samples Repository

This repository contains various samples and prototypes associated with the [IREE](https://github.com/iree-org/iree) project.

Contact the [IREE Team](https://github.com/iree-org/iree#communication-channels) for questions about this repository.

## Setting up a venv

Many samples are python based. A virtual environment can be set up to run them
via:

```
./setup_venv.sh
source iree-samples.venv/bin/activate
```

## Running sample test suites

```
# Run quick tests.
lit -v tflitehub

# Enable all tests.
lit -v -D FEATURES=hugetest tflitehub
```
