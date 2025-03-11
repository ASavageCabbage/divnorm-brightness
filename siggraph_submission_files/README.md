# Source code for "Tone mapping applications for a divisive normalization brightness model"

This archive is split into several Python modules for convenience.
- `dinos.py` contains implementations for DINOS and BRONTO.
- `bronto_example.py` contains a barebones example script for using the BRONTO tone mapper on an HDR image.
- `ml_pipeline.py` contains the PyTorch implementations of the ACES and Reinhard tone mapping operators with learnable parameters, and a Dataset implementation for setting up comparisons.
- `ml_training_example.py` contains example boilerplate for setting up a training pipeline using the classes implemented in `ml_pipeline.py`.
- `requirements.in` contains all the external libraries required by the above modules.

We also provide four HDR images from the Fairchild HDR Survey in `images/`, and the corresponding brightness responses from DINOS using default parameters and a downsample ratio of 4 in `brightnesses/`.
These can be used to test DINOS and BRONTO, or as a small training set for an ML pipeline.

Python 3.12 is recommended for running this code.

To install the required libraries, run
```
pip install -r requirements.in
```

To run a specific script (`bronto_example.py` in this example), run
```
python bronto_example.py
```
This should be run from the same directory as the Python modules, or imports may not work correctly.
