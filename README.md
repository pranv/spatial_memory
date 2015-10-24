# Spatial Memory Machine
Deforming Space for Memory

The network can be trained on 5 tasks from the Neural Turing Machines paper. Just edit the `task` variable in `train.py` along with the correct input and output dimensions in `dinput` & `doutput`

To run the training procedure:

```
	python train.py
```

The model and the memory unit is defined in `smm.py`

The layers and other utilities are in `layers.py`

`generator.py` is used to create training data.


## Current Status

Model is failing to learn correct addressing. Hence it is not utilizing the memory module, thus giving performance similar to pure LSTM baseline.
