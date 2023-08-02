import h5py
import os
import numpy as np
import matplotlib.pyplot as plt


# filepath = os.path.join(os.getenv("ASTROMATIC_PATH"), "Problems", "P2_lens_inference", "datasets", "lenses_dummy", "lenses_dummy_0000.h5")
filepath = os.path.join(os.getenv("ASTROMATIC_PATH"), "Problems", "P2_lens_inference", "datasets", "lenses_light_dummy", "lenses_light_dummy_0000.h5")

file = h5py.File(filepath, mode="r")

desc = file["base"].attrs["dataset_descriptor"]

file.close()