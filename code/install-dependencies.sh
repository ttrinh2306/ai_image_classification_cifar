#!/bin/bash

# Install the Python packages listed in requirements.txt
pip install -r requirements.txt

# Verify the installations
echo "Installed packages:"
pip list

# Run a Python script to import the installed packages
python - << END
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import time
import yaml

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pickle
END