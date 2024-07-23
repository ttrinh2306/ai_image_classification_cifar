# Image Classification project: Real vs. AI-Generated Synthetic Images
## Project Overview

For this project, I leveraged convolutional neural networks (CNNs) to differentiate between real and AI-generated synthetic images. This project highlights my skills in data preprocessing, model building, hyperparameter tuning, and model evaluation using Python (Keras & Tensorflow). 

The implementation uses class-based Python coding to stream various tasks:

1) **Sourcing images**. The **'Images'** class is used to source and organize images'.
2) **Preprocessing images**. The **'Preprocess'** class is used to preprocess images and create data generators.
3) **Defining model architecture**. The **'CNN_model'** class is used to define, compile, and train CNN models.
4) **Training & saving models**.
5) **Evaluating models**. 'The **'Evaluate_model'** class is used to load and evaluate the trained models.

## Methodology
The project uses hyperparameter tuning and model comparison techniques to enhance model performance. Different models were trained with slight variations in their configurations to identify the best hyperparameters.

## Running the Project
To execute the project, run the 'cifake-train-evaluate-models.ipynb' notebook. This notebook contains the necessary code to train and evaluate the models. To load the models: 

```python
base = load_model('/kaggle/input/cifar/tensorflow2/base/1/cifake_base.h5')
```


## References

Real images: Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.
Fake images: Bird, J.J. and Lotfi, A., 2024. CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. IEEE Access.
