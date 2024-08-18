# Image Classification project: Real vs. AI-Generated Synthetic Images
## Project Overview

This project focuses on differentiating between real images and AI-generated synthetic images using convolutional neural networks (CNNs). The core objective is to leverage deep learning techniques to enhance model performance in classifying real and fake images. In this project, I demonstrate my expertise in data preprocessing, model architecture design, hyperparameter tuning, and model evaluation using Python, Keras, and TensorFlow.

I have structured the code using Python classes, each dedicated to specific tasks:

1) **Image Sourcing**. The **'Images'** class is used for sourcing and organizing the real and AI-generated image datasets.
2) **Image Preprocessing**. The **'Preprocess'** class handles image preprocessing tasks such as resizing, normalization, and the creation of data generators for efficient loading during model training.
3) **Model Definition**. The **'CNN_model'** class defines the CNN architecture, compiles the model with various configurations, and manages the training process.
4) **Model Saving**. After training, models are saved in .h5 format for future use and comparison.
5) **Model Evaluation**. 'The **'Evaluate_model'** class is used to load trained models, evaluate their performance on the test dataset, and compare metrics.

## Input Data
800 images for the training set, 735 images for the validation set, and 1000 images for the testing set.

## Methodology
The project employs a systematic approach to model training and comparison through hyperparameter tuning. I started by training a base model and subsequently changed key hyperparameters such as the optimizer, activation functions, and the number of epochs. The models and configurations used in this project are as follows:

1) **Base**.
   * Optimizer: Adam
   * Epochs: 20
   * Activation Function in the convolution layers: ReLU
   * Validation Steps: 23
     
2) **Epoch**. Same as the base model but with a reduced number of epochs (Epochs: 10).
4) **SGD**. Same as the base model, but the optimizer is switched to Stochastic Gradient Descent (SGD).
5) **Sigmoid**. Same as the base model, but the activation function in the convolution layers is changed to Sigmoid.
6) **Val**. Same as the base model, except the validation steps are reduced to 10.

### Configuration Management with YAML
To streamline model configuration and training, I created YAML configuration files for each model. The parameters for each model (e.g., optimizer, epochs, activation function) are stored in individual YAML files, which are then loaded into the training script. This approach enables easy experimentation with different configurations and keeps the codebase clean and maintainable.

The **create_yaml_files.py** script is used to generate these YAML configuration files, allowing for seamless updates to model parameters without modifying the main code.

## Running the Project
To execute the project, follow these steps:

1) Clone the repository and navigate to the project directory.
2) Ensure the necessary Python packages are installed by running:
```bash
pip install -r requirements.txt
```
3) Open and run the **train_evaluate_compare_cnn_models_image_classification.ipynb** notebook. This notebook walks through the entire process:
  * Step 1: Generate YAML files for each model using the create_yaml_files.py script.
  * Step 2: Create training, validation, and testing datasets from the sourced images.
  * Step 3: Train and evaluate the models, comparing performance across different configurations.

## Results

The trained models were evaluated based on accuracy, loss, precision, recall, and F1-score. Detailed performance metrics for each model are included in the final notebook output, along with visualizations comparing their performance.

## Future Work 

- Grad-CAM (?)

## References

Real images: Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.
Fake images: Bird, J.J. and Lotfi, A., 2024. CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. IEEE Access.
