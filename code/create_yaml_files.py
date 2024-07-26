import yaml
import os

base_dict = {
    'info':
        {'model_name': 'base',
        'train_dir': '../input/images/train',
        'test_dir': '../input/images/test',
        'validation_dir': '../input/images/validation',
        'model_filepath': '../output/cifake_base.h5',
        'history_filepath': '../output/history_cifake_base.pkl',
        'classes': ['REAL', 'FAKE']},

    'generators':
        {'rescale': 1./255, 
        'rotation_range': 40, 
        'width_shift_range': 0.2,
        'height_shift_range': 0.2, 
        'shear_range': 0.2, 
        'zoom_range': 0.2,
        'fill_mode': 'nearest'},

    'model':
        {'optimizer': 'adam',
        'steps_per_epoch': 100,
        'epochs': 20,
        'validation_steps': 50,
        'dropout': 0.5, 
        'loss': 'categorical_crossentropy', 
        'metrics': ['accuracy']
        },

    'conv_layers': [
        {'filters': 32, 
         'kernel_size': [3, 3], 
         'activation': 'relu',
         'input_shape': [224, 224, 3]},

        {'filters': 32, 
         'kernel_size': [3, 3], 
         'activation': 'relu',
         'input_shape': [224, 224, 3]}
    ],

    'maxpool_layers':
        {'pool_size': [2,2]
         },

    'dense_layers':
        {'units': 512,
         'activation': 'relu'
        },

    'output_layer':
        {'units' : 2,
         'activation': 'softmax'
        }
}

new_folder_path = '../input'

if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

with open('../input/base_dict.yaml', 'w') as file:
    yaml.dump(base_dict, file)
    
with open('../input/base_dict.yaml', 'r') as file:
    base_dict = yaml.safe_load(file)

# Create other YAML files ---
#Sigmoid
sigmoid_dict = base_dict.copy()

for layer in sigmoid_dict['conv_layers']:
    layer['activation'] = 'sigmoid'

# Write the updated dictionary to the new YAML file
with open('../input/sigmoid_dict.yaml', 'w') as file:
    yaml.dump(sigmoid_dict, file)

#Epoch---
epoch_dict = base_dict.copy()

epoch_dict['model']['epochs'] = 30

with open('../input/epoch_dict.yaml', 'w') as file:
    yaml.dump(epoch_dict, file)

#SGD---
sgd_dict = base_dict.copy()

sgd_dict['model']['optimizer'] = 'SGD'

with open('../input/sgd_dict.yaml', 'w') as file:
    yaml.dump(sgd_dict, file)

#ValSteps---
val_dict = base_dict.copy()

val_dict['model']['validation_steps'] = 100

with open('../input/val_dict.yaml', 'w') as file:
    yaml.dump(val_dict, file)
