import yaml
import os
import copy

def update_dict(base_dict, updates, output_filename):
    """
    Update the base dictionary with the provided updates and save it to the specified output file.
    
    Parameters:
    - base_dict: The base dictionary to be updated.
    - updates: A dictionary containing the updates to be applied.
    - output_filename: The filename for the updated YAML file.
    """
    updated_dict = copy.deepcopy(base_dict)  # Make a deep copy to avoid modifying the original base_dict

    for key, value in updates.items():
        if key == 'conv_layers' and isinstance(value, dict) and 'activation' in value:
            # Special handling for updating activation in conv_layers
            for layer in updated_dict['conv_layers']:
                layer['activation'] = value['activation']
        elif isinstance(value, dict) and isinstance(updated_dict.get(key), dict):
            updated_dict[key].update(value)
        elif isinstance(value, list) and isinstance(updated_dict.get(key), list):
            updated_dict[key] = value
        else:
            updated_dict[key] = value
    
    # Save the updated dictionary to a YAML file
    with open(output_filename, 'w') as file:
        yaml.dump(updated_dict, file)

# Base dictionary
base_dict = {
    'info': {
        'model_name': 'base',
        'train_dir': '../input/images/train',
        'test_dir': '../input/images/test',
        'validation_dir': '../input/images/validation',
        'model_filepath': '../output/cifake_base.h5',
        'history_filepath': '../output/history_cifake_base.pkl',
        'classes': ['REAL', 'FAKE']
    },
    'preprocess': {
        'resize': [224, 224],
        'normalize': 255
    },
    'generators': {
        'rescale': 1./255, 
        'rotation_range': 40, 
        'width_shift_range': 0.2,
        'height_shift_range': 0.2, 
        'shear_range': 0.2, 
        'zoom_range': 0.2,
        'fill_mode': 'nearest'
    },
    'model': {
        'optimizer': 'adam',
        'steps_per_epoch': 25,
        'epochs': 20,
        'validation_steps': 23,
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
    'maxpool_layers': {'pool_size': [2, 2]},
    'dense_layers': {'units': 512, 'activation': 'relu'},
    'output_layer': {'units': 2, 'activation': 'softmax'}
}

new_folder_path = '../input'

if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

with open('../input/base_dict.yaml', 'w') as file:
    yaml.dump(base_dict, file)

# Define updates for each new dictionary
updates_sigmoid = {
    'info': {
        'model_name': 'sigmoid',
        'model_filepath': '../output/cifake_sigmoid.h5',
        'history_filepath': '../output/history_cifake_sigmoid.pkl'
    },
    'conv_layers': {
        'activation': 'sigmoid'
    }
}

updates_epoch = {
    'info': {'model_name': 'epoch',
             'model_filepath': '../output/cifake_epoch.h5',
             'history_filepath': '../output/history_cifake_epoch.pkl'}, 
    'model': {'epochs': 10}
}

updates_sgd = {
    'info': {'model_name': 'sgd',
             'model_filepath': '../output/cifake_sgd.h5',
             'history_filepath': '../output/history_cifake_sgd.pkl'}, 
    'model': {'optimizer': 'SGD'}
}

updates_val_steps = {
    'info': {'model_name': 'val',
             'model_filepath': '../output/cifake_val.h5',
             'history_filepath': '../output/history_cifake_val.pkl'}, 
    'model': {'validation_steps': 10}
}

updates_transfer_learning = {
    'info': {'model_name': 'tf',
             'model_filepath': '../output/cifake_tf.h5',
             'finetune_filepath': '../output/cifake_tf_ft.h5',
             'history_filepath': '../output/history_cifake_tf.pkl',
             'finetune_history_filepath': '../output/history_cifake_tf_ft.pkl'
    },
    'transfer_learning': {'learning_rate': 0.001,
                          'initial_epochs': 10, 
                          'loss': 'BinaryCrossentropy',
                          'optimizer':  'Adam',
                          'metrics':
                              {'name': 'BinaryAccuracy', 
                              'params': {
                                  'threshold': 0.5,
                                  'name': 'accuracy'
                              }}},
    'fine_tuning': {'fine_tune_at': 100,
                    'loss': 'BinaryCrossentropy',
                    'optimizer': 'RMSprop',
                    'learning_rate': 0.001,
                    'fine_tune_epochs': 10,
                    'metrics': 
                        {'name': 'BinaryAccuracy',
                        'params': {
                            'name': 'accuracy',
                            'threshold': 0.5
                        }
}

# Update and save new dictionaries
update_dict(base_dict, updates_sigmoid, '../input/sigmoid_dict.yaml')
update_dict(base_dict, updates_epoch, '../input/epoch_dict.yaml')
update_dict(base_dict, updates_sgd, '../input/sgd_dict.yaml')
update_dict(base_dict, updates_val_steps, '../input/val_dict.yaml')
update_dict(base_dict, updates_transfer_learning, '../input/base_tf_dict.yaml')

print("Dictionaries updated and saved.")
