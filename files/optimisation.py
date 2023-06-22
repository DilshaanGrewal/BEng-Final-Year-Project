import torch
import torch.utils.data
import copy

############### Reduce precision ###############
def reduce_precision(model_dir, model_name, is_state_dict):
    device = torch.device("cpu")
    load_model_dir = model_dir + model_name + ".pth"
    # load model
    model_load = torch.load(load_model_dir, map_location=device)
    model_to_quant = copy.deepcopy(model_load)
    # Model path can be a model or a state dict
    if is_state_dict:
        items = model_to_quant.items()
    else:
        items = model_to_quant.named_parameters()
    
    # reduce precision parameter
    reduce_precision_by = torch.float16
    
    for name, param in items:
        if 'weight' in name or 'bias' in name:
            param.data = param.data.to(reduce_precision_by)

    torch.save(model_to_quant, model_dir + model_name +
               "_quantized_" + reduce_precision_by + ".pth")


def binarise(model_dir, model_name, is_state_dict):
    device = torch.device("cpu")
    load_model_dir = model_dir + model_name + ".pth"
    # load model
    model_load = torch.load(load_model_dir, map_location=device)
    model_to_quant = copy.deepcopy(model_load)
    # Model path can be a model or a state dict
    if is_state_dict:
        items = model_to_quant.items()
    else:
        items = model_to_quant.named_parameters()
    
    # thershold for binarisation
    t = 0
    
    for name, param in items:
        if ('weight' in name or 'bias' in name) and not param.requires_grad:
            param.data = torch.where(param.data > t, torch.tensor(1.0), torch.tensor(0.0))
            # quantise to boolean data type
            param.data = param.data.to(torch.bool)

    torch.save(model_to_quant, model_dir + model_name + "_binarise" + ".pth")


def ternarise(model_dir, model_name, is_state_dict):
    device = torch.device("cpu")
    load_model_dir = model_dir + model_name + ".pth"
    # load model
    model_load = torch.load(load_model_dir, map_location=device)
    model_to_quant = copy.deepcopy(model_load)
    # Model path can be a model or a state dict
    if is_state_dict:
        items = model_to_quant.items()
    else:
        items = model_to_quant.named_parameters()
    
    # thersholds for ternarisation
    t1 = -0.33
    t2 = 0.33
    
    for name, param in items:
        if ('weight' in name or 'bias' in name) and not param.requires_grad:
            param.data = torch.where(param.data > t2, torch.tensor(1.0), torch.where(param.data < t1, torch.tensor(-1.0), torch.tensor(0.0)))
            # quantise to 8 bit integer data type (smallest non bool data type)
            param.data = param.data.to(torch.int8)

    torch.save(model_to_quant, model_dir + model_name + "_ternarise" + ".pth")

import optimisation_helper
from optimisation_helper import float32_to_uint8, float32_to_bit7, float32_to_bit6, float32_to_bit5, float32_to_bit4, float32_to_bit3
def reduce_bit(model_dir, model_name, is_state_dict):
    device = torch.device("cpu")
    load_model_dir = model_dir + model_name + ".pth"
    # load model
    model_load = torch.load(load_model_dir, map_location=device)
    model_to_quant = copy.deepcopy(model_load)
    # Model path can be a model or a state dict
    if is_state_dict:
        items = model_to_quant.items()
    else:
        items = model_to_quant.named_parameters()

    for name, param in items:
        if ('weight' in name or 'bias' in name) and not param.requires_grad:
            # quantise using appropriate function
            param.data = param.data.apply_(optimisation_helper.float32_to_fixed24)
            # for bits less than or equal to 8 bit
            # param.data = param.data.to(torch.float32)

            # for fixed point:
            param.data = param.data.to(torch.int32)

    # save model with appropriate name
    torch.save(model_to_quant, model_dir + model_name + "_24fixed" + ".pth")


############### Pruning ###############
import torch.nn.utils.prune as prune
"""
def mag_weight_pruning(model_dir, model_name, is_state_dict):
    device = torch.device("cpu")
    load_model_dir = model_dir + model_name + ".pth"
    # load model
    model_load = torch.load(load_model_dir, map_location=device)
    model_to_prune = copy.deepcopy(model_load)
    model_to_prune = model_to_prune.module
    
    pruning_percent = 0.4
    
    # Unstructured pruning is how to implement magnitude weight pruning
    for name, module in model_to_prune.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            # Apply L1 unstructured pruning on the module
            prune.l1_unstructured(module, name='weight', amount=pruning_percent)
            # Remove the pruning re-parametrisation in the module
            # Note: this does not physically remove the weights, just sets the to zero
            prune.remove(module, 'weight')
    
    model_to_prune = torch.nn.DataParallel(model_to_prune)
    torch.save(model_to_prune, model_dir + model_name + "_wPrune_" + str(pruning_percent) + ".pth")
"""


def mag_weight_pruning_soft(model_dir, model_name, is_state_dict):
    # Same as magnitude weight pruning but with soft pruning
    # Soft pruning is where the weights are not actually removed but are set to zero

    device = torch.device("cpu")
    load_model_dir = model_dir + model_name + ".pth"
    # load model
    model_load = torch.load(load_model_dir, map_location=device)
    model_to_prune = copy.deepcopy(model_load)
    
    # Specify the percentage of weights to prune
    pruning_percent = 0.4
    
    # Model path can be a model or a state dict
    if is_state_dict:
        items = model_to_prune.items()
    else:
        items = model_to_prune.named_parameters()

    for name, param in items:
        if 'weight' in name:
            param.data = prune_weight_tensor(param.data, pruning_percent)
    
    torch.save(model_to_prune, model_dir + model_name + "_wPruneSoft_" + str(pruning_percent) + ".pth")

def prune_weight_tensor(tensor, pruning_percent):
    weights = tensor.flatten()
    num_weights_to_prune = int(weights.numel() * pruning_percent)
    abs_weights = torch.abs(weights)
    sorted_weights, _ = torch.sort(abs_weights)
    threshold_index = num_weights_to_prune - 1
    threshold_value = sorted_weights[threshold_index]

    # Set the smallest weights below the threshold to zero
    prune_mask = torch.where(abs_weights <= threshold_value, torch.zeros_like(weights), torch.ones_like(weights))
    pruned_weights = weights * prune_mask
    pruned_weights = pruned_weights.reshape(tensor.shape)

    return pruned_weights

#### Filer pruning ####
def filter_pruning(model_dir, model_name, is_state_dict):
    ## Note filter pruning is only for convolutional layers
    device = torch.device("cpu")
    load_model_dir = model_dir + model_name + ".pth"
    # load model
    model_load = torch.load(load_model_dir, map_location=device)
    model_to_prune = copy.deepcopy(model_load)
    # working_model = copy.deepcopy(model_load)
    
    if is_state_dict:
        # Note: for a state dict, we can check if each layer is an instance of conv2d by checking the shape of the layer is 4
        layers = list(model_to_prune.items())
        
        # # Specify the percentage of filters to prune
        # pruning_percent = 0.05
        # num_layers = len(layers)
        # for i, prev_layer in enumerate(layers):
        #     if len(prev_layer[1].shape) == 4:
        #         # Find the next convolutional layer
        #         for j in range(i + 1, num_layers - 1):
        #             layer = layers[j]
        #             if len(layer[1].shape) == 4:
        #                 layer_in_channels = layer[1].shape[1]
        #                 prev_layer_out_channels = prev_layer[1].shape[0]
                        
        #                 # If the next layer is not a convolutional layer with the same number of input channels
        #                 # Then we cannot prune this layer
        #                 # Likely because of a non convolutional layer in between, which changes the number of channels
        #                 if layer_in_channels != prev_layer_out_channels:
        #                     break
        #                 standard_input = torch.randn(1, prev_layer_out_channels, 10, 10) # Might be better to use a real input
        #                 standard_output = torch.sum(torch.mul(layer[1], standard_input), dim=0)
        #                 smallest_error_indices = error_indices_channel(layer, standard_input, standard_output, pruning_percent)
        #                 ## Prune the weak channels in the layer (soft pruning)
        #                 for channel_idx in smallest_error_indices:
        #                     layer[1].weight.data[channel_idx] = 0.0
        #                     layer[1].bias.data[channel_idx] = 0.0
        
    else:
        model_to_prune = model_to_prune.module
        layers = list(model_to_prune.modules())
        # working_model = working_model.module
        # working_layers = list(working_model.modules())

    # Specify the percentage of filters to prune
    pruning_percent = 0.05
    
    num_layers = len(layers)
    # Iterate through the layers for pruning
    for i, prev_layer in enumerate(layers):
        if isinstance(prev_layer, torch.nn.Conv2d):
            # Find the next convolutional layer
            for j in range(i + 1, num_layers - 1):
                layer = layers[j]
                if isinstance(layer, torch.nn.Conv2d):
                    if layer.in_channels != prev_layer.out_channels:
                        # If the next layer is not a convolutional layer with the same number of input channels
                        # Then we cannot prune this layer
                        # Likely because of a non convolutional layer in between, which changes the number of channels
                        break
                    # Now we have two convolutional layers 
                    # with prev_layer.out_channels = layer.in_channels
                    # So can do filter pruning on layer
                    standard_input = torch.randn(1, prev_layer.out_channels, 10, 10) # Might be better to use a real input
                    standard_output = layer(standard_input)
                    
                    smallest_error_indices = error_indices_channel(layer, standard_input, standard_output, pruning_percent)

                    ## Prune the weak channels in the layer (soft pruning)
                    for channel_idx in smallest_error_indices:
                        layer.weight.data[channel_idx] = 0.0
                        layer.bias.data[channel_idx] = 0.0
                    
                    ## Prune the weak channels in the layer (hard pruning)
                    # Conv2d shape: (out_channels, in_channels, kernel_height, kernel_width)
                    # channel_ixd_to_keep = [k for k in range(layer.in_channels) if k not in smallest_error_indices]
                    # channel_ixd_to_keep = torch.tensor(channel_ixd_to_keep)

                    # pruned_prev_layer_w = torch.index_select(prev_layer.weight, 0, channel_ixd_to_keep)
                    # pruned_prev_layer_b = torch.index_select(prev_layer.bias, 0, channel_ixd_to_keep)
                    # pruned_layer_w = torch.index_select(layer.weight, 1, channel_ixd_to_keep)
                    # pruned_layer_b = torch.index_select(layer.bias, 0, channel_ixd_to_keep)

                    # # Update the weights in the layers of the working model
                    # working_layers[i].weight.data = pruned_prev_layer_w
                    # working_layers[i].bias.data = pruned_prev_layer_b
                    
                    # working_layers[j].weight.data = pruned_layer_w
                    # working_layers[j].bias.data = pruned_layer_b
                   

    if not is_state_dict:
        model_to_prune = torch.nn.DataParallel(model_to_prune)
    torch.save(model_to_prune, model_dir + model_name + "_fPrune_" + str(pruning_percent) + ".pth")
    # working_model = torch.nn.DataParallel(working_model)
    # torch.save(working_model, model_dir + model_name + "_fPrune___" + str(pruning_percent) + ".pth")
    
def error_indices_channel(layer, standard_input, standard_output, pruning_percent):
    # Calculate the error when setting a channel to zero for each channel in the layer
    errors = []
    for channel_idx in range(layer.in_channels):
        # Create copy of layer
        pruned_layer = copy.deepcopy(layer)
        # Set the channel to zero
        pruned_layer.weight.data[channel_idx] = 0.0
        pruned_layer.bias.data[channel_idx] = 0.0

        # Calculate the error
        pruned_output = pruned_layer(standard_input)
        error = torch.norm(pruned_output - standard_output)
        errors.append((error, channel_idx))

    # Prune the weak channels in the layer
    errors.sort(key=lambda x: x[0])
    # Remove channels which do not lead to large errors
    num_indices = int(len(errors) * pruning_percent)
    smallest_error_indices = [index for error, index in errors[:num_indices]]
    return smallest_error_indices


#### Structural Pruning ####
from dataHelper import DatasetFolder
from settings import validation_dir, test_batch_size
NUM_OF_WORKERS = 0
PIN_MEMORY_FLAG = False
import torchvision.transforms as transforms
import numpy as np
from train_and_test import test
import networkx as nx

# validation set
validation_dataset = DatasetFolder(
    validation_dir,
    loader=np.load,
    extensions=("npy",),
    transform=transforms.Compose([
        torch.from_numpy,
    ]))
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=NUM_OF_WORKERS, pin_memory=PIN_MEMORY_FLAG)


def structural_pruning_search(model_dir, model_name, is_state_dict=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_model_dir = model_dir + model_name + ".pth"
    # load model
    model_load = torch.load(load_model_dir, map_location=device)
    model_to_prune = copy.deepcopy(model_load)
    model_to_prune = model_to_prune.to(device)

    # Genetic Algorithm-based optimisation using tournament selection
    generations = 2
    population_size = 2
    mutation_rate = 0.1
    
    population = []
    
    for generation in range(generations):
        print(f"Generation {generation}/{generations}")

        new_population = population
        for _ in range(population_size):
            # Random percentage of weights to prune (between 0 and 1)
            pruning_percent = np.random.rand()
            model = copy.deepcopy(model_to_prune)
            model = structural_prune(model, pruning_percent)
            new_population.append((model, pruning_percent))
            print(f"Pruning percent: {pruning_percent}")

        new_population = sorted(new_population, key=lambda x: structural_pruning_score(x[0]), reverse=True)
        elite_percentage = 0.2
        elite_size = max(int(population_size * elite_percentage), 1) # Ensure at least one elite model
        elite = new_population[:elite_size]
        
        print("before mutation\n")

        for _ in range(elite_size, population_size):
            p1_ixd = np.random.choice(len(elite))
            parent1 = elite[p1_ixd]
            p1_prune_percent = parent1[1]
            
            p2_ixd = np.random.choice(len(elite))
            parent2 = elite[p2_ixd]
            p2_prune_percent = parent2[1]

            child_pruning_percent = (p1_prune_percent + p2_prune_percent) / 2 # Average the pruning percentages of the parents
            child_pruning_percent += np.random.randn() * mutation_rate # Add some noise to the pruning percentage
            child_pruning_percent = np.clip(child_pruning_percent, 0, 1) # Ensure the pruning percentage is between 0 and 1
            print(f"Child pruning percent: {child_pruning_percent}")
            child = structural_prune(model_to_prune, child_pruning_percent)
            
            p1_score = structural_pruning_score(parent1[0])
            p2_score = structural_pruning_score(parent2[0])
            child_score = structural_pruning_score(child)
            
            # Choose the best model out of the parents and the child
            best_score = max(p1_score, p2_score, child_score)
            print("p1_score: ", p1_score)
            print("p2_score: ", p2_score)
            print("child_score: ", child_score)
            if best_score == p1_score:
                population.append(parent1)
                print("Parent 1: ", parent1[1])
            elif best_score == p2_score:
                population.append(parent2)
                print("Parent 2", parent2[1])
            else:
                population.append((child, child_pruning_percent)) 
                print("Child", child_pruning_percent)
        
    population = sorted(population, key=lambda x: structural_pruning_score(x[0]), reverse=True)
    best_model, best_pruning_percentage = population[0]
    print(f"\n Best model: {best_pruning_percentage}")
    structural_pruning_score(best_model)
    
    torch.save(best_model, model_dir + model_name + "_sPrune_" + str(best_pruning_percentage) + ".pth")

def structural_pruning_score(model):
    # Metric to optimise for structural pruning
    with torch.no_grad():
        # !!! Set output_accuracy in train_and_test.py to True !!!
        accu = test(model=torch.nn.DataParallel(model), dataloader=validation_loader, log=print, class_specific=True)
        # accu = test(model, dataloader=validation_loader, log=print, class_specific=True)

    c, l = calculate_C_L(model.modules())
    # Convert tensors to floats
    c = c.item()
    l = l.item()
    # Scale down the characteristic path length to be in the same order of magnitude as the clustering coefficient and accuracy
    l_scaling_factor = 0.01
    l = l * l_scaling_factor
    # Differences in c and l for different models is small so we compare values after 3 decimal places
    c_trunc = int(c * 1000)/1000.0
    l_trunc = int(l * 1000)/1000.0
    c = (c - c_trunc) * 1000
    l = (l - l_trunc) * 1000
    # c = (c - round(c, 3)) * 1000
    # l = (l - round(l, 3)) * 1000
    # Fitness function: want high accuracy, high clustering coefficient, and low characteristic path length
    # But high accuracy is more important than the other two
    accu_weight = 0.5
    c_weight = 0.25
    l_weight = 0.25
    print(f"Accuracy: {accu}")
    print(f"Clustering coefficient: {c}")
    print(f"Characteristic path length: {l}")
    return (accu_weight*accu) + (c_weight*c) - (l_weight*l)

def calculate_C_L(layers):
    layers = list(layers)
    num_classes = layers[-1].out_features
    c_total = 0
    l_total = 0

    for layer in layers:
        if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
            # Features in a linear layer refer to neurons
            if isinstance(layer, torch.nn.Linear):
                num_features = layer.out_features
            # Features in a convolutional layer refer to feature maps
            elif isinstance(layer, torch.nn.Conv2d):
                num_features = layer.out_channels
            
            features_per_class = torch.zeros(num_classes)
            classes_per_feature = torch.zeros(num_features)

            for i in range(num_classes):
                for j in range(num_features):
                    # If the weight is not zero, then there is a weighted connection between 
                    # the class (next layer) and the feature (current layer)
                    if isinstance(layer, torch.nn.Linear):
                        if layer.weight[i, j] != 0:
                            classes_per_feature[j] += 1
                            features_per_class[i] += 1
                    elif isinstance(layer, torch.nn.Conv2d):
                        # Dimension of layer.weight: (out_channels, in_channels, kernel_height, kernel_width)
                        # Access the weights corresponding to a specific class using layer.weight[i, :, :, :]
                        if torch.sum(layer.weight[i, :, :, :]) != 0:
                            classes_per_feature[j] += 1
                            features_per_class[i] += 1

            # Calculate clustering coefficient (C)
            c_layer = torch.mean(classes_per_feature)
            # Calculate characteristic path length (L)
            l_layer = torch.mean(features_per_class)
            
            c_total += c_layer
            l_total += l_layer

    c_avg = c_total / len(layers)
    l_avg = l_total / len(layers)
    return c_avg, l_avg

def structural_prune(model, pruning_percent, is_state_dict=False):
    if is_state_dict:
        items = model.items()
    else:
        items = model.named_parameters()
    
    # Prune a percentage of individual weight/bias numbers in the model
    for name, param in items:
        if 'weight' in name or 'bias' in name:
            # Get the indices of the smallest weights/biases to prune based on the pruning percentage
            flatten_param = param.flatten()
            num_params_to_prune = int(flatten_param.numel() * pruning_percent)
            sorted_indices = torch.argsort(torch.abs(flatten_param))
            prune_indices = sorted_indices[:num_params_to_prune]
            # Create a mask to prune the weights/biases
            prune_mask = torch.ones_like(flatten_param)
            prune_mask[prune_indices] = 0
            prune_mask = prune_mask.reshape(param.shape)
            # Apply the mask
            param.data *= prune_mask
    return model
 
############### Combining optimisation methods ###############
## 8 bit ##
def combine_8bit_wPrune(model_dir, pruning_percent, is_state_dict=False):
    # model_name = "baseline-model_8bit"
    model_name = "vgg16-baseline_8bit"
    device = torch.device("cpu")
    load_model_dir = model_dir + model_name + ".pth"
    model = torch.load(load_model_dir, map_location=device)
    opt_model = copy.deepcopy(model)
    
    if is_state_dict:
        items = opt_model.items()
    else:
        items = opt_model.named_parameters()
    
    for name, param in items:
        if 'weight' in name or 'bias' in name:
            param.data = param.data.to(device)
            param.data = param.data.to(torch.float32)
            param.data = param.data.apply_(optimisation_helper.uint8_to_float32)
            if 'weight' in name:
                param.data = prune_weight_tensor(param.data, pruning_percent)

    torch.save(opt_model, model_dir + model_name + "_wPrune_" + str(pruning_percent) + ".pth")


def combine_8bit_fPrune(model_dir, pruning_percent, is_state_dict=False):
    # model_name = "baseline-model_8bit"
    model_name = "vgg16-baseline_8bit"
    device = torch.device("cpu")
    load_model_dir = model_dir + model_name + ".pth"
    model = torch.load(load_model_dir, map_location=device)
    opt_model = copy.deepcopy(model)
    
    if is_state_dict:
        items = opt_model.items()
    else:
        items = opt_model.named_parameters()
    
    for name, param in items:
        if 'weight' in name or 'bias' in name:
            param.data = param.data.to(device)
            param.data = param.data.to(torch.float32)
            param.data = param.data.apply_(optimisation_helper.uint8_to_float32)
    
    torch.save(opt_model, model_dir + model_name + "_in32float.pth") # Save the model with 8 bit weights and 32 bit floats
    
    # Filter pruning (define pruning percentage in filter_pruning function)
    filter_pruning(model_dir, model_name + "_in32float", is_state_dict=True)


def combine_8bit_sPrune(model_dir, pruning_percent, is_state_dict=False):
    # model_name = "baseline-model_8bit"
    model_name = "vgg16-baseline_8bit"

    device = torch.device("cpu")
    load_model_dir = model_dir + model_name + ".pth"
    model = torch.load(load_model_dir, map_location=device)
    opt_model = copy.deepcopy(model)
    pruning_percent = 0.335
    
    if is_state_dict:
        items = opt_model.items()
    else:
        items = opt_model.named_parameters()
    
    for name, param in items:
        if 'weight' in name or 'bias' in name:
            param.data = param.data.to(device)
            param.data = param.data.to(torch.float32)
            param.data = param.data.apply_(optimisation_helper.uint8_to_float32)

    opt_model = structural_prune(opt_model, pruning_percent, is_state_dict)

    torch.save(opt_model, model_dir + model_name + "_sPrune_" + str(pruning_percent) + ".pth")


## 16 bit ##
def combine_16bit_wPrune(model_dir, pruning_percent, is_state_dict=False):
    # model_name = "baseline-model_16float"
    model_name = "vgg16-baseline_16float"
    device = torch.device("cpu")
    load_model_dir = model_dir + model_name + ".pth"
    model = torch.load(load_model_dir, map_location=device)
    opt_model = copy.deepcopy(model)
    
    if is_state_dict:
        items = opt_model.items()
    else:
        items = opt_model.named_parameters()
    
    for name, param in items:
        if 'weight' in name or 'bias' in name:
            param.data = param.data.to(device)
            param.data = param.data.to(torch.float32)
            if 'weight' in name:
                param.data = prune_weight_tensor(param.data, pruning_percent)

    torch.save(opt_model, model_dir + model_name + "_wPrune_" + str(pruning_percent) + ".pth")


def combine_16bit_fPrune(model_dir, pruning_percent, is_state_dict=False):
    # model_name = "baseline-model_16float"
    model_name = "vgg16-baseline_16float"

    device = torch.device("cpu")
    load_model_dir = model_dir + model_name + ".pth"
    model = torch.load(load_model_dir, map_location=device)
    opt_model = copy.deepcopy(model)
    
    if is_state_dict:
        items = opt_model.items()
    else:
        items = opt_model.named_parameters()
    
    for name, param in items:
        if 'weight' in name or 'bias' in name:
            param.data = param.data.to(device)
            param.data = param.data.to(torch.float32)

    torch.save(opt_model, model_dir + model_name + "_in32float.pth") # Save the model with 8 bit weights and 32 bit floats
    # Filter pruning (define pruning percentage in filter_pruning function)
    filter_pruning(model_dir, model_name + "_in32float", is_state_dict)
    

def combine_16bit_sPrune(model_dir, pruning_percent, is_state_dict=False):
    # model_name = "baseline-model_16float"
    model_name = "vgg16-baseline_16float"

    device = torch.device("cpu")
    load_model_dir = model_dir + model_name + ".pth"
    model = torch.load(load_model_dir, map_location=device)
    opt_model = copy.deepcopy(model)
    
    if is_state_dict:
        items = opt_model.items()
    else:
        items = opt_model.named_parameters()
    
    for name, param in items:
        if 'weight' in name or 'bias' in name:
            param.data = param.data.to(device)
            param.data = param.data.to(torch.float32)

    opt_model = structural_prune(opt_model, pruning_percent, is_state_dict)

    torch.save(opt_model, model_dir + model_name + "_sPrune_" + str(pruning_percent) + ".pth")


#########################################

import matplotlib.pyplot as plt
def plot_graph(data):
    x_values, y_values = zip(*data)  # Unzip the list of tuples into separate lists

    # Plot the graph
    plt.plot(x_values, y_values)
    plt.xlabel('Pruning Percentage')
    plt.ylabel('Structural Pruning Score')
    plt.title('Structural Pruning Score vs Pruning Percentage')
    plt.grid(True)
    plt.savefig('graph6.png')  # Save the figure as a PNG file


if __name__ == "__main__":
    model_dir='C:\\Users\\dilsh\\OneDrive\\Documents\\Dilshaan\\Third_Year\\BEng_Individual_Project\\code\\BEng-Final-Year-Project\\model\\models\\' 
    # model_name="baseline-model"   # ppnet models are stored as models
    # model_name="vgg16-baseline"     # vgg pre-trained models are stored as state dicts
    # reduce_precision(model_dir, model_name, is_state_dict=False)
    # binarise(model_dir, model_name, is_state_dict=True)
    # ternarise(model_dir, model_name, is_state_dict=False)
    # reduce_8_bit(model_dir, model_name, is_state_dict=True)
    # reduce_bit(model_dir, model_name, is_state_dict=True)
    # mag_weight_pruning(model_dir, model_name, is_state_dict=True)
    # mag_weight_pruning_soft(model_dir, model_name, is_state_dict=True)
    # filter_pruning(model_dir, model_name, is_state_dict=True)
    # structural_pruning_search(model_dir, model_name, is_state_dict=False)
    # combine_8bit_wPrune(model_dir, 0.2, is_state_dict=True)
    # combine_8bit_fPrune(model_dir, 0.1, is_state_dict=True)
    # combine_8bit_sPrune(model_dir, 0.335, is_state_dict=True)
    # combine_16bit_wPrune(model_dir, 0.1, is_state_dict=True)
    # combine_16bit_fPrune(model_dir, 0.05, is_state_dict=True)
    # combine_16bit_sPrune(model_dir, 0.335, is_state_dict=True)
    
    
    # model_name = "baseline-model_sPrune_0.335"
    # device = torch.device("cuda:0")
    # load_model_dir = model_dir + model_name + ".pth"
    # model = torch.load(load_model_dir, map_location=device)
    # opt_model = copy.deepcopy(model)
    # i = 0
    # score = []
    # while i <= 0.45:
    #     m = structural_prune(opt_model, i, is_state_dict=False)
    #     m = m.module
    #     score.append((i, structural_pruning_score(m)))
    #     i += 0.005
    # print(score)
    # plot_graph(score)
    
    # scores = [(0, 49.74712112971713), (0.005, 49.74712112971713), (0.01, 49.74712112971713), (0.015, 49.74712112971713), (0.02, 49.74712112971713), (0.025, 49.86616874876475), (0.030000000000000002, 49.86616874876475), (0.035, 49.86616874876475), (0.04, 49.86616874876475), (0.045, 49.86616874876475), (0.049999999999999996, 49.86616874876475), (0.05499999999999999, 49.86616874876475), (0.05999999999999999, 49.86616874876475), (0.06499999999999999, 49.86616874876475), (0.06999999999999999, 49.80664493924094), (0.075, 49.80664493924094), (0.08, 49.80664493924094), (0.085, 49.80664493924094), (0.09000000000000001, 49.80664493924094), (0.09500000000000001, 49.80664493924094), (0.10000000000000002, 49.74712112971713), (0.10500000000000002, 49.74712112971713), (0.11000000000000003, 49.74712112971713), (0.11500000000000003, 49.74712112971713), (0.12000000000000004, 49.74712112971713), (0.12500000000000003, 49.628073510669516), (0.13000000000000003, 49.628073510669516), (0.13500000000000004, 49.628073510669516), (0.14000000000000004, 49.628073510669516), (0.14500000000000005, 49.628073510669516), (0.15000000000000005, 49.68759732019332), (0.15500000000000005, 49.68759732019332), (0.16000000000000006, 49.74712112971713), (0.16500000000000006, 49.74712112971713), (0.17000000000000007, 49.74712112971713), (0.17500000000000007, 49.80664493924094), (0.18000000000000008, 49.80664493924094), (0.18500000000000008, 49.74712112971713), (0.19000000000000009, 49.80664493924094), (0.1950000000000001, 49.80664493924094), (0.2000000000000001, 49.80664493924094), (0.2050000000000001, 49.74712112971713), (0.2100000000000001, 49.80664493924094), (0.2150000000000001, 49.80664493924094), (0.2200000000000001, 49.80664493924094), (0.22500000000000012, 49.80664493924094), (0.23000000000000012, 49.80664493924094),
            #   (0.23500000000000013, 49.86616874876475), (0.24000000000000013, 49.80664493924094), (0.24500000000000013, 49.80664493924094), (0.2500000000000001, 49.80664493924094), (0.2550000000000001, 49.80664493924094), (0.2600000000000001, 49.80664493924094), (0.2650000000000001, 49.80664493924094), (0.27000000000000013, 49.80664493924094), (0.27500000000000013, 49.80664493924094), (0.28000000000000014, 49.80664493924094), (0.28500000000000014, 49.86616874876475), (0.29000000000000015, 49.80664493924094), (0.29500000000000015, 49.80664493924094), (0.30000000000000016, 49.80664493924094), (0.30500000000000016, 49.74712112971713), (0.31000000000000016, 49.80664493924094), (0.31500000000000017, 49.92569255828856), (0.3200000000000002, 49.92569255828856), (0.3250000000000002, 49.92569255828856), (0.3300000000000002, 49.86616874876475), (0.3350000000000002, 49.86616874876475), (0.3400000000000002, 49.80664493924094), (0.3450000000000002, 49.80664493924094), (0.3500000000000002, 49.86616874876475), (0.3550000000000002, 49.74712112971713), (0.3600000000000002, 49.74712112971713), (0.3650000000000002, 49.80664493924094), (0.3700000000000002, 49.74712112971713), (0.3750000000000002, 49.80664493924094), (0.3800000000000002, 49.74712112971713), (0.38500000000000023, 49.74712112971713), (0.39000000000000024, 49.74712112971713), (0.39500000000000024, 49.80664493924094), (0.40000000000000024, 48.73521636781237), (0.40500000000000025, 48.55664493924094), (0.41000000000000025, 49.270930653526655), (0.41500000000000026, 49.330454463050465), (0.42000000000000026, 49.21140684400285)]
    # plot_graph(scores)
    
    model_name="vgg16-baseline" 
    device = torch.device("cuda:0")
    load_model_dir = model_dir + model_name + ".pth"
    m = torch.load(load_model_dir, map_location=device)
    model = copy.deepcopy(m)
    model_ = structural_prune(model, 0.335, is_state_dict=True)
    torch.save(model_, model_dir + model_name + "_sPrune___" + str(0.335) + ".pth")
    

