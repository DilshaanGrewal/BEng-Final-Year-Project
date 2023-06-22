import os
import torch
from torchinfo import summary

def get_model_size(model_path, is_state_dict, log=print):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the state dict from the file
    model = torch.load(model_path, map_location=device)
    
    # Model path can be a model or a state dict    
    if is_state_dict:
        items = model.items()
    else:
        items = model.named_parameters()
    
    type_of_w_and_b = None
    num_w_and_b = 0
    size_of_w_and_b = None
    num_w = 0
    num_b = 0
    num_nonzero_w = 0  # Counter for non-zero weights
    for name, param in items:
        # if 'weight' in name or 'bias' in name:
        #     # check if all weights and biases are of the same type 
        #     if type_of_w_and_b is None:
        #         type_of_w_and_b = param.dtype
        #         size_of_w_and_b = param.element_size()
        #     # assert type_of_w_and_b == param.dtype # <- optional check
            
        #     # count number of weights and biases
        #     num_w_and_b += param.numel()
        if type_of_w_and_b is None:
            type_of_w_and_b = param.dtype
            size_of_w_and_b = param.element_size()
        if 'weight' in name:
            num_w += param.numel()
            num_w_and_b += param.numel()
            num_nonzero_w += torch.count_nonzero(param)
        elif 'bias' in name:
            num_b += param.numel()
            num_w_and_b += param.numel()
        
    # if is_state_dict:
    #     # As state dict does not contain the information 'per layer' (only per item)
    #     # we need to iterate over the state dictionary to get the number of non-zero weights per channel
    #     is_model_filter_pruned = True
    #     if is_model_filter_pruned:
    #         channel_non_zero = 0
    #         # Iterate over the state dictionary
    #         for name, param in items:
    #             # Check if the parameter corresponds to a convolutional layer
    #             if len(param.shape) == 4:
    #                 # Iterate over the channels in the weight tensor
    #                 # for channel_index in range(param.shape[0]):
    #                     # channel = param[channel_index, :, :, :]
    #                     # channel_non_zero += torch.count_nonzero(channel)
    #                 channel_non_zero += torch.count_nonzero(param)
    #         num_nonzero_w = channel_non_zero

    log('model name: {}'.format(os.path.splitext(os.path.basename(model_path))[0]))
    # log('\tnumber of weights and biases: {}'.format(num_w_and_b))
    log('number of weights: {}'.format(num_w))
    log('number of biases: {}'.format(num_b))
    log('type of weights and biases in model: {}'.format(type_of_w_and_b))
    log('size of type of weights and biases in model (Bytes): {}'.format(size_of_w_and_b))

    size_of_model = num_w_and_b * size_of_w_and_b  # in Bytes
    log('size of model (MB): {}'.format(size_of_model / 1e6))
    
    log('number of non-zero weights: {}'.format(num_nonzero_w))
    
    # convert model to float32 as torchinfo only supports float32
    # for name, param in items:
    #     if 'weight' in name or 'bias' in name:
    #         param.data = param.data.to(torch.float32)
    # print(summary(model, input_size=(1, 3, 224, 224)))
    # layers = list(model.module.modules())
    # print(layers)


if __name__ == "__main__":
    model_dir = 'C:\\Users\\dilsh\\OneDrive\\Documents\\Dilshaan\\Third_Year\\BEng_Individual_Project\\code\\BEng-Final-Year-Project\\model\\models\\'
    # model_name = 'baseline-model_16float_in32float_fPrune_0.1'
    # model_name = 'baseline-model'
    # get_model_size(model_dir + model_name + '.pth', is_state_dict=False)
    # print('')
    model_name = 'vgg16-baseline'
    get_model_size(model_dir + model_name + '.pth', is_state_dict=True)
