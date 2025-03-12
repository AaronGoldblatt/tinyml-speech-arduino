import torch
import numpy as np
import torch.nn as nn


def flop(model, input_shape, device):
    total = {}

    def count_flops(name):
        def hook(module, input, output):
            "Hook that calculates number of floating point operations"
            flops = {}
            batch_size = input[0].shape[0]
            
            if isinstance(module, nn.Linear):
                input_features = input[0].shape[1]
                output_features = output.shape[1]
                flops['multiply'] = batch_size * input_features * output_features
                flops['add'] = batch_size * input_features * output_features
                if module.bias is not None:
                    flops['add'] += batch_size * output_features

            if isinstance(module, nn.Conv2d):
                in_channels = input[0].shape[1]
                out_channels = output.shape[1]
                output_height, output_width = output.shape[2:]
                kernel_operations = module.kernel_size[0] * module.kernel_size[1] * (in_channels // module.groups)
                flops['multiply'] = batch_size * output_height * output_width * out_channels * kernel_operations
                flops['add'] = batch_size * output_height * output_width * out_channels * kernel_operations
                if module.bias is not None:
                    flops['add'] += batch_size * out_channels * output_height * output_width

            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                num_elements = np.prod(input[0].shape)
                flops['add'] = num_elements * 2
                flops['multiply'] = num_elements * 2
            
            total[name] = flops
        return hook

    handle_list = []
    for name, module in model.named_modules():
        handle = module.register_forward_hook(count_flops(name))
        handle_list.append(handle)
    input = torch.ones(input_shape).to(device)
    model(input)

    # Remove forward hooks
    for handle in handle_list:
        handle.remove()
    return total


def count_trainable_parameters(model):
    """
    Return the total number of trainable parameters for [model]
    :param model:
    :return:
    """
    params = 0
    for param in model.parameters():
        if param.requires_grad:
            params += param.numel()
    return params

def compute_forward_memory(model, input_shape, device):
    """
    Compute the memory required for the forward pass of the model
    :param model:
    :param input_shape:
    :param device:
    :return:
    """
    memory_bytes = 0
    input_tensor = torch.ones(input_shape).to(device)
    memory_bytes += np.prod(input_tensor.shape) * input_tensor.element_size()
    model = model.to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    memory_bytes += np.prod(output_tensor.shape) * output_tensor.element_size()
    return memory_bytes
