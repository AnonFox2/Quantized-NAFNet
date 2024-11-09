import os
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn as nn

class Model():
    def __init__(self, training):
        self.device = torch.device('cuda')
        self.training = training
        self.schedulers = []
        self.optimizers = []
        self.save_folder = "./experiments/training/models"
    def process_model(self, network):
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def load(self, network, path, strict=True, key='params'):
        network = self.process_model(network)
        data = torch.load(path, map_location=lambda storage, loc: storage)
        
        if key is not None:
            data = data[key]
        print(' model keys', data.keys)
        
        for name, val in deepcopy(data).items():
            if name.startswith('module.'):
                data[name[7:]] = val
                data.pop(name)
        network.load_state_dict(data, strict=strict)

    def save_checkpoint(self, epoch, step):
        if step != -1:
            data = {
                'epoch': epoch,
                'step': step,
                'optimizers': [],
                'schedulers': []
            }
            data['optimizers'].append(self.optimizer_g.state_dict())
            data['schedulers'].append(self.schedulers.state_dict())
            
            filename = f'{step}.state'
            save_folder = "./experiments/training/training_states"
            path = os.path.join(save_folder, filename)
            os.makedirs(save_folder, exist_ok=True)
            torch.save(data, path)

    def calculate_losses(self, losses):
        with torch.no_grad():
            results = OrderedDict()
            for name, value in losses.items():
                results[name] = value.mean().item()
            return results
    def save_model(self, model, model_name, iteration, param_key='params'):
        if iteration == -1:
            iteration = 'latest'
        
        filename = f'{model_name}_{iteration}.pth'
        save_path = os.path.join(self.save_folder, filename)
        os.makedirs(self.save_folder, exist_ok=True)
        model_list = model if isinstance(model, list) else [model]
        key_list = param_key if isinstance(param_key, list) else [param_key]
        assert len(model_list) == len(key_list), 'model and param_key lengths must match.'

        model_data = {}
        for single_model, key in zip(model_list, key_list):
            # bare_model = self.get_core_model(single_model)
            model_state = single_model.state_dict()
            cleaned_state = {k[7:] if k.startswith('module.') else k: v.cpu() 
                             for k, v in model_state.items()}
            model_data[key] = cleaned_state

        torch.save(model_data, save_path)
