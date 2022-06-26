from .mymodel import CustomizationNet 
import torch

def get_model(model_name, n_classes, input_size, ckpt_path = None):
    model = CustomizationNet( n_classes, model_name, input_size )


    if ckpt_path is not None:
        print (f'Loading trained weight from {ckpt_path}..')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['weight'])

    model.cuda()

    return model