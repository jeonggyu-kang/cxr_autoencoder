from .autoencoder import CXRAutoencoder
import torch

def get_model(global_avg_pool, z_dim, input_size=(2,3,448,448),n_class=None, ckpt_path = None):
    model = CXRAutoencoder(global_avg_pool=global_avg_pool, z_dim=z_dim, input_shape = input_size )


    if ckpt_path is not None:
        print (f'Loading trained weight from {ckpt_path}..')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['weight'])

    model.cuda()

    return model