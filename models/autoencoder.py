import torch
import torch.nn as nn
import numpy as np 

try:
    from . import modules
except:
    import modules

def _get_eleme_num(model, x):
    y = model(x)
    _, C, H, W = y.shape
    return (C,H,W)

class CXRAutoencoder(nn.Module):
    def __init__(self, global_avg_pool, z_dim = 512, input_shape=(2, 3, 448, 448)):
        super(CXRAutoencoder, self).__init__()

        self.encoder = modules.resnet50(pretrained = True)
        bottleneck_shape = _get_eleme_num(self.encoder, torch.randn(input_shape))

        self.global_avg_pool = global_avg_pool
        if self.global_avg_pool:
            self.encoder_fc = nn.Linear(2048, z_dim)
            self.decoder = modules.ResDeconv(
                block=modules.BasicBlock,
                global_avg_pool = True,
                z_all = z_dim,
                bottleneck_shape = bottleneck_shape

            )

        else:

            self.decoder = modules.ResDeconv(modules.BasicBlock)


    def forward(self, x): # x: x-ray
        latent_code = self.encoder(x)

        if self.global_avg_pool:
            latent_code = latent_code.mean(-1).mean(-1) # (2048 X 14 x 14) -> (2048)
            latent_code = self.encoder_fc(latent_code)

        x_hat = self.decoder(latent_code)

        return x_hat, latent_code

if __name__ == '__main__':
    model1 = CXRAutoencoder(global_avg_pool = False, input_shape=(2,3,448*2, 448*2)).cuda()
    model2 = CXRAutoencoder(global_avg_pool = True, z_dim = 1024, input_shape=(2,3,448*2, 448*2)).cuda()

    image = torch.rand(2, 3, 448*2, 448*2).cuda()

    x_hat1, latent_code1 = model1(image) # w/o bottleneck-linear
    x_hat2, latent_code2 = model2(image) # w/ bottleneck-linear

    print(x_hat1.shape, latent_code1.shape)
    print(x_hat2.shape, latent_code2.shape)
