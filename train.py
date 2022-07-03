# train process main script
import os

import torch

from models import get_model                     # "models" is a directory 

from config import get_hyperparameters            
from dataset import get_dataloader

from logger import get_logger
from runner import trainer


def main():
    # TODO : apply easydict
    args = get_hyperparameters()                 #             

    model = get_model(
        global_avg_pool = args['global_avg_pool'],
        z_dim = args['z_dim'],
        input_size = (2,3, *args['image_size']),
        n_class = args['n_class']
    )    


    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    if args.get('mile_stone') is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones = args['mile_stone'], gamma = 0.1)            
    else:
        scheduler = None

    # loss_fn = torhc.nn.L1Loss()
    loss_fn = torch.nn.MSELoss()
    
    mode = 'train'

    small_loader = get_dataloader(
        dataset = args['dataset'],
        data_dir = args[mode]['img_dir'],
        ann_path = args[mode]['ann_file'],
        mode = mode,
        batch_size = args['batch_size'],
        num_workers = args['workers_per_gpu'],
        pipeline = args[mode]['pipeline'],
        csv = False,
        small_set = True
    )

    '''
    train_loader = get_dataloader(
        dataset = args['dataset'],
        data_dir = args[mode]['img_dir'],
        ann_path = args[mode]['ann_file'],
        mode = mode,
        batch_size = args['batch_size'],
        num_workers = args['workers_per_gpu'],
        pipeline = args[mode]['pipeline'],
        csv = False
    )

    
    mode = 'test'
    test_loader = get_dataloader(
        dataset = args['dataset'],
        data_dir = args[mode]['img_dir'],
        ann_path = args[mode]['ann_file'],
        mode = mode,
        batch_size = args['batch_size'],
        num_workers = args['workers_per_gpu'],
        pipeline = args[mode]['pipeline'],
        csv = False
    )
    '''
    writer = get_logger(args['save_root'] )



    trainer(                                      # from runner.py
        max_epoch = args['max_epoch'],
        model = model,
        # train_loader = train_loader,
        # test_loader = test_loader,
        train_loader = small_loader,
        test_loader = small_loader,
        loss_fn = loss_fn,
        optimizer = optimizer,
        scheduler = scheduler,
        meta = {
            'save_every' : 10,
            'print_every' : 5,
            'test_every' : 10
        },
        writer = writer
        
    )


if __name__ == '__main__':
    main()

