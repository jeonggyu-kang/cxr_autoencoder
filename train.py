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

    for model_name in args['name']:              # loop in various models

        input_size = args['image_size']
        model = get_model(model_name, args['n_class'], input_size = (2,3, *input_size))

        if args['loss'] in ['ce', 'cross-entropy']:
            loss = torch.nn.CrossEntropyLoss()
            task_type = 'classification'

        elif args['loss'] in ['mse', 'mean_squared_error']:
            loss = torch.nn.MSELoss()
            task_type = 'regression'
        else:
            raise ValueError

        optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

        if args.get('mile_stone') is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones = args['mile_stone'], gamma = 0.1)            
        else:
            scheduler = None

        mode = 'train'

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

        #writer = get_logger(args['save_root'] )

        save_path = os.path.join(args['save_root'], model_name)
        writer = get_logger(save_path)


        trainer(                                      # from runner.py
            max_epoch = args['max_epoch'],
            model = model,
            train_loader = train_loader,
            test_loader = test_loader,
            loss_fn = loss,
            optimizer = optimizer,
            scheduler = scheduler,
            meta = {
                'save_every' : 10,
                'print_every' : 5,
                'test_every' : 10
            },
            writer = writer,
            task_type = task_type
        )


if __name__ == '__main__':
    main()

