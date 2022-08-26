# Hyperparameter control
import os, shutil

def get_hyperparameters(config = None):
    if config is None: # Train
        _save()
    else: # test
        _load(config)

    ret = {}
    ret.update( model_dict )         # updata: function for updating dictionary
    ret.update( data_dict )

    return ret

_IMAGE_WIDTH = 896
_IMAGE_HEIGHT = 896

# model-related params
model_dict = dict(                  
    n_class = 4,
    max_epoch = 200,
    learning_rate = 1e-4,
    # mile_stone = None,
    mile_stone = [150, 180],
    decay_rate = 0.1,
    loss = 'ce',   # cross-entropy (classification)
    #loss = 'mse',    # mean squared error (regresion)
    image_size = (_IMAGE_WIDTH, _IMAGE_HEIGHT),   # width, height
    global_avg_pool = True,
    z_dim = 512,
    # z_cac = None, # all-latent code
    z_cac = 64,    # partial latent code
    train_target = 'joint', # classifier, fine-tune
    extra = ['autoencoder-test']    
)

train_pipeline = [
    dict(
        type = 'Resize',
        width = _IMAGE_WIDTH,
        height = _IMAGE_HEIGHT
    ),


    dict(
        type = 'Contrastive',
<<<<<<< HEAD
        p = 0.0,
        w = 1.5
=======
        p = 0.5,
        w = 1.3
>>>>>>> 7e70fcd8193f0a3a6226ce3c49387db63de0e1cb
    ),

    dict(
        type = 'Sharpness',
<<<<<<< HEAD
        p = 0.0
=======
        p = 0.5
>>>>>>> 7e70fcd8193f0a3a6226ce3c49387db63de0e1cb
    ),

    dict(
        type= 'ToTensor'
    ),
]

test_pipeline = [
    dict(
        type = 'Resize',
        width = _IMAGE_WIDTH,
        height = _IMAGE_HEIGHT
    ),
    dict(
        type= 'ToTensor'
    ),
]


# dataset-related params
data_dict = dict(
    dataset = 'CoronaryArteryDataset',
    #dataset = 'AGEDataset',
    save_root = './work_dir',
    batch_size = 2,
    workers_per_gpu = 1,

    train = dict(
        img_dir = '/mnt/project_classifiers/data',
        ann_file = '/mnt/project_classifiers/data/train_dataset_cac.parquet',
        pipeline = train_pipeline
    ),
    test = dict(
        img_dir = '/mnt/project_classifiers/data',
        ann_file = '/mnt/project_classifiers/data/test_dataset_cac.parquet',
        pipeline = test_pipeline
    ),
)

def _save():
    model_version = []
    for k in ['name', 'imagenet_pretrained', 'extra']:
        if k in model_dict:
            if isinstance(model_dict[k], list):
                model_version += model_dict[k]
            else:
                model_version.append(str(model_dict[k]))
    
    os.makedirs(data_dict['save_root'], exist_ok = True)
    VERSION = '.'.join(model_version)
    VERSION = str('{:04d}'.format(len(os.listdir(data_dict['save_root'])) + 1) + '_') + VERSION

    SAVE_ROOT_DIR = os.path.join(data_dict['save_root'], VERSION)
    os.makedirs(SAVE_ROOT_DIR, exist_ok=True)
    shutil.copy2(os.path.abspath(__file__), os.path.join(SAVE_ROOT_DIR, __name__ + '.py'))
    data_dict['save_root'] = SAVE_ROOT_DIR

def _load(config):
    data_dict['save_root'] = os.path.join(os.path.dirname(config), 'eval')
    data_dict['max_epoch'] = None