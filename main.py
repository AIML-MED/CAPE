import numpy as np
import os, sys, copy, json, importlib, time, datetime, shutil
import torch, torchvision
import matplotlib.pyplot as plt
import helpers
from configs.arguments import parse_args
from configs import transforms
from path import Path
from torch.utils.data import DataLoader
import torch.nn as nn
import datasets
from models.model import Net
from trainer import Trainer

torch.cuda.empty_cache()

def main(args):
    if args.seed is not None:
        helpers.set_random_seed(args.seed)
        
    experiment_config = helpers.import_module_from_file(args.config_file) # importlib.import_module(f'configs.{args.config_file}')
    
    opts =  vars(args)
    opts.update({v: getattr(experiment_config, v)
                  for v in experiment_config.__dict__
                  if not v.startswith("_")})
    
    opts['cls_cols_dict'] = {'orig': opts['num_class'], 'cape': opts['num_class']}
    
    # dataset
    dataset_transform_func = getattr(transforms, f"{opts['dataset_transform']}")
    train_transform, val_transform = dataset_transform_func(opts['image_size'])
    
    dataset_class = getattr(datasets, opts['dataset_class'])
    train_dataset = dataset_class(train=True, transform=train_transform, root=opts['data_path'])
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=opts['batch_size'], shuffle=True,
                                    num_workers=opts['num_workers'], pin_memory=True,
                                    worker_init_fn=helpers.worker_init_fn)
    val_dataset = dataset_class(train=False, transform=val_transform, root=opts['data_path'])
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=opts['batch_size'], shuffle=False,
                                     num_workers=opts['num_workers'], pin_memory=True)
    
    
    # need to save opts here
    if opts['eval']:
        opts['experiment_dir'] = None
    else:
        experiment_dir = f'./experiments/{args.dataset}'
        Path(experiment_dir).mkdir_p()
        experiment_dir += f"/{helpers.random_experiment_name()}_{opts['settings']}"
        opts['experiment_dir'] = Path(experiment_dir).mkdir_p()
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Experiment directory:", opts['experiment_dir'])

        with open(os.path.join(opts['experiment_dir'], 'opts.json'), "w") as outfile:
            outfile.write(json.dumps(opts, indent=4))
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Configuration is saved!!!')
    
    print("Configurations:", opts)
    
    opts['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
    model = Net(opts)

    if opts['settings']=='PF' or opts['eval']:
        # reload trained model weights from a checkpoint
        if opts['reload_from_checkpoint']:
            print('loading from checkpoint: {}'.format(opts['reload_path']))
            if os.path.exists(opts['reload_path']):
                model.load_state_dict(torch.load(opts['reload_path']))        
            else:
                raise ValueError('File not exists in the reload path: {}'.format(opts['reload_path']))
        
        if not opts['eval']:
            model.classifiers['cape'].load_state_dict(model.classifiers['orig'].state_dict()) 
            for p in model.net.parameters():
                p.requires_grad = False
            for p in model.classifiers['orig'].parameters():
                p.requires_grad = False
            
    if torch.cuda.is_available():
        if torch.cuda.device_count()>1:
            print(f'Multiple GPUS: {torch.cuda.device_count()} .......')
            model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    
    model = model.to(opts['device'])
            
    params = list(p for p in model.parameters() if p.requires_grad)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    
    # optimizer
    if opts['eval']:
        scheduler, optimizer = None, None
    else:
        if opts['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(params, lr=opts['learning_rate'],
                                        weight_decay=opts['weight_decay'], momentum=0.9)
        elif opts['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(params, lr=opts['learning_rate'], betas=(0.9, 0.999),
                                        eps=1e-8, weight_decay=opts['weight_decay'])
        else:
            raise ValueError('Unexpected optimizer: {}'.format(opts['optimizer']))
        
        # scheduler
        if opts['settings'] == 'TS':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif opts['settings'] == 'PF':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            raise ValueError('Invalid settings!!!!')
        
    
    # trainer
    trainer = Trainer(opts, model, optimizer, scheduler, train_data_loader, val_data_loader)
    if opts['eval']:
        trainer.eval()  
    else:
        trainer.run() 
        
        
        
if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    main(args)
