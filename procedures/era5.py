import logging
import wandb
from time import time

from utils.loss import LpLoss
from datasets import ERA5Dataset

from trainers import TRAINER_DICT


def era5_procedure(args):
    if args['model_name'] not in TRAINER_DICT.keys():
        raise NotImplementedError("Model {} not implemented".format(args['model_name']))
    
    if args['verbose']:
        logger = logging.info if args['log'] else print

    if args['wandb']:
        wandb.init(
            project=args['wandb_project'], 
            name=args['saving_name'],
            tags=[args['model'], args['dataset']],
            config=args)
    
    if args['verbose']:
        logger("Loading {} dataset".format(args['dataset']))
    start = time()
    dataset = ERA5Dataset(
        data_path=args['data_path'],
        raw_resolution=args['raw_resolution'],
        sample_resolution=args['sample_resolution'],
        eval_resolution=args['eval_resolution'],
        in_t=args['in_t'],
        out_t=args['out_t'],
        duration_t=args['duration_t'],
        train_day=args['train_day'],
        valid_day=args['valid_day'],
        test_day=args['test_day'],
        train_batchsize=args['train_batchsize'],
        eval_batchsize=args['eval_batchsize'],
        normalize=args['normalize'],
        normalizer_type=args['normalizer_type'],
        prop=args['prop'],
        sub=args['sub'],
    )
    train_loader = dataset.train_loader
    valid_loader = dataset.valid_loader
    test_loader = dataset.test_loader
    if args['verbose']:
        logger("Loading data costs {: .2f}s".format(time() - start))
        
    if args['verbose']:
        logger("Building models")
    start = time()
    trainer = TRAINER_DICT[args['model_name']](args)
    model = trainer.build_model(args)
    model = model.to(args['device'])
    optimizer = trainer.build_optimizer(model, args)
    scheduler = trainer.build_scheduler(optimizer, args)
    criterion = LpLoss(d=2, p=2, size_average=False)
    if args['verbose']:
        logger("Model: {}".format(model))
        logger("Criterion: {}".format(criterion))
        logger("Optimizer: {}".format(optimizer))
        logger("Scheduler: {}".format(scheduler))
        logger("Building models costs {: .2f}s".format(time() - start))

    trainer.process(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        prop=args['prop'],
    )
