"""
Main entry for AmpHGT, Train, Fine-Tune, Inference in one go.

===============================================================================
Copyright (C) 2025 by Yongcheng He
Center for Sustainable Antimicrobials, Department of Pharmacy, College of Veterinary Medicine
Sichuan Agricultural University, Chengdu

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
===============================================================================
"""

import os
import mlflow
import torch.optim as optim
import shutil
import argparse

from model.PepMAE import PepMAE
from utils.read_params import lcfig, Cfig
from utils.data_rework import make_pretrain_loaders, make_binary_loaders, make_regression_loaders
from utils.MaskMethod import MaskGraph
from utils.std_logger import initialize_logger, info
from utils.seed_device import setup_seed_reproducibility

from trainers.pretrain_trainer import Masking_Trainer
from trainers.finetune_trainer import Finetune_Trainer
from trainers.inferencer import Inferencer_Binary

from model.PharmHGT_refactor import PharmHGT, PharmHGT_FP, AmpHGT_FT
from model.scheduler import NoamLR
# from model.pretrain_heads import PretrainingHeads, PretrainingMetrics, ReconstructionLoss

"""
Message from 2024/10/24
在这里写一份备忘录。

目前正在重构AmpHGT的代码，训练，微调，推理一体化。
现在的进度是：抛弃了原PepLand设计的预测原子序数与fragment idx的一层linear。
重写了trainer，以及添加了PepMAE（Masked Autoencoders）
参考GraphMAE，下游任务只需要移除decoder。使用GRU readout即可。

PEPland：
Atom:
分类原子序数
Fragments:
直接片段类型分类（从词汇表中预测哪个片段）

PepMAE：
重建了一个更加强大的预测体系，PepMAE，包含了对于特征的重建，对原子的序数的预测，预测fragments的分类。详见代码。
重建特征的代码与方法编写的正确吗？
鉴于目前的PharmHGT结构与返回特征表示，我们确实需要PepMAE重建junction_features吗？
我认为我们应该仔细思考PharmHGT的返回格式，并且基于新的返回格式思考重建特征的细节。（是否不应该重建junction_features）
需要参考新的特征表示方法。

PharmHGT：
目前的深度够了吗？结构足够好吗？参数量足够吗？

目前的训练方法任然不完善。试着运行代码发现了如下问题：
1. VRAM会逐渐累计直到溢出。（似乎解决了）
2. Validation epoch还没有进行过测试。
3. 预测方法可能还需要调整。
4. 使用MLflow保存模型，目前抛弃了这个方式。

#BUG:
Random seed的运行方式出问题了，在不同的py文件中，import random似乎会设置独立的random seed。
必须找到一种方式来保存切割方法FraSCESS。Mask的frag与atom。
在cfg中提供random。
# FIXED

#BUG:
Traceback (most recent call last):
File "G:/AmpHGT/refactor/amphgt_refactor/main.py", line 189, in <module>
    pretrain_main()
File "G:/AmpHGT/refactor/amphgt_refactor/utils/read_params.py", line 100, in wrapper
    return func(params, *args, **kwargs)
File "G:/AmpHGT/refactor/amphgt_refactor/main.py", line 179, in pretrain_main
    trainer.run()
File "G:/AmpHGT/refactor/amphgt_refactor/trainers/pretrain_trainer.py", line 168, in run
    self._log_epoch_metrics(train_metrics, val_metrics, epoch)
File "G:/AmpHGT/refactor/amphgt_refactor/trainers/pretrain_trainer.py", line 191, in _log_epoch_metrics
    for name, value in val_metrics.items():
AttributeError: 'NoneType' object has no attribute 'items'
# FIXED

记录于 2024/10/27
"""

@lcfig(config_path = 'configs/pretrain.yaml', output_dir = "out_pretrain")
def pretrain_main(cfg: Cfig):
    """
    Main function to pretrain the model.
    """
    ensure_dir_exists(cfg.logger.log_dir)

    initialize_logger(cfg.logger.log_dir)

    # copy the config file to the log directory
    shutil.copy('configs/pretrain.yaml', os.path.join(cfg.logger.log_dir, "pretrain.yaml"))

    # Initialize the random seed and device configuration for reproducibility
    devices = setup_seed_reproducibility(cfg.train.seed, cfg.train.cuda_deterministic)

    # Set up MLflow tracking if enabled
    if cfg.logger.log:
        info("Setting up MLflow tracking...")
        mlflow.set_tracking_uri(cfg.logger.tracking_uri)
        mlflow.set_experiment(cfg.logger.mlflow_exp_name)

    devices = devices[0]

    info("Creating DataLoaders...")

    pretrain_loader, vocab_dict = make_pretrain_loaders(
        cfg, 
        batch_size=cfg.train.batch_size,
        )

    info("Setting up model...")
    encoder = PharmHGT(
        cfg.train.hid_dim, 
        cfg.train.act, 
        cfg.train.num_layer,
        cfg.train.atom_dim, 
        cfg.train.bond_dim,
        cfg.train.pharm_dim, 
        cfg.train.reac_dim,
        devices
    ).to(devices)

    ### New head added 24.10.24
    info("Setting up PepMAE decoder...")
    # Create decoder (PepMAE)
    decoder = PepMAE(
        hidden_dim=cfg.train.hid_dim,
        atom_feat_dim=cfg.train.atom_dim,
        fragment_feat_dim=cfg.train.pharm_dim,
        num_atom_classes=119,
        num_fragment_classes=len(vocab_dict),
        decoder_type=cfg.train.decoder_type
    ).to(devices)

    info("Setting up optimizers...")
    # Combined optimizer for encoder and decoder
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.decay
    )

    with mlflow.start_run():
        info("Logging hyper-parameters...")
        # log hyper-parameters
        for p, v in cfg.train.items():
            mlflow.log_param(p, v)

        info("Initializing trainer...")
        trainer = Masking_Trainer(
            cfg=cfg,
            dataloaders=pretrain_loader, # this is a dict with "train"/"valid"/"test"
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            device=devices,
            outputdir=cfg.logger.log_dir
        )
        
        info("Starting training...")
        trainer.run()

    info("finished training......")

@lcfig(config_path = 'configs/finetune_binary.yaml', output_dir = "out_finetune_binary")
def finetune_binary_main(cfg: Cfig):
    """
    Main function to fine-tune the model.
    """
    ensure_dir_exists(cfg.logger.log_dir)

    initialize_logger(cfg.logger.log_dir)

    # copy the config file to the log directory
    shutil.copy('configs/finetune_binary.yaml', os.path.join(cfg.logger.log_dir, "finetune_binary.yaml"))

    # Initialize the random seed and device configuration for reproducibility
    devices = setup_seed_reproducibility(cfg.train.seed, cfg.train.cuda_deterministic)

    # Set up MLflow tracking if enabled
    if cfg.logger.log:
        info("Setting up MLflow tracking...")
        mlflow.set_tracking_uri(cfg.logger.tracking_uri)
        mlflow.set_experiment(cfg.logger.mlflow_exp_name)

    devices = devices[0]

    info("Creating DataLoaders...")

    finetune_loader, vocab_dict = make_binary_loaders(
        cfg, 
        batch_size=cfg.train.batch_size,
        mask_graph=MaskGraph(
            num_atom_type=119,
            num_edge_type=5,
            mask_edge=cfg.train.mask_edge,
            mask_atom=cfg.train.mask_atom,
            mask_fragment=cfg.train.mask_fragment,
            mask_rsd=cfg.train.mask_rsd,
            mask_rate=cfg.train.mask_rate,
            )
        )
    
    info("Setting up model...")
    model = PharmHGT_FP(
        cfg,
        cfg.train.hid_dim, 
        cfg.train.act, 
        cfg.train.num_layer,
        cfg.train.atom_dim, 
        cfg.train.bond_dim,
        cfg.train.pharm_dim, 
        cfg.train.reac_dim,
        devices,
        cfg.train.load_pretrained
    ).to(devices)
    
    info("Setting up optimizers...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.decay
    )
    scheduler = NoamLR(
        optimizer=optimizer,
        warmup_epochs=[2],
        total_epochs=[cfg.train.epochs],
        steps_per_epoch=len(finetune_loader["train"]),
        init_lr=[cfg.train.lr],
        max_lr=[1e-03],
        final_lr=[1e-05]
    )

    with mlflow.start_run():
        info("Logging hyper-parameters...")
        # log hyper-parameters
        for p, v in cfg.train.items():
            mlflow.log_param(p, v)

        info("Initializing trainer...")
        trainer = Finetune_Trainer(
            cfg=cfg,
            dataloaders=finetune_loader, # this is a dict with "train"/"valid"/"test"
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=devices,
            outputdir=cfg.logger.log_dir
        )
        
        info("Starting training...")
        trainer.run()

    info("finished training......")

@lcfig(config_path = 'configs/finetune_regression.yaml', output_dir = "out_finetune_regression")
def finetune_regression_main(cfg: Cfig):
    """
    Main function to fine-tune the model with regression targets. testing amphgt with hemolytic/inhibitory concentration dataset.
    """
    ensure_dir_exists(cfg.logger.log_dir)

    initialize_logger(cfg.logger.log_dir)

    # copy the config file to the log directory
    shutil.copy('configs/finetune_regression.yaml', os.path.join(cfg.logger.log_dir, "finetune_regression.yaml"))

    # Initialize the random seed and device configuration for reproducibility
    devices = setup_seed_reproducibility(cfg.train.seed, cfg.train.cuda_deterministic)

    # Set up MLflow tracking if enabled
    if cfg.logger.log:
        info("Setting up MLflow tracking...")
        mlflow.set_tracking_uri(cfg.logger.tracking_uri)
        mlflow.set_experiment(cfg.logger.mlflow_exp_name)

    devices = devices[0]

    info("Creating DataLoaders...")
    finetune_loader, vocab_dict = make_regression_loaders(
        cfg,
        batch_size=cfg.train.batch_size,
    )

    info("Setting up model...")
    model = PharmHGT_FP(
        cfg,
        cfg.train.hid_dim, 
        cfg.train.act, 
        cfg.train.num_layer,
        cfg.train.atom_dim, 
        cfg.train.bond_dim,
        cfg.train.pharm_dim, 
        cfg.train.reac_dim,
        devices,
        cfg.train.load_pretrained
    ).to(devices)

    info("Setting up optimizers...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.decay
    )
    scheduler = NoamLR(
        optimizer=optimizer,
        warmup_epochs=[2],
        total_epochs=[cfg.train.epochs],
        steps_per_epoch=len(finetune_loader["train"]),
        init_lr=[cfg.train.lr],
        max_lr=[1e-03],
        final_lr=[1e-05]
    )

    with mlflow.start_run():
        info("Logging hyper-parameters...")
        # log hyper-parameters
        for p, v in cfg.train.items():
            mlflow.log_param(p, v)

        info("Initializing trainer...")
        trainer = Finetune_Trainer(
            cfg=cfg,
            dataloaders=finetune_loader, # this is a dict with "train"/"valid"/"test"
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=devices,
            outputdir=cfg.logger.log_dir
        )
        
        info("Starting training...")
        trainer.run()

    info("finished training......")

@lcfig(config_path = 'configs/inference_binary.yaml', output_dir = "out_inference_binary")
def inference_binary_main(cfg: Cfig):
    """
    Main function to inference with binary targets.
    """
    ensure_dir_exists(cfg.logger.log_dir)

    initialize_logger(cfg.logger.log_dir)

    # copy the config file to the log directory
    shutil.copy('configs/inference_binary.yaml', os.path.join(cfg.logger.log_dir, "inference_binary.yaml"))

    # Initialize the random seed and device configuration for reproducibility
    devices = setup_seed_reproducibility(cfg.train.seed, cfg.train.cuda_deterministic)

    devices = devices[0]

    info("Creating DataLoaders...")

    inference_loader, vocab_dict = make_binary_loaders(
        cfg, 
        batch_size=cfg.train.batch_size,
        mask_graph=None,
        inference=True
    ) # inference loader only load test path data.

    info("Setting up model...")
    model = AmpHGT_FT.from_checkpoint(
        checkpoint_path=cfg.train.checkpoint_path,
        cfg=cfg,
        device=devices
    ).to(devices)

    inferencer = Inferencer_Binary(
        cfg=cfg,
        dataloaders=inference_loader, # this is a dict with "test" only.
        model=model,
        device=devices
    )

    info("Starting inference...")

    inferencer.infer()

    info("finished inference......")

@lcfig(config_path = 'configs/extract_embeddings_encoder.yaml', output_dir = "out_extract_encoder_embeddings")
def extract_embeddings_encoder_main(cfg: Cfig):
    pass

@lcfig(config_path = 'configs/extract_embeddings_amphgt.yaml', output_dir = "out_extract_amphgt_embeddings")
def extract_embeddings_amphgt_main(cfg: Cfig):
    pass
    
def ensure_dir_exists(path):
    if not os.path.exists(path):
        info(f"Creating: {path}")
        os.makedirs(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AmpHGT')
    parser.add_argument('mode', type=str, default='pt', help='Mode to run the program in: pt, ftb, ftr, ifb, fe, fa')
    args = parser.parse_args()

    if args.mode == 'pt':
        pretrain_main()
    elif args.mode == 'ftb':
        finetune_binary_main()
    elif args.mode == 'ftr':
        finetune_regression_main()
    elif args.mode == 'ifb':
        inference_binary_main()
    elif args.mode == 'fe':
        extract_embeddings_encoder_main()
    elif args.mode == 'fa':
        extract_embeddings_amphgt_main()
    else:
        raise ValueError(f"Invalid mode: {args.mode}")