"""
Main entry for AmpHGT, Train, Fine-Tune, Inference in one go.
"""

import os
import mlflow
import torch.optim as optim

from model.PepMAE import PepMAE
from utils.read_params import lcfig, Cfig
from utils.data import make_loaders
from utils.MaskMethod import MaskGraph
from utils.std_logger import initialize_logger, info

from trainers.pretrain_trainer import Masking_Trainer

from model.PharmHGT_refactor import PharmHGT
# from model.pretrain_heads import PretrainingHeads, PretrainingMetrics, ReconstructionLoss
from seed_device import setup_seed_reproducibility

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
重建了一个更加强大的预测体系，PepMAE，包含了对于特征的重建。详见代码。
#TODO: 下列在PepMAE中被丢弃了，应该重新实施。目前是重建了特征预测。
对原子的序数，带点，质量，芳香性等综合进行预测。
对于fragments，特征重建（预测片段属性和特征），我们也需要预测fragments的分类。#TODO:加上这个任务。

目前的训练方法任然不完善。试着运行代码发现了如下问题：
1. VRAM会逐渐累计直到溢出。
2. Validation epoch还没有进行过测试。
3. 预测方法可能还需要调整。
4. 使用MLflow保存模型，目前抛弃了这个方式。

记录于 2024/10/24
"""

@lcfig(config_path = 'configs/pretrain.yaml', output_dir = "out_pretrain")
def pretrain_main(cfg: Cfig):
    """
    Main function to pretrain the model.
    """
    ensure_dir_exists(cfg.logger.log_dir)

    initialize_logger(cfg.logger.log_dir)

    # Initialize the random seed and device configuration for reproducibility
    devices = setup_seed_reproducibility(cfg.train.seed, cfg.train.cuda_deterministic)

    # Set up MLflow tracking if enabled
    if cfg.logger.log:
        info("Setting up MLflow tracking...")
        mlflow.set_tracking_uri(cfg.logger.tracking_uri)
        mlflow.set_experiment(cfg.logger.mlflow_exp_name)

    devices = devices[0]

    info("Creating DataLoaders...")

    pretrain_loader, vocab_dict = make_loaders(
        cfg, 
        batch_size=cfg.train.batch_size,
        mask_graph=MaskGraph(
            num_atom_type=119,
            num_edge_type=5,
            mask_edge=cfg.train.mask_edge,
            mask_atom=cfg.train.mask_atom,
            mask_fragment=cfg.train.mask_fragment,
            mask_edge_r=cfg.train.mask_edge_r,
            mask_atom_r=cfg.train.mask_atom_r,
            mask_fragment_r=cfg.train.mask_fragment_r
            )
        )

    info("Setting up model...")
    encoder = PharmHGT(
        cfg.train.hid_dim, 
        cfg.train.act, 
        cfg.train.num_layer,
        cfg.train.atom_dim, 
        cfg.train.bond_dim,
        cfg.train.pharm_dim, 
        cfg.train.reac_dim).to(devices)

    ### New head added 24.10.24
    info("Setting up PepMAE decoder...")
    # Create decoder (PepMAE)
    decoder = PepMAE(
        hidden_dim=cfg.train.hid_dim,
        atom_feat_dim=cfg.train.atom_dim,
        fragment_feat_dim=cfg.train.pharm_dim,
        num_atom_classes=119,
        num_fragment_classes=len(vocab_dict)
    ).to(devices)

    info("Setting up optimizers...")
    # Combined optimizer for model and pretraining heads
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.decay
    )
    
    '''
    info("Setting up linear prediction layers...")
    linear_pred_atoms = torch.nn.Linear(cfg.train.hid_dim, 119).to(devices)
    # check if hid_dim at least >= vocab size
    if cfg.train.hid_dim < len(vocab_dict):
        info("Warning: hid_dim is less than vocab size. This may lead to underfitting.")
    if cfg.train.frag_method != "AdaFrag":
        linear_pred_pharms = torch.nn.Linear(cfg.train.hid_dim, len(vocab_dict)).to(devices)
    else:
        linear_pred_pharms = torch.nn.Linear(cfg.train.hid_dim, 264).to(devices)
    linear_pred_bonds = torch.nn.Linear(cfg.train.hid_dim, 4).to(devices)

    model_list = [
        model, linear_pred_atoms, linear_pred_pharms, linear_pred_bonds
    ]

    info("Setting up optimizers...")

    optimizer_model = optim.Adam(model.parameters(),
                                 lr=cfg.train.lr,
                                 weight_decay=cfg.train.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(),
                                             lr=cfg.train.lr,
                                             weight_decay=cfg.train.decay)
    optimizer_linear_pred_pharms = optim.Adam(linear_pred_pharms.parameters(),
                                              lr=cfg.train.lr,
                                              weight_decay=cfg.train.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(),
                                             lr=cfg.train.lr,
                                             weight_decay=cfg.train.decay)
    optimizer_list = [
        optimizer_model, 
        optimizer_linear_pred_atoms,
        optimizer_linear_pred_pharms, 
        optimizer_linear_pred_bonds
    ]

    criterion = nn.CrossEntropyLoss()
    '''

    with mlflow.start_run():
        info("Logging hyper-parameters...")
        # log hyper-parameters
        for p, v in cfg.train.items():
            mlflow.log_param(p, v)

        info("Initializing trainer...")
        trainer = Masking_Trainer(
            cfg=cfg,
            dataloaders=pretrain_loader,
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            device=devices,
            outputdir=cfg.logger.log_dir
        )
        
        info("Starting training...")
        trainer.run()

    info("finished training......")

def ensure_dir_exists(path):
    if not os.path.exists(path):
        info(f"Creating: {path}")
        os.makedirs(path)

if __name__ == '__main__':
    pretrain_main()