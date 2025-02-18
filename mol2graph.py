"""For fast mol2graph generation, we use the parallel version of mol2graph function in mol2graph_parallel.py."""

import argparse
from utils.read_params import lcfig, Cfig
from utils.MaskMethod import MaskGraph
from utils.mol2graph_parallel import mol2graph
from utils.seed_device import setup_seed_reproducibility

@lcfig(config_path = 'configs/pretrain.yaml', output_dir = "out_pretrain")
def main(cfg: Cfig):
    parser = argparse.ArgumentParser(description="Standalone script to pre-process molecules into DGL graph chunks.")
    parser.add_argument("stage", type=str, help="Stage name (e.g., 'train', 'valid', 'test') for naming chunks.")
    parser.add_argument("--num_workers", "-nw", type=int, default=None, help="Number of worker processes to use. Default: Number of CPU cores.")
    args = parser.parse_args()
    # set randomness
    setup_seed_reproducibility(cfg.train.seed, cfg.train.cuda_deterministic)
    # get mask method for pretraining.
    mask_graph=MaskGraph(
        num_atom_type=119,
        num_edge_type=5,
        mask_edge=cfg.train.mask_edge,
        mask_atom=cfg.train.mask_atom,
        mask_fragment=cfg.train.mask_fragment,
        mask_rate=cfg.train.mask_rate,
        )
    
    # get sdf_path according to stage.
    if args.stage == "train":
        sdf_path = cfg.data.train_sdf
    elif args.stage == "valid":
        sdf_path = cfg.data.valid_sdf
    elif args.stage == "test":
        sdf_path = cfg.data.test_sdf
    else:
        raise ValueError(f"Invalid stage: {args.stage}")

    mol2graph(cfg, args, mask_graph, sdf_path)
    
if __name__ == "__main__":
    main()