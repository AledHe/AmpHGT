"""For fast mol2graph generation, we use the parallel version of mol2graph function in mol2graph_parallel.py."""

import argparse
from utils.read_params import lcfig, Cfig
from utils.MaskMethod import MaskGraph
from utils.mol2graph_parallel import mol2graph
from utils.seed_device import setup_seed_reproducibility

@lcfig(config_path = 'configs/ncaas_generate_graph.yaml', output_dir = "out_general_generate_graph")
def main(cfg: Cfig):
    parser = argparse.ArgumentParser(description="Standalone script to pre-process molecules into DGL graph chunks.")
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
    paths = get_paths(cfg)
    
    print(paths)

    for path in paths:
        if "train" in path:
            stage = "train"
        elif "valid" in path or "val" in path:
            stage = "valid"
        elif "test" in path:
            stage = "test"
        else:
            raise ValueError(f"Invalid Stage: {stage}")
        mol2graph(cfg, args, mask_graph, path, stage)

def get_paths(cfg):
    paths = [
        cfg.data.train_sdf_pos, 
        cfg.data.train_sdf_neg, 
        cfg.data.valid_sdf_pos, 
        cfg.data.valid_sdf_neg, 
        cfg.data.test_sdf_pos, 
        cfg.data.test_sdf_neg
    ]
    valid_paths = [path for path in paths if path]
    return valid_paths
    
if __name__ == "__main__":
    main()