# AmpHGT

**AmpHGT: Expanding Prediction of Antimicrobial Activity in Peptides Containing Non-Canonical Amino Acids Using Multi-View Constrained Heterogeneous Graph Transformer**

This repository contains the code for the paper and provides all necessary interfaces for model training and inference.

---

- **Main Script:**
  `main.py` provides the interface for model training and inference.

- **Training Data:**
  The training data used in the paper can be downloaded from [AmpHGT_db](#).

- **Configuration Options:**
  You can either modify the YAML configuration files in the `configs/` directory or override parameters via command-line arguments.

---

## Env Setup

It is recommend to create a new conda environment with our environment.yml by

```bash
conda env create -f environment.yml
```

## Training the Model

There are two primary approaches for training:

1. **Using YAML Configuration Files:**
   Modify the parameters in the YAML files located in the `configs/` directory.

2. **Using Command-Line Arguments:**
   Directly provide detailed parameters when running the command.

### Quick Start for Training

To begin training with default parameters (defined in `configs/finetune_binary.yaml`), simply run:

```bash
python main.py ftb
```

> **Note:**
> Despite the name `finetune_binary`, this mode does not use any pretrained PharmHGT as `load_pretrained` is set to `False` by default.

### Recommended Command for Reproducing Results

For a more controlled setup and to reproduce our reported results, try the following command:

```bash
python main.py ftb -c configs/finetune_binary.yaml -o out_finetune_binary/gru_npt \
  -train*readout gru -train*fusion attention -train*sq_embed ESM2 \
  -train*decay 1e-2 -train*patience 10 -train*seed 512
```

- `-c`: Specifies which configuration file to use.
- `-o`: Specifies the output directory for the run.
- All other parameters (like `-train*readout`, `-train*fusion`, etc.) override the corresponding settings in the YAML file.

### Configuration Shortcuts

- **`ft`** uses `configs/pretrain.yaml` and outputs to `out_pretrain`.
- **`ftb`** uses `configs/finetune_binary.yaml` and outputs to `out_finetune_binary`.
- **`ifb`** uses `configs/inference_binary.yaml` and outputs to `out_inference_binary`.

---

## Inference

To perform inference using your trained model, use a command similar to the following:

```bash
python main.py ifb -o out_test -train*readout gru -train*fusion attention \
  -train*sq_embed ESM2 -train*checkpoint_path your/model/path/model.pt -train*batch_size 512
```

- Replace `your/model/path/model.pt` with the actual path to your model checkpoint.
- The `-o` option defines the output directory for the inference run.

### Additional Note on Data Processing

If a new `.smi` file (raw SMILES data) is provided, the model will automatically preprocess the file into DGL-saved graphs in the `tmp/` folder. Please allow extra processing time for this step.