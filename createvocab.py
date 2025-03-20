import argparse
import json

from utils.read_params import lcfig
from tokenizer.generate_vocablist import main as generate_vocablist

def rearrange(file_name):
    # Load JSON data
    with open(file_name, 'r') as file:
        data = json.load(file)

    fragments = data.get("frequent_fragments", {})

    # Target SMILES string to be placed at index 0, which is alpha backbone.
    target_smiles = "NCC=O"

    # Create a new dictionary for rearranged data
    rearranged_fragments = {}

    # If target SMILES is present, add it to the start of the dictionary
    if target_smiles in fragments:
        rearranged_fragments[target_smiles] = 0
        del fragments[target_smiles]

    # Append remaining SMILES with incremented index
    index = 1
    for smiles in fragments:
        rearranged_fragments[smiles] = index
        index += 1

    # Add the empty string with the length of the original frequent_fragments
    rearranged_fragments[""] = index

    # Save the rearranged data back to the JSON file
    data["frequent_fragments"] = rearranged_fragments

    # Save the rearranged data to a new JSON file
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Rearrangement complete. Data saved to {file_name}.")

# Warning Unfound fragments CC(=O)NCCC=O in generate_1621850_ACT

@lcfig(config_path = 'configs/genvocab.yaml', output_dir = "out_vocabgen")
def main(cfg):
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--count_threshold", type=int, help="Minimum count threshold for frequent fragments")
    argparser.add_argument("--output_path", type=str, required=True, help="Output path for the fragment vocabulary list")
    argparser.add_argument("--method", type=str, help="Method used to generate the vocabulary list")
    argparser.add_argument("--workers", type=int, help="Number of workers for multiprocessing")
    argparser.add_argument("--rearrange", type=str, help="Rearrange a json")
    args = argparser.parse_args()
    if not args.rearrange:
        generate_vocablist(cfg, args.count_threshold, args.output_path, args.method, args.workers)
    elif args.rearrange == "True": # --rearrange "True"
        rearrange(args.output_path)

if __name__ == "__main__":
    main()