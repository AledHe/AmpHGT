"""
Call Pretrained ChemBERTa2 for SMILES feature embedding.

Extract embeddings for whole graph level.
"""

import torch
from transformers import AutoTokenizer, RobertaModel, PreTrainedTokenizerFast

ChemBERTa_77M_MLM = "DeepChem/ChemBERTa-77M-MLM"

def init_model():
    tokenizer = AutoTokenizer.from_pretrained(ChemBERTa_77M_MLM, cache_dir="./huggingface")
    model = RobertaModel.from_pretrained(ChemBERTa_77M_MLM, cache_dir="./huggingface")
    return tokenizer, model

def smi_to_embedding(smi: str, tokenizer: PreTrainedTokenizerFast, model: RobertaModel):
    inputs = tokenizer(smi, return_tensors="pt", padding=False, truncation=False)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

    return embeddings[0, 0, :]

if __name__ == "__main__":
    tokenizer, model = init_model()
    smi = "CC(C)C[C@H](NC(=O)CN(CCCCN)C(=O)CN(CCCCN)C(=O)CNCCCCN)C(=O)N[C@@H](Cc1ccc2ccccc2c1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccc2ccccc2c1)C(N)=O"
    print(smi_to_embedding(smi, tokenizer, model))