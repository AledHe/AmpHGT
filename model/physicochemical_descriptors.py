"""
implementation of AAC, DPC, and DTC features.

codes here are modified to accommodate "X" as non-canonical amino acids.

implement with modified Propy Library.

for a simple test purpose, we only consider AAC as handcraft features.
for most of NCAAs, the physicochemical properties are not available.
"""

from .AAComposition import CalculateAAComposition, CalculateDipeptideComposition, GetSpectrumDict, CalculateAADipeptideComposition

def aac_fasta(fasta, **kwargs):
    """
    Calculate the Amino Acid Composition (AAC) for a given protein sequence.

    Parameters
    ----------
    fasta : str
        The protein sequence in fasta format.

    Returns
    -------
    dict
        A dictionary containing the AAC of the protein sequence.
    """
    return CalculateAAComposition(fasta, **kwargs)