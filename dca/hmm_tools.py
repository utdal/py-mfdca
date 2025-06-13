import sys
if sys.platform == "win32":
    raise ImportError("dca.hmm_tools is only supported on Unix systems.")

# safe to import pyhmmer and do other Unix-specific work
import pyhmmer


def MapSeq2hmmProfile(hmmFile, Sequence2Align, whichHit=1) -> dict:
    """
    Align a protein sequence to a profile HMM using pyhmmer and build a mapping dictionary.

    This function loads a protein sequence from a FASTA file and a profile HMM from an HMMER3 file,
    performs a profile-sequence alignment using pyhmmer.hmmscan, and constructs a dictionary mapping
    HMM match state positions to the corresponding positions in the target sequence.

    Parameters
    ----------
    hmmFile : str or Path
        Path to the HMMER3 profile HMM file (e.g., 'PF10531.hmm').
    Sequence2Align : str or Path
        Path to the FASTA file containing the amino acid sequence to align.

    Returns
    -------
    mapping : dict
        A dictionary where keys are 1-based HMM match state positions and values are 1-based positions
        in the target sequence. Only aligned (non-gap, non-insert) positions are included.

    Notes
    -----
    - The function assumes there is only one sequence in the FASTA file and one significant domain hit.
    - The mapping is useful for relating MSA/HMM positions to PDB or other sequence coordinates.
    - Requires pyhmmer to be installed and available in the Python environment.

    Example
    -------
    >>> mapping = MapSeq2hmmProfile("PF10531.hmm", "3M9S_1.fasta")
    >>> print(mapping)
    {1: 3, 2: 4, 3: 5, ...}
    """
    # Load the input sequence (assumes single sequence in file)
    with open(Sequence2Align, "rb") as sf:
        # Read the sequence as a digital sequence for use with pyhmmer
        sequence = pyhmmer.easel.SequenceFile(sf, digital=True).read_block()[0]

    # Load the HMM profile(s) and scan the sequence
    with open(hmmFile, "rb") as hf:
        # Read all HMM profiles from the file
        hmmProfiles = list(pyhmmer.plan7.HMMFile(hf))
        # Run hmmscan: returns an iterator of (query, hits)
        topHit = next(pyhmmer.hmmscan([sequence], hmmProfiles))
        if len(topHit) == 0:
            raise ValueError("No hits found for the provided sequence in the HMM profile.")
        else:
            topHit = topHit[0]  # Get the first hit

    # Get the domain alignment object from the hit
    try:
        domainObj = topHit.domains[whichHit - 1]  # whichHit is 1-based index
    except IndexError:
        raise ValueError(f"There are {len(topHit.domains)} hits to the HMM profile.")
    
    alignObj = domainObj.alignment

    # Positions in HMM and target sequence (1-based, inclusive)
    hmm_start, hmm_end = alignObj.hmm_from, alignObj.hmm_to
    target_start, target_end = alignObj.target_from, alignObj.target_to

    # Aligned sequences (with gaps and insertions)
    hmm_seq = alignObj.hmm_sequence  # e.g. 'A-C.D'
    target_seq = alignObj.target_sequence  # e.g. 'AGC--'

    mapping = {}
    hmm_pos = hmm_start - 1  # adjust for 0-based indexing
    target_pos = target_start - 1

    # Iterate over the aligned columns
    for hmm_char, target_char in zip(hmm_seq, target_seq):
        # Only increment hmm_pos if not a gap in HMM ('.' means insert in target)
        if hmm_char != ".":
            hmm_pos += 1
        # Only increment target_pos if not a gap in target ('-' means gap in target)
        if target_char != "-":
            target_pos += 1
        # Only map positions where both are not gaps/inserts
        if hmm_char != "." and target_char != "-":
            mapping[hmm_pos] = target_pos

    return mapping