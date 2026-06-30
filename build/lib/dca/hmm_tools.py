import sys
if sys.platform == "win32":
    raise ImportError("dca.hmm_tools is only supported on Unix systems.")

# safe to import pyhmmer and do other Unix-specific work
import pyhmmer
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import tempfile

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

def MapNuc2hmmProfile(hmmFile, NucSequence2Align,whichHit=1) -> dict:
    """
    Map a nucleotide sequence to an HMM profile via its translated protein sequence.

    This function:
      1. Translates a nucleotide FASTA to protein.
      2. Aligns the protein to the HMM profile using pyhmmer.
      3. Builds a mapping from HMM match states to nucleotide codons.
      4. Writes aligned protein and nucleotide sequences to disk.

    Parameters
    ----------
    hmmFile : str or Path
        Path to the HMMER3 profile HMM file.
    NucSequence2Align : str or Path
        Path to the nucleotide FASTA file.
    whichHit : int, optional
        Which domain hit to use from the alignment (default: 1).

    Returns
    -------
    alignment_dictionary : dict
        Dictionary mapping 1-based HMM match state positions to 1-based codon positions in the nucleotide sequence.

    Notes
    -----
    - Aligned sequences are saved as FASTA files with '.aligned.fasta' suffix.
    - Requires pyhmmer and biopython.
    """

    if not isinstance(hmmFile, Path):
        hmmFile = Path(hmmFile)
    if not isinstance(NucSequence2Align, Path):
        NucSequence2Align = Path(NucSequence2Align)

    with open(NucSequence2Align, "r") as nt_handle:
        nucleotide_record = SeqIO.read(nt_handle, "fasta")

    # Convert nucleotide sequence to protein sequence
    protein_seq = nucleotide_record.seq.translate()
    protein_record = SeqRecord(
        protein_seq,
        id=nucleotide_record.id,
        name=nucleotide_record.name,
        description=nucleotide_record.description
    )

    TransProtFile = NucSequence2Align.with_suffix('.translated.fasta')
    with open(TransProtFile, "w") as protein_handle:
        SeqIO.write(protein_record, protein_handle, "fasta")

    with open(hmmFile, "rb") as hf:
        # Read all HMM profiles from the file
        profL = list(pyhmmer.plan7.HMMFile(hf))[0].M
    
    alignment_dictionary = MapSeq2hmmProfile(hmmFile, TransProtFile, whichHit)

    aligned_sequence = []
    aligned_nucleotide_sequence = []
    for resIdx in range(1, profL+1):
        # Get the corresponding position in the target sequence
        # If the position is not mapped, it will return None
        target_position = alignment_dictionary.get(resIdx, None)
        aligned_sequence.append(protein_record.seq[target_position-1] if target_position is not None else '-')
        aligned_nucleotide_sequence.append(str(nucleotide_record.seq[3*(target_position-1):3*target_position]) if target_position is not None else '---')

    aligned_sequence = ''.join(aligned_sequence)
    aligned_nucleotide_sequence = ''.join(aligned_nucleotide_sequence)

    aligned_protein_record = SeqRecord(
        Seq(aligned_sequence),
        id=protein_record.id,
        name=protein_record.name,
        description=protein_record.description
    )

    aligned_nucleotide_record = SeqRecord(
        Seq(aligned_nucleotide_sequence),
        id=nucleotide_record.id,
        name=nucleotide_record.name,
        description=nucleotide_record.description
    )

    # Save aligned sequences to files
    aligned_protein_file = TransProtFile.with_suffix('.aligned.fasta')
    with open(aligned_protein_file, "w") as protein_handle:
        SeqIO.write(aligned_protein_record, protein_handle, "fasta")

    aligned_nucleotide_file = NucSequence2Align.with_suffix('.aligned.fasta')
    with open(aligned_nucleotide_file, "w") as nucleotide_handle:
        SeqIO.write(aligned_nucleotide_record, nucleotide_handle, "fasta")

    return alignment_dictionary

def MapMSA2hmmProfile(hmmFile, MSA2Align, whichHit=1):
    """
    Aligns each sequence in a multiple sequence alignment (MSA) file to a given HMM profile and generates an aligned MSA in HMM profile coordinates.
    Parameters
    ----------
    hmmFile : str or pathlib.Path
        Path to the HMM profile file (in HMMER format).
    MSA2Align : str or pathlib.Path
        Path to the input MSA file (in FASTA format) whose sequences will be mapped to the HMM profile.
    whichHit : int, optional
        Specifies which alignment hit to use when mapping each sequence to the HMM profile (default is 1).
    Returns
    -------
    None
        The function writes the aligned sequences to a new FASTA file with the suffix '.aligned.fasta' in the same directory as the input MSA.
    Notes
    -----
    - Each sequence in the input MSA is realigned to the HMM profile using a temporary FASTA file.
    - The output aligned MSA will have the same number of columns as the HMM profile length, with gaps ('-') inserted where residues do not align.
    - Requires `pyhmmer`, `Bio.SeqIO`, and `tempfile` modules.
    """

    # Ensure hmmFile and MSA2Align are Path objects
    if not isinstance(hmmFile, Path):
        hmmFile = Path(hmmFile)
    if not isinstance(MSA2Align, Path):
        MSA2Align = Path(MSA2Align)

    with pyhmmer.plan7.HMMFile(hmmFile) as hmmfile:
        hmm_profile = next(hmmfile)
        hmm_length = hmm_profile.M
        print(f"HMM profile length: {hmm_length}")

    alignedRecords = []
    with open(MSA2Align, "r") as handle:
        msa_iterator = SeqIO.parse(handle, "fasta")
        for record in msa_iterator:
            # Create temp file and ensure it's properly closed before reading
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as temp_fasta:
                temp_filename = Path(temp_fasta.name)
                record.seq = Seq(str(record.seq).replace('-', ''))
                SeqIO.write(record, temp_fasta, "fasta")
            try:
                # print(f"Processing temp file: {temp_filename}")
                alignDict = MapSeq2hmmProfile(hmmFile, str(temp_filename), whichHit=whichHit)
                # print(f"Alignment for {record.id}: {alignDict[record.id]}")
            finally:
                # Clean up the temporary file
                temp_filename.unlink()
            
            aligned_sequence = []

            for resIdx in range(1, hmm_length+1):
                # Get the corresponding position in the target sequence
                # If the position is not mapped, it will return None
                target_position = alignDict.get(resIdx, None)
                aligned_sequence.append(record.seq[target_position-1] if target_position is not None else '-')
        
            aligned_sequence = ''.join(aligned_sequence)

            aligned_protein_record = SeqRecord(
                Seq(aligned_sequence),
                id=record.id,
                name=record.name,
                description=record.description)

            alignedRecords.append(aligned_protein_record)

    # Save aligned sequences to files
    alignedMSA = MSA2Align.with_suffix('.aligned.fasta')
    with open(alignedMSA, "w") as protein_handle:
        SeqIO.write(alignedRecords, protein_handle, "fasta")    