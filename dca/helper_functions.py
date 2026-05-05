import linecache
from Bio.PDB import PDBParser
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import math
import os
import warnings
import numpy as np
import string
from itertools import groupby
from typing import Union


def clean_pfam(input_filename: str, output_filename: str) -> None:
    """Fasta files acquired by hmmsearch often have lowercase characters and periods, which we remove for our analysis.
    This function does that, producing sequences with only gaps and uppercase letters.
    """
    # Define string translator
    translator = {c: "" for c in string.ascii_lowercase}
    translator["."] = ""
    trans = str.maketrans(translator)

    # define mapper function
    def clean_record(record: SeqRecord) -> SeqRecord:
        sequence = str(record.seq)
        record.seq = Seq(sequence.translate(trans))
        return record

    clean_sequences = map(clean_record, SeqIO.parse(input_filename, "fasta"))

    SeqIO.write(clean_sequences, output_filename, "fasta")


def filter_pfam(filename: str, gap_cutoff: Union[float, int], output: str) -> None:
    """Large contiguous gapped regions in sequences negatively affect our analysis and results.
    This removes sequences which contain contiguous gaps whose length is >= the supplied percentage of the total length of the sequence.
    Empirically, 0.2 (20%) is a good starting point."""
    # Get max gap size based on provided gap_cutoff. If float, assume percentage and calculate it. If integer, assume max gap count provided.
    if isinstance(gap_cutoff, float):
        maxgap = round(gap_cutoff * len(next(SeqIO.parse(filename, "fasta")).seq))
    elif isinstance(gap_cutoff, int):
        maxgap = gap_cutoff

    # Find all gap substrings in a sequence, if max is > gaps, remove from set.
    def max_chunksize(record: SeqRecord) -> int:
        chunk_lengths = list(
            len(list(chunk)) for char, chunk in groupby(str(record.seq)) if char == "-"
        )
        if len(chunk_lengths) == 0:
            return 0
        else:
            return max(chunk_lengths)

    # parse file and keep only ones with <=gaps
    original_set = list(SeqIO.parse(filename, "fasta"))
    filtered_set = [record for record in original_set if max_chunksize(record) < maxgap]
    SeqIO.write(filtered_set, output, "fasta")

    # print statistics
    percentage = (len(original_set) - len(filtered_set)) / len(original_set) * 100
    print(f"Original number of sequences: {len(original_set)}\n")
    print(
        f"Number of sequences excluded: {len(original_set) - len(filtered_set)} ({round(percentage, 2)}%)"
    )


def interface_contacts_allatoms(
    input_filename: str, first_chain: str, second_chain: str, angstrom_cutoff: float
) -> str:
    linecache.clearcache()
    size = open(input_filename, "r")
    lines = len(size.readlines())
    # create output_filename, considering directories as inputs
    input_dir = os.path.sep.join(input_filename.split(os.path.sep)[:-1])
    pdb_name = input_filename.split(os.path.sep)[-1].split(".")[0]
    output_filename = os.path.join(
        input_dir,
        "contactmap_calpha_"
        + pdb_name
        + "_"
        + first_chain
        + second_chain
        + "_"
        + str(angstrom_cutoff),
    )
    n = 0
    an1 = []
    an2 = []
    r1 = []
    r2 = []
    rn1 = []
    rn2 = []
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    z1 = []
    z2 = []
    ch1 = []
    ch2 = []

    res1 = ""
    res2 = ""

    for i in range(1, lines):
        a = linecache.getline(input_filename, i)
        if a[0:4] == "ATOM" and a[21:22] == first_chain:
            an1.append(str(a[6:11]))
            r1.append(str(a[22:26]))
            rn1.append(str(a[17:20]))
            x1.append(float(str(a[30:38])))
            y1.append(float(str(a[38:46])))
            z1.append(float(str(a[46:54])))
            ch1.append(str(a[21:22]))
        if a[0:4] == "ATOM" and a[21:22] == second_chain:
            an2.append(str(a[6:11]))
            r2.append(str(a[22:26]))
            rn2.append(str(a[17:20]))
            x2.append(float(str(a[30:38])))
            y2.append(float(str(a[38:46])))
            z2.append(float(str(a[46:54])))
            ch2.append(str(a[21:22]))
    output = open(output_filename, "w")
    output.write(
        "at numb".ljust(10)
        + "res numb".ljust(10)
        + "res name".ljust(10)
        + "chain".ljust(8)
        + "at numb".ljust(10)
        + "res numb".ljust(10)
        + "res name".ljust(10)
        + "chain".ljust(8)
        + "distance".ljust(15)
        + "\n"
    )

    for j in range(0, len(r1)):
        for k in range(j + 1, len(r2)):
            dx = math.pow(x1[j] - x2[k], 2)
            dy = math.pow(y1[j] - y2[k], 2)
            dz = math.pow(z1[j] - z2[k], 2)
            dist = math.pow(dx + dy + dz, 0.5)
            if dist <= angstrom_cutoff:
                output.write(
                    an1[j].ljust(10)
                    + r1[j].ljust(10)
                    + rn1[j].ljust(10)
                    + ch1[j].ljust(8)
                    + an2[k].ljust(10)
                    + r2[k].ljust(10)
                    + rn2[k].ljust(10)
                    + ch2[k].ljust(8)
                    + str(dist).ljust(15)
                    + "\n"
                )
                n += 1

    print("\n\tNumber of interactions found: " + str(n) + "\n")
    print(
        "\n\tFile saved as: "
        + "contactmap_allatom_"
        + input_filename[:-4]
        + "_"
        + first_chain
        + second_chain
        + "_"
        + str(angstrom_cutoff)
        + "\n"
    )
    return output_filename


def interface_contacts_calpha(
    input_filename: str, first_chain: str, second_chain: str, angstrom_cutoff: float
) -> str:
    linecache.clearcache()
    size = open(input_filename, "r")
    lines = len(size.readlines())
    # create output_filename, considering directories as inputs
    input_dir = os.path.sep.join(input_filename.split(os.path.sep)[:-1])
    pdb_name = input_filename.split(os.path.sep)[-1].split(".")[0]
    output_filename = os.path.join(
        input_dir,
        "contactmap_calpha_"
        + pdb_name
        + "_"
        + first_chain
        + second_chain
        + "_"
        + str(angstrom_cutoff),
    )
    n = 0
    r1 = []
    r2 = []
    rn1 = []
    rn2 = []
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    z1 = []
    z2 = []
    ch1 = []
    ch2 = []

    res1 = ""
    res2 = ""

    for i in range(1, lines):
        a = linecache.getline(input_filename, i)
        if a[0:4] == "ATOM" and a[21:22] == first_chain:
            if a[22:26] != res1:
                if a[13:15] == "CA":
                    r1.append(str(a[22:26]))
                    rn1.append(str(a[17:20]))
                    x1.append(float(str(a[30:38])))
                    y1.append(float(str(a[38:46])))
                    z1.append(float(str(a[46:54])))
                    ch1.append(str(a[21:22]))
                    res1 = a[22:26]
        if a[0:4] == "ATOM" and a[21:22] == second_chain:
            if a[22:26] != res2:
                if a[13:15] == "CA":
                    r2.append(str(a[22:26]))
                    rn2.append(str(a[17:20]))
                    x2.append(float(str(a[30:38])))
                    y2.append(float(str(a[38:46])))
                    z2.append(float(str(a[46:54])))
                    ch2.append(str(a[21:22]))
                    res2 = a[22:26]
    output = open(output_filename, "w")
    output.write(
        "res numb".ljust(10)
        + "res name".ljust(10)
        + "chain".ljust(8)
        + "res numb".ljust(10)
        + "res name".ljust(10)
        + "chain".ljust(8)
        + "distance".ljust(15)
        + "\n"
    )

    for j in range(0, len(r1)):
        for k in range(j, len(r2)):
            dx = math.pow(x1[j] - x2[k], 2)
            dy = math.pow(y1[j] - y2[k], 2)
            dz = math.pow(z1[j] - z2[k], 2)
            dist = math.pow(dx + dy + dz, 0.5)
            if dist <= angstrom_cutoff and dist > 0:
                output.write(
                    r1[j].ljust(10)
                    + rn1[j].ljust(10)
                    + ch1[j].ljust(8)
                    + r2[k].ljust(10)
                    + rn2[k].ljust(10)
                    + ch2[k].ljust(8)
                    + str(dist).ljust(15)
                    + "\n"
                )
                n += 1

    print("\n\tNumber of interactions found: " + str(n) + "\n")
    print(
        "\n\tFile saved as: "
        + "contactmap_calpha_"
        + input_filename[:-4]
        + "_"
        + first_chain
        + second_chain
        + "_"
        + str(angstrom_cutoff)
        + "\n"
    )
    return output_filename


def backmap_alignment(align: str) -> dict:
    # get information from manual alignment file
    linecache.clearcache()
    domain = linecache.getline(align, 1)
    protein_id = linecache.getline(align, 6)[:4]

    d_init = int(linecache.getline(align, 2))
    d_end = int(linecache.getline(align, 4))
    p_init = int(linecache.getline(align, 7))
    p_end = int(linecache.getline(align, 9))

    # domain sequence string
    l1 = linecache.getline(align, 3)
    # protein sequence string
    l2 = linecache.getline(align, 8)

    x1 = len(l1)
    x2 = len(l2)

    # get the difference between initial positions
    delta = max(x1, x2)
    # delta = max(d_end-d_init,p_end-p_init)

    # domain code and respective number
    d = []
    dn = []
    # protein code and respective number
    p = []
    pn = []

    # fill d and p arrays with domain and protein sequences
    for i in range(0, delta - 1):
        d.append(l1[i])
        p.append(l2[i])

    # compute the original positions in the system

    j1 = -1
    j2 = -1
    for i in range(0, len(d)):
        if d[i] != ".":
            j1 += 1
            dn.append(str(d_init + j1))
        if d[i] == ".":
            dn.append("")
        if p[i] != "-":
            j2 += 1
            pn.append(str(p_init + j2))
        if p[i] == "-":
            pn.append("")

    dic = {}
    for i in range(0, len(dn)):
        if d[i] != "." and p[i] != "-":
            dic[int(dn[i])] = int(pn[i])
    return dic


def get_allatom_contacts(
    pdb_file: str, chain1_id: str, chain2_id: str, distance_cutoff: int
) -> tuple:
    warnings.filterwarnings("ignore")
    # Create a PDB parser object

    suffix_struc = pdb_file.split(".")[-1]
    if suffix_struc == "pdb":
        parser = PDBParser()
    elif suffix_struc == "cif":
        parser = MMCIFParser(auth_chains=False, auth_residues=False)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix_struc}. Only PDB and CIF formats are supported."
        )

    # Parse the PDB file
    structure = parser.get_structure("protein", pdb_file)

    # Get the specified chains
    chain1 = structure[0][chain1_id]
    chain2 = structure[0][chain2_id]

    # Function to get atom coordinates from a residue
    def get_atom_coords(residue):
        atom_coords = []
        for atom in residue:
            if atom.is_disordered():
                for position in atom:
                    atom_coords.append(position.coord)
            else:
                atom_coords.append(atom.coord)
        return np.array(atom_coords)

    # Get coordinates and residue IDs for both chains
    coord_ids1 = [(get_atom_coords(res), res.id[1]) for res in chain1 if is_aa(res)]
    coord_ids2 = [(get_atom_coords(res), res.id[1]) for res in chain2 if is_aa(res)]

    if len(coord_ids1) == 0 or len(coord_ids2) == 0:
        # no valid aa residues in the chain, so we skip
        return ()

    coords1, res_ids1 = zip(*coord_ids1)
    coords2, res_ids2 = zip(*coord_ids2)

    # Combine all atom coordinates for each chain
    all_coords1 = np.vstack(coords1)
    all_coords2 = np.vstack(coords2)

    # Calculate distances between all atoms
    distances = np.linalg.norm(all_coords1[:, np.newaxis] - all_coords2, axis=2)

    # Find pairs of atoms within the cutoff distance
    close_atom_pairs = np.argwhere(distances <= distance_cutoff)

    # Map atom pairs to residue pairs
    res_index1 = np.cumsum([len(c) for c in coords1])
    res_index2 = np.cumsum([len(c) for c in coords2])

    close_residue_pairs = set()
    for i, j in close_atom_pairs:
        res1 = np.searchsorted(res_index1, i, side="right")
        res2 = np.searchsorted(res_index2, j, side="right")
        close_residue_pairs.add((res_ids1[res1], res_ids2[res2]))

    print(f"Finished {pdb_file} {chain1_id} {chain2_id}")
    results = (
        pdb_file,
        chain1_id,
        chain2_id,
        np.array(list(close_residue_pairs)),
    )
    return np.array(list(close_residue_pairs))
