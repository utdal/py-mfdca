import numpy as np
from Bio import Align
from matplotlib import pyplot as plt
from numba import jit, prange


def map_strings(alignment_sequence, pdb_sequence):
    """Returns a dictionary of {alignment_sequence : pdb_sequence}.
    Does not currently handle merged sequence alignments (seq1->pdb1, seq2->pdb2)"""

    aligner = Align.PairwiseAligner()
    aligner.target_internal_open_gap_score = -10
    aligner.query_internal_extend_gap_score = -10
    alignment = aligner.align(pdb_sequence, alignment_sequence)
    aligned_dict = {}
    for pairs in range(len(alignment[0].aligned[0])):
        newdict = {
            x: y
            for x, y in zip(
                range(
                    alignment[0].aligned[1][pairs][0],
                    alignment[0].aligned[1][pairs][1],
                ),
                range(
                    alignment[0].aligned[0][pairs][0],
                    alignment[0].aligned[0][pairs][1],
                ),
            )
        }
        aligned_dict.update(newdict)
    return aligned_dict, alignment


def map_filter_DIpairs(alignment_dictionary, DI_pairs):
    mapped_di = []
    for pair in DI_pairs:
        if pair[1] - pair[0] > 4:
            if (
                pair[0] in alignment_dictionary
                and pair[1] in alignment_dictionary
            ):
                newpair = [
                    alignment_dictionary[pair[0]],
                    alignment_dictionary[pair[1]],
                ]
                if len(pair) == 3:  # DI pairs are included
                    newpair.append(pair[2])
                mapped_di.append(newpair)

    mapped_di = np.array(mapped_di)
    mapped_di[:, :2] = mapped_di[:, :2] + 1
    return mapped_di


@jit(nopython=True, parallel=True)
def find_pair_hits(array_one: np.ndarray, array_two: np.ndarray) -> np.ndarray:
    """Returns boolean array of length of array_one, indicating for each pair in array_one if it occurs in array_two."""
    output = np.zeros(array_one.shape[0], dtype=np.bool_)
    for i in prange(array_one.shape[0]):
        for j in range(array_two.shape[0]):
            if (
                array_one[i][0] == array_two[j][0]
                and array_one[i][1] == array_two[j][1]
            ):
                output[i] = True
                break
    return output


def plot_top_pairs(pdb_pairs, mapped_DI_pairs, figure_size):
    if mapped_DI_pairs.shape[1] == 3:  # ranks DI pairs for you
        sort_map = np.argsort(mapped_DI_pairs[:, 2])[::-1]
    elif (
        mapped_DI_pairs.shape[1] == 2
    ):  # you have provided ranked pairs with no DI scores
        sort_map = range(mapped_DI_pairs.shape[0])
    pair_count = mapped_DI_pairs.shape[0]

    sorted_DI = mapped_DI_pairs[sort_map]
    hits_and_misses = find_pair_hits(sorted_DI, pdb_pairs)

    hits = sorted_DI[hits_and_misses == 1]
    misses = sorted_DI[hits_and_misses == 0]
    di_results = np.hstack((sorted_DI[:, :2], hits_and_misses[:, None]))

    plot = plt.figure(figsize=figure_size, dpi=400)
    _ = plt.grid(alpha=0.5, linestyle="--", linewidth="0.3")
    ax = plt.gca()
    ax.set_axisbelow(True)
    _ = plt.scatter(
        x=pdb_pairs[:, 1],
        y=pdb_pairs[:, 0],
        s=0.05,
        c="gray",
        marker=".",
        label="PDB contacts",
    )
    _ = plt.scatter(
        x=pdb_pairs[:, 0],
        y=pdb_pairs[:, 1],
        s=0.05,
        marker="o",
        label="PDB contacts",
    )
    _ = plt.scatter(
        x=misses[:, 1],
        y=misses[:, 0],
        s=0.05,
        marker="x",
        color="black",
        label="DI pair misses",
    )
    _ = plt.scatter(
        x=hits[:, 1],
        y=hits[:, 0],
        s=0.1,
        marker="x",
        color="red",
        label="DI pair hits",
    )

    # Get nice tickmarks
    data_min = np.min(pdb_pairs[:, 0])
    data_max = np.max(pdb_pairs[:, 1])
    data_range = data_max - data_min

    # Find appropriate interval size rounded to the nearest 10
    interval = np.ceil(data_range / 10 / 10) * 10

    # Create tick marks
    start = np.floor(data_min / interval) * interval
    end = np.ceil(data_max / interval) * interval
    ticker = np.arange(start, end + interval, interval)

    _ = plt.xticks(ticker, fontsize=3)
    _ = plt.yticks(ticker, fontsize=3)
    _ = plt.margins(0.001)

    return plot, len(hits) / pair_count, di_results
