import numpy as np
from scipy.spatial.distance import cdist
from numba import jit, prange
from Bio import SeqIO

# numerical sequence converter


def create_numerical_MSA(sequences, alphabet):
    # if the letter is not in the alphabet, it is considered a gap.
    seq_parser = SeqIO.parse(sequences, "fasta")
    numerical_seqs = []
    headers = []
    for item in seq_parser:
        headers.append(item.id)
        newseq = []
        for letter in item.seq:
            try:
                newseq.append(alphabet[letter])
            except:
                newseq.append(alphabet["-"])
        numerical_seqs.append(newseq)
    numerical_seqs = np.array(numerical_seqs, dtype=int)
    return numerical_seqs, headers


# loading couplings from Mi3GPU, compute gaugeless DI


def load_couplings(N, q, eij_file):
    if ".npy" in eij_file:
        couplings = np.load(eij_file)
    else:
        couplings = np.loadtxt(eij_file)
    # check that the input is correct
    if couplings.shape[1] == q * q:
        pass
    elif couplings.shape[0] == q * q:
        couplings = np.transpose(couplings)
    else:
        exit(
            "The couplings do not have a dim of q*q (q*q ij block, flattened). Please reformat your data!"
        )

    formatted_couplings = np.zeros((N, N, q, q))
    count = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            formatted_couplings[i, j, :, :] = np.reshape(couplings[count], (q, q))
            formatted_couplings[j, i, :, :] = formatted_couplings[i, j, :, :]
            count += 1
    return formatted_couplings


def compute_DI_justcouplings(N, q, couplings):
    Pdir = np.zeros((N, N, q, q))
    DI = np.zeros((int(N * (N - 1) / 2), 3))
    count = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            pdir_unnormalized = np.exp(couplings[i, j, :, :])
            Pab = pdir_unnormalized / pdir_unnormalized.sum()
            fa = np.sum(Pab, 1)
            fb = np.sum(Pab, 0)
            temp = np.zeros((q, q))
            for qi in range(q):
                for qj in range(q):
                    temp[qi, qj] = Pab[qi, qj] / (fa[qi] * fb[qj])
            temp = np.log(temp)
            DI[count, :] = [i, j, np.trace(np.dot(Pab.conj().T, temp))]
            count += 1
    return DI


# functions for mfDCA


def compute_W(sequences, theta, batch_size):
    """handles very large data so that it fits into memory"""
    msa_len = sequences.shape[0]
    seq_len = sequences.shape[1]
    output_W = np.zeros(msa_len, dtype=np.float64)
    for idx in range(0, msa_len, batch_size):
        if idx + batch_size > msa_len:
            computation = (cdist(sequences[idx:], sequences, "hamm") < theta).sum(1)
            output_W[idx:] = computation
        else:
            computation = (
                cdist(sequences[idx : idx + batch_size], sequences, "hamm") < theta
            ).sum(1)
            output_W[idx : idx + batch_size] = computation
    return 1 / output_W


@jit(nopython=True)
def compute_Pi(
    sequences, pseudocount, N, M, q, Meff, W
):  # count single site statistics
    """np.array((N,N,q,q)), returns non-diagonal Pij matrix"""
    Pi = np.zeros((N, q), dtype=np.float64)
    for j in range(N):
        for l in range(M):
            Pi[j, sequences[l, j]] = Pi[j, sequences[l, j]] + W[l]

    Pi = Pi / Meff

    return Pi


@jit(nopython=True, parallel=True)
def compute_Pij(
    sequences, pseudocount, N, M, q, Meff, W, Pi
):  # count pairwise statistics
    Pij = np.zeros((N, N, q, q), dtype=np.float64)
    for i in prange(N - 1):
        for j in range(i + 1, N):
            for l in range(M):
                Pij[i, j, sequences[l, i], sequences[l, j]] = (
                    Pij[i, j, sequences[l, i], sequences[l, j]] + W[l]
                )
                Pij[j, i, sequences[l, j], sequences[l, i]] = Pij[
                    i, j, sequences[l, i], sequences[l, j]
                ]

    Pij = Pij / Meff

    qq_id = np.eye(q, q, dtype=np.float64)
    for i in range(N):
        for alpha in range(q):
            for beta in range(q):
                Pij[i, i, alpha, beta] = Pi[i, alpha] * qq_id[alpha, beta]
    return Pij


@jit(nopython=True)
def add_pseudocount(
    Pi, Pij, pseudocount_weight, N, q
):  # add pseudocount to counting matrices
    Pij_pc = (1 - pseudocount_weight) * Pij + pseudocount_weight / q / q * np.ones(
        (N, N, q, q), dtype=np.float64
    )
    Pi_pc = (1 - pseudocount_weight) * Pi + pseudocount_weight / q * np.ones(
        (N, q), dtype=np.float64
    )

    qq_id = np.eye(q, q, dtype=np.float64)

    for i in range(N):
        for alpha in range(q):
            for beta in range(q):
                Pij_pc[i, i, alpha, beta] = (1 - pseudocount_weight) * Pij[
                    i, i, alpha, beta
                ] + pseudocount_weight / q * qq_id[alpha, beta]
    return Pi_pc, Pij_pc


@jit(nopython=True)
def computeC(Pi, Pij, N, q):  # produce connected correlation matrix
    C = np.zeros((N * (q - 1), N * (q - 1)), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            for alpha in range(q - 1):
                for beta in range(q - 1):
                    C[(q - 1) * (i) + alpha, (q - 1) * (j) + beta] = (
                        Pij[i, j, alpha, beta] - Pi[i, alpha] * Pi[j, beta]
                    )
    return C


def invC_to_4D(invC, N, q):  # create 4D eij array from inverted C matrix
    reshaped_invC = np.zeros((N, N, q, q), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            reshaped_invC[i, j, :, :] = np.pad(
                invC[
                    i * (q - 1) : (i + 1) * (q - 1),
                    j * (q - 1) : (j) * (q - 1) + (q - 1),
                ],
                ((0, 1), (0, 1)),
                mode="constant",
            )
    return reshaped_invC


@jit(nopython=True)
def Compute_Results(Pi, invC, N, q):  # calculate DI score for each pair of positions
    Pairwisehfield = np.zeros((N * q, 2 * N), dtype=np.float64)
    DI_pairs = np.zeros((int(N * (N - 1) / 2), 3), dtype=np.float64)
    epsilon = 0.00004
    count = 0
    tiny = np.nextafter(0, 1)

    for i in range(N):
        for j in range(i + 1, N):
            # direct information from mean-field
            W_mf = np.ones((q, q), dtype=np.float64)
            W_mf[: q - 1, : q - 1] = np.exp(-invC[i, j, : q - 1, : q - 1])
            diff = 1.0
            mu1 = np.ones(q, dtype=np.float64) / q
            mu2 = np.ones(q, dtype=np.float64) / q
            pi = Pi[i, :]
            pj = Pi[j, :]
            while diff > epsilon:
                calc1 = np.dot(mu2, W_mf.T)
                calc2 = np.dot(mu1, W_mf)

                new1 = np.divide(pi, calc1)
                new1 = new1 / new1.sum()
                new1_diff = np.absolute(new1 - mu1)

                new2 = np.divide(pj, calc2)
                new2 = new2 / new2.sum()
                new2_diff = np.absolute(new2 - mu2)

                diff = np.maximum(new1_diff, new2_diff).max()
                mu1 = new1
                mu2 = new2

            # DI pair calc
            mu1_t = np.expand_dims(mu1, axis=1)
            Pdir = W_mf * (mu1_t * mu2)
            Pdir = Pdir / Pdir.sum()

            Pi_t = np.expand_dims(pi, axis=1)
            Pfac = Pi_t * pj
            DI = np.trace(np.dot(Pdir.T, np.log((Pdir + tiny) / (Pfac + tiny))))
            DI_pairs[count, :] = np.array([i, j, DI])
            count += 1

            # pairwise fields
            mu1 = np.log(mu1 / mu1[-1])
            mu2 = np.log(mu2 / mu2[-1])
            hihj = np.vstack((mu1, mu2)).T
            Pairwisehfield[(i) * q : (i * q) + q, 2 * j : 2 * j + 2] = hihj
            Pairwisehfield[(j) * q : (j * q) + q, 2 * i : 2 * i + 2] = hihj[:, ::-1]
    return Pairwisehfield, DI_pairs


def Compute_AverageLocalField(Pairwisehfield, N, q):  # caluclate local fields (h)
    hi = np.zeros((q, N), dtype=np.float64)
    # average i->rest pairwise fields, returns 21*N matrix
    # 1st,3rd,5th...2*N-1 columns, excluding the column at pos i.
    # split the matrix at these intervals, pull i+i*21 rows and average these N-1 values together. Append to position i.
    for i in prange(N):
        pairwise_chunk = (
            Pairwisehfield[i * q : (i * q) + q, range(0, N * 2, 2)].sum(1)
            - Pairwisehfield[i * q : (i * q) + q, i]
        ) / (N - 1)
        hi[:, i] = pairwise_chunk
    return hi


@jit(nopython=True)
def return_Hamiltonian(
    numerical_sequences, couplings, localfields, interDomainCutoff=None
):  # calculates Hamiltonian for input fasta
    M = numerical_sequences.shape[0]
    L = numerical_sequences.shape[1]
    hamiltonians = np.zeros(M, dtype=np.float64)

    first, second = np.triu_indices(couplings.shape[0], k=1)
    pairs = np.stack((first, second), axis=1)

    if isinstance(interDomainCutoff, int):
        pairs = pairs[
            (pairs[:, 0] < interDomainCutoff) & (pairs[:, 1] >= interDomainCutoff)
        ]

    for seq in range(M):
        for i in range(L):
            hamiltonians[seq] = (
                hamiltonians[seq] + localfields[numerical_sequences[seq, i], i]
            )
    for seq in range(M):
        for pair in pairs:
            hamiltonians[seq] = (
                hamiltonians[seq]
                + couplings[
                    pair[0],
                    pair[1],
                    numerical_sequences[seq, pair[0]],
                    numerical_sequences[seq, pair[1]],
                ]
            )

    return -hamiltonians
