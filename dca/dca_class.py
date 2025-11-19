import numpy as np
from numpy import linalg as LA
from .dca_functions import (
    compute_W,
    compute_Pi,
    compute_Pij,
    add_pseudocount,
    computeC,
    invC_to_4D,
    Compute_Results,
    Compute_AverageLocalField,
    create_numerical_MSA,
    return_Hamiltonian,
    load_couplings,
    compute_DI_justcouplings,
    return_EffAlphabet,
)


class dca:
    def __init__(self, fasta, couplings="", localfields="", stype="protein"):
        if stype == "protein":
            self.symboldict = {
                keys: index for index, keys in enumerate("-ACDEFGHIKLMNPQRSTVWY")
            }
        elif stype == "dna":
            self.symboldict = {keys: index for index, keys in enumerate("ACGTU-")}
        elif len(stype) > 0:
            self.symboldict = {keys: index for index, keys in enumerate(stype)}
        else:
            raise ("Provide valid sequence alphabet")

        self.q = len(self.symboldict)
        # create numeric MSA.
        self.sequences, self.headers = create_numerical_MSA(fasta, self.symboldict)
        self.M = self.sequences.shape[0]
        self.N = self.sequences.shape[1]
        if len(couplings) > 0:
            self.couplings = -load_couplings(self.N, self.q, couplings)
            self.DI = compute_DI_justcouplings(self.N, self.q, -self.couplings)
        if len(localfields) > 0:
            self.localfields = np.loadtxt(localfields)
        if len(couplings) > 0 and len(localfields) == 0:
            self.localfields = np.zeros(
                (self.q, self.N)
            )  # assumes you use a no-gauge solution

    def mean_field(
        self,
        pseudocount_weight=0.5,
        theta=0.2,
        cdist_batch_size=50000,
        save_sequence_info=True,
    ):
        """captures a coupling matrix np.array((N,N,q,q)) and local fields np.array((N,q)) to self.couplings and self.localfields.
        See original paper for description of pseudocount_weight and theta.
        cdist_batch_size can be modified if your MSA does not fit easily in your memory when doing pairwise comparisons for the
        reweighting step (lower values are slower but requires less memory).
        save_sequence_info is set to True, and setting this to false will remove the self.sequences variable so that saved models
        do not contain the original MSA as an array (to cut down on final size). Leave on if you'd like a record of the data from which
        couplings/localfields were derived."""
        # compute m_a, then M_eff
        W = compute_W(self.sequences, theta=theta, batch_size=cdist_batch_size)
        self.Meff = W.sum()
        # compute reweighted frequences
        Pi = compute_Pi(
            self.sequences, pseudocount_weight, self.N, self.M, self.q, self.Meff, W
        )
        Pij = compute_Pij(
            self.sequences, pseudocount_weight, self.N, self.M, self.q, self.Meff, W, Pi
        )
        Pi_pc, Pij_pc = add_pseudocount(Pi, Pij, pseudocount_weight, self.N, self.q)
        # compute couplings matrix
        C = computeC(Pi_pc, Pij_pc, self.N, self.q)
        invC = np.linalg.inv(C)
        self.couplings = invC_to_4D(-invC, self.N, self.q)  # save "pretty" couplings
        pairwisefield, self.DI = Compute_Results(Pi_pc, -self.couplings, self.N, self.q)
        self.localfields = Compute_AverageLocalField(pairwisefield, self.N, self.q)
        if not save_sequence_info:
            self.sequences = None

    def compute_Hamiltonian(self, sequences, interDomainCutoff=None):
        numerical_sequences, headers = create_numerical_MSA(sequences, self.symboldict)
        return (
            return_Hamiltonian(
                numerical_sequences,
                self.couplings,
                self.localfields,
                interDomainCutoff=interDomainCutoff,
            ),
            headers,
        )
    
    def Frobenius(self):
        """
        Compute pairwise coupling strengths using the (centered) Frobenius norm and apply
        Average Product Correction (APC) to mitigate background / entropic effects.
        For each pair of positions (i, j), the method:
          1. Extracts the q x q coupling submatrix (excluding the final gauge/state if present).
          2. Mean-centers the matrix first across columns, then across rows (double-centering).
          3. Computes the Frobenius norm of the centered matrix as a raw coupling score.
          4. Applies APC: F_apc(i, j) = F(i, j) - (mean_i * mean_j) / global_mean,
             where mean_i is the average of row i over all partners, and global_mean is the
             average of all row means.
        Returns
        -------
        F_array : np.ndarray, shape (M, 3)
            Array of raw Frobenius coupling scores.
            Each row: [i, j, F_ij] with i < j, where
            M = L * (L - 1) / 2 and L = self.N (sequence length).
        Fapc_array : np.ndarray, shape (M, 3)
            Array of APC-corrected coupling scores.
            Each row: [i, j, F_apc_ij] with i < j.
        Notes
        -----
        - self.couplings is expected to have shape (L, L, q, q) or compatible,
          where q includes the gauge state; the last state (index -1) is excluded
          before centering.
        - Double-centering removes both row and column means, yielding a matrix
          with zero mean across rows and columns before norm evaluation.
        - APC reduces spurious correlations arising from positional conservation
          rather than genuine coevolution.
        References
        ----------
        - Frobenius norm: ||A||_F = sqrt(sum_{i,j} A_{ij}^2).
        See Also
        --------
        LA.norm : NumPy linear algebra norm function used internally.
        """

        L = self.N
        F=np.zeros((L,L))
        Fapc=np.zeros((L, L))
        q= self.q
        for i in range (L):
            for j in range(i+1,L):
                #matrix  21x21 including also the gauge symbol
                jinf2=np.zeros((q,q))
                jinf2[:-1,:-1]= self.couplings[i,j,:-1,:-1]
                J_norm=jinf2 - np.mean(jinf2,0)
                J_norm=J_norm - np.mean(J_norm,1,keepdims=True)
                #Frobenius norm
                F[j,i]=F[i,j]
        avcoupl1 = np.sum(F, 1) / L
        sumj = np.sum(avcoupl1) / L
        for i in range(L):
            for j in range(i + 1, L):
                Fapc[i, j] = F[i, j] - avcoupl1[i] * avcoupl1[j] / sumj
                Fapc[j, i] = Fapc[i, j]
        F_array = np.column_stack([ np.repeat(np.arange(L), L),
            np.tile(np.arange(L), L),F.flatten()])
        Fapc_array = np.column_stack([ np.repeat(np.arange(L), L),
            np.tile(np.arange(L), L),Fapc.flatten()])

        # Filter to keep only upper triangular (i < j)
        F_array = F_array[F_array[:, 0] < F_array[:, 1]]
        Fapc_array = Fapc_array[Fapc_array[:, 0] < Fapc_array[:, 1]]
        
        return F_array, Fapc_array

    def compute_EffAlphabet(self, sequences):
        numerical_sequences, _ = create_numerical_MSA(sequences, self.symboldict)
        return return_EffAlphabet(numerical_sequences, self.couplings, self.localfields)