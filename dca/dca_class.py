import numpy as np
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
    
    def compute_EffAlphabet(self, sequences):
        numerical_sequences, _ = create_numerical_MSA(sequences, self.symboldict)
        return return_EffAlphabet(numerical_sequences, self.couplings, self.localfields)