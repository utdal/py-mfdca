import numpy as np
from dca.dca_functions import (
    compute_W,
    compute_Pi,
    compute_Pij,
    add_pseudocount,
    computeC,
    invC_to_4D,
    Compute_Results,
    create_numerical_MSA,
    return_Hamiltonian,
    load_couplings,
    compute_DI_justcouplings,
    Compute_AverageLocalField,
)
from scipy.io import loadmat
from scipy.stats import spearmanr
import unittest


# Unit tests for loading non mfDCA function outputs
class test_NumericalMSA(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        alphabet = {keys: index for index, keys in enumerate("-ACDEFGHIKLMNPQRSTVWY")}
        self.msa, self.headers = create_numerical_MSA(
            "unit_testing/main_function/dummy_msa.fasta", alphabet
        )

    def test_check_header_accuracy(self):
        self.assertEqual(len(self.headers), 25)

    def test_check_sequence_conversion(self):
        self.assertEqual(self.msa.sum(), 500)


class test_load_couplings(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.loaded_couplings = load_couplings(
            20, 21, "unit_testing/main_function/dummy_couplings.npy"
        )

    def test_check_correct_dimensions(self):
        self.assertEqual(self.loaded_couplings.shape, (20, 20, 21, 21))

    def test_correct_filling(self):
        self.assertEqual(self.loaded_couplings.sum(), 167580)


class test_computeDI_justcouplings(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.loaded_couplings = load_couplings(
            20, 21, "unit_testing/main_function/dummy_couplings.npy"
        )
        self.computed_DI = compute_DI_justcouplings(20, 21, self.loaded_couplings)
        self.dummy_DI = np.load("unit_testing/main_function/dummy_di.npy")

    def test_DI_calculation_closeness(self):
        np.testing.assert_allclose(self.computed_DI, self.dummy_DI)


# test of essential functions for mfDCA
class test_compute_Pi_Pij_pseudocount_DI_Hamiltonians(
    unittest.TestCase
):  # theta=0.2,pseudo=0.5
    @classmethod
    def setUpClass(self):
        # run mfDCA in its entirety
        alphabet = {keys: index for index, keys in enumerate("-ACDEFGHIKLMNPQRSTVWY")}
        self.sequences, _ = create_numerical_MSA(
            "unit_testing/main_function/pdz_nodupes.fa", alphabet
        )
        self.W = compute_W(self.sequences, 0.2, 50000)
        self.Meff = self.W.sum()
        self.pi_calc = compute_Pi(
            self.sequences, 0.5, 82, self.sequences.shape[0], 21, self.Meff, self.W
        )
        self.pij_calc = compute_Pij(
            self.sequences,
            0.5,
            82,
            self.sequences.shape[0],
            21,
            self.Meff,
            self.W,
            self.pi_calc,
        )
        self.pi_pseudo, self.pij_pseudo = add_pseudocount(
            self.pi_calc, self.pij_calc, 0, 82, 21
        )
        pi_real, pij_real = add_pseudocount(self.pi_calc, self.pij_calc, 0.5, 82, 21)
        self.C = computeC(pi_real, pij_real, 82, 21)
        self.invC = np.linalg.inv(self.C)
        self.couplings = invC_to_4D(-self.invC, 82, 21)  # save "pretty" couplings
        pairwisefield, self.calc_di = Compute_Results(pi_real, -self.couplings, 82, 21)
        self.localfield = Compute_AverageLocalField(pairwisefield, 82, 21)
        self.full_hamiltonian = return_Hamiltonian(
            self.sequences, self.couplings, self.localfield
        )
        self.domain_hamiltonian = return_Hamiltonian(
            self.sequences, self.couplings, self.localfield, interDomainCutoff=50
        )

        # load MATLAB mfDCA arrays
        self.mat_w = np.squeeze(loadmat("unit_testing/main_function/w_pdz.mat")["W"].T)
        self.pi_matlab = loadmat("unit_testing/main_function/pi_pdz.mat")["Pi_true"]
        self.pij_matlab = loadmat("unit_testing/main_function/pij_pdz.mat")["Pij_true"]
        self.C_matlab = loadmat("unit_testing/main_function/C_pdz.mat")["C"]
        self.invC_matlab = loadmat("unit_testing/main_function/invC_pdz.mat")["invC"]
        opendi = open("unit_testing/main_function/pdz.DI", "r")
        self.loaded_di = np.array(
            [[float(y) for y in x.rstrip().split()] for x in opendi]
        )
        self.loaded_di[:, :2] = self.loaded_di[:, :2] - 1
        opendi.close()
        self.known_full_ham = np.loadtxt(
            "unit_testing/main_function/pdz_hamiltonians.txt"
        )
        self.known_domain_ham = np.loadtxt(
            "unit_testing/main_function/pdz_domain50_hamiltonian.txt"
        )

        # rank DI values for spearmans test at different range cutoffs
        self.ranked_DI_calc = np.argsort(self.calc_di[:, 2])
        self.ranked_DI_matlab = np.argsort(self.loaded_di[:, 2])
        self.spearman_corr_50, _ = spearmanr(
            self.ranked_DI_calc[:50], self.ranked_DI_matlab[:50]
        )
        self.spearman_corr_100, _ = spearmanr(
            self.ranked_DI_calc[:100], self.ranked_DI_matlab[:100]
        )
        self.spearman_corr_500, _ = spearmanr(
            self.ranked_DI_calc[:500], self.ranked_DI_matlab[:500]
        )
        self.spearman_corr_1000, _ = spearmanr(
            self.ranked_DI_calc[:1000], self.ranked_DI_matlab[:1000]
        )

    def test_equivalent_reweighted_sequences(self):
        np.testing.assert_equal(self.W, self.mat_w)

    def test_pi_closeness(self):
        np.testing.assert_allclose(self.pi_matlab, self.pi_calc)

    def test_pij_closeness(self):
        np.testing.assert_allclose(self.pij_matlab, self.pij_calc)

    def test_pseudocount_pij(self):  # no pseudocount added, should be identical
        np.testing.assert_equal(self.pij_calc, self.pij_pseudo)

    def test_pseudocount_pi(self):
        np.testing.assert_equal(self.pi_calc, self.pi_pseudo)

    def test_C_closeness(self):
        np.testing.assert_allclose(self.C, self.C_matlab)

    def test_invC_closeness(self):
        np.testing.assert_allclose(self.invC, self.invC_matlab)

    def test_DI_arraysize(self):
        self.assertEqual(len(self.calc_di[:, 0]), len(self.loaded_di[:, 0]))

    def test_DI_closeness(self):
        np.testing.assert_allclose(self.calc_di[:, 2], self.loaded_di[:, 2])

    def test_spearmans_DI_50(self):
        self.assertGreater(self.spearman_corr_50, 0.999)

    def test_spearmans_DI_100(self):
        self.assertGreater(self.spearman_corr_100, 0.999)

    def test_spearmans_DI_500(self):
        self.assertGreater(self.spearman_corr_500, 0.999)

    def test_spearmans_DI_1000(self):
        self.assertGreater(self.spearman_corr_1000, 0.999)

    def test_full_hamiltonians(self):
        np.testing.assert_equal(self.full_hamiltonian, self.known_full_ham)

    def test_domain_hamiltonians(self):
        np.testing.assert_equal(self.domain_hamiltonian, self.known_domain_ham)


# run tests
unittest.main(verbosity=2)
