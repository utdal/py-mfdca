The data used in the DCA computation via the MATLAB script given a set of input sequences.

The same input sequences are used as input for py-mfdca, and the methods are compared at various points in the computation.

*****

Reasons for discrepencies:

- condition number on C matrix is high (~3000), so the inversion step is sensitive to numerical differences.
- Coupling matrix values are extremely similar, will not affect significantly sequence Hamiltonian scores.
- Spearmans correlation of Top 1000 DI pairs is > 99.9% compared to MATLAB results, will not significantly differ from MATLAB results.
- Spearmans correlation of all DI pairs is 98.5%.

**Update March 8, 2022. Same results as below with new compute_W function.**


```shell
(dca) unit_testing.py
test_check_header_accuracy (__main__.test_NumericalMSA) ... ok
test_check_sequence_conversion (__main__.test_NumericalMSA) ... ok
test_DI_calculation_closeness (__main__.test_computeDI_justcouplings) ... ok
test_C_closeness (__main__.test_compute_Pi_Pij_pseudocount_DI) ... ok
test_DI_arraysize (__main__.test_compute_Pi_Pij_pseudocount_DI) ... ok
test_DI_closeness (__main__.test_compute_Pi_Pij_pseudocount_DI) ... FAIL
test_equivalent_reweighted_sequences (__main__.test_compute_Pi_Pij_pseudocount_DI) ... ok
test_invC_closeness (__main__.test_compute_Pi_Pij_pseudocount_DI) ... FAIL
test_pi_closeness (__main__.test_compute_Pi_Pij_pseudocount_DI) ... ok
test_pij_closeness (__main__.test_compute_Pi_Pij_pseudocount_DI) ... ok
test_pseudocount_pi (__main__.test_compute_Pi_Pij_pseudocount_DI) ... ok
test_pseudocount_pij (__main__.test_compute_Pi_Pij_pseudocount_DI) ... ok
test_spearmans_DI_100 (__main__.test_compute_Pi_Pij_pseudocount_DI) ... ok
test_spearmans_DI_1000 (__main__.test_compute_Pi_Pij_pseudocount_DI) ... ok
test_spearmans_DI_50 (__main__.test_compute_Pi_Pij_pseudocount_DI) ... ok
test_spearmans_DI_500 (__main__.test_compute_Pi_Pij_pseudocount_DI) ... ok
test_check_correct_dimensions (__main__.test_load_couplings) ... ok
test_correct_filling (__main__.test_load_couplings) ... ok

======================================================================
FAIL: test_DI_closeness (__main__.test_compute_Pi_Pij_pseudocount_DI)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "dca/unit_testing.py", line 89, in test_DI_closeness
    np.testing.assert_allclose(self.calc_di[:,2],self.loaded_di[:,2])
  File "miniconda/base/envs/dca/lib/python3.9/site-packages/numpy/testing/_private/utils.py", line 1528, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
  File "miniconda/base/envs/dca/lib/python3.9/site-packages/numpy/testing/_private/utils.py", line 842, in assert_array_compare
    raise AssertionError(msg)
AssertionError: 
Not equal to tolerance rtol=1e-07, atol=0

Mismatched elements: 3100 / 3321 (93.3%)
Max absolute difference: 0.00159403
Max relative difference: 0.00971624
 x: array([0.088409, 0.018345, 0.013299, ..., 0.325436, 0.025644, 0.079542])
 y: array([0.088411, 0.018345, 0.013299, ..., 0.325342, 0.025643, 0.079542])

======================================================================
FAIL: test_invC_closeness (__main__.test_compute_Pi_Pij_pseudocount_DI)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "dca/unit_testing.py", line 85, in test_invC_closeness
    np.testing.assert_allclose(self.invC,self.invC_matlab)
  File "miniconda/base/envs/dca/lib/python3.9/site-packages/numpy/testing/_private/utils.py", line 1528, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
  File "miniconda/base/envs/dca/lib/python3.9/site-packages/numpy/testing/_private/utils.py", line 842, in assert_array_compare
    raise AssertionError(msg)
AssertionError: 
Not equal to tolerance rtol=1e-07, atol=0

Mismatched elements: 181749 / 2689600 (6.76%)
Max absolute difference: 1.40259779e-11
Max relative difference: 384275.22300591
 x: array([[ 7.098221e+01,  4.200000e+01,  4.200000e+01, ..., -3.923955e-11,
        -3.924689e-11, -1.184982e-02],
       [ 4.200000e+01,  8.400000e+01,  4.200000e+01, ..., -3.845147e-11,...
 y: array([[ 7.098221e+01,  4.200000e+01,  4.200000e+01, ..., -2.842980e-11,
        -2.842981e-11, -1.184982e-02],
       [ 4.200000e+01,  8.400000e+01,  4.200000e+01, ..., -2.506304e-11,...

----------------------------------------------------------------------
Ran 18 tests in 147.081s

FAILED (failures=2)
