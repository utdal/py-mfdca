# Direct Coupling Analysis for Python

## mfDCA implementation, adapted from MATLAB version (http://dca.rice.edu/portal/dca/)

[Direct Coupling Analysis](https://www.pnas.org/doi/10.1073/pnas.1111471108) was developed using MATLAB. This is a python port.

Unit testing Readme and script details output differences from original MATLAB script.

Installation:

Simplest is:
```bash
pip install git+https://github.com/utdal/py-mfdca.git
```

Or:
```bash
git clone https://github.com/utdal/py-mfdca.git
pip install py-mfdca
```
Which will also download the unit testing.

Example Usage: 

```python
from dca.dca_class import dca
protein_family = dca('sequence_file')
protein_family.mean_field()
protein_family.DI # contains DI scores for each pair
protein_family.couplings # NxNxqxq matrix of couplings (eij)
protein_family.localfields # Nxq matrix of localfields (h)
protein_family.compute_Hamiltonian('sequence_file') # returns (Hamiltonians,sequence_headers) for input sequences
```
Runtimes, Random MSA Average of 2 (M1 Pro, 16GB RAM):

| Sequence Length | # of sequences | Runtime (s) | 
| --- | --- | --- |
| 100 | 1,000 | 2.2 | 
| 300 | 1,000  | 5.4 | 
| 500 | 1,000 | 21 |
| 100 | 10,000  | 0.7 |
| 300 | 10,000  | 6.7 |
| 500 | 10,000 | 24.1 |
| 100 | 100,000  | 98.5 |
| 300 | 100,000  | 266.8 |
| 500 | 100,000 | 391 |