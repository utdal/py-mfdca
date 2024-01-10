# Direct Coupling Analysis for Python

## mfDCA implementation, adapted from MATLAB version (http://dca.rice.edu/portal/dca/)

[Direct Coupling Analysis](https://www.pnas.org/doi/10.1073/pnas.1111471108) was developed using MATLAB. This is a python port.

Unit testing Readme and script details output differences from original MATLAB script.

Runtime comparison:

| Sequence Length | # of sequences | MATLAB runtime (s) | Python runtime (s) | 
| --- | --- | --- | --- |
| 82 | 36,540 | 42.3 | 148.8 | 
| 298 | 11,507 | 30.7 | 48.8 | 
| 217 | 422 | 5.4 | 8.1 |
| 79 | 31,575 | 26.1 | 92.7 |
| 302 | 461 | 13.6 | 14.7 |

Installation:
When in directory with setup.py,
```bash
pip install .
```

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
