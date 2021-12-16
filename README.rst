======
MapLib
======
**MapLib** provides algorithms for generating mapping of processing elements to processor unities to three 3-D topologies: mesh, torus, and HAEC Box. MapLib implements: (i) Communication- and topology-oblivious mapping strategies, which do not take into account the communication matrices of the applications nor the target processor topologies. These algorithms follow a predetermined node ordering to map all processes to the available nodes and produce deterministic mappings. (ii) Communication- and topology-aware mapping strategies, which consider both communication matrices of the applications and target processor topologies. These algorithms produce different mappings for a given application–system pair.

**Paper references**


- [J. H. M. Korndörfer, M. Bielert, L. L. Pilla, and F. M. Ciorba, Mapping matters: Application process mapping on 3-d processor topologies, arXiv preprint arXiv:2005.10413 (2020).](https://arxiv.org/abs/2005.10413)

- Paper presentation recording: https://drive.switch.ch/index.php/s/VaO9f7zT79lvRe2


**MapLib contains the following process mapping techniques:**

**Communication- and Topology-Oblivious Mapping:**

| ``Peano``  
| ``Hilbert``  
| ``Gray``  
| ``sweep``  
``scan``

**Communication- and Topology-Aware Mapping:**

| ``Bokhari``  
| ``topo-aware``  
| ``greedy``  
| ``FHgreedy``  
| ``greedyALLC``  
| ``bipartition``   
``PaCMap``

| The folder **mapping-matters-commMatrices** contains all input communication matrices  
| and  
| The folder **mapping-matters-Mappings** contains all mappings that were used in the paper:  
Mapping Matters: Application Process Mapping on 3-D Processor Topologies available at: https://arxiv.org/abs/2005.10413

The sub-folder **MapLib** contains all Python files where mapping.py and mapping_compl.py consist of the implemented mapping algorithms.

**To install the program:**

* Run ``make``
* Run ``pip3 install --editable .``

**As an example of how to use the library, one can run the provided** ``wrapper.sh`` **script**

**Acknowledgments**

Acknowledgment to Daniel Besmer and Viacheslav Sharunov for their earlier contribution to the library.
