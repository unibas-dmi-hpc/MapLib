======
MapLib
======
**MapLib** provides algorithms for generating mapping of processing elements to processor unities

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
