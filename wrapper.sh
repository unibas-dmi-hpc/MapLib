#!/bin/bash

comm_matrix="./comm_matrix.p2p.size.csv" #Update this line with the target communication matrix path
#comm_matrix.p2p.size.csv
# mappings=(peano hilbert gray sweep scan bokhari topo_aware greedy FHGreedy greedyALLC bipartition pacmap)
mappings=(peano hilbert gray)
topologies=(mesh torus haec)
dimensions="4 4 4"
#dimensions="16 8 8" #Example of other topology dimensions. The mappings are not limited to the 4 4 4 example. However, a compatible communication matrix is required.
for topology in "${topologies[@]}"; do
    for mapping in "${mappings[@]}"; do
        echo "Mapping: $mapping Topology: $topology Dimensions: $dimensions"
        outfile="$(basename $comm_matrix).$mapping.$topology.${dimensions// /x}.csv"
        mapper map -i $comm_matrix -m $mapping -t $topology -d $dimensions save -o "$outfile" stat
        # mv $outfile $(basename)
    done
done




