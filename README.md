# MaxWeight Scheduling for Wireless Networks

## For Experiment Case 1-3:
### Directory Structure
├── max1.py                     # Main implementation of the MaxWeight scheduling framework

├── network_topology_tests.py   # Test script for different network topologies

├── analysis_plots.py           # Utilities for generating analysis plots

├── output_[topology]/          # Output data for each topology test

└── plots_[topology]/           # Generated plots and visualizations

### Prerequisites
Python 3.6+

Required packages: numpy, matplotlib, networkx, csv

### Running the Tests
The main script network_topology_tests.py can be run with various parameters to test different network topologies:
```bash
# Test with butterfly network
python network_topology_tests.py --topology butterfly --resolution 30 --time 8000 --threshold 5000

# Test with dense 5-node network (3D capacity region)
python network_topology_tests.py --topology dense_5node --resolution 10 --time 5000 --threshold 200 --xmax 1.0 --ymax 1.0 --zmax 1.0
```

### Command Line Arguments
--topology: Network topology to test (line, ring, butterfly, random_conflict, dense_5node)

--xmax, --ymax, --zmax: Maximum rates for flows (default: 2.0)

--resolution: Number of points to test in each dimension (default: 20)

--time: Simulation time for each test (default: 2000)

--threshold: Stability threshold for queue lengths (default: 1000)

--num_links, --num_flows: Parameters for random topologies

--p_conflict: Probability of conflicts in random topologies


### Example Results
Running the tests generates several visualization files that help understand the behavior of the MaxWeight algorithm:

Network Topology: Visualization of nodes, links, and flows

Capacity Region: 2D or 3D plots showing stable and unstable operating points

Queue Evolution: Time-series plots of queue lengths for different nodes and flows

Scheduling Decisions: Analysis of link and node activation patterns
