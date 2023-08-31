# aggregation_md

## minimal documentation for the aggregation_md project

The aggregation_md project is a series of python scripts that can be used to evaluate the distribution of aggregates from molecular dynamics simulations.

The scripts are written in python 3 and require the following packages:
* MDAnalysis
* numpy
* matplotlib
* networkx
* tqdm

The scripts are designed to be run from the command line.  The scripts are:
* calc_distribution_1_type.py
	* Written for lammps trajectories, it looks for the aggregate made by one type of atom.
	* It can be tested using the following command:
		* `python3 calc_distribution_1_type.py -c yes -p yes -pl yes -a "type 1" -e 2.0 -f examples/lammps/traj_1_type.lammpstrj -o distr`
* calc_distribution_2_types.py
	* Written for lammps trajectories, it looks for the aggregate made by two types of atoms and three distances between them.
	* It can be tested using the following command:
		* `python3 calc_distribution_2_types.py -c yes -p yes -pl yes -a "type 1" "type 2" -e 2.0 2.0 -f examples/lammps/traj_2_types.lammpstrj -o distr`
* calc_distribution_3_types_coordination_amber.py
	* Written for amber trajectories, it looks for the coordination of an ion with organic molecules. It considers three types of atoms, but only two distances since it only considers the distance between the ion and the oxygen atom of the two possible organic molecules, forgetting about the distance between the two organic molecules, which is not relevant for the coordination.
	* It can be tested using the following command:
		* `python3 calc_distribution_3_types_coordination_amber.py -c yes -p no -a "resname ECA and type o" "resname PCA and type o" "resname Na+" -e 3.1 3.1 -tr examples/amber/10frames.nc -to examples/amber/boite.prmtop -pl yes -o distr` 


