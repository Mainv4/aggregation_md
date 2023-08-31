# This program is used to calculate the distribution of complexes according to their composition from AMBER simulations.

"""
    This program is used to calculate the distribution of aggregates from molecular dynamics simulations.
    The distribution of aggregates is the number of aggregates composed by n atoms of a given type along the simulation.
    The distribution of aggregates is saved in a .npy file.
    The mean distribution of aggregates is saved in a .dat file.
    The distribution of aggregates is plotted vs time.
    The mean distribution of aggregates is plotted.
    
    The program is written for amber simulations. And is supposedly used for a system composed by 3 types of atoms, A, B and C and 2 distances, epsilon_AC and epsilon_BC with the idea of studying the complexation of atoms of type C with atoms of type A and B.
"""

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from plot_style import set_plot_style
import MDAnalysis as mda
import networkx as nx
import numpy as np
import argparse
import tqdm
import sys
import os

def loadUnivers(traj_file, top_file):
    """ 
    Load the trajectory file and return the universe

    Parameters
    ----------
    traj_file : str
        The name of the trajectory file
    top_file : str
        The name of the topology file

    Returns
    -------
    u : MDAnalysis.Universe
        The universe object of the trajectory file

    Examples
    --------
    >>> import MDAnalysis as mda
    >>> u = loadUnivers('traj.nc', 'top.prmtop')
    >>> u
    <Universe with 1000 atoms>
    """

    try:
        if traj_file.split('.')[-1] == 'lammpstrj':
            u = mda.Universe(traj_file, topology_format='LAMMPSDUMP')
        elif traj_file.split('.')[-1] == 'nc':
            u = mda.Universe(top_file, traj_file)
    except FileNotFoundError:
        print("Trajectory file not found")
        sys.exit()
    return u

def getAtoms(u, atom_types):
    """
    Get the wanted atoms from the universe

    Parameters
    ----------
    u : MDAnalysis.Universe
        The universe object of the trajectory file
    atom_type : list of str
        The type of the wanted atoms

    Returns
    -------
    atoms : MDAnalysis.AtomGroup
        The wanted atoms

    Examples
    --------
    >>> import MDAnalysis as mda
    >>> u = mda.Universe('traj.lammpstrj', topology_format='LAMMPSDUMP')
    >>> atoms = getAtoms(u, atom_types)
    >>> atoms
    <AtomGroup with 1000 atoms>
    """
    atom_type_A = atom_types[0]
    atom_type_B = atom_types[1]
    atom_type_C = atom_types[2]
    atoms_A = u.select_atoms(atom_type_A)
    atoms_B = u.select_atoms(atom_type_B)
    atoms_C = u.select_atoms(atom_type_C)
    #atoms = atoms_A + atoms_B
    return atoms_A, atoms_B, atoms_C

def computeAggregates(u, atoms, epsilons, atom_types, cutoff, parallel):
    """
    Compute the list of pairs of atoms separated by a distance less than epsilon_r for each frame of the trajectory

    Parameters
    ----------
    u : MDAnalysis.Universe
        The universe object of the trajectory file
    atoms : list of MDAnalysis.AtomGroup
        The wanted atoms (atoms_A, atoms_B, atoms_C)
    epsilons : list of float
        Distances between atoms to be considered as a pair
    cutoff : int
        Number of frames to be skipped at the beginning of the trajectory
    parallel : bool
        If True, the computation will be done in parallel

    Returns
    -------
    aggregates : numpy.ndarray (n_frames - cutoff, n_atoms, 2)
        1st dimension: frame
        2nd dimension: index of the aggregate
        3rd dimension: number of atoms of type A, number of atoms of type B in the aggregate

    Examples
    --------
    >>> import MDAnalysis as mda
    >>> u = mda.Universe('traj.lammpstrj', topology_format='LAMMPSDUMP')
    >>> atoms = getAtoms(u, atom_types)
    >>> aggregates = computeAggregates(u, atoms, epsilon_r, atom_type, 100, True)
    >>> aggregates.shape
    (900, 1000, 2)
    >>> aggregates[0]
    array([[0, 0], [0, 0], [0, 0], ..., [2, 1], [2, 1], [2, 1]], dtype=int32)
    """

    def computeAgg_step(i):
        """ 
        Compute the list of aggregates for a given frame

        Parameters
        ----------
        i : int
            The index of the frame

        Returns
        -------
        a : numpy.ndarray
            The list of aggregates for the given frame
        """
        u.trajectory[i]
        if i == cutoff:
            last = True
        else:
            last = False
        a = computeAgg(atoms, epsilons, atom_types, last)
        return a
    n_frames = u.trajectory.n_frames
    n_atoms = atoms[0].n_atoms + atoms[1].n_atoms
    # The output is a 3D matrix (n_frames - cutoff, n_atoms, 3) where the last dimension is used to store the number of atoms in the aggregate
    aggregates = np.zeros((n_frames - cutoff, n_atoms, 3), dtype=np.int32) - 1

    if parallel == True:
        a_tot = Parallel(n_jobs=-1)(delayed(computeAgg_step)(i) for i in tqdm.tqdm(range(cutoff, n_frames), desc='Aggregates list'))
    else:
        a_tot = [computeAgg_step(i) for i in tqdm.tqdm(range(cutoff, n_frames), desc='Aggregates list')]
    for i in tqdm.tqdm(range(cutoff, n_frames), desc='Putting aggregates in the matrix'):
        aggregates[i - cutoff][n_atoms-a_tot[i - cutoff].shape[0]:] = a_tot[i - cutoff]
    print("aggregates[-1]: " + str(aggregates[-1][-100:]))
    return aggregates

def computeAgg(atoms, epsilons, atom_type, last):
    """ 
    Compute the list of aggregates for a given frame

    Parameters
    ----------
    atoms : list of MDAnalysis.AtomGroup
        The wanted atoms (atoms_A, atoms_B, atoms_C)
    epsilons : float
        Distances between atoms to be considered as a pair

    Returns
    -------
    agg : numpy.ndarray
        The list of aggregates for the given frame

    Examples
    --------
    >>> import MDAnalysis as mda
    >>> u = mda.Universe('traj.lammpstrj', topology_format='LAMMPSDUMP')
    >>> atoms = getAtoms(u, atom_types)
    >>> agg = computeAgg(atoms, 1.27, atom_types)
    >>> agg.shape
    (1000, 2)
    >>> agg
    array([[0, 0], [0, 0], [0, 0], ..., [2, 1], [2, 1], [2, 1]], dtype=int32)
    """

    """ We use here a simplified version of the algorithm in which we compute only the list of pairs A-C and B-C. We do not compute the list of 
    pairs A-A, B-B C-C and A-B since we are interested of the complexation of the atoms of type C with the atoms of type A and B.
    The code just below is the code for the full algorithm but in the case in which we have only 2 types of atoms, and the all possibles 3      
    distances. """
    #   # We compute the list of pairs of atoms of type A separated by a distance less than epsilon_AA
    #   pairs_AA = mda.lib.nsgrid.FastNS(epsilons[0], atoms[0].positions, atoms[0].universe.dimensions, pbc=True).self_search().get_pairs()
    #   # We compute the list of pairs of atoms of type B separated by a distance less than epsilon_BB
    #   pairs_BB = mda.lib.nsgrid.FastNS(epsilons[1], atoms[1].positions, atoms[1].universe.dimensions, pbc=True).self_search().get_pairs()
    #   # We have to add an offset to the indices of the atoms of type B since the indices of the atoms of type B start at 0
    #   pairs_BB = np.array([[pair[0] + atoms[0].n_atoms, pair[1] + atoms[0].n_atoms] for pair in pairs_BB])
    #   # We compute the list of pairs of atoms of type A and B separated by a distance less than epsilon_AB
    #   atoms_all = atoms[0] + atoms[1]
    #   pairs_AB = mda.lib.nsgrid.FastNS(epsilons[2], atoms_all.positions, atoms_all.universe.dimensions, pbc=True).self_search().get_pairs()
    #   # In this last case, we have to remove the pairs of atoms of type A and B that are also in pairs_AA and pairs_BB
    #   # since they are counted twice
    #   #pairs_AB = np.array([pair for pair in pairs_AB if pair not in pairs_AA and pair not in pairs_BB])
    #   # We concatenate the three lists of pairs
    #   try:
    #       pairs = np.concatenate((pairs_AA, pairs_BB, pairs_AB))
    #   except ValueError:
    #       # If one of the lists is empty, we have to concatenate only the two other lists
    #       if pairs_AA.size == 0 and pairs_BB.size == 0 and pairs_AB.size == 0:
    #           pairs = np.array([])
    #       elif pairs_AA.size == 0 and pairs_BB.size == 0:
    #           pairs = pairs_AB
    #       elif pairs_AA.size == 0 and pairs_AB.size == 0:
    #           pairs = pairs_BB
    #       elif pairs_BB.size == 0 and pairs_AB.size == 0:
    #           pairs = pairs_AA
    #       elif pairs_AA.size == 0:
    #           pairs = np.concatenate((pairs_BB, pairs_AB))
    #       elif pairs_BB.size == 0:
    #           pairs = np.concatenate((pairs_AA, pairs_AB))
    #       elif pairs_AB.size == 0:
    #           pairs = np.concatenate((pairs_AA, pairs_BB))
    #       else:
    #           pairs = np.concatenate((pairs_AA, pairs_BB, pairs_AB))

    # Distance between atoms to be considered as a pair
    epsilon_AC = epsilons[0]
    epsilon_BC = epsilons[1]

    # Group of atoms
    atoms_A = atoms[0]
    atoms_B = atoms[1]
    atoms_C = atoms[2]

    # Type of atoms
    resname_A = atom_type[0].split(' ')[1]
    resname_B = atom_type[1].split(' ')[1]
    resname_C = atom_type[2].split(' ')[1]

    # We define a group of atoms containing all the atoms
    atoms_all = atoms_A + atoms_B + atoms_C

    # We compute the list of pairs of atoms of type A and C separated by a distance less than epsilon_AC
    pairs_AC = mda.lib.nsgrid.FastNS(epsilon_AC, atoms_all.positions, atoms_all.universe.dimensions, pbc=True).self_search().get_pairs()
    # We compute the list of pairs of atoms of type B and C separated by a distance less than epsilon_BC
    pairs_BC = mda.lib.nsgrid.FastNS(epsilon_BC, atoms_all.positions, atoms_all.universe.dimensions, pbc=True).self_search().get_pairs()

    # We put the pairs together
    pairs = np.concatenate((pairs_AC, pairs_BC))

    # We remove all pairs that are not composed by an atom of type C
    for i in range(pairs.shape[0]):
        if resname_C not in atoms_all[pairs[i]].resnames:
            pairs[i] = -1, -1
    # We create a graph
    G = nx.Graph()
    G.add_edges_from(pairs)
    # We compute the number of atoms in each aggregate
    agg = np.zeros((nx.number_connected_components(G), 3), dtype=np.int32)
    j = 0
    str_type_A = "x.resnames == '" + resname_A + "'"
    str_type_B = "x.resnames == '" + resname_B + "'"
    str_type_C = "x.resnames == '" + resname_C + "'"
    for component in nx.connected_components(G):
        x = atoms_all[list(component)]
        if last == True:
            print(component)
            for i in range(x.n_atoms):
                print(x[i])
            print()
        a,b,c= np.where(eval(str_type_A))[0].shape[0], np.where(eval(str_type_B))[0].shape[0], np.where(eval(str_type_C))[0].shape[0]
        agg[j] = a, b, c
        j += 1
    return agg

def computeDistribution(aggregates_list, atoms, parallel):
    """ 
    Compute the distribution of aggregates from the list of aggregates

    Parameters
    ----------
    aggregates_list : numpy.ndarray (n_frames - cutoff, n_atoms, 1)
        The list of aggregates
    atoms : MDAnalysis.AtomGroup
        The wanted atoms
    parallel : bool
        If True, the computation will be done in parallel

    Returns
    -------
    distribution : numpy.ndarray (n_frames - cutoff, n_aggregates_max)
        1st dimension: frame
        2nd dimension: size of the aggregate
        The value is the number of aggregates of a given size

    Examples
    --------
    >>> import MDAnalysis as mda
    >>> u = mda.Universe('traj.lammpstrj', topology_format='LAMMPSDUMP')
    >>> atoms = getAtoms(u, atom_types)
    >>> aggregates_list = computeAggregates(u, atoms, 1.27,  atom_type, 100, True)
    >>> distribution = computeDistribution(aggregates_list, atoms)
    """
    print(aggregates_list.shape)

    # We remove the last column of the matrix aggregates_list
    aggregates_list = aggregates_list[:, :, :-1]
    print(aggregates_list.shape)
    print("aggregates_list[-1]: " + str(aggregates_list[-1][-100:]))

    # We compute the number of frames
    n_frames = aggregates_list.shape[0]
    # We get the number of atoms in the system
    atoms_all = atoms[0] + atoms[1]
    n_atoms = atoms_all.n_atoms
    # We compute the number of aggregates
    n_aggregates_max = np.max(aggregates_list)
    print("Maximum number of aggregates : {}".format(n_aggregates_max))
    # We compute the distribution of aggregates for each frame
    distribution = np.zeros((n_frames, n_aggregates_max, n_aggregates_max), dtype=np.int32)
    aggregat_list = np.ma.masked_equal(aggregates_list, -1)
    def computeDistribution_step(t):
        """
        Compute the distribution of aggregates for a given frame t

        Parameters
        ----------
        t : int
            The index of the frame

        Returns
        -------
        distribution_t : numpy.ndarray (n_aggregates_max, n_aggregates_max)
            The distribution of aggregates for the given frame
        """
        distribution_t = np.zeros((n_aggregates_max + 1, n_aggregates_max + 1), dtype=np.int32)
        #for i in range(n_aggregates_max):
        #    for j in range(n_aggregates_max):
        #        distribution_t[i][j] = np.where(aggregates_list[t] == [i, j])[0].shape[0]
        #        print("distribution_t[{}][{}] : {}".format(i, j, np.where(aggregates_list[t] == [i, j])))
        v = np.unique(aggregat_list[t], return_counts=True, axis=0) 
        for i in range(1,v[0].shape[0]):
            distribution_t[v[0][i][0]][v[0][i][1]] = v[1][i]
        return distribution_t
    if parallel == True:
        distribution = Parallel(n_jobs=-1)(delayed(computeDistribution_step)(t) for t in tqdm.tqdm(range(n_frames), desc='Aggregates distribution'))
    else:
        distribution = [computeDistribution_step(t) for t in tqdm.tqdm(range(n_frames), desc='Aggregates distribution')]
    print("distribution[-1]: " + str(distribution[-1]))
    return distribution

def computeMeanDistribution(distribution):
    """ 
    Compute the mean distribution of aggregates over time

    Parameters
    ----------
    distribution : numpy.ndarray (n_frames - cutoff, n_aggregates_max)
        The distribution of aggregates

    Returns
    -------
    distribution_mean : numpy.ndarray (n_aggregates_max)
        The mean distribution of aggregates over time
    """
    distribution_mean = np.mean(distribution, axis=0)
    return distribution_mean

def saveDistribution(distribution, output_file):
    """ 
    Save the distribution of aggregates. The mean distribution in a .dat file and the distribution for each frame in a .npy file

    Parameters
    ----------
    distribution : numpy.ndarray (n_frames - cutoff, n_aggregates_max)
        The distribution of aggregates
    output_file : str
        The name of the output file
    """

    np.save("DATA_distribution/" + output_file + '.npy', distribution)
    distribution_mean = computeMeanDistribution(distribution)
    np.savetxt("DATA_distribution/" + output_file + '_mean_distribution.dat', distribution_mean)



# We write a function that will modifiy the distribution of aggregates in order to have only monomers and dimers
def modifyDistributionOnlyDimers(distribution):
    """ Modify the distribution of aggregates in order to have only monomers and dimers.
    If we have 3 atoms in an aggregate, we consider that we have 1 dimer and 1 monomer;
    If we have 4 atoms in an aggregate, we consider that we have 2 dimers;
    If we have 5 atoms in an aggregate, we consider that we have 2 dimers and 1 monomer;
    ... """
    # We compute the number of frames
    n_frames = distribution.shape[0]
    # We compute the number of aggregates
    n_aggregates_max = distribution.shape[1]
    # We create a new distribution
    distribution_new = np.zeros((n_frames, 3), dtype=np.int32)
    # We compute the new distribution
    for i in tqdm.tqdm(range(n_frames), desc='Modify distribution'):
        for j in range(1, n_aggregates_max):
            if j == 1:
                distribution_new[i][1] += distribution[i][j]
            elif j == 2:
                distribution_new[i][2] += distribution[i][j]
            elif j > 2:
                distribution_new[i][1] += distribution[i][j] * (j % 2)
                distribution_new[i][2] += distribution[i][j] * (j // 2)
    return distribution_new
        

def plotDistributionVsTime(distribution, k_AB, k_BA):
    """ Plot the distribution of aggregates vs time """
    time = np.arange(distribution.shape[0])
    plt.plot(time, distribution[:, 1], label='Monomers')
    plt.plot(time, distribution[:, 2], label='Dimers')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$N$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('distribution_vs_time.png')

def plotDistribution(file_name, atom_type):
    """
    Plot the mean distribution of aggregates

    Parameters
    ----------
    file_name : str
        The name of the distribution file
    atom_type : list of str
        The type of the wanted atoms
    """
    try:
        distribution = np.load("DATA_distribution/" + file_name + '.npy')
        distribution_mean = computeMeanDistribution(distribution)

        # Linscale
        set_plot_style()
        plt.imshow(distribution_mean, cmap='jet', origin='lower')
        plt.colorbar(aspect=5, pad=0.01, fraction=0.15 * 2, label=r'$N$')
        resname_A = atom_type[0].split(' ')[1]
        resname_B = atom_type[1].split(' ')[1]
        plt.xlabel(r'$n_\mathrm{%s}$' % resname_B)
        plt.ylabel(r'$n_\mathrm{%s}$' % resname_A)
        if max(distribution_mean.shape) < 10:
            plt.xticks(np.arange(0, distribution_mean.shape[0], 1))
            plt.yticks(np.arange(0, distribution_mean.shape[1], 1))
        else:
            plt.xticks(np.arange(0, distribution_mean.shape[0], 2))
            plt.yticks(np.arange(0, distribution_mean.shape[1], 2))
        plt.tight_layout()
        plt.savefig("FIGURES/" + file_name + '.png')
        plt.close()
        # Logscale
        from matplotlib.colors import LogNorm
        set_plot_style()
        distribution_mean = np.ma.masked_equal(distribution_mean, 0)
        plt.imshow(distribution_mean, cmap='jet', origin='lower', norm=LogNorm())
        plt.colorbar(aspect=5, pad=0.01, fraction=0.15 * 2, label=r'$N$')
        plt.xlabel(r'$n_\mathrm{%s}$' % resname_B)
        plt.ylabel(r'$n_\mathrm{%s}$' % resname_A)
        if max(distribution_mean.shape) < 10:
            plt.xticks(np.arange(0, distribution_mean.shape[0], 1))
            plt.yticks(np.arange(0, distribution_mean.shape[1], 1))
        else:
            plt.xticks(np.arange(0, distribution_mean.shape[0], 2))
            plt.yticks(np.arange(0, distribution_mean.shape[1], 2))
        plt.tight_layout()
        plt.savefig("FIGURES/" + file_name + '_log.png')
        plt.close()


    except FileNotFoundError:
        print("Distribution file not found")
        print("Please compute the distribution first")
        print("Looking for the file : {}".format("DATA_distribution/" + file_name + '.npy'))
        print("We have the following files in the directory DATA_distribution :")
        for file in os.listdir("DATA_distribution"):
            print(file)

# main function
def main():
    # parse the command line arguments with argparse
    parser = argparse.ArgumentParser(description="Calculate the number of formed dimers from a LJ gas simulated using LAMMPS.")
    parser.add_argument("-c", "--compute", type=str, help="Compute the distribution of aggregates", choices=['yes', 'no'])
    parser.add_argument('-p', '--parallel', type=str, help='If True, the computation will be done in parallel', choices=['yes', 'no'])
    #parser.add_argument('-a', '--atom_type', type=str, help='The type of the wanted atoms')
    # We now have to choose two types of atoms
    parser.add_argument('-a', '--atom_type', type=str, nargs='+', help='The type of the wanted atoms: "type A" "type B" "type C"')
    # And three distance
    parser.add_argument('-e', '--epsilon', type=float, nargs='+', help='Distance between atoms to be considered as a pair: epsilon_AC psilon_BC')
    parser.add_argument('-tr', '--traj-file', type=str, help='Trajectory file or distribution file')
    parser.add_argument('-to', '--top-file', type=str, help='Topology file')
    parser.add_argument('-pl', '--plot', type=str, help='Plot the distribution of aggregates', choices=['yes', 'no'])
    parser.add_argument('-o', '--output', type=str, help='Output file')

    # create the directory DATA_distribution if it does not exist
    os.makedirs('DATA_distribution', exist_ok=True)
    os.makedirs('FIGURES', exist_ok=True)


    args = parser.parse_args()

    # Parse the types
    atom_types = args.atom_type
    print("Types of atoms:")
    for i in range(len(atom_types)):
        print(atom_types[i])

    # Parse the distances
    epsilons = args.epsilon
    print("Distances between atoms to be considered as a pair:")
    for i in range(len(epsilons)):
        print(epsilons[i])

    if args.compute == 'yes':
        # We compute the distribution of aggregates
        print("Compute the distribution of aggregates")

        # load the trajectory file
        print("Load the trajectory file...")
        u = loadUnivers(args.traj_file, args.top_file)
        print("Trajectory file loaded")

        print("In the trajectory file, we found {} resnames".format(np.unique(u.atoms.resnames).shape[0]))
        print("The resnames are:")
        for i in range(np.unique(u.atoms.resnames).shape[0]):
            print(np.unique(u.atoms.resnames)[i])
        print()
        
        # get the wanted atoms
        print("Get the wanted atoms...")
        atom_type = args.atom_type
        atoms = getAtoms(u, atom_types)
        print("Wanted atoms got")
        print(atoms)
 
        # Number of frames to be skipped at the beginning of the trajectory: cutoff
        cutoff = np.int32(0.5 * u.trajectory.n_frames) 
        
        # compute the list of aggregates
        print("Compute the list of aggregates...")
        if args.parallel == 'yes':
            parallel = True
        else:
            parallel = False
        aggregates_list = computeAggregates(u, atoms, epsilons, atom_types, cutoff, parallel)
        print("List of aggregates computed")
 
        # compute the distribution of aggregates
        print("Compute the distribution of aggregates...")
        distribution = computeDistribution(aggregates_list, atoms, parallel)
        print("Distribution of aggregates computed")
        print()
 
        # compute the mean distribution of aggregates over time
        print("Compute the mean distribution of aggregates over time...")
        distribution_mean = computeMeanDistribution(distribution)
        print("Mean distribution of aggregates over time computed")
        print()
 
        # save the distribution of aggregates
        print("Save the distribution of aggregates into the file {}...".format(args.output))
        output = args.output + "_" + args.traj_file.replace('.lammpstrj', '').replace('.nc', '').replace('/', '_')
        saveDistribution(distribution, output)
        print("Distribution of aggregates saved")
        print("The data are saved in the directory DATA_distribution")
    elif args.compute == 'no':
        try:
            distribution = np.load(args.file)
        except FileNotFoundError:
            print("Distribution file not found")
            print("Please compute the distribution first")
            sys.exit()
        # compute the mean distribution of aggregates over time
        distribution_mean = computeMeanDistribution(distribution)
        # Print with 2 decimals the mean distribution without scientific notation
        distribution_new = modifyDistributionOnlyDimers(distribution)
        distribution_mean_new = computeMeanDistribution(distribution_new)
        np.set_printoptions(precision=2, suppress=True)
        for i in range(distribution_mean_new.shape[0]):
            print("Number of aggregates of size {} : {}".format(i, distribution_mean_new[i]))
        s = 0
        for i in range(distribution_mean_new.shape[0]):
            s += distribution_mean_new[i] * (i)
    if args.plot == 'yes':
        file_name = args.output + "_" + args.traj_file.replace('.lammpstrj', '').replace('.nc', '').replace('/', '_')
        plotDistribution(file_name, atom_types)

if __name__ == "__main__":
    main()
