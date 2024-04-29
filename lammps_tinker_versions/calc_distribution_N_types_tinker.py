# This program is used to calculate the distribution of complexes according to their composition from tinker simulations.

"""
    This program is used to calculate the distribution of aggregates from molecular dynamics simulations.
    The distribution of aggregates is the number of aggregates composed by n atoms of a given type along the simulation.
    The distribution of aggregates is saved in a .npy file.
    The mean distribution of aggregates is saved in a .dat file.
    The distribution of aggregates is plotted vs time.
    The mean distribution of aggregates is plotted.
    
    The program is written for tinker simulations. And is supposedly used for a system composed by 3 types of atoms, A, B and C and 2 distances, epsilon_AC and epsilon_BC with the idea of studying the complexation of atoms of type C with atoms of type A and B.
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
            #u = mda.Universe(traj_file, topology_format='LAMMPSDUMP')
            print("The trajectory file is a LAMMPS trajectory file")
            print("Please consider an tinker trajectory file")
            sys.exit()
        elif traj_file.split('.')[-1] == 'nc':
            u = mda.Universe(top_file, traj_file)
        elif traj_file.split('.')[-1] == 'arc' or traj_file.split('.')[-1] == 'txyz':
            u = mda.Universe(traj_file)
    except FileNotFoundError:
        print("Trajectory file not found")
        sys.exit()
    return u

def getAtoms(u, atom_types, n_atoms):
    """
    Get the wanted atoms from the universe

    Parameters
    ----------
    u : MDAnalysis.Universe
        The universe object of the trajectory file
    atom_type : list of str
        The type of the wanted atoms
    n_atoms : int
        The number of types of atoms considered in the analysis

    Returns
    -------
    atoms : MDAnalysis.AtomGroup
        The wanted atoms

    Examples
    --------
    >>> import MDAnalysis as mda
    >>> u = mda.Universe('traj.lammpstrj', topology_format='LAMMPSDUMP')
    >>> atoms = getAtoms(u, atom_types, n_atoms)
    >>> atoms
    (<AtomGroup with X atoms>, <AtomGroup with Y atoms>, ...)
    """
    #atom_type_A = atom_types[0]
    #atom_type_B = atom_types[1]
    #atom_type_C = atom_types[2]
    #atoms_A = u.select_atoms(atom_type_A)
    #atoms_B = u.select_atoms(atom_type_B)
    #atoms_C = u.select_atoms(atom_type_C)
    ##atoms = atoms_A + atoms_B
    atoms_type = []
    for i in range(n_atoms):
        atoms_type.append(u.select_atoms(atom_types[i]))
    atoms = tuple(atoms_type)
    return atoms

def computeAggregates(u, atoms, epsilons_matrix, atom_types, n_atoms, cutoff, parallel):
    """
    Compute the list of pairs of atoms separated by a distance less than epsilon_r for each frame of the trajectory

    Parameters
    ----------
    u : MDAnalysis.Universe
        The universe object of the trajectory file
    atoms : list of MDAnalysis.AtomGroup
        The wanted atoms (atoms_A, atoms_B, atoms_C)
    epsilons_matrix : numpy.ndarray (n_atoms, n_atoms)
        Distances between atoms to be considered as a pair
    n_atoms : int
        The number of types of atoms considered in the analysis
    cutoff : int
        Number of frames to be skipped at the beginning of the trajectory
    parallel : bool
        If True, the computation will be done in parallel

    Returns
    -------
    aggregates : numpy.ndarray (n_frames - cutoff, n_atoms, 2)
        1st dimension: frame
        2nd dimension: index of the aggregate
        3rd dimension: number of atoms of type A, number of atoms of type B, ..., number of atoms of type Z in the aggregate

    Examples
    --------
    >>> aggregates = computeAggregates(u, atoms, epsilons_matrix, atom_types, n_atoms, cutoff, parallel)
    >>> aggregates.shape
    (n_frames - cutoff, n_aggregates_max, n_atoms)
    (900, 1000, 2)
    >>> aggregates[0]
    array([[0, 0, ..., 0, 0], [0, 0, ..., 0, 0], [0, 0, ..., 0, 0], ..., [2, 1, ..., 0, 0]], dtype=int32)
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
        a = computeAgg(atoms, epsilons_matrix, atom_types, n_atoms, last, i)
        return a
    n_frames = u.trajectory.n_frames
    #n_atoms = atoms[0].n_atoms + atoms[1].n_atoms
    total_number_of_atoms = np.sum([atoms[i].n_atoms for i in range(n_atoms)])
    # The output is a 3D matrix (n_frames - cutoff, n_atoms, 3) where the last dimension is used to store the number of atoms in the aggregate
    aggregates = np.zeros((n_frames - cutoff, total_number_of_atoms, n_atoms), dtype=np.int32) - 1

    if parallel == True:
        a_tot = Parallel(n_jobs=-1)(delayed(computeAgg_step)(i) for i in tqdm.tqdm(range(cutoff, n_frames), desc='Aggregates list'))
    else:
        a_tot = [computeAgg_step(i) for i in tqdm.tqdm(range(cutoff, n_frames), desc='Aggregates list')]
    for i in tqdm.tqdm(range(cutoff, n_frames), desc='Putting aggregates in the matrix'):
        aggregates[i - cutoff][total_number_of_atoms-a_tot[i - cutoff].shape[0]:] = a_tot[i - cutoff]
    #print("aggregates[-1]: " + str(aggregates[-1][-100:]))
    return aggregates

def computeAgg(atoms, epsilons_matrix, atom_types, n_atoms, last, frame_current):
    """ 
    Compute the list of aggregates for a given frame

    Parameters
    ----------
    atoms : list of MDAnalysis.AtomGroup
        The wanted atoms (atoms_A, atoms_B, atoms_C)
    epsilons_matrix : numpy.ndarray (n_atoms, n_atoms)
        Distances between atoms to be considered as a pair
    n_atoms : int
        The number of types of atoms considered in the analysis
    last : bool
        If True, we are at the last frame of the trajectory
    frame_current : int
        The index of the current frame

    Returns
    -------
    agg : numpy.ndarray
        The list of aggregates for the given frame

    Examples
    --------
    >>> agg = computeAgg(atoms, epsilons_matrix, atom_types, n_atoms, last, frame_current)
    >>> agg.shape
    (n_aggregates_max, n_atoms)
    >>> agg
    array([[0, 0, ..., 0, 0], [0, 0, ..., 0, 0], [0, 0, ..., 0, 0], ..., [2, 1, ..., 0, 0]], dtype=int32)
    """
    ## We define a group of atoms containing all the atoms
    for i in range(n_atoms):
        if i == 0:
            atoms_all = atoms[i]
        else:
            atoms_all += atoms[i]
    n_pairs = int(n_atoms * (n_atoms + 1) / 2)
    pairs = []
    # We will start with the pairs of atoms of the same type
    for i in range(n_atoms):
        pair_type = mda.lib.nsgrid.FastNS(epsilons_matrix[i][i], atoms_all.positions, atoms_all.universe.dimensions, pbc=True).self_search().get_pairs()
        pairs.append(pair_type)
    # We will now consider the pairs of atoms of different types
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pair_type = mda.lib.nsgrid.FastNS(epsilons_matrix[i][j], atoms_all.positions, atoms_all.universe.dimensions, pbc=True).self_search().get_pairs()
            pairs.append(pair_type)
    # Now, we have to do the difficult part of the exercise: for each list of pairs, we have to remove the pairs that are not relevant
    # For exemple, the list of pairs of atoms of type AA should only contain pairs of atoms of type AA
    # The list of pairs of atoms of type BB should only contain pairs of atoms of type BB
    # The list of pairs of atoms of type AB should only contain pairs of atoms of type AB
    # Etc. IT IS VERY IMPORTANT TO DO THAT
    
    # Pairs of atoms of same type
    for i in range(n_atoms):
        # We create a mask to remove the pairs that are not relevant (all A[i]: A[i,0] != A[i,1] and A[i,0] != type)
        type = atom_types[i].split(' ')[1]
        print("type : {}".format(type))
        print("indice : {}".format(i))
        mask = np.ma.masked_equal(atoms_all[pairs[i][:, 0]].types, atoms_all[pairs[i][:, 1]].types).mask
        pairs[i] = pairs[i][mask]
        try:
            mask = np.ma.masked_equal(atoms_all[pairs[i][:, 0]].types, type).mask
        except IndexError:
            pass
        pairs[i] = pairs[i][mask]
        print("atoms_all[pairs[i]].types : {}".format(atoms_all[pairs[i]].types))
        print()
        print()
        print()
    
    # Pairs of atoms of different types. We have corrected all the pairs from 0 to n_atoms - 1. 
    # We now have to correct the pairs from n_atoms to n_pairs - 1
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # We create a mask to remove the pairs that are not relevant (all A[i]: A[i,0] != A[i,1] and A[i,0] != type)
            type_i = atom_types[i].split(' ')[1]
            type_j = atom_types[j].split(' ')[1]
            
            # We should NOT to this test if type_i == type_j because:
            # 1. We are dealing with atoms belonging to the same type so the test is useless
            # 2. If we do this test on pairs having the same type, but different atom types, we will remove all the pairs
            if type_i == type_j:
                continue
            indice = int(n_atoms + i * n_atoms - i * (i + 1) / 2 + j - i - 1)
            print("type_i : {}".format(type_i))
            print("type_j : {}".format(type_j))
            print("indice : {}".format(indice))
            # We create a mask to remove the pairs that are not relevant (all A[i]: A[i,0] == A[i,1])
            mask = np.ma.masked_not_equal(atoms_all[pairs[indice][:, 0]].types, atoms_all[pairs[indice][:, 1]].types).mask
            pairs[indice] = pairs[indice][mask]
            # We create a mask to remove the pairs that are not relevant
            # (all A[i]: (A[i,0] != type_i and A[i,1] != type_j) or (A[i,0] != type_j and A[i,1] != type_i))
            # Attention, the two conditions have to be evaluated at the same time
            print()
            try:
                mask_ij_res_i_res_j = np.ma.masked_equal(atoms_all[pairs[indice][:, 0]].types, type_i).mask
                mask_ij_res_j_res_i = np.ma.masked_equal(atoms_all[pairs[indice][:, 0]].types, type_j).mask
                mask_ji_res_i_res_j = np.ma.masked_equal(atoms_all[pairs[indice][:, 1]].types, type_i).mask
                mask_ji_res_j_res_i = np.ma.masked_equal(atoms_all[pairs[indice][:, 1]].types, type_j).mask
                mask_ij = (mask_ij_res_i_res_j) & (mask_ji_res_j_res_i) | (mask_ij_res_j_res_i) & (mask_ji_res_i_res_j)
                pairs[indice] = pairs[indice][mask_ij]
            except IndexError:
                pass
            print("atoms_all[pairs[i]].types : {}".format(atoms_all[pairs[indice]].types))
            print()
            print()
            print()

    # We put the pairs together
    try:
        pairs = np.concatenate(pairs)
        print("Pairs : {}".format(pairs))
    except ValueError:
        # One of the lists is empty, we remove it and concatenate again
        pairs = np.concatenate([pair for pair in pairs if pair.size != 0])
        print("We have removed an empty list of pairs")
        print("Pairs : {}".format(pairs))
    #print("Unique pairs : {}".format(np.unique(atoms_all[pairs].types, axis=0)))
    #for i in range(len(pairs)):
    #    print("Pair {} : {}".format(i, atoms_all[pairs[i]]))

    # We create a graph
    G = nx.Graph()
    G.add_edges_from(pairs)
    # We compute the number of atoms in each aggregate
    agg = np.zeros((nx.number_connected_components(G), n_atoms), dtype=np.int32)
    print("Number of connected components : {}".format(nx.number_connected_components(G)))
    print("Connected components : {}".format(nx.connected_components(G)))
    j = 0
    str_type = []
    for i in range(n_atoms):
        type = atom_types[i].split(' ')[1]
        str_type.append("x.types == '" + type + "'")
    str_type = np.array(str_type)

    for component in nx.connected_components(G):
        x = atoms_all[list(component)]
        print("Component : {}".format(x))
        # We sort x according to the resid
        x = x[np.argsort(x.resids)]
        # We remove in x the duplicates according to the resid
        #x = x[np.unique(x.resids, return_index=True)[1]]
        # If at the end, the component contains only one atom, we remove it
        if len(x) == 1:
            continue
        else:
            # We have, for each of the connected components, to remove the atoms that are if their resids is already in the list
            # We search all the uniques resid in the component 
            list_resid = []
            for i in range((len(x))):
                list_resid.append(x[i].resid)
            list_resid = np.array(list_resid)
            list_resid = np.unique(list_resid)
            for i in range(n_atoms):
                agg[j][i] = np.where(eval(str_type[i]))[0].shape[0]
            j += 1
    if last == True:
        try:
            os.remove("check/aggregates_types_frame_" + str(frame_current) + ".txt")
        except FileNotFoundError:
            pass
        j = 0
        for component in nx.connected_components(G):
            x = atoms_all[list(component)]
            # We sort x according to the resid
            x = x[np.argsort(x.resids)]
            # We remove in x the duplicates according to the resid
            x = x[np.unique(x.resids, return_index=True)[1]]
            if len(x) == 1:
                continue
            else:
                j += 1
                # We will save this information in a file in order to be able to check it
                resid_list = []
                with open("check/aggregates_types_frame_" + str(frame_current) + ".txt", "a") as f:
                    f.write("Component {}\n".format(j))
                    for i in range(x.n_atoms):
                        f.write("{}\n".format(x[i]))
                        resid_list.append(x[i].resid)
                    # I want to print a line like following:
                    # resid resids_list[0] resids_list[1] ... resids_list[-1]
                    # I have to convert resid_list to a string
                    resid_list = np.array(resid_list)
                    resid_list = np.array2string(resid_list, separator=' ')
                    f.write("resid ")
                    f.write(resid_list[1:-1])
                    f.write("\n")
                    f.write("\n")
    return agg

def computeDistribution(aggregates_list, atoms, n_atoms ,parallel):
    """ 
    Compute the distribution of aggregates from the list of aggregates

    Parameters
    ----------
    aggregates_list : numpy.ndarray (n_frames - cutoff, n_atoms, 1)
        The list of aggregates
    atoms : MDAnalysis.AtomGroup
        The wanted atoms
    n_atoms : int
        The number of types of atoms considered in the analysis
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
    >>> distribution = computeDistribution(aggregates_list, atoms, n_atoms, parallel)
    """

    # We remove the last column of the matrix aggregates_list
    #aggregates_list = aggregates_list[:, :, :-1]
    #print(aggregates_list.shape)
    #print("aggregates_list[-1]: " + str(aggregates_list[-1][-100:]))

    # We compute the number of frames
    n_frames = aggregates_list.shape[0]
    # We compute the number of aggregates
    n_aggregates_max = np.max(aggregates_list)
    print("Maximum number of aggregates : {}".format(n_aggregates_max))
    # We compute the distribution of aggregates for each frame
    distribution = np.zeros((n_frames, n_aggregates_max, n_aggregates_max), dtype=np.int32)
    def computeDistribution_step(t):
        """
        Compute the distribution of aggregates for a given frame t

        Parameters
        ----------
        t : int
            The index of the frame

        Returns
        -------
        distribution_t : numpy.ndarray (n_aggregates_max, n_aggregates_max, ..., n_aggregates_max) # n_atoms times
            The distribution of aggregates for the given frame
        """
        distribution_t = np.zeros(tuple([n_aggregates_max + 1 for i in range(n_atoms)]), dtype=np.int32)
        aggregates = aggregates_list[t]
        # We remove the entries that do not correspond to an aggregate (if all the entries of a single line are -1)
        #for i in range(aggregates.shape[0]):
        #    if aggregates[i][0] < 1 and aggregates[i][1] < 1:
        #        aggregates[i] = np.zeros(n_atoms, dtype=np.int32) - 1
        # We do that by creating a mask in order to avoid to do a loop
        aggregates = aggregates[~np.all(aggregates == -1, axis=1)]
        print('t : {}'.format(t))
        print('aggregates :')
        print(aggregates)
        v = np.unique(aggregates, return_counts=True, axis=0) 
        print('v : ')
        for i in range(0,len(v[0])):
            print(str(i) + ' ' + str(v[1][i]) + ' ' + str(v[0][i]))
        print()
        for i in range(0,v[0].shape[0]):
            distribution_t[tuple(v[0][i])] = v[1][i]
        print('distribution_t : ')
        print(distribution_t)
        return distribution_t
    if parallel == True:
        distribution = Parallel(n_jobs=-1)(delayed(computeDistribution_step)(t) for t in tqdm.tqdm(range(n_frames), desc='Aggregates distribution'))
    else:
        distribution = [computeDistribution_step(t) for t in tqdm.tqdm(range(n_frames), desc='Aggregates distribution')]
    #print("distribution[-1]: " + str(distribution[-1]))
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

def plotDistribution(file_name, atom_type, n_atoms, total_number_of_atoms):
    """
    Plot the mean distribution of aggregates
    There is as many plots as the number of types of atoms considered in the analysis
    If we have atoms of type A and B, we will have one plot:
    - d(n_A, n_B)
    If we have atoms of type A, B and C, we will have three plots:
    - d(n_A, n_B)
    - d(n_A, n_C)
    - d(n_B, n_C)
    Generally, we will have n_plots = n_atoms * (n_atoms - 1) / 2

    Parameters
    ----------
    file_name : str
        The name of the distribution file
    atom_type : list of str
        The type of the wanted atoms
    n_atoms : int
        The number of types of atoms considered in the analysis
    total_number_of_atoms : int
        The total number of atoms in the system
    """
    distribution = np.load("DATA_distribution/" + file_name + '.npy')
    distribution_mean = computeMeanDistribution(distribution)
    #print("distribution_mean: " + str(distribution_mean))
    print(distribution.shape)
    print(distribution_mean.shape)
    print("n_atoms : {}".format(n_atoms))

    # Linscale
    set_plot_style()
    if n_atoms > 1:
        n_plots = int(n_atoms * (n_atoms - 1) / 2)
        print("Plotting {} plots".format(n_plots))
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                try:
                    atom_type_A = atom_type[i]
                    atom_type_B = atom_type[j]
                    type_A = atom_type_A.split(' ')[1]
                    type_B = atom_type_B.split(' ')[1]
                    # distribution_mean is a n_atoms dimensional matrix. Each dimension corresponds to the number of atoms of a given type
                    # Example: distribution_mean[1, 2, 3, ... N] is the number of aggregates with 1 atom of type A, 2 atoms of type B, 3 atoms of type C, ..., N atoms of type Z
                    # We have to sum over all the dimensions except the two we want to plot
                    distribution_AB = np.sum(distribution_mean, axis=tuple([k for k in range(n_atoms) if k != i and k != j]))
                    set_plot_style()
                    plt.imshow(distribution_AB, cmap='jet', origin='lower')
                    plt.colorbar(aspect=5, pad=0.01, fraction=0.15 * 2, label=r'$N$')
                    plt.xlabel(r'$n_\mathrm{%s}$' % type_B)
                    plt.ylabel(r'$n_\mathrm{%s}$' % type_A)
                    if max(distribution_AB.shape) < 10:
                        plt.xticks(np.arange(0, distribution_AB.shape[0], 1))
                        plt.yticks(np.arange(0, distribution_AB.shape[1], 1))
                    else:
                        plt.xticks(np.arange(0, distribution_AB.shape[0], 2))
                        plt.yticks(np.arange(0, distribution_AB.shape[1], 2))
                    plt.tight_layout()
                    plt.savefig("FIGURES/" + file_name + '_' + type_A + '_' + type_B + '.png')
                    plt.close()
                    print("Plotting {}_{}_{}.png".format(file_name, type_A, type_B))
                except IndexError:
                    print("IndexError")
                    print("i, j : {}, {}".format(i, j))
                    print("n_atoms : {}".format(n_atoms))
    else:
        print("Plotting 1 plot since we have only one type of atoms")
        print("It cannot be a heatmap because we have only one dimension")
        print("Plotting histogram...")
        atom_type_A = atom_type[0]
        type_A = atom_type_A.split(' ')[1]
        distribution_A = np.sum(distribution_mean, axis=tuple([k for k in range(n_atoms) if k != 0]))
        # We add the number of monomers to the distribution. It corresponds to the total number of atoms of type A - the number of aggregates of size i time i
        n_monomers = total_number_of_atoms[0] - np.sum([distribution_A[i] * i for i in range(distribution_A.shape[0])])
        print("n_monomers : {}".format(n_monomers))
        distribution_A[1] += int(n_monomers)
        print(distribution_A.shape)
        print(distribution_A)
        set_plot_style()
        plt.bar(np.arange(distribution_A.shape[0]), distribution_A)
        plt.xlabel(r'$n_\mathrm{%s}$' % type_A)
        plt.ylabel(r'$N$')
        plt.tight_layout()
        plt.savefig("FIGURES/" + file_name + '_' + type_A + '_' + type_A + '.png')
        plt.close()

def printDistribution(file_name, atom_type, n_atoms, total_number_of_atoms):
    """
    Print the mean distribution of aggregates
    There is as many matrice to plot as the number of types of atoms considered in the analysis
    If we have atoms of type A and B, we will have one plot:
    - d(n_A, n_B)
    If we have atoms of type A, B and C, we will have three plots:
    - d(n_A, n_B)
    - d(n_A, n_C)
    - d(n_B, n_C)
    Generally, we will have n_matrices = n_atoms * (n_atoms - 1) / 2

    Parameters
    ----------
    file_name : str
        The name of the distribution file
    atom_type : list of str
        The type of the wanted atoms
    n_atoms : int
        The number of types of atoms considered in the analysis
    total_number_of_atoms : int
        The total number of atoms in the system
    """
    distribution = np.load("DATA_distribution/" + file_name + '.npy')
    distribution_mean = computeMeanDistribution(distribution)
    #print("distribution_mean: " + str(distribution_mean))
    print(distribution.shape)
    print(distribution_mean.shape)
    print("n_atoms : {}".format(n_atoms))

    if n_atoms > 1:
        n_matrices = int(n_atoms * (n_atoms - 1) / 2)
        #print("Plotting {} plots".format(n_plots))
        print("Printing {} matrices".format(n_matrices))
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                try:
                    atom_type_A = atom_type[i]
                    atom_type_B = atom_type[j]
                    type_A = atom_type_A.split(' ')[1]
                    type_B = atom_type_B.split(' ')[1]
                    # distribution_mean is a n_atoms dimensional matrix. Each dimension corresponds to the number of atoms of a given type
                    # Example: distribution_mean[1, 2, 3, ... N] is the number of aggregates with 1 atom of type A, 2 atoms of type B, 3 atoms of type C, ..., N atoms of type Z
                    # We have to sum over all the dimensions except the two we want to plot
                    distribution_AB = np.sum(distribution_mean, axis=tuple([k for k in range(n_atoms) if k != i and k != j]))
        # We print the distribution and we specify to which type i, j correspond
        # We also specify wich type has been summed over (if we have more than 2 types of atoms)
                    print("distribution_{}_{} :".format(type_A, type_B))
                    print('columns : {}'.format(type_B))
                    print('rows : {}'.format(type_A))
                    print(distribution_AB)
                    print()
                    print()
                except IndexError:
                    print("IndexError")
                    print("i, j : {}, {}".format(i, j))
                    print("n_atoms : {}".format(n_atoms))
    else:
        #print("Plotting 1 plot since we have only one type of atoms")
        #print("It cannot be a heatmap because we have only one dimension")
        #print("Plotting histogram...")
        print("Print the distribution of aggregates")
        print("It cannot be a matrix because we have only one dimension")
        print("Print the histogram...")
        atom_type_A = atom_type[0]
        type_A = atom_type_A.split(' ')[1]
        distribution_A = np.sum(distribution_mean, axis=tuple([k for k in range(n_atoms) if k != 0]))
        # We add the number of monomers to the distribution. It corresponds to the total number of atoms of type A - the number of aggregates of size i time i
        n_monomers = total_number_of_atoms[0] - np.sum([distribution_A[i] * i for i in range(distribution_A.shape[0])])
        print("n_monomers : {}".format(n_monomers))
        distribution_A[1] += int(n_monomers)
        print(distribution_A.shape)
        print(distribution_A)
        print("distribution_{}_{} :".format(type_A))
        print(distribution_A)
        print()
        print()





# main function
def main():
    # parse the command line arguments with argparse
    parser = argparse.ArgumentParser(description="Calculate the distribution of aggregates composed by given atoms of given types thanks by distances evaluation from Amber trajectory file")
    parser.add_argument("-c", "--compute", type=str, help="Compute the distribution of aggregates (if no, the distribution will be loaded, if it exists, in order to plot it)", choices=['yes', 'no'])
    parser.add_argument('-p', '--parallel', type=str, help='If True, the computation will be done in parallel', choices=['yes', 'no'])
    # cutoff can be int or float (if it is int: number of frames, if it is float: proportion of frames to be skipped)
    parser.add_argument('-ct', '--cutoff', type=int, help='If int>1: number of frames to be skipped at the beginning of the trajectory, if float<1=: proportion of frames to be skipped at the beginning of the trajectory')
    parser.add_argument('-na', '--n-atoms', type=int, help='Number of types of atoms considered in the analysis')
    parser.add_argument('-a', '--atom_type', type=str, nargs='+', help='The type of the wanted atoms: "type A" "type B" "type C", ...')
    parser.add_argument('-e', '--epsilon', type=float, nargs='+', help='Distance between atoms to be considered as a pair (beware of the order) : epsilon_AA, epsilon_BB, ..., espilon_ZZ, epsilon_AB, epsilon_AC, ..., epsilon_YZ')
    parser.add_argument('-tr', '--traj-file', type=str, help='Trajectory file or distribution file')
    parser.add_argument('-to', '--top-file', type=str, help='Topology file')
    parser.add_argument('-pl', '--plot', type=str, help='Plot the distribution of aggregates', choices=['yes', 'no'])
    parser.add_argument('-o', '--output', type=str, help='Output file')

    # create the directory DATA_distribution if it does not exist
    os.makedirs('DATA_distribution', exist_ok=True)
    os.makedirs('FIGURES', exist_ok=True)
    os.makedirs('check', exist_ok=True)


    args = parser.parse_args()

    # Parse the types
    atom_types = args.atom_type

    # Parse the distances
    epsilons = args.epsilon

    # Parse the number of atoms
    n_atoms = args.n_atoms

    cutoff = args.cutoff

    if args.compute == 'yes':
        # We compute the distribution of aggregates
        print()
        print("#" * 80)
        print("Compute the distribution of aggregates")
        print()
        # load the trajectory file
        print("Load the trajectory file...")
        u = loadUnivers(args.traj_file, args.top_file)
        print("Trajectory file loaded")
        print()

        #print("In the trajectory file, we found {} types:".format(np.unique(u.atoms.types).shape[0]))
        #count_type = 0
        #for i in range(np.unique(u.atoms.types).shape[0]):
        #    print(" -", np.unique(u.atoms.types)[i])
        #    count_type += 1
        #print()
        print("In the trajectory file, we found {} types of atoms:".format(np.unique(u.atoms.types).shape[0]))
        count_atom_type = 0
        for i in range(np.unique(u.atoms.types).shape[0]):
            print(" -", np.unique(u.atoms.types)[i])
            count_atom_type += 1
        print()

        print("Types of atoms asked for the analysis:")
        count_atom_type = 0
        for i in range(len(atom_types)):
            print(" -",atom_types[i])
            count_atom_type += 1
        print()
        if n_atoms > count_atom_type:
            print("You cannot consider more types than the number of types in the system")
            sys.exit()
        #if count_atom_type != n_atoms:
        if count_atom_type != n_atoms:
            print("You said that you wanted {} types of atoms but you gave {} types of atoms".format(n_atoms, count_atom_type))
            print("Please give the correct number of types of atoms")
            sys.exit()

        ## Check if all the names in atom_types contain the word 'type'
        #for i in range(n_atoms):
        #    if 'type' not in atom_types[i]:
        #        print("All the names in atom_types have to contain the word 'type'")
        #        sys.exit()

        print("Distances between atoms to be considered as a pair:")
        count_epsilons = 0
        n_epsilons_expected = n_atoms * (n_atoms + 1) / 2
        if len(epsilons) != n_epsilons_expected:
            print("You said that you wanted {} types of atoms, you should give {} distances (n * (n + 1) / 2), but you gave {} distances".format(int(n_atoms), int(n_epsilons_expected), int(len(epsilons))))
            print("Please give the correct number of distances (you can use 0.0 for the distances that you do not want to consider)")
            print("The expected input for the distances is:")
            counter = 1
            epsilons_matrix_theo = [[" " for i in range(n_atoms)] for j in range(n_atoms)]
            for i in range(n_atoms):
                epsilons_matrix_theo[i][i] = str("    e({})  ".format(count_epsilons))
                count_epsilons += 1
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    epsilons_matrix_theo[i][j] = str("    e({})  ".format(count_epsilons))
                    count_epsilons += 1
            print(" " * 10, end='')
            for i in range(args.n_atoms):
                print(" {:^10s}".format(atom_types[i].split(' ')[1] ), end='')
            print()
            for i in range(args.n_atoms):
                print(" {:^10s}".format(atom_types[i].split(' ')[1]), end='')
                for j in range(args.n_atoms):
                # If we are below the diagonal, we put a * instead of the value
                    if j < i:
                        print(" {:^10s}".format("*"), end='')
                    else:
                        print(" " + str(epsilons_matrix_theo[i][j]), end='')
                print()
            print("Where e(i) is the distance given at position i in the list of distances and corresponds to the distance between the atoms of types specified in the title of the column and the row")
            sys.exit()


        # We put the distances in a matrix (n_atoms, n_atoms)
        epsilons_matrix = np.zeros((args.n_atoms, args.n_atoms))
        # We fill the matrix (starting from the diagonal)
        k = 0
        for i in range(args.n_atoms):
            epsilons_matrix[i][i] = epsilons[k]
            k += 1
        # We fill the other half of the matrix
        for i in range(args.n_atoms):
            for j in range(i + 1, args.n_atoms):
                epsilons_matrix[i][j] = epsilons[k]
                k += 1
        # We mask the entries below the diagonal
        print(" " * 10, end='')
        for i in range(args.n_atoms):
            print(" {:^10s}".format(atom_types[i].split(' ')[1] ), end='')
        print()
        for i in range(args.n_atoms):
            print(" {:^10s}".format(atom_types[i].split(' ')[1]), end='')
            for j in range(args.n_atoms):
            # If we are below the diagonal, we put a * instead of the value
                if j < i:
                    print(" {:^10s}".format("*"), end='')
                else:
                    print(" {:^10.2f}".format(epsilons_matrix[i][j]), end='')
            print()
        print()
        print("You have chosen {} distances and {} types of atoms for the analysis".format(int(len(epsilons)), args.n_atoms))
        # get the wanted atoms
        print("Get the wanted atoms...")
        atom_type = args.atom_type
        atoms = getAtoms(u, atom_types, n_atoms)
        print("Wanted atoms got")
        print(atoms)
        total_number_of_atoms = [atoms[i].n_atoms for i in range(n_atoms)]
        print("Total number of atoms for the analysis: {}".format(total_number_of_atoms))

        # Number of frames to be skipped at the beginning of the trajectory: cutoff
        if cutoff <= 1:
            cutoff = np.int32(u.trajectory.n_frames) - 1
        print("Number of frames to be skipped at the beginning of the trajectory: {}".format(cutoff))
        print("Total number of frames: {}".format(u.trajectory.n_frames))
        
        # compute the list of aggregates
        print("Compute the list of aggregates...")
        if args.parallel == 'yes':
            parallel = True
        else:
            parallel = False
        #aggregates_list = computeAggregates(u, atoms, epsilons, atom_types, cutoff, parallel)
        aggregates_list = computeAggregates(u, atoms, epsilons_matrix, atom_types, n_atoms, cutoff, parallel)
        print("List of aggregates computed")
 
        # compute the distribution of aggregates
        print("Compute the distribution of aggregates...")
        distribution = computeDistribution(aggregates_list, atoms, n_atoms, parallel)
        print("Distribution of aggregates computed")
        print()
 
        # compute the mean distribution of aggregates over time
        print("Compute the mean distribution of aggregates over time...")
        distribution_mean = computeMeanDistribution(distribution)
        print("Mean distribution of aggregates over time computed")
        print()
 
        # save the distribution of aggregates
        print("Save the distribution of aggregates into the file {}...".format(args.output))
        output = args.output + "_" + args.traj_file.replace('.arc', '').replace('.txyz', '').replace('/', '_')
        saveDistribution(distribution, output)
        print("Distribution of aggregates saved")
        print("The data are saved in the directory DATA_distribution")
    elif args.compute == 'no':
        try:
            distribution = np.load("DATA_distribution/" + args.output + "_" + args.traj_file.replace('.lammpstrj', '').replace('.nc', '').replace('/', '_') + '.npy')
        except FileNotFoundError:
            print("Distribution file not found")
            print("Please compute the distribution first")
            sys.exit()
        # compute the mean distribution of aggregates over time
        distribution_mean = computeMeanDistribution(distribution)
    if args.plot == 'yes':
        file_name = args.output + "_" + args.traj_file.replace('.arc', '').replace('.txyz', '').replace('/', '_')
        plotDistribution(file_name, atom_types, n_atoms, total_number_of_atoms)
    elif args.plot == 'no':
        file_name = args.output + "_" + args.traj_file.replace('.arc', '').replace('.txyz', '').replace('/', '_')
        printDistribution(file_name, atom_types, n_atoms, total_number_of_atoms)

if __name__ == "__main__":
    main()
