import json
import os
import sys

import numpy as np

from Bio.PDB import *

hl = "../Data/HL_67042.17.json"
PDB_PATH = "../Data/hl_pdbs/"

def calc_residue_dist(res_one, res_two):
    res_one_array = np.array(res_one["P"].get_vector().get_array())
    res_two_array = np.array(res_two["P"].get_vector().get_array())
    diff_vector = res_one_array - res_two_array

    return np.sqrt(np.sum(diff_vector ** 2))

def calc_dist_matrix(chain_one, chain_two):
    distmat = np.zeros((len(chain_one), len(chain_two)), dtype=np.float32)

    for row, res_one in enumerate(chain_one):
        for col, res_two in enumerate(chain_two):
            distmat[row, col] = calc_residue_dist(res_one, res_two) 
    return distmat

def get_cmap(pdbid):
    pass


def struc_align(ref, nonref, ref_res, nonref_res):
    """
        Align models at given residues.
    """
    ref_atoms = [r["P"] for r in ref_res]
    nonref_atoms = [r["P"] for r in nonref]
    super_imposer = Superimposer()
    super_imposer.set_atoms(ref_atoms, nonref_atoms )
    super_imposer.apply(nonref.get_atoms())

    pass
def get_motif_cmaps(motifs):
    #get motifs json
    pdbs = [] 

    with open(motifs, 'r') as m:
        mot = json.load(m)

    #for each motif, load PDB
    maps = []
    for i, struc in enumerate(sorted(mot['alignment'].keys())):

        struc_nucs_info = mot['alignment'][struc]
        # print(struc_nucs_info)
        # print(struc_nucs_info)
        pdbid = struc_nucs_info[0].split('|')[0]
        model = struc_nucs_info[0].split('|')[1]
        chain = struc_nucs_info[0].split('|')[2]

        motif_start = int(struc_nucs_info[0].split('|')[4])
        motif_end = int(struc_nucs_info[-1].split('|')[4])

        if motif_end - motif_start != 5:
            continue

        parser = PDBParser(QUIET=True)
        struc_path = os.path.join(PDB_PATH, pdbid.lower() + ".pdb")
        try:
            structure = parser.get_structure('NIG', struc_path)
        except FileNotFoundError as e:
            print(e)
            continue
        print(pdbid)
        print(motif_start, motif_end)
        residues = structure[int(model)-1][chain]
        motif_residues = []
        for res in residues:
            _, pos, _ = res.get_id()
            if motif_start <= pos <= motif_end:
                motif_residues.append(res)


        contact_map = calc_dist_matrix(motif_residues, motif_residues)
        maps.append(contact_map.flatten())
    return maps

if __name__ == "__main__":
    get_motif_cmaps(hl)
    pass
