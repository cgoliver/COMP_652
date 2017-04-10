import json
import os
import sys

from Bio.PDB import *

hl = "../Data/HL_67042.17.json"
PDB_PATH = "../Data/hl_pdbs/"

def get_cmap(pdbid):
    pass


def load_cmaps(motifs):
    #get motifs json
    pdbs = [] 

    with open(motifs, 'r') as m:
        mot = json.load(m)

    #for each motif, load PDB
    for i, struc in enumerate(sorted(mot['alignment'].keys())):

        struc_nucs_info = mot['alignment'][struc]
        print(struc_nucs_info)
        # print(struc_nucs_info)
        pdbid = struc_nucs_info[0].split('|')[0]
        model = struc_nucs_info[0].split('|')[1]
        chain = struc_nucs_info[0].split('|')[2]

        motif_start = struc_nucs_info[0].split('|')[4]
        motif_end = struc_nucs_info[-1].split('|')[4]

        print(chain, motif_start, motif_end)
        parser = PDBParser()
        struc_path = os.path.join(PDB_PATH, pdbid.lower() + ".pdb")
        
        try:
            structure = parser.get_structure(model, struc_path)
            extract(structure, chain, int(motif_start), int(motif_end),\
                'test{0}.pdb'.format(i))
        except Exception as e:
            print(e)
            continue

    return mot

if __name__ == "__main__":
    load_cmaps(hl)
    pass
