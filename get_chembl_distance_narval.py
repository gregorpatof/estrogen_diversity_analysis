from omltk.fingerprints import get_fp_from_smiles, get_tanimoto_array
import numpy as np
from multiprocessing import Pool
import pandas as pd
from rdkit import Chem


def make_chembl_smi():
    fout = open("/home/mailhoto/projects/rrg-mailhoto/share/chembl36/all_smiles.smi", "w")
    chembl_df = pd.read_csv(f"/home/mailhoto/projects/rrg-mailhoto/share/chembl36/all_chembl36.csv",
                            sep=";", quotechar='"', error_bad_lines=False, warn_bad_lines=True)
    cids = [x for x in chembl_df["ChEMBL ID"].values]
    smiles_list = [x for x in chembl_df["Smiles"].values]
    for smiles, cid in zip(smiles_list, cids):
        if 'nan' in str(smiles):
            continue
        fout.write(f"{smiles} {cid}\n")
    fout.close()

def compute_valid_smiles():
    smiles_list, names_list = get_chembl_smiles_names(path="/home/mailhoto/projects/rrg-mailhoto/share/chembl36/all_smiles.smi")
    with Pool(64) as p:
        is_valid = p.map(is_valid_smiles, smiles_list)
    fout = open("/home/mailhoto/projects/rrg-mailhoto/share/chembl36/all_smiles_valid.smi", "w")
    for i, v in enumerate(is_valid):
        if v:
            fout.write(f"{smiles_list[i]} {names_list[i]}\n")
    fout.close()



def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False

def get_chembl_smiles_names(path='/home/mailhoto/projects/rrg-mailhoto/share/chembl36/all_smiles_valid.smi'):
    with open(path) as f:
        lines = f.readlines()
    smiles_list = []
    names_list = []
    for line in lines:
        smiles, name = line.split()
        smiles_list.append(smiles)
        names_list.append(name)
    return smiles_list, names_list


def compute_chembl_fps():
    smiles_list, names_list = get_chembl_smiles_names()
    with Pool(64) as p:
        results = p.map(get_fp_from_smiles, smiles_list)
    fps = np.array(results)
    np.save('data/chembl_fps.npy', fps)


if __name__ == "__main__":
    # make_chembl_smi()
    # compute_valid_smiles()
    compute_chembl_fps()

