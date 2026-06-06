from omltk.fingerprints import get_fp_from_smiles, get_tanimoto_array
import numpy as np
from multiprocessing import Pool
import pandas as pd

def make_chembl_smi():
    fout = open("/home/mailhoto/projects/rrg-mailhoto/share/chembl36/all_smiles.smi", "w")
    chembl_df = pd.read_csv(f"/home/mailhoto/projects/rrg-mailhoto/share/chembl36/all_chembl36.csv",
                            sep=";", quotechar='"', error_bad_lines=False, warn_bad_lines=True)
    cids = [x for x in chembl_df["ChEMBL ID"].values]
    smiles_list = [x for x in chembl_df["Smiles"].values]
    for smiles, cid in zip(smiles_list, cids):
        fout.write(f"{smiles} {cid}\n")
    fout.close()


if __name__ == "__main__":
    make_chembl_smi()

