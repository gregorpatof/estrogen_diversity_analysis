from omltk.fingerprints import get_fp_from_smiles, get_tanimoto_array
import numpy as np
from multiprocessing import Pool
import pandas as pd



if __name__ == "__main__":
    chembl_df = pd.read_csv("/home/mailhoto/projects/rrg-mailhoto/share/chembl36/all_chembl36_part1.csv",
                            sep=";", quotechar='"', on_bad_lines='skip')

