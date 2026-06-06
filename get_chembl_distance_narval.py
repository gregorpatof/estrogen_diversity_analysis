from omltk.fingerprints import get_fp_from_smiles, get_tanimoto_array, compute_morgan_fps
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




def get_smiles_list(filename):
    with open(filename) as f:
        lines = f.readlines()
    smiles_list = []
    for line in lines:
        smiles = line.split()[0]
        smiles_list.append(smiles)
    return smiles_list


def get_closest_chembl_df():
    for target in ['6B0F', '7UJ7']:
        filename = f"data/{target}_raw.smi"
        smiles_list = get_smiles_list(filename)
        fps, names = compute_morgan_fps(filename)
        chunks = np.array_split(fps, 1000)
        with Pool(64) as p:
            results = p.map(get_closest_chembl_chunk, chunks)
        fout = open(f'dataframes/{target}_chembl_closest.df', 'w')
        fout.write('smiles id chembl_closest_td closest_smiles closest_id\n')
        count = 0
        for chunk in results:
            for data in chunk:
                fout.write(f'{smiles_list[count]} {names[count]} {" ".join([str(x) for x in data])}\n')
                count += 1
        fout.close()


def get_closest_chembl_chunk(fps):
    CHEMBL_SMILES, CHEMBL_IDS = get_chembl_smiles_names()
    CHEMBL_FPS = np.load('data/chembl_fps.npy')
    print('starting')
    data = []
    array = get_tanimoto_array(CHEMBL_FPS, fps)
    for row in array:
        closest = np.argmax(row)
        td = 1 - row[closest]
        data.append([td, CHEMBL_SMILES[closest], CHEMBL_IDS[closest]])
    print(data[-1])
    return data


if __name__ == "__main__":
    # make_chembl_smi()
    # compute_valid_smiles()
    # compute_chembl_fps()
    get_closest_chembl_df()


