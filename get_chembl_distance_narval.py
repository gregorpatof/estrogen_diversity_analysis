from omltk.fingerprints import get_fp_from_smiles, get_tanimoto_array, compute_morgan_fps
import numpy as np
from multiprocessing import Pool
import pandas as pd
from rdkit import Chem

CHEMBL_FPS_PATH = "data/chembl_fps.npy"

_CHEMBL_FPS = None  # populated once per worker; never pickled across the boundary

def _init_worker(path):
    global _CHEMBL_FPS
    _CHEMBL_FPS = np.load(path, mmap_mode="r")

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


def get_closest_chembl_chunk(query_fps):
    sim = get_tanimoto_array(_CHEMBL_FPS, query_fps)   # (n_query, n_chembl)
    closest = np.argmax(sim, axis=1)
    td = 1.0 - sim[np.arange(sim.shape[0]), closest]
    return closest.astype(np.int32), td.astype(np.float32)

def get_closest_chembl_df():
    chembl_smiles, chembl_ids = get_chembl_smiles_names()   # parent only

    with Pool(64, initializer=_init_worker, initargs=(CHEMBL_FPS_PATH,)) as p:
        for target in ("6B0F", "7UJ7"):
            filename = f"data/{target}_raw.smi"
            smiles_list = get_smiles_list(filename)
            fps, names = compute_morgan_fps(filename)
            chunks = np.array_split(fps, 1000)              # ~70 queries/chunk
            results = p.map(get_closest_chembl_chunk, chunks)

            with open(f"dataframes/{target}_chembl_closest.df", "w") as fout:
                fout.write("smiles id chembl_closest_td closest_smiles closest_id\n")
                count = 0
                for closest_idx, td in results:
                    for ci, d in zip(closest_idx, td):
                        fout.write(
                            f"{smiles_list[count]} {names[count]} "
                            f"{d} {chembl_smiles[ci]} {chembl_ids[ci]}\n"
                        )
                        count += 1


if __name__ == "__main__":
    # make_chembl_smi()
    # compute_valid_smiles()
    # compute_chembl_fps()
    get_closest_chembl_df()


