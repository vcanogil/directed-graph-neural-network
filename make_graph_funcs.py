#%%
import networkx as nx
import torch
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from torch_geometric.utils import from_networkx
from molgraph.chemistry import Featurizer, features

atom_encoder = Featurizer([
    features.Symbol(),
    features.TotalNumHs(),
    features.ChiralCenter(),
    features.Aromatic(),
    features.Ring(),
    features.Hetero(),
    features.HydrogenDonor(),
    features.HydrogenAcceptor(),
    features.RingSize(),
    features.GasteigerCharge()
])


def make_pyg_graph(input_str, format="InChI"):
    if format == "InChI":
        mol = Chem.MolFromInchi(input_str)
    elif format == "SMILES":
        mol = Chem.MolFromSmiles(input_str)
    else:
        raise ValueError("format kwarg needs to InChI or SMILES")

    if not mol:
        return None

    rdPartialCharges.ComputeGasteigerCharges(mol)
    G = nx.DiGraph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
        x = atom_encoder(atom)
        )

    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        if float(begin_atom.GetProp('_GasteigerCharge')) > float(end_atom.GetProp('_GasteigerCharge')):
            G.add_edge(begin_atom.GetIdx(), end_atom.GetIdx(), bond_type=bond.GetBondType())
        else:
            G.add_edge(end_atom.GetIdx(), begin_atom.GetIdx(), bond_type=bond.GetBondType())

    G = from_networkx(G)
    # G.x = torch.FloatTensor(G.x)

    return G

a = make_pyg_graph("CCO", format="SMILES")
# %%
a

# %%
