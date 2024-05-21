#%%
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

# Load the molecule
mol = Chem.MolFromSmiles('COCCCCl')

# Compute Gasteiger charges
rdPartialCharges.ComputeGasteigerCharges(mol)

# Create a directed graph
G = nx.DiGraph()

# Add nodes for atoms
for atom in mol.GetAtoms():
    G.add_node(atom.GetIdx(),
               atomic_num=atom.GetAtomicNum(),
               formal_charge=atom.GetFormalCharge(),
               chiral_tag=atom.GetChiralTag(),
               hybridization=atom.GetHybridization(),
               num_explicit_hs=atom.GetNumExplicitHs(),
               is_aromatic=atom.GetIsAromatic(),
               gasteiger_charge=atom.GetProp('_GasteigerCharge'))

# Add edges for bonds
for bond in mol.GetBonds():
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    if float(begin_atom.GetProp('_GasteigerCharge')) > float(end_atom.GetProp('_GasteigerCharge')):
        G.add_edge(begin_atom.GetIdx(), end_atom.GetIdx(), bond_type=bond.GetBondType())
    else:
        G.add_edge(end_atom.GetIdx(), begin_atom.GetIdx(), bond_type=bond.GetBondType())
#%%
import matplotlib.pyplot as plt

for atom in mol.GetAtoms():
    print(atom.GetIdx(), atom.GetSymbol(), atom.GetProp("_GasteigerCharge"))
# Draw the graph
nx.draw(G, with_labels=True)

# Show the plot
plt.show()
# %%
from torch_geometric.utils import from_networkx
a = from_networkx(G)

# %%
