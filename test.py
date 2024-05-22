#%%
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
import numpy as np

# Load the molecule
mol = Chem.MolFromSmiles('CN1CCC[C@H]1c1cccnc1')

# Compute Gasteiger charges
rdPartialCharges.ComputeGasteigerCharges(mol)

# Create a directed graph
G = nx.DiGraph()

# Add nodes for atoms
for atom in mol.GetAtoms():
    G.add_node(atom.GetIdx(),
    x = np.array([atom.GetAtomicNum(),
         atom.GetFormalCharge(),
        #  atom.GetChiralTag(),
         atom.GetHybridization(),
         atom.GetNumExplicitHs(),
         atom.GetIsAromatic(),
         atom.GetProp('_GasteigerCharge')]))

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
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Sample array
vector = np.array([1, 2, 3, 1, 1, 8])

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the vector
encoded_vector = label_encoder.fit_transform(vector)

# If the maximum possible number is 10, we need to limit the encoded values
encoded_vector[encoded_vector >= 10] = 10

print(encoded_vector)


# %%
