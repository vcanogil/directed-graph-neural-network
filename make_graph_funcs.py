#%%
import networkx as nx
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from torch_geometric.data import DataLoader
from torch_geometric.utils import from_networkx
from molgraph.chemistry import Featurizer, features
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

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


def make_pyg_graph(input_str, label,format="InChI"):
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
        G.add_node(atom.GetIdx(), x = atom_encoder(atom))

    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        if float(begin_atom.GetProp('_GasteigerCharge')) > float(end_atom.GetProp('_GasteigerCharge')):
            G.add_edge(begin_atom.GetIdx(), end_atom.GetIdx())
                    #    bond_type=bond.GetBondType()
        else:
            G.add_edge(end_atom.GetIdx(), begin_atom.GetIdx())

    G = from_networkx(G)
    G.x = torch.FloatTensor(G.x)
    G.y = torch.LongTensor([label])
    G.string = input_str

    return G

def make_train_data(train_csv):
    df = pd.read_csv(train_csv)
    train_data = [make_pyg_graph(i, j) for i,j in zip(df['InChI'].values, df['covalent'].values)]
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    return train_loader

class GCNNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dense = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        x = torch.selu(self.conv1(x, edge_index))
        x = torch.selu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch=batch)
        x = self.dense(x)
        return torch.sigmoid(x)

# a = make_pyg_graph("InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H")
train_loader = make_train_data("./victor_data/test_data_all.csv")
# train_loader = make_train_data("./victor_data/training_data_all.csv")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #use CUDA if available
device = "cpu"

#%%
model = GCNNet(input_dim=train_loader.dataset[0].num_node_features,
               hidden_dim=10, output_dim=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4) #use Adam optimizer
criterion = torch.nn.BCELoss() #define loss
model.train()
def train():
    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data, data.batch).reshape(-1)  # Perform a single forward pass.
         loss = criterion(out, data.y.float())  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients
         print(loss)
train()

# with torch.no_grad():
#   y_pred = []
#   for g in train_data:
#     y_pred.append(model(g))

# y_pred
# %%
