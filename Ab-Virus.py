
import numpy as np
from rdkit import Chem
import multiprocessing
import logging
import pandas as pd
# The code for featurization was borrowed from deepchem. Please refer https://deepchem.io for more information
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from xgboost import plot_importance
from xgboost import plot_importance
import matplotlib.pyplot as plt
from matplotlib import pyplot
 
def one_of_k_encoding(x, allowable_set):
  if x not in allowable_set:
    raise Exception("input {0} not in allowable set{1}:".format(
        x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))
 
 
def one_of_k_encoding_unk(x, allowable_set):
  """Maps inputs not in the allowable set to the last element."""
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))
 
 
 
def atom_features(atom,explicit_H=False,use_chirality=False):
  from rdkit import Chem
  results = one_of_k_encoding_unk(
    atom.GetSymbol(),
    [
      'C',
      'N',
      'O',
      'S',
      'H',  # H?
      'Unknown'
    ]) + one_of_k_encoding(atom.GetDegree(),
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
            one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
            [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
            one_of_k_encoding_unk(atom.GetHybridization(), [
              Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
              Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                  SP3D, Chem.rdchem.HybridizationType.SP3D2
            ]) + [atom.GetIsAromatic()]
  # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
  if not explicit_H:
    results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                              [0, 1, 2, 3, 4])
  if use_chirality:
    try:
      results = results + one_of_k_encoding_unk(
          atom.GetProp('_CIPCode'),
          ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
      results = results + [False, False
                          ] + [atom.HasProp('_ChiralityPossible')]

  return np.array(results) 
 
def mol2vec(mol): # creates features
    atoms = mol.GetAtoms()
    node_f= [atom_features(atom) for atom in atoms]
    return node_f

df= pd.read_csv('2F5_4E10_10E8_final.csv') # dataset file here

mol_virus = []
for i in range (len(df['FASTA_Virus'])):
    mol = Chem.MolFromFASTA(df['FASTA_Virus'].loc[i])
    mol_virus.append(mol)

mol_ab = []
for i in range (len(df['FASTA_Ab'])):
    mol = Chem.MolFromFASTA(df['FASTA_Ab'].loc[i])
    mol_ab.append(mol)


a = np.zeros((len(mol_ab),))
ab_feature = []
for i in range (len(mol_ab)):
    ab = mol2vec(mol_ab[i])
    for i in range(len(ab)):
    	if ((ab[i][29]==1)):
    		print(i)
    ab_feature.append(ab)

virus_feature = []
for i in range (len(mol_virus)):

    ab = mol2vec(mol_virus[i])
    for i in range(len(ab)):
    	if ((ab[i][29]==1)):
    		print(i)
    virus_feature.append(ab)

ab_adj = []
for i in range(len(mol_ab)):
    p1 = [Chem.rdmolops.GetAdjacencyMatrix(mol_ab[i])+np.eye(Chem.rdmolops.GetAdjacencyMatrix(mol_ab[i]).shape[0])]
    ab_adj.append(p1)

virus_adj = []
for i in range(len(mol_ab)):
    p2 = [Chem.rdmolops.GetAdjacencyMatrix(mol_virus[i])+np.eye(Chem.rdmolops.GetAdjacencyMatrix(mol_virus[i]).shape[0])]
    virus_adj.append(p2)

arr_vir_feature = []
for i in range (len(virus_feature)):
    arr1 = np.asarray(virus_feature[i])
    arr_vir_feature.append(arr1)


arr_ab_feature = []
for i in range (len(ab_feature)):
    arr1 = np.asarray(ab_feature[i])
    arr_ab_feature.append(arr1)

arr_virus_adj = []
for i in range (len(virus_adj)):
    arr_adj1 = np.asarray(virus_adj[i])
    arr_virus_adj.append(arr_adj1)

arr_ab_adj = []
for i in range (len(ab_adj)):
    arr_adj1 = np.asarray(ab_adj[i])
    arr_ab_adj.append(arr_adj1)

for i in range (len(arr_virus_adj)):
    arr_virus_adj[i] = arr_virus_adj[i].reshape(arr_vir_feature[i].shape[0],arr_vir_feature[i].shape[0])


for i in range (len(arr_ab_adj)):
    arr_ab_adj[i] = arr_ab_adj[i].reshape(arr_ab_feature[i].shape[0],arr_ab_feature[i].shape[0])


matmul_virus = []
for i in range (len(arr_virus_adj)):
    feature_virus = np.matmul(arr_virus_adj[i],arr_vir_feature[i]) # adjacency x feat
    matmul_virus.append(feature_virus)

matmul_ab = []
for i in range (len(arr_ab_adj)):
    feature_ab = np.matmul(arr_ab_adj[i],arr_ab_feature[i]) # adjacency x feat
    matmul_ab.append(feature_ab)


mean_virus = []
for i in range (len(matmul_virus)):
    mean1 = np.mean(matmul_virus[i],axis=0).reshape(37) # pooling
    mean_virus.append(mean1)

mean_ab = []
for i in range (len(matmul_ab)):
    mean1 = np.mean(matmul_ab[i],axis=0).reshape(37) # pooling
    mean_ab.append(mean1)

mean_virus_arr = np.asarray(mean_virus)

mean_ab_arr = np.asarray(mean_ab)

mean_final = []
for i in range (len(mean_ab)):
    mean = np.sum((mean_ab[i],mean_virus[i]),axis = 0)
    mean_final.append(mean)

mean_pass = np.asarray(mean_final)

X = mean_pass
y = df['Class'].values

kf = KFold(n_splits=5)
mean_acc = []
mean_roc = []
tra_idx = []
tes_idx = []
for train_index, test_index in kf.split(X):
  tra_idx.append(train_index)
  tes_idx.append(test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  model = RandomForestClassifier(max_depth=5, n_estimators=10) # change the classifier here
  model.fit(X_train, y_train)
# make predictions for test data
  y_pred = model.predict(X_test)
  predictions = [round(value) for value in y_pred]
# evaluate predictions
  accuracy = accuracy_score(y_test, predictions)
  mean_acc.append(accuracy)
  print("Accuracy: %.2f%%" % (accuracy * 100.0))
  score_roc = roc_auc_score(y_test, predictions)
  mean_roc.append(score_roc)
  print(roc_auc_score(y_test, predictions))

te_idx = np.asarray(tes_idx)
tr_idx = np.asarray(tra_idx)
print("Mean roc",np.mean(mean_roc))
accuracy_arr = np.asarray(mean_acc)
mean_acc = np.mean(accuracy_arr)
max_acc = np.max(accuracy_arr)
min_acc = np.min(accuracy_arr)
print(mean_acc)
print(max_acc)
print(min_acc)






