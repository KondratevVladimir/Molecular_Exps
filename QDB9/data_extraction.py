from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np
df_train, df_test = train_test_split(pd.read_csv("gdb9_prop_smiles.csv", index_col="Unnamed: 0"), test_size = 0.4, shuffle=True, random_state=42)

print(df_train.describe())
print(df_train.head(10))
#def fun(x):
#    if Chem.MolFromSmiles(x) != None:
#        return x
#    return None

#smiles = df_train.smiles
#smiles = smiles.apply(lambda x: fun(x))
#smiles = smiles[smiles.isna() == False].sort_index()
#smiles.to_csv(r'./data/train.txt', header=None, index=None, sep='\n')

#smiles = df_test.smiles
#smiles = smiles.apply(lambda x: fun(x))
#smiles = smiles[smiles.isna() == False].sort_index()
#smiles.to_csv(r'./data/test.txt', header=None, index=None, sep='\n')
#print(smiles.head(20))