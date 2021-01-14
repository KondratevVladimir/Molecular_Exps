from rdkit import Chem
smiles = "C1NC[C:2]123N[CH:2]2[NH:2][CH2:2]3"
with open('./QDB9/data/train.txt') as f:
    data = [line.strip("\r\n ").split()[0] for line in f]

mol = Chem.MolFromSmiles(smiles, sanitize=False)

Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)

print(Chem.MolToSmiles(mol), " ", smiles)