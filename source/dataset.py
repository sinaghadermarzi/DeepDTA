
import dask.dataframe as dd
import numpy as np

def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind))) #+1

	for i, ch in enumerate(line[:MAX_SMI_LEN]):
		X[i, (smi_ch_ind[ch]-1)] = 1

	return X #.tolist()

def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind)))
	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i, (smi_ch_ind[ch])-1] = 1

	return X #.tolist()


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros(MAX_SMI_LEN)
	for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
		X[i] = smi_ch_ind[ch]

	return X #.tolist()

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros(MAX_SEQ_LEN)

	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i] = smi_ch_ind[ch]

	return X #.tolist()




class dta_dataset:
	def __init__(self, conf, csv_path=None,unlabeled=False):
		self.conf = conf
		if csv_path != None:
			self.read_from_csv(csv_path,unlabeled=unlabeled)
		pass

	def read_from_csv(self,csv_path,unlabeled=False):
		self.df = dd.read_csv(csv_path, dtype="str", error_bad_lines=False).compute()
		if unlabeled:
			self.df["affinity[-log10(Ki/10e9)]"]= [None] * len(self.df)
		else:
			self.df= self.df.loc[self.df["affinity[-log10(Ki/10e9)]"]!= "none",:]
			self.df.set_index(np.array(list(range(len(self.df)))),inplace=True)

	def get_num_pairs(self):
		return len(self.df)

	def get_objects(self, index_list, with_label= True):
		cids = self.df.loc[index_list,"compound_pubchem_cid"].values
		pids = self.df.loc[index_list, "protein_uniprot_id"].values
		seqs  = self.df.loc[index_list, "protein_sequence"].values
		SMILEs = self.df.loc[index_list, "compound_SMILES"].values
		affs = self.df.loc[index_list, "affinity[-log10(Ki/10e9)]"].astype(float).values
		num_obj = len(index_list)
		X_compound = [None] * num_obj
		X_protein = [None] * num_obj
		if with_label:
			for i in range(num_obj):
				X_compound[i] = label_smiles(SMILEs[i], self.conf.max_smi_len, self.conf.CHARISOSMISET)
				X_protein[i] = label_sequence(seqs[i], self.conf.max_seq_len, self.conf.CHARPROTSET)
		else:
			for i in range(num_obj):
				X_compound[i] =one_hot_smiles(SMILEs[i], self.conf.max_smi_len, self.conf.CHARISOSMISET)
				X_protein[i] = one_hot_sequence(seqs[i], self.conf.max_seq_len, self.conf.CHARPROTSET)
		Y = affs
		return np.array(X_compound), np.array(X_protein), Y , cids, pids




