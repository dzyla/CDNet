import os

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from Bio.PDB import PDBList, PDBParser, DSSP
import random
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

path = 'data/'

def read_header(file):

    header = pd.read_csv(file, nrows=17, sep='\t', names=['description', 'data'])

    return header.iloc[15]['data']



def get_ss(file):

    p = PDBParser()
    pdbl = PDBList()
    try:

        p.get_structure("structure", pdbl.retrieve_pdb_file(file, file_format='pdb'))

        f = os.popen('find /mnt/c/linux/python_kurs/deep_learning/CD_net/ -iname *{}*.ent'.format(file))
        path = f.read().replace('\n', '')

        structure = p.get_structure("", path)
        model = structure[0]
        dssp = DSSP(model, path)

        ss_holder = {'-':0, 'T':0, 'S':0, 'H':0, 'B':0, 'E':0, 'G':0, 'I':0}

        for entry in dssp:
            if entry[2] not in ss_holder:
                ss_holder[entry[2]] = 1
            else:
                ss_holder[entry[2]] += 1

        sorted(ss_holder.keys())

        return ss_holder

    except FileNotFoundError:
        pass

cd_files = glob.glob(path+'*.gen', recursive=True)
output_file = []

for filename in cd_files:

    pdb_id = read_header(filename).replace('\n','')

    print(pdb_id)

    if pdb_id != 'No data provided':


        ss_structure = get_ss(pdb_id)

        ss_list = []

        try:
            for key, value in ss_structure.items():
                temp = value
                ss_list.append(temp)

            ss_list = ss_list / np.sum(ss_list)
            ss_list = np.array(ss_list)

            dataframe = pd.read_csv(filename, header=18, sep='\t', index_col=False,
                                    names=['nm', 'mdeg', '??', '4', '5', '6', '7'])


            holder_list = [pdb_id, ss_list, dataframe['nm'].values.astype('float32'), dataframe['mdeg'].values.astype('float32')]


            print(holder_list)
            output_file.append(holder_list)

            #print(output_file)
            #plt.plot(dataframe['nm'], dataframe['mdeg'])
            #plt.show()

            np.save('CD_output.npy', np.array(output_file, dtype=object))


        except AttributeError or ValueError:
            print('?')
            pass






