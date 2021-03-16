from copy import deepcopy
import numpy as np
import pickle
from collections import Counter
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import argparse
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--ambfile', required=True, type=str, help='path to the output file')
parser.add_argument('--ifile', required=True, type=str, help='path to the input file containing unique sudoku puzzles and their solutions')
parser.add_argument('--jsolve', required=True, type=str, help='path to Jsolve binary file')
parser.add_argument('--unqfile', default=None, type=str, help='path to the output file')
args = parser.parse_args()


def generate_query(datapoint,num_add=5):
    zero_ind = np.nonzero(datapoint["query"]==0)[0]
    nonzero_ind = np.nonzero(datapoint["query"])[0]
    new_cells_ind = np.random.choice(zero_ind,size=num_add,replace=False)
    query = deepcopy(datapoint["query"])
    query[new_cells_ind] = datapoint["target"][new_cells_ind]
    mask_essential = np.random.choice(nonzero_ind,1)
    query[mask_essential]=0
    return query

def generate_input(queries, inp_file):
    with open(inp_file,"w") as f:
        for query in queries:
            print("".join(map(str,query)).replace("0","."),file=f)

def get_output(output_file):
    with open(output_file,"r") as f:
        l = f.readlines()
    l = l[:-1]
    ret_set = []
    line_count = len(l)
    index = 0
    while(index<line_count):
        n = int(l[index])
        target_set = [np.array(list(x.strip())).astype(np.int8) for x in set(l[index+1:n+index+1])]
        index += n+1
        ret_set.append(target_set)
        if len(ret_set)%2000==1999:
            print("Read solutions for {} queries".format(len(ret_set)))
    return ret_set

# read unique solutions puzzles and generate queries to be solved by Jsolve
rrn_data = pd.read_csv(args.ifile,header=None)
sudoku_queries = np.array([np.array(list(x)).astype(np.int8) for x in rrn_data[0]])
sudoku_sols = np.array([np.array(list(x)).astype(np.int8) for x in rrn_data[1]])

z = [dict(query=sudoku_queries[i],target=sudoku_sols[i]) for i in range(len(rrn_data)) 
     if (len(np.nonzero(sudoku_queries[i])[0])<18)]
# upsample 17 and 18 given queries since many of them tend to get rejected
queries = [generate_query(x,i) for i in ([1]*15 + [2]*5 + list(range(1,19))*2) for x in z]
print("Generated {} queries".format(len(queries)))

# generate input for Jsolve
generate_input(queries,"temp.in")

# call Jsolve
print("Running Jsolve")
subprocess.check_output(args.jsolve+ " ./temp.in > ./temp.out", shell=True)
print("All queries solved")

# read Jsolve output 
target_set = get_output("temp.out")

dataset = [dict(query=queries[i], target_set=target_set[i], count=len(target_set[i])) for i in range(len(target_set)) if (len(target_set[i])<50)]

counter = dict(Counter([len(np.nonzero(x["query"])[0]) for x in dataset]))
for key in sorted(counter.keys()):
    print(key,"givens:",counter[key])
    
print("Dumping multi-solution data to",args.ambfile)
with open(args.ambfile,"wb") as f:
    pickle.dump(dataset,f)

if args.unqfile is not None:
    print("Dumping unique-solution data to",args.unqfile)
    ch = list(range(len(sudoku_queries)))
    np.random.shuffle(ch)
    unique_dataset = [dict(query=sudoku_queries[i], target_set=[sudoku_sols[i]], count=1) for i in ch]
    with open(args.unqfile,"wb") as f:
        pickle.dump(unique_dataset,f)