import numpy as np 
from numba import jit,prange
from itertools import combinations 
from Bio import Align
import string 
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import linecache


def map_strings(alignment_sequence,pdb_sequence):
    '''Returns a dictionary of {alignment_sequence : pdb_sequence}. 
    Does not currently handle merged sequence alignments (seq1->pdb1, seq2->pdb2) '''
    
    aligner = Align.PairwiseAligner()
    aligner.target_internal_open_gap_score = -10
    aligner.query_internal_extend_gap_score = -10
    alignment=aligner.align(pdb_sequence,alignment_sequence)
    aligned_dict = {}
    for pairs in range(len(alignment[0].aligned[0])):
        newdict = {x:y for x,y in zip(range( alignment[0].aligned[1][pairs][0], alignment[0].aligned[1][pairs][1] ), range( alignment[0].aligned[0][pairs][0], alignment[0].aligned[0][pairs][1]))}
        aligned_dict.update(newdict)
    return aligned_dict,alignment

def map_filter_DIpairs(alignment_dictionary,DI_pairs):
    mapped_di = []
    for pair in DI_pairs:
        if pair[1] - pair[0] > 4:
            try:
                newpair = [alignment_dictionary[pair[0]],alignment_dictionary[pair[1]]]
                newpair.append(pair[2])
                mapped_di.append(newpair)
            except:
                pass
    mapped_di = np.array(mapped_di)
    mapped_di[:,:2] = mapped_di[:,:2] + 1
    return mapped_di

def plot_top_pairs(pdb_pairs,mapped_DI_pairs,figure_size):
    if mapped_DI_pairs.shape[1] == 3: #ranks DI pairs for you
        sort_map = np.argsort(mapped_DI_pairs[:,2])[::-1]
    elif mapped_DI_pairs.shape[1] == 2: # you have provided ranked pairs with no DI scores
        sort_map = range(mapped_DI_pairs.shape[0])
    pair_count = mapped_DI_pairs.shape[0]

    # define true and false hits as separate arrays.
    mask_DI_true = np.zeros((len(pdb_pairs),3))
    mask_DI_false = np.zeros((len(pdb_pairs),3))
    count_true = 0
    count_false = 0
    got_hit = False

    for pair in mapped_DI_pairs[sort_map][:,:2]:
        for native_pair in pdb_pairs:
            if (native_pair == pair).all():
                mask_DI_true[count_true,:] = [pair[0],pair[1],1]
                count_true += 1
                got_hit = True
        if not got_hit:
            mask_DI_false[count_false,:] = [pair[0],pair[1],0]
            count_false += 1
        got_hit = False
    mask_DI_true[count_true:len(pdb_pairs),:] = [native_pair[0],native_pair[1],0]
    mask_DI_false[count_false:len(pdb_pairs),:] = [native_pair[0],native_pair[1],0]

    plot = figure(figsize=figure_size,dpi=400);
    _ = plt.grid(alpha=0.5,linestyle='--',linewidth='0.3');
    ax = plt.gca();
    ax.set_axisbelow(True);
    _ = plt.scatter(x=pdb_pairs[:,1],y=pdb_pairs[:,0],s=0.05,c='gray',marker='.',label='PDB contacts');
    _ = plt.scatter(x=pdb_pairs[:,0],y=pdb_pairs[:,1],s=0.05,marker='o',label='PDB contacts');
    _ = plt.scatter(x=mask_DI_false[:,1],y=mask_DI_false[:,0],s=0.05,marker='x',color='black',label='DI pair misses');
    _ = plt.scatter(x=mask_DI_true[:,1],y=mask_DI_true[:,0],s=0.1,marker='x',color='red',label='DI pair hits');

    ticker = np.arange( np.around( min( pdb_pairs[:,0] ),decimals=-2),np.around( max( pdb_pairs[:,1] ),decimals=-2 ) , 40)
    _ = plt.xticks(ticker,fontsize=3);
    _ = plt.yticks(ticker,fontsize=3);
    _ = plt.margins(0.001)
    return plot, count_true / pair_count, mask_DI_true

def backmap_alignment(align):
    #get information from manual alignment file
    domain = linecache.getline(align, 1)
    protein_id = linecache.getline(align, 6)[:4]

    d_init = int(linecache.getline(align, 2))
    d_end = int(linecache.getline(align, 4))
    p_init = int(linecache.getline(align, 7))
    p_end = int(linecache.getline(align, 9))

    #domain sequence string
    l1 = linecache.getline(align, 3)
    #protein sequence string
    l2 = linecache.getline(align, 8)

    x1 = len(l1)
    x2 = len(l2)

    #get the difference between initial positions
    delta = max(x1,x2)
    #delta = max(d_end-d_init,p_end-p_init)

    #domain code and respective number
    d = []
    dn = []
    #protein code and respective number
    p = []
    pn = []

    #fill d and p arrays with domain and protein sequences
    for i in range(0,delta-1):
        d.append(l1[i])
        p.append(l2[i])

    #compute the original positions in the system

    j1=-1
    j2=-1
    for i in range(0,len(d)):
        if d[i]!='.':
            j1+=1
            dn.append(str(d_init+j1))
        if d[i]=='.':
            dn.append('')
        if p[i]!='-':
            j2+=1
            pn.append(str(p_init+j2))
        if p[i]=='-':
            pn.append('')

    dic = {}
    for i in range(0,len(dn)):
        if d[i]!='.' and p[i]!='-':
            dic[int(dn[i])]=int(pn[i])
    return dic

