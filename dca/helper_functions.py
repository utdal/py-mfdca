import linecache
import textwrap
from Bio import SeqIO as s
from Bio.PDB import PDBParser
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
import math
import pylab
import os
import warnings
import numpy as np

def clean_pfam(input_filename: str, output_filename: str):
    dont_write_newline = True
    with open(input_filename,'r') as fd:
        with open(output_filename,'w') as od:
            for line in fd:
                if '>' in line:
                    if dont_write_newline:
                        od.writelines(line)
                        dont_write_newline = False
                    else:
                        od.writelines('\n')
                        od.writelines(line)
                else:
                    newline = ''.join(x for x in line.rstrip() if not x.islower() and x != '.')
                    od.writelines(newline)

def filter_pfam(filename: str, gaps: int, output: str):
    linecache.clearcache()
    data = filename
    size = open(data,"r")
    limit = int(gaps)
    
    ###################################################################################
    lim =""
    for k in range(0,limit):
        lim+="-"
    #print lim
    ###################################################################################


    i=1

    l = len(size.readlines())


    output = open(output,"w")

    ####################################################################################

    nseq = 0
    excluded =0

    ####################################################################################

    while i < l:
        sequence = ""
        n = linecache.getline(data, i)
        counter = 0
        if n[0] == ">":
            name = n
            next =  linecache.getline(data, i+1)
            try:
                while next[0] != ">":
                    sequence=sequence+next
                    i+=1
                    next =  linecache.getline(data, i+1)
                nseq+=1
            except IndexError:
                pass
        i+=1
        x=""
        for j in range(0,len(sequence)):
            if sequence[j]!="." and sequence[j]!="\n" and sequence[j].islower()==False:
                x+=sequence[j]
        if len(x.split(lim)) == 1:
            output.write(name+x+"\n")
        else:
            excluded+=1

    output.close()


    print("Original number of sequences: "+str(nseq)+"\n")
    percentage = float(excluded)/nseq*100

    print("Number(%) of sequences excluded: "+str(excluded)+" ("+str("{0:.2f}".format(percentage))+"%)\n")

def interface_contacts_allatoms(input_filename: str, first_chain: str, second_chain: str, angstrom_cutoff: float) -> str:
    linecache.clearcache()
    size = open(input_filename,"r")
    lines = len(size.readlines())
    # create output_filename, considering directories as inputs
    input_dir = os.path.sep.join(input_filename.split(os.path.sep)[:-1])
    pdb_name = input_filename.split(os.path.sep)[-1].split('.')[0]
    output_filename = os.path.join(input_dir,"contactmap_calpha_"+pdb_name+"_"+first_chain+second_chain+"_"+str(angstrom_cutoff))
    n=0
    an1 = []
    an2 = []
    r1 = []
    r2 = []
    rn1 = []
    rn2 = []
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    z1 = []
    z2 = []
    ch1 = []
    ch2 = []

    res1 = ""
    res2 = ""

    for i in range(1,lines):
        a = linecache.getline(input_filename,i)
        if a[0:4]=="ATOM" and a[21:22]==first_chain:
                an1.append(str(a[6:11]))
                r1.append(str(a[22:26]))
                rn1.append(str(a[17:20]))
                x1.append(float(str(a[30:38]))) 
                y1.append(float(str(a[38:46])))
                z1.append(float(str(a[46:54])))
                ch1.append(str(a[21:22]))
        if a[0:4]=="ATOM" and a[21:22]==second_chain:
                an2.append(str(a[6:11]))    
                r2.append(str(a[22:26]))
                rn2.append(str(a[17:20]))
                x2.append(float(str(a[30:38]))) 
                y2.append(float(str(a[38:46])))
                z2.append(float(str(a[46:54])))
                ch2.append(str(a[21:22]))            
    output = open(output_filename,"w")
    output.write("at numb".ljust(10)+"res numb".ljust(10)+"res name".ljust(10)+"chain".ljust(8)+"at numb".ljust(10)+"res numb".ljust(10)+"res name".ljust(10)+"chain".ljust(8)+"distance".ljust(15)+"\n")
        
    for j in range(0,len(r1)):
        for k in range(j+1,len(r2)):
            dx = math.pow(x1[j]-x2[k],2)
            dy = math.pow(y1[j]-y2[k],2)
            dz = math.pow(z1[j]-z2[k],2)
            dist = math.pow(dx+dy+dz,0.5)        
            if dist <= angstrom_cutoff:
                output.write(an1[j].ljust(10)+r1[j].ljust(10)+rn1[j].ljust(10)+ch1[j].ljust(8)+an2[k].ljust(10)+r2[k].ljust(10)+rn2[k].ljust(10)+ch2[k].ljust(8)+str(dist).ljust(15)+"\n")        
                n+=1

    print("\n\tNumber of interactions found: "+str(n)+"\n")
    print("\n\tFile saved as: "+"contactmap_allatom_"+input_filename[:-4]+"_"+first_chain+second_chain+"_"+str(angstrom_cutoff)+"\n")
    return output_filename

def interface_contacts_calpha(input_filename: str, first_chain: str, second_chain: str, angstrom_cutoff: float) -> str:
    linecache.clearcache()
    size = open(input_filename,"r")
    lines = len(size.readlines())
    # create output_filename, considering directories as inputs
    input_dir = os.path.sep.join(input_filename.split(os.path.sep)[:-1])
    pdb_name = input_filename.split(os.path.sep)[-1].split('.')[0]
    output_filename = os.path.join(input_dir,"contactmap_calpha_"+pdb_name+"_"+first_chain+second_chain+"_"+str(angstrom_cutoff))
    n=0
    r1 = []
    r2 = []
    rn1 = []
    rn2 = []
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    z1 = []
    z2 = []
    ch1 = []
    ch2 = []

    res1 = ""
    res2 = ""

    for i in range(1,lines):
        a = linecache.getline(input_filename,i)
        if a[0:4]=="ATOM" and a[21:22]==first_chain:
            if a[22:26] != res1: 
                if a[13:15]=="CA":
                    r1.append(str(a[22:26]))
                    rn1.append(str(a[17:20]))
                    x1.append(float(str(a[30:38]))) 
                    y1.append(float(str(a[38:46])))
                    z1.append(float(str(a[46:54])))
                    ch1.append(str(a[21:22]))
                    res1 = a[22:26]
        if a[0:4]=="ATOM" and a[21:22]==second_chain:
            if a[22:26] != res2:
                if a[13:15]=="CA":            
                    r2.append(str(a[22:26]))
                    rn2.append(str(a[17:20]))
                    x2.append(float(str(a[30:38]))) 
                    y2.append(float(str(a[38:46])))
                    z2.append(float(str(a[46:54])))
                    ch2.append(str(a[21:22]))            
                    res2 = a[22:26]
    output = open(output_filename,"w")
    output.write("res numb".ljust(10)+"res name".ljust(10)+"chain".ljust(8)+"res numb".ljust(10)+"res name".ljust(10)+"chain".ljust(8)+"distance".ljust(15)+"\n")
        
    for j in range(0,len(r1)):
        for k in range(j,len(r2)):
            dx = math.pow(x1[j]-x2[k],2)
            dy = math.pow(y1[j]-y2[k],2)
            dz = math.pow(z1[j]-z2[k],2)
            dist = math.pow(dx+dy+dz,0.5)        
            if dist <= angstrom_cutoff and dist > 0:
                output.write(r1[j].ljust(10)+rn1[j].ljust(10)+ch1[j].ljust(8)+r2[k].ljust(10)+rn2[k].ljust(10)+ch2[k].ljust(8)+str(dist).ljust(15)+"\n")
                n+=1        

    print("\n\tNumber of interactions found: "+str(n)+"\n")
    print("\n\tFile saved as: "+"contactmap_calpha_"+input_filename[:-4]+"_"+first_chain+second_chain+"_"+str(angstrom_cutoff)+"\n")
    return output_filename

def backmap_alignment(align: str) -> dict:
    #get information from manual alignment file
    linecache.clearcache()
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

def get_allatom_contacts( pdb_file: str, chain1_id: str, chain2_id: str, distance_cutoff: int) -> tuple:
    warnings.filterwarnings("ignore")
    # Create a PDB parser object

    suffix_struc = pdb_file.split(".")[-1]
    if suffix_struc == "pdb":
        parser = PDBParser()
    elif suffix_struc == "cif":
        parser = MMCIFParser()
    else:
        raise ValueError(f"Unsupported file format: {suffix_struc}. Only PDB and CIF formats are supported.")

    # Parse the PDB file
    structure = parser.get_structure("protein", pdb_file)

    # Get the specified chains
    chain1 = structure[0][chain1_id]
    chain2 = structure[0][chain2_id]

    # Function to get atom coordinates from a residue
    def get_atom_coords(residue):
        atom_coords = []
        for atom in residue:
            if atom.is_disordered():
                for position in atom:
                    atom_coords.append(position.coord)
            else:
                atom_coords.append(atom.coord)
        return np.array(atom_coords)
        
    # Get coordinates and residue IDs for both chains
    coord_ids1 = [
        (get_atom_coords(res), res.id[1]) for res in chain1 if is_aa(res)
    ]
    coord_ids2 = [
        (get_atom_coords(res), res.id[1]) for res in chain2 if is_aa(res)
    ]

    if len(coord_ids1) == 0 or len(coord_ids2) == 0:
        # no valid aa residues in the chain, so we skip
        return ()

    coords1, res_ids1 = zip(*coord_ids1)
    coords2, res_ids2 = zip(*coord_ids2)

    # Combine all atom coordinates for each chain
    all_coords1 = np.vstack(coords1)
    all_coords2 = np.vstack(coords2)

    # Calculate distances between all atoms
    distances = np.linalg.norm(all_coords1[:, np.newaxis] - all_coords2, axis=2)

    # Find pairs of atoms within the cutoff distance
    close_atom_pairs = np.argwhere(distances <= distance_cutoff)

    # Map atom pairs to residue pairs
    res_index1 = np.cumsum([len(c) for c in coords1])
    res_index2 = np.cumsum([len(c) for c in coords2])

    close_residue_pairs = set()
    for i, j in close_atom_pairs:
        res1 = np.searchsorted(res_index1, i, side="right")
        res2 = np.searchsorted(res_index2, j, side="right")
        close_residue_pairs.add((res_ids1[res1], res_ids2[res2]))

    print(f"Finished {pdb_file} {chain1_id} {chain2_id}")
    results = (
        pdb_file,
        chain1_id,
        chain2_id,
        np.array(list(close_residue_pairs)),
    )
    return np.array(list(close_residue_pairs))    
