import linecache
import textwrap
from Bio import SeqIO as s
import math
import pylab

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

