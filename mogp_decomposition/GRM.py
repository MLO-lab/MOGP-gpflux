from sklearn.neighbors import NearestNeighbors
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import itertools
import matplotlib.patches as mpatches
import gseapy as gp

def convert_to_List_of_List(mcs_genes_sorted_SEACell):
        big_list = mcs_genes_sorted_SEACell
        sublists = np.ones(len(mcs_genes_sorted_SEACell))
        assert sum(sublists) == len(big_list)
        lol = [[] for _ in sublists]
        i = -1
        
        for v in big_list:
            while True:
                i += 1
                if i == len(lol):
                    i = 0
                if len(lol[i]) < sublists[i]:
                    lol[i].append(v)
                    break       
        return lol
######################################################GENES###########################################################
def rank_dhatgc_score(RNA_struct_MOGP, Genes_struct_MOGP, RNA_Normalized_2000, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(RNA_Normalized_2000)
    distances, NN_idxs = nbrs.kneighbors(RNA_Normalized_2000)
    
    [cells,embs1_num]=RNA_struct_MOGP.obsm['X_umap'].shape
    [genes,embs2_num]=Genes_struct_MOGP.obsm['X_umap'].shape
    num_allcells=[]
    dom_allcells=[]
    for c in range(cells):
        num=[]
        dom=[]
        for k in NN_idxs[c,1:n_neighbors]:
            num.append((RNA_Normalized_2000[c,:]-RNA_Normalized_2000[k,:]))
            dom.append((RNA_struct_MOGP.obsm['X_umap'][c,:]-RNA_struct_MOGP.obsm['X_umap'][k,:]))
        num_allcells.append(num)
        dom_allcells.append(dom)
      
    dhatgc=[]
    for c in range(cells):
        num=np.array(num_allcells[c])
        dom=np.array(dom_allcells[c])
        
        dhatg_p=[]
        for p in range(embs1_num):
            dhatg_p.append(num/np.repeat(dom[:,p], (n_neighbors-1)*[genes]).reshape(n_neighbors-1,genes)) 
        dhatg=np.sqrt(np.sum(np.square(np.median(dhatg_p,1)),0))
        dhatgc.append(dhatg)
    dhatgc=np.array(dhatgc)
    print(dhatgc.shape)
    rank_dhatgc=[]
    for c in range(cells):
        rank=[sorted(dhatgc[c,:] , reverse=True).index(x) for x in dhatgc[c,:]]
        rank_dhatgc.append(rank)
        print(c)
    rank_dhatgc=np.array(rank_dhatgc) 

    return rank_dhatgc

def LRgPsi_avgPsi_score(RNA_struct_MOGP, Genes_struct_MOGP, rank_dhatgc, rg_max):
    [cells,embs1_num]=RNA_struct_MOGP.obsm['X_umap'].shape
    [genes,embs2_num]=Genes_struct_MOGP.obsm['X_umap'].shape
    all_seacells=np.unique(RNA_struct_MOGP.obs['SEACell'])
    LR_gPsi=np.zeros((cells,genes))
    avg_Psi = np.zeros((cells,2))
    for i in range(len(all_seacells)):
        Psi=np.where(RNA_struct_MOGP.obs['SEACell']==all_seacells[i])
        avg_Psi=RNA_struct_MOGP.obsm['X_umap']
        for g in range(genes):
            if len(Psi)==0:
                LR_gPsi[i,g]=0
            else:
                nom=np.sum([rank_dhatgc[Psi,g]<=np.repeat(rg_max, len(Psi))])
                s = nom/len(Psi)
                LR_gPsi[Psi,g]=s         
    avg_Psi=np.asarray(avg_Psi)   
    return LR_gPsi, avg_Psi

def Gene_Relevance_Map(rna, gene_list, LR_gPsi, avg_Psi):
    data_list=[]
    for i in range(len(gene_list)):
        gene=np.where(rna.var_names==gene_list[i])[0]
        data_list.append(LR_gPsi[:,gene])   
    fig, axs = plt.subplots(int(np.floor(len(gene_list)/4)),4, figsize=(15, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()

    for i in range(len(gene_list)):
        points=axs[i].scatter(avg_Psi[:, 0], avg_Psi[:, 1], s=2, c=data_list[i])
        axs[i].set_title(gene_list[i])
        axs[i].axis('off')
        axs[i].grid(False)
        
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(points, cax=cax, orientation='vertical')    
    plt.show()
######################################################META_GENES###########################################################    
def rank_dhatMetagc_score(RNA_struct_MOGP, Genes_struct_MOGP, rank_dhatgc):
    tmp = pd.DataFrame(rank_dhatgc.T).set_index(Genes_struct_MOGP.obs_names).join(Genes_struct_MOGP.obs['SEACell'])
    tmp['SEACell'] = tmp['SEACell'].astype("category")
    tmp = tmp.groupby('SEACell').mean().reset_index()
    rank_dhatMetagc=tmp.iloc[:,1:].values.T
    return rank_dhatMetagc

def LRgPsi_avgPsi_GlobalMetascore(RNA_struct_MOGP, Genes_struct_MOGP, rank_dhatMetagc, rg_max):
    [cells,embs1_num]=RNA_struct_MOGP.obsm['X_umap'].shape
    metaGenes=rank_dhatMetagc.shape[1]
    GR_g=np.zeros(metaGenes)
    for g in range(metaGenes):
        nom=np.sum([rank_dhatMetagc[:,g]<=np.repeat(rg_max, cells)])
        s = nom/cells
        GR_g[g]=s  
    idx = GR_g.argsort()[-metaGenes:][::-1]
    rank_dhatMetagc=rank_dhatMetagc[:,idx]
    
    all_seacells=np.unique(RNA_struct_MOGP.obs['SEACell'])
    LR_gPsi=np.zeros((cells,metaGenes))
    avg_Psi = np.zeros((cells,2))
    for i in range(len(all_seacells)):
        Psi=np.where(RNA_struct_MOGP.obs['SEACell']==all_seacells[i])
        avg_Psi=RNA_struct_MOGP.obsm['X_umap']
        for g in range(metaGenes):
            if len(Psi)==0:
                LR_gPsi[Psi,g]=0
            else:
                nom=np.sum([rank_dhatMetagc[Psi,g]<=np.repeat(rg_max, len(Psi))])
                s = nom/len(Psi)
                LR_gPsi[Psi,g]=s         
    avg_Psi=np.asarray(avg_Psi) 
    return LR_gPsi, avg_Psi, idx

def Global_Gene_Relevance_Map_CITE(rna, RNA_struct_MOGP, Genes_struct_MOGP, col_dict1, mcs_genes_sorted0, mcs_genes_sorted1,mcs_genes_sorted_SEACell, LR_gPsi, avg_Psi, offset=0, topRG=16, n_cell_types=13, n_marker_genes=100):
    
    cell_type_list=['CD4+ naïve T','CD4+ memory T','intermediate mono','CD8+ memory T','CD14 mono',
                   'NK','intermediate mono','mature B','pre-B','memory-like NK','intermediate mono', 
                   'CD16 mono','Treg','CD8+ naïve T','pDC']
    rank_genes_groups=rna.uns["rank_genes_groups"]["names"]
    
    [cells,genes]=rna.X.shape
    C1 = np.array([None] * genes)
    for i in range(len(cell_type_list)):
        for j in range(n_marker_genes):
            C1[np.where(rna.var_names.values==rank_genes_groups[j][i])]=(cell_type_list[i])
    
    cnt_tmp=0
    C2 = np.array([None] * genes)
    if not isinstance(mcs_genes_sorted_SEACell[0], pd.core.arrays.categorical.Categorical):
        mcs_genes_sorted_SEACell=convert_to_List_of_List(mcs_genes_sorted_SEACell)
    for i in range(len(mcs_genes_sorted_SEACell)):
        for j in range(len(mcs_genes_sorted_SEACell[i])):
            if (cnt_tmp % len(cell_type_list))==0:
                cnt_tmp=0  
            C2[np.where(Genes_struct_MOGP.obs['SEACell']==mcs_genes_sorted_SEACell[i][j])]=(cell_type_list[cnt_tmp])
        cnt_tmp=cnt_tmp+1
    
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    ax1.scatter(Genes_struct_MOGP.obsm["X_umap"][:, 0], Genes_struct_MOGP.obsm["X_umap"][:, 1], s=2,  c=[col_dict1[i] for i in C2]) 
    ax2.scatter(Genes_struct_MOGP.obsm["X_umap"][:, 0], Genes_struct_MOGP.obsm["X_umap"][:, 1], s=2,  c=[col_dict1[i] for i in C1])
    ax2.scatter(mcs_genes_sorted0[offset:offset+topRG], mcs_genes_sorted1[offset:offset+topRG], s=5,  c='black',marker='s') #c='black')
    ax3.scatter(RNA_struct_MOGP.obsm["X_umap"][:, 0], RNA_struct_MOGP.obsm["X_umap"][:, 1], s=2, c=[col_dict1[i] for i in rna.obs["celltype"]]) #c='black')
    
    lst = list(itertools.chain(range(offset, offset+topRG)))
    for i in lst:
        ax1.annotate(i+1, (mcs_genes_sorted0[i], mcs_genes_sorted1[i]))
        ax2.annotate(i+1, (mcs_genes_sorted0[i], mcs_genes_sorted1[i]))
   
    
    # Creating legend with color box
    handles = []
    for i in range(n_cell_types):
        handles.append(mpatches.Patch(color=list(col_dict1.values())[i], label=list(col_dict1.keys())[i]))
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    #ax1.title.set_text('Genes associated to each meta gene')
    #ax2.title.set_text('Marker genes for each cell type')
    #ax3.title.set_text('Cell types in scRNA-seq embedded data')
    plt.show()
        
    fig, axs = plt.subplots(int(len(lst)/4),4, figsize=(15, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()

    for i in lst:
        points=axs[i-offset].scatter(avg_Psi[:, 0], avg_Psi[:, 1], s=2, c=LR_gPsi[:,i])
        axs[i-offset].set_title(str(i+1))
        axs[i-offset].axis('off')
        axs[i-offset].grid(False)
        divider = make_axes_locatable(axs[i-offset])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(points, cax=cax, orientation='vertical')
        
        
def Global_Gene_Relevance_Map_PBMC10k(rna, RNA_struct_MOGP, Genes_struct_MOGP, col_dict1, mcs_genes_sorted0, mcs_genes_sorted1,mcs_genes_sorted_SEACell, LR_gPsi, avg_Psi, offset=0, topRG=16, n_cell_types=13, n_marker_genes=100):
    
    cell_type_list=['CD4+ memory T', 'CD4+ naïve T','intermediate mono','CD8+ naïve T','CD14 mono',
                   'CD8+ activated T', 'memory B', 'NK', 'CD16 mono', 'naïve B', 'mDC', 'MAIT', 'pDC']
    
    rank_genes_groups=rna.uns["rank_genes_groups"]["names"]
    cell_type_list_modified=['CD4+ memory T', 'CD4+ naïve T','intermediate mono','CD8+ naïve T','CD14 mono',
                   'CD8+ activated T', 'memory B', 'NK', 'CD16 mono', 'ignore',  'naïve B', 'mDC','ignore',  'MAIT'
                             , 'pDC','ignore','ignore' ]
    
    [cells,genes]=rna.X.shape
    C1 = np.array([None] * genes)
    for i in range(len(cell_type_list_modified)):
        if i not in {9,12,15,16}:
            for j in range(n_marker_genes):
                C1[np.where(rna.var_names.values==rank_genes_groups[j][i])]=(cell_type_list_modified[i])
    
    cnt_tmp=0
    C2 = np.array([None] * genes)
    if not isinstance(mcs_genes_sorted_SEACell[0], pd.core.arrays.categorical.Categorical):
        mcs_genes_sorted_SEACell=convert_to_List_of_List(mcs_genes_sorted_SEACell)
    for i in range(len(mcs_genes_sorted_SEACell)):
        for j in range(len(mcs_genes_sorted_SEACell[i])):
            if (cnt_tmp % len(cell_type_list))==0:
                cnt_tmp=0  
            C2[np.where(Genes_struct_MOGP.obs['SEACell']==mcs_genes_sorted_SEACell[i][j])]=(cell_type_list[cnt_tmp])
        cnt_tmp=cnt_tmp+1
    
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    ax1.scatter(Genes_struct_MOGP.obsm["X_umap"][:, 0], Genes_struct_MOGP.obsm["X_umap"][:, 1], s=2,  c=[col_dict1[i] for i in C2]) 
    ax2.scatter(Genes_struct_MOGP.obsm["X_umap"][:, 0], Genes_struct_MOGP.obsm["X_umap"][:, 1], s=2,  c=[col_dict1[i] for i in C1])
    ax2.scatter(mcs_genes_sorted0[offset:offset+topRG], mcs_genes_sorted1[offset:offset+topRG], s=5,  c='black',marker='s') #c='black')
    ax3.scatter(RNA_struct_MOGP.obsm["X_umap"][:, 0], RNA_struct_MOGP.obsm["X_umap"][:, 1], s=2, c=[col_dict1[i] for i in rna.obs["celltype"]]) #c='black')
    
    lst = list(itertools.chain(range(offset, offset+topRG)))
    for i in lst:
        ax1.annotate(i+1, (mcs_genes_sorted0[i], mcs_genes_sorted1[i]))
        ax2.annotate(i+1, (mcs_genes_sorted0[i], mcs_genes_sorted1[i]))
   
    
    # Creating legend with color box
    handles = []
    for i in range(n_cell_types):
        handles.append(mpatches.Patch(color=list(col_dict1.values())[i], label=list(col_dict1.keys())[i]))
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    #ax1.title.set_text('Genes associated to each meta gene')
    #ax2.title.set_text('Marker genes for each cell type')
    #ax3.title.set_text('Cell types in scRNA-seq embedded data')
    plt.show()
        
    fig, axs = plt.subplots(int(len(lst)/4),4, figsize=(15, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()

    for i in lst:
        points=axs[i-offset].scatter(avg_Psi[:, 0], avg_Psi[:, 1], s=2, c=LR_gPsi[:,i])
        axs[i-offset].set_title(str(i+1))
        axs[i-offset].axis('off')
        axs[i-offset].grid(False)
        divider = make_axes_locatable(axs[i-offset])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(points, cax=cax, orientation='vertical')

######################################################MERGED_META_GENES###########################################################
def LRgPsi_avgPsi_MergedMetascore(RNA_struct_MOGP, Genes_struct_MOGP, metaGenesList_new, rank_dhatMetagc, rg_max):
    [cells,embs1_num]=RNA_struct_MOGP.obsm['X_umap'].shape
    all_seacells=np.unique(RNA_struct_MOGP.obs['SEACell'])
    LR_gPsi=np.zeros((cells,len(metaGenesList_new)))
    avg_Psi = np.zeros((cells,2))
    for i in range(len(all_seacells)):
        Psi=np.where(RNA_struct_MOGP.obs['SEACell']==all_seacells[i])
        avg_Psi=RNA_struct_MOGP.obsm['X_umap']
        for j in range(len(metaGenesList_new)):
            metaGenes_new=metaGenesList_new[j]
            nom_new=[]
            for g in metaGenes_new:
                nom_new.append(rank_dhatMetagc[Psi,g])
                
            if len(Psi)==0:
                LR_gPsi[Psi,g]=0
            else:
                nom=np.sum([np.mean(np.asarray(nom_new),0)<=np.repeat(rg_max, len(Psi))])
                s = nom/len(Psi)
                LR_gPsi[Psi,j]=s         
    avg_Psi=np.asarray(avg_Psi)     
    return LR_gPsi, avg_Psi

def GEA(rna, RNA_struct_MOGP, Genes_struct_MOGP, mcs_genes_sorted_SEACell_new, gene_sets_all, gene_sets_CellType):
    background=rna.var_names
    #gene_sets_all = os.path.join(data_folder_path,  "h.all.v2023.1.Hs.symbols.gmt")
    #gene_sets_CellType = os.path.join(data_folder_path,  "c8.all.v2023.1.Hs.symbols.gmt")
    Dict_all = gp.read_gmt(path=gene_sets_all)
    Dict_CellType = gp.read_gmt(path=gene_sets_CellType)
    
    tmp = pd.DataFrame(rna.var_names).set_index(Genes_struct_MOGP.obs_names).join(Genes_struct_MOGP.obs['SEACell'])
    tmp['SEACell'] = tmp['SEACell'].astype("category")
    Genes_grouped = tmp.groupby('SEACell')
    frames = []
    for i in range(12):
        print("Top relenance Gene: "+str(i+1))
        gene_list=[]
        for j in range(mcs_genes_sorted_SEACell_new[i].shape[0]):
            gene_list.append(Genes_grouped.get_group(mcs_genes_sorted_SEACell_new[i][j])[0].values.tolist())
        gene_list = list(itertools.chain(*gene_list))
        enr_CellType = gp.enrich(gene_list=gene_list, # or gene_list=glist
                     gene_sets=[gene_sets_CellType, "unknown", Dict_CellType ], # kegg is a dict object
                     background=background, # or "hsapiens_gene_ensembl", or int, or text file, or a list of genes
                     outdir=None,
                     verbose=True)
        
        #enr_all = gp.enrich(gene_list=gene_list, # or gene_list=glist
        #             gene_sets=[gene_sets_all, "unknown", Dict_all ], # kegg is a dict object
        #             background=background, # or "hsapiens_gene_ensembl", or int, or text file, or a list of genes
        #             outdir=None,
        #             verbose=True)
        print("Enrichment_CellType:")
        idx=np.argsort(-enr_CellType.res2d['Combined Score'])
        sorted_enr=enr_CellType.res2d.iloc[idx]
        idx_hey=[]
        for j in range(sorted_enr.shape[0]):
            if (np.char.startswith(sorted_enr['Term'].iloc[j], 'HAY', start = 0, end = None)):
                idx_hey.append(j)
        if len(idx_hey)>1:
            display(sorted_enr.iloc[idx_hey].head(2))  
            df2 = sorted_enr.iloc[idx_hey][['Term', 'Adjusted P-value', 'Combined Score']].head(2).copy()
            df2.insert(0, "Meta Gene", [i+1,i+1])
        elif len(idx_hey)==1:
            display(sorted_enr.iloc[idx_hey].head(1))  
            df2 = sorted_enr.iloc[idx_hey][['Term', 'Adjusted P-value', 'Combined Score']].head(1).copy()
            df2.insert(0, "Meta Gene", [i+1])
        else:
            df2 = pd.DataFrame(
                   {
                       "Meta Gene": [i+1, i+1],
                       "Term": ["NaN", "NaN"],
                       "Adjusted P-value": ["NaN", "NaN"],
                       "Combined Score": ["NaN", "NaN"],
                   },
                   index=[0, 1],
                   )
        #display(sorted_enr.iloc[idx_hey].head(2))  
        #df2 = sorted_enr.iloc[idx_hey][['Term', 'Adjusted P-value', 'Combined Score']].head(2).copy()
        #df2.insert(0, "Meta Gene", [i+1,i+1])
        frames.append(df2)
        plt.show()
    
    result = pd.concat(frames)
    display(result)
######################################################PROTS###########################################################            
def rank_dhatpc_score(RNA_struct_MOGP, prot_Normalized_2000, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(prot_Normalized_2000)
    distances, NN_idxs = nbrs.kneighbors(prot_Normalized_2000)
    
    [cells,embs1_num]=RNA_struct_MOGP.obsm['X_umap'].shape
    prots=prot_Normalized_2000.shape[1]
    num_allcells=[]
    dom_allcells=[]
    for c in range(cells):
        num=[]
        dom=[]
        for k in NN_idxs[c,1:n_neighbors]:
            num.append((prot_Normalized_2000[c,:]-prot_Normalized_2000[k,:]))
            dom.append((RNA_struct_MOGP.obsm['X_umap'][c,:]-RNA_struct_MOGP.obsm['X_umap'][k,:]))
        num_allcells.append(num)
        dom_allcells.append(dom)
      
    dhatpc=[]
    for c in range(cells):
        num=np.array(num_allcells[c])
        dom=np.array(dom_allcells[c])
        
        dhatpe_p=[]
        for p in range(embs1_num):
            dhatpe_p.append(num/np.repeat(dom[:,p], (n_neighbors-1)*[prots]).reshape(n_neighbors-1,prots)) 
        dhatp=np.sqrt(np.sum(np.square(np.median(dhatpe_p,1)),0))
        dhatpc.append(dhatp)
    dhatpc=np.array(dhatpc)
    rank_dhatpc=[]
    for c in range(cells):
        rank=[sorted(dhatpc[c,:] , reverse=True).index(x) for x in dhatpc[c,:]]
        rank_dhatpc.append(rank)
    rank_dhatpc=np.array(rank_dhatpc) 
    return rank_dhatpc

def LRpPsi_avgPsi_score(RNA_struct_MOGP, prot_Normalized_2000, rank_dhatpc, rg_max):
    [cells,prots]=prot_Normalized_2000.shape
    all_seacells=np.unique(RNA_struct_MOGP.obs['SEACell'])
    LR_pPsi=np.zeros((cells,prots))
    avg_Psi = np.zeros((cells,2))
    for i in range(len(all_seacells)):
        Psi=np.where(RNA_struct_MOGP.obs['SEACell']==all_seacells[i])
        avg_Psi=RNA_struct_MOGP.obsm['X_umap']
        for p in range(prots):
            if len(Psi)==0:
                LR_pPsi[i,p]=0
            else:
                nom=np.sum([rank_dhatpc[Psi,p]<=np.repeat(rg_max, len(Psi))])
                s = nom/len(Psi)
                LR_pPsi[Psi,p]=s         
    avg_Psi=np.asarray(avg_Psi) 
    return LR_pPsi, avg_Psi

    
def Prot_Relevance_Map(rna, prot, RNA_struct_MOGP, prot_list, LR_pPsi, avg_Psi, col_dict1, n_cell_types=13):
    
    data_list=[]
    for i in range(len(prot_list)):
        prot_tmp=np.where(prot.var_names==prot_list[i])[0]
        data_list.append(LR_pPsi[:,prot_tmp])  
    fig, axs = plt.subplots(4,3, figsize=(12, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .2, wspace=.1)
    axs = axs.ravel()

    for i in range(len(prot_list)-1):
        points=axs[i].scatter(avg_Psi[:, 0], avg_Psi[:, 1], s=2, c=data_list[i])
        axs[i].set_title(prot_list[i])
        axs[i].axis('off')
        axs[i].grid(False)
        
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(points, cax=cax, orientation='vertical')

    axs[len(prot_list)-1].scatter(RNA_struct_MOGP.obsm["X_umap"][:, 0], RNA_struct_MOGP.obsm["X_umap"][:, 1], s=2, c=[col_dict1[i] for i in rna.obs["celltype"]]) #c='black')
    # Creating legend with color box
    handles = []
    for i in range(n_cell_types):
        handles.append(mpatches.Patch(color=list(col_dict1.values())[i], label=list(col_dict1.keys())[i]))
    axs[len(prot_list)-1].grid(False)
    axs[len(prot_list)-1].axis('off')
    axs[len(prot_list)-1].set_position([0.66, 0.115, 0.22, 0.16])
    axs[len(prot_list)-1].legend(handles=handles, loc='lower right', bbox_to_anchor=(1.15, 0.5), fontsize=5)
    
    plt.show()
    fig.savefig('CITEProtRelevanceMAP.pdf', dpi=100, format='pdf', bbox_inches='tight')
    