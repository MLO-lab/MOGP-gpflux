import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from operator import itemgetter
from sklearn.metrics import *

def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    y_true = [int(numeric_string) for numeric_string in y_true]    
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        idx=list(np.where(y_pred==cluster)[0])
        hist, _ = np.histogram(np.array(itemgetter(*idx)(y_true)), bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels),adjusted_rand_score(y_true, y_voted_labels)

def Plot_emb1_tsne(rna=None, RNA_struct_MOGP=None, col_dict1=None, n_cell_types=13):
    
    fig, ax = plt.subplots()
    
    ax.scatter(RNA_struct_MOGP.obsm["X_tsne"][:, 0], RNA_struct_MOGP.obsm["X_tsne"][:, 1], s=2, c=[col_dict1[i] for i in rna.obs["celltype"]])
    
    # Creating legend with color box
    handles = []
    for i in range(n_cell_types):
        handles.append(mpatches.Patch(color=list(col_dict1.values())[i], label=list(col_dict1.keys())[i]))
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(False)
    ax.set_xlabel('MOGP1', fontsize=10)
    ax.set_ylabel('MOGP2', fontsize=10)
    #ax.axis('off')
    #ax.axis('off')
    #ax1.title.set_text('Cell types in scRNA-seq embedded data')
    #ax2.title.set_text('Marker genes for each cell type')
    plt.show() 
###############################################PBMC10k########################################################
def Relating_clusters_of_emb1_to_clusters_of_emb2_PBMC10k(rna=None, atac=None, RNA_struct_MOGP=None, Genes_struct_MOGP=None, Peaks_struct_MOGP=None, col_dict1=None, cell_type_list_modified1=None, cell_type_list_modified2=None,  n_cell_types=13, n_marker_genes=100, use_umap=True, n_marker_peaks=100):
    
    rank_genes_groups=rna.uns["rank_genes_groups"]["names"]
    [cells,genes]=rna.X.shape
    C1 = np.array([None] * genes)
    for i in range(len(cell_type_list_modified1)):
        if cell_type_list_modified1[i] not in {'ignore'}:
            for j in range(n_marker_genes):
                C1[np.where(rna.var_names.values==rank_genes_groups[j][i])]=(cell_type_list_modified1[i])
       
    if Peaks_struct_MOGP is not None:
        rank_peaks_groups=atac.uns["rank_genes_groups"]["names"]
        [cells,K] =atac.X.shape
        C2 = np.array([None] * K)
        for i in range(len(cell_type_list_modified2)):
            if cell_type_list_modified2[i] not in {'ignore'}:
                for j in range(n_marker_peaks):
                    C2[np.where(atac.var_names.values==rank_peaks_groups[j][i])]=(cell_type_list_modified2[i])
        fig = plt.figure(figsize=(15,5))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax3.grid(False)
        ax3.axis('off')
    else: 
        fig = plt.figure(figsize=(15,5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
    if use_umap:
        ax1.scatter(RNA_struct_MOGP.obsm["X_umap"][:, 0], RNA_struct_MOGP.obsm["X_umap"][:, 1], s=2, c=[col_dict1[i] for i in rna.obs["celltype"]]) 
        ax2.scatter(Genes_struct_MOGP.obsm["X_umap"][:, 0], Genes_struct_MOGP.obsm["X_umap"][:, 1], s=2,  c=[col_dict1[i] for i in C1]) 
        if Peaks_struct_MOGP is not None:
            ax3.scatter(Peaks_struct_MOGP.obsm["X_umap"][:, 0], Peaks_struct_MOGP.obsm["X_umap"][:, 1], s=2,  c=[col_dict1[i] for i in C2])
    else:
        ax1.scatter(RNA_struct_MOGP.obsm["X_tsne"][:, 0], RNA_struct_MOGP.obsm["X_tsne"][:, 1], s=2, c=[col_dict1[i] for i in rna.obs["celltype"]]) #c='black')
        ax2.scatter(Genes_struct_MOGP.obsm["X_tsne"][:, 0], Genes_struct_MOGP.obsm["X_tsne"][:, 1], s=2,  c=[col_dict1[i] for i in C1]) 
        if Peaks_struct_MOGP is not None:
            ax3.scatter(Peaks_struct_MOGP.obsm["X_tsne"][:, 0], Peaks_struct_MOGP.obsm["X_tsne"][:, 1], s=2,  c=[col_dict1[i] for i in C2]) 
    
    # Creating legend with color box
    handles = []
    for i in range(n_cell_types):
        handles.append(mpatches.Patch(color=list(col_dict1.values())[i], label=list(col_dict1.keys())[i]))
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(False)
    ax2.grid(False)
    
    ax1.axis('off')
    ax2.axis('off')
    
    #ax1.title.set_text('Cell types in scRNA-seq embedded data')
    #ax2.title.set_text('Marker genes for each cell type')
    plt.show()
    
###############################################CITEseq########################################################
def Relating_clusters_of_emb1_to_clusters_of_emb2_CITEseq(rna=None, prot=None, RNA_struct_MOGP=None, Genes_struct_MOGP=None, Prot_struct_MOGP=None, col_dict1=None, cell_type_list_modified1=None, cell_type_list_modified2=None, n_cell_types=13, n_marker_genes=100, use_umap=True, n_marker_prot=5):
    
    rank_genes_groups=rna.uns["rank_genes_groups"]["names"]
    [cells,genes]=rna.X.shape
    C1 = np.array([None] * genes)
    for i in range(len(cell_type_list_modified1)):
        if cell_type_list_modified1[i] not in {'ignore'}:
            for j in range(n_marker_genes):
                C1[np.where(rna.var_names.values==rank_genes_groups[j][i])]=(cell_type_list_modified1[i])
    
    if Prot_struct_MOGP is not None:
        rank_prots_groups=prot.uns["rank_genes_groups"]["names"]
        embs3_list=[]
        for i in range(n_cell_types):
            for j in range(n_marker_prot):
                embs3_list.append(Prot_struct_MOGP.obsm["X_tsne"][np.where(prot.var_names.values==rank_prots_groups[j][i])[0][0],:])
    
    fig1 = plt.figure(figsize=(10,5))
    ax1 = fig1.add_subplot(121)
    ax2 = fig1.add_subplot(122)
    if use_umap:
        ax1.scatter(RNA_struct_MOGP.obsm["X_umap"][:, 0], RNA_struct_MOGP.obsm["X_umap"][:, 1], s=2, c=[col_dict1[i] for i in rna.obs["celltype"]]) 
        ax2.scatter(Genes_struct_MOGP.obsm["X_umap"][:, 0], Genes_struct_MOGP.obsm["X_umap"][:, 1], s=2,  c=[col_dict1[i] for i in C1])
    else:
        ax1.scatter(RNA_struct_MOGP.obsm["X_tsne"][:, 0], RNA_struct_MOGP.obsm["X_tsne"][:, 1], s=2, c=[col_dict1[i] for i in rna.obs["celltype"]])
        ax2.scatter(Genes_struct_MOGP.obsm["X_tsne"][:, 0], Genes_struct_MOGP.obsm["X_tsne"][:, 1], s=2,  c=[col_dict1[i] for i in C1])
        
    # Creating legend with color box
    handles = []
    for i in range(n_cell_types):
        handles.append(mpatches.Patch(color=list(col_dict1.values())[i], label=list(col_dict1.keys())[i]))
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(False)
    ax2.grid(False)
    
    ax1.axis('off')
    ax2.axis('off')
    
    plt.show()
    
    if Prot_struct_MOGP is not None:
        col_dict1_list = [key for key in col_dict1.keys() if key is not None]
        # Try to match the column of rank_genes_groups based on col_dict1_list
        result = prot.uns['rank_genes_groups']
        groups = result['names'].dtype.names
        pd.set_option('display.max_columns', 50)
        tmp=pd.DataFrame(
            {group + '_' + key[:1]: result[key][group]
            for group in groups for key in ['names']}).values
        matching_indices = {}
        for element in col_dict1_list:
            indices = [i for i, x in enumerate(cell_type_list_modified2) if x == element]
            matching_indices[element] = indices
        # Swap columns in the other array based on the matching indices
        swapped_array = tmp.copy()
        for element, indices in matching_indices.items():
            for i, index in enumerate(indices):
                swapped_array[:,col_dict1_list.index(element)]=tmp[:,index]
        
        embs3_list_list_X=[]
        embs3_list_list_Y=[]
        for i in range(n_cell_types):
            embs3_list_X=[]
            embs3_list_Y=[]
            for j in range(n_marker_prot):
                embs3_list_X.append(Prot_struct_MOGP.obsm['X_tsne'][np.where(prot.var_names.values==swapped_array[j,i])[0][0],0])
                embs3_list_Y.append(Prot_struct_MOGP.obsm['X_tsne'][np.where(prot.var_names.values==swapped_array[j,i])[0][0],1])
            embs3_list_list_X.append(embs3_list_X)   
            embs3_list_list_Y.append(embs3_list_Y)  
        
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        bp1 = ax1.boxplot(embs3_list_list_X, patch_artist=True)
        bp2 = ax2.boxplot(embs3_list_list_Y, patch_artist=True)
        for patch, color in zip(bp1['boxes'], col_dict1.values()):
            patch.set_facecolor(color)
        for patch, color in zip(bp2['boxes'], col_dict1.values()):
            patch.set_facecolor(color)    
        ax1.set_xticklabels([])
        ax2.set_xticklabels(col_dict1_list, rotation='vertical')
        ax1.set_ylabel("Latent dim 1")
        ax2.set_ylabel("Latent dim 2")
        fig.canvas.draw()
        plt.tight_layout(rect=[0.1, 0.1, 1.5, 1.5])
        plt.show()
        
        