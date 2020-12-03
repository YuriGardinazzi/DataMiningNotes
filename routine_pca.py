# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:55:51 2020

@author: Yuri
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import plotly.express as px
import seaborn as sns


'''
Restituisce il DataFrame ripulito dato un file excel in input
'''
def get_df(filename):
    input_df = df=pd.read_excel(filename)

    #sostituisco il nome della colonna delle variabili contente il tipo con "Class"
    df.rename(columns={'Unnamed: 0':'Class'}, inplace=True )
    #rimuovo eventuali spazi nei nomi
    df.rename(columns=lambda x: x.strip(), inplace=True)
    return input_df
'''
Stampa a terminale le prime 5 righe del DataFrame
'''
def show_first_rows(df):
    print(df.head())

'''
Effettua la PCA sul dataframe utilizzando SVD
'''
def pca(df):

    
    #Assegno ad X solo le colonne coi dati numerici
    X=  df.loc[:, df.columns != 'Class']   
    n = X.shape[0]
    #applicazione del metodo SVD
    U,s,Vt = np.linalg.svd(X,full_matrices=False,compute_uv=True)
    V = Vt.T
    S = np.diag(s)
    
    T = np.dot(U,S) #Scores 
    P = V   #Loadings
    data = pd.DataFrame(data=np.dot(T,P.T), columns=df.columns[1:])
    #data = pd.DataFrame(data=np.dot(T,P.T), columns=list(range(n-1)))
    #data = pd.DataFrame(data=np.dot(T,P.T), columns=(list(range(n))))
    E = X.subtract(data) #Residuals 
    #E = E*E
    

    Eigen_pca = np.power(S,2)/(n-1)
    return T,P,E, Eigen_pca
 
'''
Effettua uno scatter plot della pca. 
component_1 e component_2 indicano le due componenti che si vogliono plottare
1 e 2 sono i valori di default

show_zero_axes è True di default e permette di plottare gli assi X e Y passanti
per (0,0)
'''   
def plot_pca(scores,dataf,component_1=1, component_2=2, show_zero_axes = True):
    pc1 = component_1 - 1
    pc2 = component_2 - 1
    pc1string = f"PC{component_1}"
    pc2string = f"PC{component_2}"
    class_column = dataf['Class']
    
    pc1df = pd.DataFrame(scores[:,pc1])
    pc1df.columns = [f"PC{component_1}"]
    
    pc2df = pd.DataFrame(scores[:,pc2])
    pc2df.columns = [f"PC{component_2}"]
    
    
    outputdf =pd.concat([class_column,pc1df,pc2df], axis = 1)
    
    
    plt.figure(2)
    ax = plt.subplot()
    if show_zero_axes:  
        ax.axhline(y=0, color='k', linewidth=1)
        ax.axvline(x=0, color='k', linewidth=1)


    for key,group in outputdf.groupby('Class'):
         sns.scatterplot(x=group[pc1string], y=group[pc2string],label=key)

    plt.show()
    plt.close()
    return
'''
Plot dei residui
'''
def plot_residuals(E, eigen_val, T):
    plt.figure(1)
    columns_name =  E.columns
    
    ax = plt.subplot()
    #for col in columns_name:
    #    sns.scatterplot(x=range(len(E[col])), y=E[col])
    sns.scatterplot(x=range(len(E[columns_name[0]])), y=E[columns_name[0]])
    #sns.scatterplot(x=range(len(E['linoleico'])), y=E['linoleico'])
    #sns.lineplot(x=range(len(E['linoleico'])), y=E['linoleico'])
    ax.set(title=f"residuals")
    plt.show()
    plt.close()

'''
Applicazione del preprocessing
'''
def custom_preprocessing(df):
    #prendo un dataframe con solo i dati numerici
    df_num = df.select_dtypes(include=[np.number])
    
    #applico il preprocessing
    df_result= (df_num - df_num.mean())/df_num.std() 
    
    #assegno le colonne preprocessate al dataframe originale
    df[df_result.columns] = df_result
    return df

'''
Plot data matrix without "Class" variable
'''
def plot_matrix(df):
    df_matrix = df.loc[:, df.columns != 'Class'] 
    fig, ax = plt.subplots(figsize=(10,10))
    ax.matshow(df_matrix, interpolation='nearest',\
                     aspect='auto',cmap='jet')
        
    plt.xticks(range(len(df_matrix.columns)), df_matrix.columns, \
                fontsize=14, rotation=90)
    ax.xaxis.set_ticks_position('bottom')
    plt.show()
'''
Clustering and Plot
'''
def clustering(df):
    X=  df.loc[:, df.columns != 'Class']   
    mat = X.values
    # Using sklearn
    km = KMeans(n_clusters=10)
    km.fit(mat)
    # Get cluster assignment labels
    labels = km.labels_

    centroids = km.cluster_centers_
    
    #Eseguo la PCA 
    scores, loadings, residuals, eigen_pca = pca(df)
    
    
    pc1df = pd.DataFrame(scores[:,0])
    pc1df.columns = ['PC1']
    pc2df = pd.DataFrame(scores[:,1])
    pc2df.columns = ['PC2']
    

    
    clustering_labels = pd.DataFrame(data =labels, columns=['Clusters'])
    

#    clustering_labels.columns['Clusters']
    print(clustering_labels.head())
    outputdf = pd.concat([clustering_labels,pc1df,pc2df],axis=1)
    
    for key,group in outputdf.groupby('Clusters'):
        sns.scatterplot(x=group['PC1'], y=group['PC2'],label=key)

    plt.show()
if __name__ == '__main__':
    print("ROUTINE PCA\n \
          Inserisci il nome del file Excel da leggere: ")
    filename = "olive_oil.xlsx"
    #filename = input("file: ")
    df = get_df(filename)
    #show_first_rows(df)
    df = custom_preprocessing(df)

    scores, loadings, residuals, eigen_pca = pca(df)
    plot_pca(scores,dataf = df)
    plot_matrix(df)
    
    clustering(df)
    #plot_residuals(residuals)
    #plt.show()