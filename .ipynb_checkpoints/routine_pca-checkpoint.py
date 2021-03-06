# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:55:51 2020

@author: Yuri
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
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
    X = df
    X = df.drop('Class', axis=1)
        
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
def plot_pca(scores,component_1=1, component_2=2, show_zero_axes = True):
    pc1 = component_1 - 1
    pc2 = component_2 - 1
   
    plt.figure(2)
    ax = plt.subplot()
    if show_zero_axes:  
        ax.axhline(y=0, color='k', linewidth=1)
        ax.axvline(x=0, color='k', linewidth=1)
    ax.set(xlabel=f"PC{component_1}", ylabel=f"PC{component_2}",)
   
    sns.scatterplot(x=scores[:,pc1], y=scores[:,pc2])
    plt.show()
    plt.close()
    return
'''
Plot dei residui
'''
def plot_residuals(E):
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

if __name__ == '__main__':
    print("ROUTINE PCA\n \
          Inserisci il nome del file Excel da leggere: ")
    filename = "olive_oil.xlsx"
    #filename = input("file: ")
    df = get_df(filename)
    #show_first_rows(df)

    df = (df - df.mean())/df.std()  #Little Preprocessing
    scores, loadings, residuals, eigen_pca = pca(df)
    
    plot_pca(scores)
    plot_residuals(residuals)
    #plt.show()