#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import math
import numpy as np
df = pd.read_csv('train.csv')


# In[47]:





# In[9]:


#mydf = df[['LotArea','Street','BldgType','HouseStyle','OverallQual', 'OverallCond','YearBuilt','TotalBsmtSF','BedroomAbvGr','KitchenAbvGr']]
mydf = df[['LotArea','OverallQual', 'OverallCond','YearBuilt','TotalBsmtSF','BedroomAbvGr','KitchenAbvGr']]
k    = math.floor(1 + ((math.log(mydf['LotArea'].shape[0],2))))
print('ElementosK')
print(k)
cols = ['amplitud']
amplitud = pd.DataFrame(index = mydf.columns.values, columns =cols)
amplitud = amplitud.fillna(0)
for column in mydf:
    hist, bin_edges = np.histogram(mydf[column])
    amplitud.loc[column] = bin_edges[1] - bin_edges[0]    
print(amplitud)

def obtenerAmplitudCorrecta(column, data):
    val_min = math.floor(mydf[column].min())     
    amp = amplitud.at[column,'amplitud']        
    return math.floor((data-val_min) / amp)     

freq    = np.array(np.zeros((7, k)), dtype='int64')
idx_row = 0
for column in mydf:
    values = np.zeros(k)
    for i in range(0, mydf[column].shape[0]):
        element = mydf[column][i]
        amp_cor = obtenerAmplitudCorrecta(column, element)
        freq[idx_row][amp_cor] += 1
    idx_row +=1
        
print('FRECUENCIA')
frecuencia = pd.DataFrame(freq)
frecuencia = frecuencia.set_index(amplitud.index.values)
print(frecuencia)
print('FIN FRECUENCIA')

#Momento original
cols_original     = ['k1']
momentos_original = pd.DataFrame(index = mydf.columns.values, columns=cols_original)
momentos_original = momentos_original.fillna(0)

cols_medio     = ['k1', 'k2', 'k3', 'k4']
momentos_medio = pd.DataFrame(index = mydf.columns.values, columns=cols_medio)
momentos_medio = momentos_medio.fillna(0)
media          = mydf.mean()


cols_padronizado     = ['k1', 'k2', 'k3', 'k4']
momentos_padronizado = pd.DataFrame(index = mydf.columns.values, columns=cols_padronizado)
momentos_padronizado = momentos_padronizado.fillna(0)
varianza             = mydf.var()
for column in mydf:
    temp_original    = 0
    temp_medio_k1    = 0
    temp_medio_k2    = 0
    temp_medio_k3    = 0
    temp_medio_k4    = 0
    temp_padronizado_k1 = 0
    temp_padronizado_k2 = 0
    temp_padronizado_k3 = 0
    temp_padronizado_k4 = 0
    for i in range(0, mydf[column].shape[0]):
        element = mydf[column][i]
        amplitud
        amp_cor = obtenerAmplitudCorrecta(column, element)
        amp     = amplitud.at[column,'amplitud']
        minimo  =  mydf[column].min()
        min_amp = minimo + (amp * amp_cor)
        max_amp = minimo + (amp * (amp_cor + 1))
        elemnt  = (min_amp + max_amp) / 2
        
        frq_temp= frecuencia.at[column, amp_cor]
        
        # Momento Original                
        temp_original += frq_temp * element
        
        #Momento Medio
        aux = element - media[column]
        temp_medio_k1 += aux ** 1
        temp_medio_k2 += aux ** 2
        temp_medio_k3 += aux ** 3
        temp_medio_k4 += aux ** 4
        
        #Momento Padronizado
        # Los momentos k = 1 , k = 2 y k = 4 el calculo inicial es parecido al del momento medio
        temp_padronizado_k3 += (aux ** 3) * frq_temp
        
    
    # Agrupando momento original
    momentos_original.loc[column] = temp_original
    
    # Agrupando momento medio
    temp_medio_k1 /= (mydf.shape[0] - 1)
    temp_medio_k2 /= (mydf.shape[0] - 1)
    temp_medio_k3 /= (mydf.shape[0] - 1)
    temp_medio_k4 /= (mydf.shape[0] - 1)    
    momentos_medio.loc[column] = pd.Series({'k1':temp_medio_k1, 'k2':temp_medio_k2, 'k3':temp_medio_k3, 'k4':temp_medio_k4})
    
    #Agrupando momento padronizado
    temp_padronizado_k1 = temp_medio_k1 / (varianza[column])
    temp_padronizado_k2 = temp_medio_k2 / (varianza[column])
    temp_padronizado_k3 /= temp_padronizado_k3 / (varianza[column])
    temp_padronizado_k4 = temp_medio_k4 / (varianza[column] ** 2)
    momentos_padronizado.loc[column] = pd.Series({'k1':temp_padronizado_k1, 'k2':temp_padronizado_k2, 'k3':temp_padronizado_k3, 'k4':temp_padronizado_k4})

print('******* MOMENTO ORIGINAL *******')
print(momentos_original)
print('******* MOMENTO MEDIO *******')
print(momentos_medio)
print('******* MOMENTO PADRONIZADO *******')
print(momentos_padronizado)


# In[ ]:





# In[ ]:



        

    
    
    


# In[ ]:





# In[ ]:




