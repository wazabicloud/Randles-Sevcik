from pandas.core.frame import DataFrame
import AdmiralDecode, BiologicDecode
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from typing import Tuple

#Opzioni

get_files = False
mode = "capacity"

#Area campione
area = 2.0

#Parametri per ricerca picchi
E_range = (-3, 3)
max_order = 200
min_order = 200

#Nomi finali delle colonne per unificare i dataframe
pot = "Potential (V)"
curr = "Current (A)"
time = "Time (s)"
scrate = "Scan rate (V/s)"
cathodic = "Cathodic Branch"
anodic = "Anodic Branch"
peaks = "peaks"

#Altre definizioni
branches = [cathodic, anodic]


#====================================
# Acquisizione dati
#====================================

file_dir = input("In che percorso si trovano i file?\n")

df_list = []

#Funzione per importare dati direttamente da csv di admiral
def admiral_import(file):

    data_pot_name = "Working Electrode (V)"
    data_curr_name = "Current (A)"
    data_time_name = "Elapsed Time (s)"

    last_test = AdmiralDecode.extract_simple(file, normalize=True)[-1]

    last_test["new_cycle"] = last_test["Step number"].ne(last_test["Step number"].shift())

    last_cycle_begin = last_test.loc[(last_test["new_cycle"])].index[-1]

    last_cycle = last_test.loc[last_cycle_begin:].copy()

    last_cycle.loc[:,data_curr_name] = last_cycle[data_curr_name].div(area)
    last_cycle.rename({data_pot_name: pot, data_curr_name: curr, data_time_name: time}, axis="columns", inplace=True)

    return last_cycle

#Funzione per importare dati direttamente da mpt del biologic
def biologic_import(file):

    data_pot_name = "Ewe/V"
    data_curr_name = "<I>/A"
    data_time_name = "time/s"

    last_test = BiologicDecode.extract_simple(file, normalize=True)[-1]

    last_test["new_cycle"] = last_test["cycle number"].ne(last_test["cycle number"].shift())

    last_cycle_begin = last_test.loc[(last_test["new_cycle"])].index[-1]

    last_cycle = last_test.loc[last_cycle_begin:].copy()

    last_cycle.loc[:,data_curr_name] = last_cycle[data_curr_name].div(area)
    last_cycle.rename({data_pot_name: pot, data_curr_name: curr, data_time_name: time}, axis="columns", inplace=True)

    return last_cycle

for file in os.listdir(file_dir):

    file = file_dir + "\\" + file
        
    if ".csv" in file:
        df_list.append(admiral_import(file))
    elif ".mpt" in file:
        df_list.append(biologic_import(file))
    else:
        continue

#====================================
# Trasformazione dati acquisiti
#====================================

#Si parte direttamente da un ciclo specifico di una CV

total_df = pd.DataFrame()

for i in range(len(df_list)):

    #Prendo solo potenziale, corrente e tempo
    df_list[i] = df_list[i][[pot, curr, time]].copy()

    df_list[i][scrate] = ((df_list[i][pot].diff(1).abs())/df_list[i][time].diff(1)).mean()

    #Alla fine viene popolata una lista di dizionari
    #Ogni dizionario contiene un dataframe per i rami catodico e anodico e lo scan rate

    total_df = pd.concat([total_df, df_list[i]])

print(total_df.groupby(scrate))