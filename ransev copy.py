import AdmiralDecode, BiologicDecode
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema


file_dir = input("In che percorso si trovano i file?\n")

#Inizializzazione variabili

#Lista di dataframe estratti direttamente dai file e divisi in ramo anodico e catodico
df_list = []
area = 1.0

pot = "Potential (V)"
curr = "Current (A)"
time = "Time (s)"
scrate = "Scan rate (V/s)"

#====================================
# Acquisizione dati
#====================================

def admiral_import(file, df_list):

    data_pot_name = "Working Electrode (V)"
    data_curr_name = "Current (A)"
    data_time_name = "Elapsed Time (s)"

    last_test = AdmiralDecode.extract_simple(file, normalize=True)[-1]

    last_test["new_cycle"] = last_test["Step number"].ne(last_test["Step number"].shift())

    last_cycle_begin = last_test.loc[(last_test["new_cycle"])].index[-1]

    last_cycle = last_test.loc[last_cycle_begin:]

    last_cycle.loc[:,data_curr_name] = last_cycle[data_curr_name].div(area)
    last_cycle.rename({data_pot_name: pot, data_curr_name: curr, data_time_name: time}, axis="columns", inplace=True)

    df_list.append(last_cycle)

def biologic_import(file, df_list):

    data_pot_name = "Ewe/V"
    data_curr_name = "<I>/A"
    data_time_name = "time/s"

    last_test = BiologicDecode.extract_simple(file, normalize=True)[-1]

    last_test["new_cycle"] = last_test["cycle number"].ne(last_test["cycle number"].shift())

    last_cycle_begin = last_test.loc[(last_test["new_cycle"])].index[-1]

    last_cycle = last_test.loc[last_cycle_begin:]

    last_cycle.loc[:,data_curr_name] = last_cycle[data_curr_name].div(area)
    last_cycle.rename({data_pot_name: pot, data_curr_name: curr, data_time_name: time}, axis="columns", inplace=True)

    df_list.append(last_cycle)

# Li estraggo normalizzati a V e A
# Da completare con amel e biologic
for file in os.listdir(file_dir):

    file = file_dir + "\\" + file
        
    if ".csv" in file:
        admiral_import(file, df_list)
    elif ".mpt" in file:
        biologic_import(file, df_list)
    else:
        continue

#====================================
# Trasformazione dati acquisiti
#====================================

#Si parte direttamente da un ciclo specifico di una CV

for i in range(len(df_list)):

    #Prendo solo potenziale, corrente e tempo
    df_list[i] = df_list[i][[pot, curr, time]].copy()

    #Divido la curva in ramo catodico e anodico
    cat_branch = df_list[i][(df_list[i][pot].diff(1) < 0)].sort_values(by=[pot])
    anod_branch = df_list[i][(df_list[i][pot].diff(1) > 0)].sort_values(by=[pot])

    #Calcolo lo scan rate in automatico
    scan_rate = ((df_list[i][pot].diff(1).abs())/df_list[i][time].diff(1)).mean()

    #Trovo posizioni dei picchi

    max_list = anod_branch.iloc[argrelextrema(anod_branch[curr].values, np.greater_equal, order=20)].copy()
    max_list.sort_values(by=[curr], inplace=True, ignore_index=True, ascending=False)

    min_list = cat_branch.iloc[argrelextrema(cat_branch[curr].values, np.less_equal, order=20)].copy()
    min_list.sort_values(by=[curr], inplace=True, ignore_index=True, ascending=True)

    #Alla fine viene popolata una lista di dizionari
    #Ogni dizionario contiene un dataframe per i rami catodico e anodico e lo scan rate
    df_list[i] = {
        "cat": cat_branch,
        "anod": anod_branch,
        "scan_rate": scan_rate,
        "max": max_list,
        "min": min_list
    }

#====================================
# Analisi dati
#====================================

dict_list = []
sub_df_list = []
mode = 1

if mode == 1:
    fig, (ax1, ax2) = plt.subplots(1, 2)

    #Slicing del potenziale in intervalli
    slice_list = np.linspace(1.10, 2.60, num=21)

    for i in range(len(df_list)):
        ax1.plot(df_list[i]["cat"][pot], df_list[i]["cat"][curr], c="blue")
        ax1.plot(df_list[i]["anod"][pot], df_list[i]["anod"][curr], c="red")

    ax1.vlines(slice_list, -0.5, 0.5, colors="black", linestyles="dashed", lw=0.5)

    for target_pot in slice_list:

        target_pot = target_pot.__round__(3)

        x = []
        y = []

        for i in range(len(df_list)):
            df = df_list[i]["cat"]
            plot_curr = df.iloc[(df[pot]-target_pot).abs().argsort()[:1]][[curr]].values.tolist()[0][0]
            plot_pot = df.iloc[(df[pot]-target_pot).abs().argsort()[:1]][[pot]].values.tolist()[0][0]

            ax1.scatter(plot_pot, plot_curr, s=10, c="black")

            x.append(np.sqrt(df_list[i]["scan_rate"]))
            y.append(plot_curr/x[i])
            
        sub_df_list.append([pd.DataFrame({"x": x, "y": y}).sort_values(by="x"), target_pot])

    for sub_df in sub_df_list:    
        ax2.plot(sub_df[0]["x"], sub_df[0]["y"], label=sub_df[1])
        ax2.legend()

    ax1.set_xlabel(pot)
    ax1.set_ylabel(curr)

    ax2.set_xlabel(r"$\nu ^{1/2}$")
    ax2.set_ylabel(r"i / $\nu ^{1/2}$")

elif mode == 2:

    fig, (ax1, ax2) = plt.subplots(1, 2)

    cat_peak_list = []
    anod_peak_list = []

    for i in range(len(df_list)):
        #Rinomino variabili
        df_anod = df_list[i]["anod"]
        df_cat = df_list[i]["cat"]
        df_max = df_list[i]["max"]
        df_min = df_list[i]["min"]

        #Prendo corrente e potenziale del primo massimo
        current_anodpeak_curr = df_max[curr][0]
        current_anodpeak_pot = df_max[pot][0]
        anod_peak_list.append([current_anodpeak_pot, current_anodpeak_curr, df_list[i]["scan_rate"]])

        #Prendo corrente e potenziale del primo minimo
        current_catpeak_curr = df_min[curr][0]
        current_catpeak_pot = df_min[pot][0]
        cat_peak_list.append([current_catpeak_pot, current_catpeak_curr, df_list[i]["scan_rate"]])

        ax1.scatter(df_anod[pot], df_anod[curr], c="black", s=1)
        ax1.scatter(df_cat[pot], df_cat[curr], c="black", s=1)

        ax1.scatter(current_anodpeak_pot, current_anodpeak_curr, c="blue", s=40)
        ax1.scatter(current_catpeak_pot, current_catpeak_curr, c="red", s=40)

    #Creo due dataframe per i massimi e minimi che userò per plottare e fare i calcoli
    cat_peaks = pd.DataFrame(cat_peak_list, columns=[pot, curr, scrate])
    anod_peaks = pd.DataFrame(anod_peak_list, columns=[pot, curr, scrate])

    ax2.scatter(cat_peaks[scrate], cat_peaks[curr], c="red")
    ax2.scatter(anod_peaks[scrate], anod_peaks[curr], c="blue")


    #Fit della power law
    def power_law(x, a, b):
        return a*np.power(x, b)

    #Ramo anodico
    popt, pcov = curve_fit(power_law, anod_peaks[scrate], anod_peaks[curr], method="lm")

    residuals = anod_peaks[curr] - power_law(anod_peaks[scrate], *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((anod_peaks[curr] - np.mean(anod_peaks[curr]))**2)

    anod_r_squared = 1 - (ss_res/ss_tot)

    x_data_for_curve = np.linspace(anod_peaks[scrate].min(), anod_peaks[scrate].max(), num=50)
    ax2.plot(x_data_for_curve, power_law(x_data_for_curve, *popt), c="blue", linestyle="dashed", label=r'i = a$\nu^b$ with a=%5.3f, b=%5.3f' % tuple(popt))

    #Ramo catodico
    popt, pcov = curve_fit(power_law, cat_peaks[scrate], cat_peaks[curr], method="lm")

    residuals = cat_peaks[curr] - power_law(cat_peaks[scrate], *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((cat_peaks[curr] - np.mean(cat_peaks[curr]))**2)

    cat_r_squared = 1 - (ss_res/ss_tot)

    x_data_for_curve = np.linspace(cat_peaks[scrate].min(), cat_peaks[scrate].max(), num=50)
    ax2.plot(x_data_for_curve, power_law(x_data_for_curve, *popt), c="red", linestyle="dashed", label=r'i = a$\nu^b$ with a=%5.3f, b=%5.3f' % tuple(popt))

    ax1.set_xlabel("E (V)")
    ax1.set_ylabel("i (A)")

    ax2.set_xlabel(r"$\nu$ (V/s)")
    ax2.set_ylabel("i (A)")

    ax2.legend()

    cat_peaks.to_excel(file_dir + "\\Picchi catodici.xlsx")
    anod_peaks.to_excel(file_dir + "\\Picchi anodici.xlsx")

plt.show()