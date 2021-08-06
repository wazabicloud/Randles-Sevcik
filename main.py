from pandas.core.frame import DataFrame
import AdmiralDecode, BiologicDecode
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

# Opzioni
# get_files genera i files dei dati e dei report alla fine
# mode seleziona il tipo di analisi:
#   peakfit:    fit dei picchi con la randles-sevcik
#   slice:      divisione della cv in contributi capacitivi e faradici

get_files = False
mode = "peakfit"

#Area campione
area = 1.0

#Parametri per ricerca picchi
E_range = (0.2, 0.8)
max_order = 300
min_order = 300

#Parametri per slices
slices = 200
plot_cap_contrib = True
plot_far_contrib = True
plot_total_contrib = False
plot_R2 = True

#Nomi finali delle colonne per unificare i dataframe
pot = "Potential (V)"
curr = "Current (A)"
time = "Time (s)"
scrate = "Scan rate (V/s)"
cathodic = "Cathodic Branch"
anodic = "Anodic Branch"
peaks = "peaks"

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

def musca_import(file):

    data_pot_name = "Average Potential (V)"
    data_curr_name = "Average Cumulative Current (A)"
    data_time_name = "Equivalent Time (s)"

    df = pd.read_excel(file, index_col=0)

    df = df[[data_pot_name, data_curr_name, data_time_name]].copy()
    df.rename({data_pot_name: pot, data_curr_name: curr, data_time_name: time}, axis="columns", inplace=True)

    return df

for file in os.listdir(file_dir):

    file = file_dir + "\\" + file
        
    if ".csv" in file:
        df_list.append(admiral_import(file))
    elif ".mpt" in file:
        df_list.append(biologic_import(file))
    elif ".xlsx" in file:
        df_list.append(musca_import(file))
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
    cathodic_branch = df_list[i][(df_list[i][pot].diff(1) < 0)].sort_values(by=[pot])
    anodic_branch = df_list[i][(df_list[i][pot].diff(1) > 0)].sort_values(by=[pot])

    #Calcolo lo scan rate in automatico
    scan_rate = ((df_list[i][pot].diff(1).abs())/df_list[i][time].diff(1)).mean()

    #Alla fine viene popolata una lista di dizionari
    #Ogni dizionario contiene un dataframe per i rami catodico e anodico e lo scan rate

    df_list[i] = {
        cathodic: cathodic_branch,
        anodic: anodic_branch,
        scrate: scan_rate
    }

#====================================
# Analisi dati
#====================================

#Funzioni per fit
def power_law(x, a, b):
    return a*np.power(x, b)

def linear(x, a, b):
    return (a*x) + b

def square_root(x, a):
    return a * np.power(x, 0.5)

#Plot delle curve normali

if mode == "slice" and plot_R2 == True:
    fig = plt.figure()
    spec = fig.add_gridspec(2, 2, height_ratios=(2, 7))
    
    ax1 = fig.add_subplot(spec[:, 0])
    ax2 = fig.add_subplot(spec[1, 1])
    ax3 = fig.add_subplot(spec[0, 1], sharex=ax2)
else:
    fig, (ax1, ax2) = plt.subplots(1, 2)

for df in df_list:
    ax1.scatter(df[anodic][pot], df[anodic][curr], c="black", s=2)
    ax1.scatter(df[cathodic][pot], df[cathodic][curr], c="black", s=2)

ax1.set_xlabel("E (V)")
ax1.set_ylabel("j (A/cm$^2$)")

ax1.tick_params(which="both", direction="in")
ax2.tick_params(which="both", direction="in")

#Funzione per trovare quanti picchi sono contenuti all'interno di un range di potenziale
def peaks_in_range(max_order:float, min_order:float, E_range: tuple[float, float], df:DataFrame):

    if E_range[0] > E_range[1]:
        E_max = E_range[0]
        E_min = E_range[1]
    else:
        E_min = E_range[0]
        E_max = E_range[1]

    #Escludo i picchi esterni al range
    anodic_in_range = df[anodic].loc[((df[anodic][pot] > E_min) & (df[anodic][pot] < E_max))]
    cathodic_in_range = df[cathodic].loc[((df[cathodic][pot] > E_min) & (df[cathodic][pot] < E_max))]

    #Trovo quanti massimi e minimi ho nel range    
    anodic_peaks_in_range = anodic_in_range.iloc[argrelextrema(anodic_in_range[curr].values, np.greater_equal, order=max_order)].copy()
    cathodic_peaks_in_range = cathodic_in_range.iloc[argrelextrema(cathodic_in_range[curr].values, np.less_equal, order=min_order)].copy()

    return anodic_peaks_in_range,cathodic_peaks_in_range

#Funzione da chiamare per fittare le funzioni e plottare i risultati
def fit_and_plot(peak_set, fit_func, x_var, y_var):
    
    #Bisogna cambiare il try/except o almeno cercare di migliorare il fit
    
    try:
        #Fit della power law
        popt, pcov = curve_fit(fit_func, x_var, y_var)

        #Calcolo R^2
        residuals = y_var - fit_func(x_var, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_var - np.mean(y_var))**2)
        r_squared = 1 - (ss_res/ss_tot)

        labels = np.append(popt, r_squared)

        #Plot del fit                
        x_data_for_curve = np.linspace(x_var.min(), x_var.max(), num=50)
        ax2.plot(x_data_for_curve, fit_func(x_data_for_curve, *popt), linestyle="dashed", label=r'i = a$\nu^b$ with a=%5.3f b=%5.3f and R$^2$=%5.3f' % tuple(labels))

    except:
        pass

    ax1.scatter(peak_set[pot], peak_set[curr], s=30, edgecolors="black")
    ax2.scatter(x_var, y_var, edgecolors="black")

if mode == "peakfit":

    # while True:
    #     E_range[0] = input("Da che valore di potenziale devo cercare i picchi?\n")
    #     try:
    #         E_range[0] = float(E_range[0])
    #         break
    #     except:
    #         print(E_range[0], "non è un numero valido")

    # while True:
    #     E_range[1] = input("Fino a che valore di potenziale devo cercare i picchi?\n")
    #     try:
    #         E_range[1] = float(E_range[1])
    #         break
    #     except:
    #         print(E_range[1], "non è un numero valido")

    num_of_cat_peaks = -1
    num_of_anod_peaks = -1

    #Trovo quanti picchi anodici e catodici ci sono
    for df in df_list:

        anodic_peaks_in_range, cathodic_peaks_in_range = peaks_in_range(max_order, min_order, E_range, df)

        #Alla fine del ciclo for il numero di picchi deve essere il più basso tra tutti quelli trovati
        #perché il fit si può fare solo se i picchi sono presenti in ogni scan rate

        if len(anodic_peaks_in_range) < num_of_anod_peaks or num_of_anod_peaks == -1:
            num_of_anod_peaks = len(anodic_peaks_in_range)

        if len(cathodic_peaks_in_range) < num_of_cat_peaks or num_of_cat_peaks == -1:
            num_of_cat_peaks = len(cathodic_peaks_in_range)

    #Se la ricerca trova qualcosa (uno dei due num_of_peaks è > 0), raggruppo i picchi in set
    if num_of_anod_peaks > 0 or num_of_cat_peaks > 0:
        for i in range(len(df_list)):
            anodic_peaks_in_range, cathodic_peaks_in_range = peaks_in_range(max_order, min_order, E_range, df_list[i])

            # Filtro temporaneo
            # if i == 0:
            #     temp_range = (-3, 2)
            #     anodic_peaks_in_range, cathodic_peaks_in_range = peaks_in_range(max_order, min_order, temp_range, df_list[i])

            #Per capire quali sono i picchi più prominenti uso come criterio la corrente, mentre
            #per raggrupparli in un set uso il potenziale

            if num_of_anod_peaks > 0:
                anodic_peaks_in_range.sort_values(by=[curr], inplace=True, ignore_index=True, ascending=False)
                anodic_peaks_in_range = anodic_peaks_in_range.iloc[0:num_of_anod_peaks]
                anodic_peaks_in_range.sort_values(by=[pot], inplace=True, ignore_index=True)
            else:
                anodic_peaks_in_range = pd.DataFrame()

            if num_of_cat_peaks > 0:
                cathodic_peaks_in_range.sort_values(by=[curr], inplace=True, ignore_index=True, ascending=True)
                cathodic_peaks_in_range = cathodic_peaks_in_range.iloc[0:num_of_cat_peaks]
            else:
                cathodic_peaks_in_range = pd.DataFrame()

            df_list[i][peaks] = pd.concat([anodic_peaks_in_range, cathodic_peaks_in_range], ignore_index=True)
            peak_num = num_of_anod_peaks + num_of_cat_peaks

        #Vado a prendere il primo picco in ordine da ogni curva a diverso scan rate e creo
        #un set di picchi considerato come singolo dataframe

        peakset_df_list = []

        for i in range(peak_num):

            current_peakset = []

            for df in df_list:
                current_peak_curr = df[peaks][curr][i]
                current_peak_pot = df[peaks][pot][i]

                current_peakset.append([current_peak_pot, current_peak_curr, df[scrate]])

            current_peak_df = pd.DataFrame(current_peakset, columns=[pot, curr, scrate])

            peakset_df_list.append(current_peak_df)

        #A questo punto faccio il fit della randles-sevcik e plotto i risultati

        for peak_set in peakset_df_list:

            x = peak_set[scrate]
            y = peak_set[curr]

            fit_and_plot(peak_set, power_law, x, y)

        ax2.legend()

        ax2.set_xlabel(r"$\nu$ (V/s)")
        ax2.set_ylabel("j (A/cm$^2$)")

    #Se richiesto, tiro fuori i files
    if get_files == True:
        for i in range(len(peakset_df_list)):
            peak_set.to_excel(file_dir + "\\peak" + str(i) + ".xlsx")

        for i in range(len(df_list)):
            cv_df = pd.concat([df_list[i][cathodic], df_list[i][anodic]], ignore_index=True)
            cv_df[scrate] = df_list[i][scrate]
            cv_df.sort_values(by=[time], inplace=True, ignore_index=True)
            cv_df.to_excel(file_dir + "\\CV_" + str(i) + ".xlsx")

elif mode == "slice":

    #Trovo gli estremi più interni
    min_pot = None
    max_pot = None

    for df in df_list:

        if min_pot == None:
            min_pot = df[cathodic][pot].min()
        if max_pot == None:
            max_pot = df[cathodic][pot].max()

        for branch in [cathodic, anodic]:
            if min_pot < df[branch][pot].min():
                min_pot = df[branch][pot].min()
            
            if max_pot > df[branch][pot].max():
                max_pot = df[branch][pot].max()

    #slice_fit è un dataframe che viene popolato con i risultati del fit
    slice_fit_df = pd.DataFrame(columns=["k1", "k2", pot, "branch", "R2"])

    #Divido il potenziale in tanti intervalli
    slice_list = np.linspace(min_pot, max_pot, num=slices)

    #Per ogni intervallo trovo i punti nelle curve più vicini e faccio il fit lineare
    for target_pot in slice_list:

        for branch in [anodic, cathodic]:

            closer_points_df = pd.DataFrame()

            for i in range(len(df_list)):
            
                df = df_list[i][branch]

                closer_point = df.iloc[(df[pot]-target_pot).abs().argsort()[:1]].copy()

                closer_point[scrate] = df_list[i][scrate]

                closer_points_df = pd.concat([closer_points_df, closer_point], ignore_index=True)

            x_to_fit = np.sqrt(closer_points_df[scrate])
            y_to_fit = closer_points_df[curr].div(np.sqrt(closer_points_df[scrate]))

            popt, pcov = curve_fit(linear, x_to_fit, y_to_fit)

            k1 = popt[0]
            k2 = popt[1]

            #Calcolo R^2
            residuals = y_to_fit - linear(x_to_fit, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_to_fit - np.mean(y_to_fit))**2)
            r_squared = 1 - (ss_res/ss_tot)

            slice_fit_df = slice_fit_df.append({"k1":k1, "k2":k2, pot:target_pot, "branch":branch, "R2":r_squared}, ignore_index=True)

    #Trovo la curva a scan rate più lento

    smallest_scrate = -1
    smallest_curve = None

    for i in range(len(df_list)):

        df_list[i][cathodic].sort_values(by=[pot], ascending=True, ignore_index=True, inplace=True)
        df_list[i][anodic].sort_values(by=[pot], ascending=False, ignore_index=True, inplace=True)

        df_list[i][cathodic].reset_index(drop=True, inplace=True)
        df_list[i][anodic].reset_index(drop=True, inplace=True)

        if df_list[i][scrate] < smallest_scrate or smallest_scrate == -1:
            smallest_scrate = df_list[i][scrate]
            smallest_curve = df_list[i]

    #Evidenzio la curva nel grafico di destra colorandola
    ax1.plot(smallest_curve[cathodic][pot], smallest_curve[cathodic][curr], c="red", lw=2)
    ax1.plot(smallest_curve[anodic][pot], smallest_curve[anodic][curr], c="red", lw=2)

    ax2.plot(smallest_curve[cathodic][pot], smallest_curve[cathodic][curr], c="black")
    ax2.plot(smallest_curve[anodic][pot], smallest_curve[anodic][curr], c="black")

    #Calcolo contributi capacitivi e faradici della curva più lenta
    slice_fit_df["cap_current"] = slice_fit_df["k1"].mul(smallest_scrate)
    slice_fit_df["far_current"] = slice_fit_df["k2"].mul(np.sqrt(smallest_scrate))
    slice_fit_df["total_current"] = slice_fit_df["cap_current"] + slice_fit_df["far_current"]

    #Sistemo le curve in modo da avere un fill corretto
    slice_fit_df.sort_values(by=pot, ascending=True, inplace=True)

    slice_fit_df["graph_index"] = np.where(slice_fit_df["branch"] == anodic, slice_fit_df.index, -slice_fit_df.index)

    slice_fit_df.sort_values(by=["graph_index"], ascending=True, inplace=True)

    #Plot dei contributi
    if plot_cap_contrib == True:
        ax2.fill(slice_fit_df[pot], slice_fit_df["cap_current"], c="red", alpha=0.5)

    if plot_far_contrib == True:
        ax2.fill(slice_fit_df[pot], slice_fit_df["far_current"], c="blue", alpha=0.5)

    if plot_total_contrib == True:
        ax2.fill(slice_fit_df[pot], slice_fit_df["total_current"], c="purple", alpha=0.5)

    #Se richiesto plotto R^2
    if plot_R2 == True:

        ax3.plot(slice_fit_df[slice_fit_df["branch"] == anodic][pot], slice_fit_df[slice_fit_df["branch"] == anodic]["R2"], label="Anodic")
        ax3.plot(slice_fit_df[slice_fit_df["branch"] == cathodic][pot], slice_fit_df[slice_fit_df["branch"] == cathodic]["R2"], label="Cathodic")

        ax3.tick_params(which="both", direction="in", labelbottom=False)
        ax3.set_ylabel(r"R$^2$")
        ax3.set_ylim(0, 1)
        ax3.legend(frameon=False)

    ax2.set_xlabel(r"E (V)")
    ax2.set_ylabel(r"j (A/cm$^2$)")

    #Se richiesto, tiro fuori i files
    if get_files == True:

        for i in range(len(df_list)):
            cv_df = pd.concat([df_list[i][cathodic], df_list[i][anodic]], ignore_index=True)
            cv_df[scrate] = df_list[i][scrate]
            cv_df.sort_values(by=[time], inplace=True, ignore_index=True)
            cv_df.to_excel(file_dir + "\\CV_" + str(i) + ".xlsx")

        results = slice_fit_df[["k1", "k2", pot, "branch", "cap_current", "far_current"]]
        results.to_excel(file_dir + "\\report.xlsx")

plt.show()