# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:17:19 2018

@author: petit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scistats

df = pd.read_csv("toimport.csv")

df[["WACh", "WAPa", "WAEx", "WATa", "WATr", "WARe", "WADi", "WAFr"]] = df[["WACh", "WAPa", "WAEx", "WATa", "WATr", "WARe", "WADi", "WAFr"]].fillna(value="")

df["allwords"] = df.WACh + ":" + df.WAPa + ":" + df.WAEx + ":" + df.WATa + ":" + df.WATr + ":" + df.WARe + ":" + df.WADi + ":" + df.WAFr + ":"

text = ""

for num, row in df.iterrows():
    text = text + row["allwords"] + ":"
    
text = text.lower()
words = text.split(":")
words, counts = np.unique(words, return_counts=True)
wordsm = words[counts >=50]
words = words[counts >= 3]

plotcounts = counts.copy()
plotcounts[counts>10] = 10
plt.hist(plotcounts[1:], bins=10, range=(1,10), color="green")
plt.title("Total association frequency")
plt.ylabel("Number of words")
plt.xlabel("Frequency of occurrence")
plt.savefig("AssFreq_Histogram.png")
plt.show()

ad = pd.read_csv("toimportadj.csv")

wordlist = []
for colname in ad.columns[1:]:
    word = colname.split("[")[1]
    word = word.split("|")[0]
    wordlist.append(word)

ad = ad-1
means = ad.mean(axis=0)
means = means.values[1:]

plt.hist(means, bins=15, range=(-1,1), color="green")
plt.title("Emotional loading distribution")
plt.ylabel("Number of words")
plt.savefig("EmLo_Histogram.png")
plt.show()

adjloadings = dict(zip(wordlist, means))

def mean_loading_of_string(s, adjloadings, sep=":"):
    toeval = s.lower()
    words = toeval.split(sep)
    meanlist = []
    for w in words:
        if w == "":
            continue
        try:
            meanlist.append(adjloadings[w])
        except Exception as E:
            continue
    if meanlist == []:
        return 0        
    else:
        return np.mean(meanlist)

MLCh_list = []
MLPa_list = []
MLEx_list = []
MLTa_list = []
MLTr_list = []
MLRe_list = []
MLDi_list = []
MLFr_list = []

for num, row in df.iterrows():
    MLCh = mean_loading_of_string(row["WACh"], adjloadings)
    MLCh_list.append(MLCh)

    MLPa = mean_loading_of_string(row["WAPa"], adjloadings)
    MLPa_list.append(MLPa)
    
    MLEx = mean_loading_of_string(row["WAEx"], adjloadings)
    MLEx_list.append(MLEx)
    
    MLTa = mean_loading_of_string(row["WATa"], adjloadings)
    MLTa_list.append(MLTa)
    
    MLTr = mean_loading_of_string(row["WATr"], adjloadings)
    MLTr_list.append(MLTr)
    
    MLRe = mean_loading_of_string(row["WARe"], adjloadings)
    MLRe_list.append(MLRe)
    
    MLDi = mean_loading_of_string(row["WADi"], adjloadings)
    MLDi_list.append(MLDi)
    
    MLFr = mean_loading_of_string(row["WAFr"], adjloadings)
    MLFr_list.append(MLFr)
 

df["MLCh"] = MLCh_list
df["MLPa"] = MLPa_list
df["MLEx"] = MLEx_list
df["MLTa"] = MLTa_list
df["MLTr"] = MLTr_list
df["MLRe"] = MLRe_list
df["MLDi"] = MLDi_list
df["MLFr"] = MLFr_list

del MLCh_list, MLPa_list, MLEx_list, MLTa_list, MLTr_list, MLRe_list, MLDi_list, MLFr_list, MLCh, MLDi, MLEx, MLFr, MLPa, MLRe, MLTa, MLTr

df["WHO5-Score"] = df[["WHOA1", "WHOA2", "WHOA3", "WHOA4", "WHOA5"]].sum(axis=1)
plt.hist(df["WHO5-Score"], bins=15, color="green")
plt.title("WHO5 Scores")
plt.ylabel("Number of participants")
plt.savefig("WHO_Histogram.png")
plt.show()

corr = df[["WHO5-Score", "MLCh", "MLPa", "MLEx", "MLTa", "MLTr", "MLRe", "MLDi", "MLFr"]].corr()

plt.hist(df["Age"], bins=17, color="green")
plt.title("Age of participants")
plt.ylabel("Numer of participants")
plt.savefig("Age_Histogram.png")
plt.show()

df["MeanTotal"] = df[["MLCh", "MLPa", "MLEx", "MLTa", "MLTr", "MLRe", "MLDi", "MLFr"]].sum(axis=1)

correl = df[["WHO5-Score", "MeanTotal"]].corr()

r, p = scistats.pearsonr(df["WHO5-Score"], df["MLCh"])
print ("WHO5-Score - Chair")
print("Correlation:", r, "P-Value", p)
plt.scatter(df["WHO5-Score"], df["MLCh"])
plt.title("Chair")
plt.ylabel("Emotional loading average")
plt.xlabel("WHO-5 Score sum")
plt.savefig("Chair_Histogram.png")
plt.show()

r, p = scistats.pearsonr(df["WHO5-Score"], df["MLPa"])
print ("WHO5-Score - Party")
print("Correlation:", r, "P-Value", p)
plt.scatter(df["WHO5-Score"], df["MLPa"])
plt.title("Party")
plt.ylabel("Emotional loading average")
plt.xlabel("WHO-5 Score sum")
plt.savefig("Party_Histogram.png")
plt.show()

r, p = scistats.pearsonr(df["WHO5-Score"], df["MLEx"])
print ("WHO5-Score - Exam")
print("Correlation:", r, "P-Value", p)
plt.scatter(df["WHO5-Score"], df["MLEx"])
plt.title("Exam")
plt.ylabel("Emotional loading average")
plt.xlabel("WHO-5 Score sum")
plt.savefig("Exam_Histogram.png")
plt.show()

r, p = scistats.pearsonr(df["WHO5-Score"], df["MLTa"])
print ("WHO5-Score - Table")
print("Correlation:", r, "P-Value", p)
plt.scatter(df["WHO5-Score"], df["MLTa"])
plt.title("Table")
plt.ylabel("Emotional loading average")
plt.xlabel("WHO-5 Score sum")
plt.savefig("Table_Histogram.png")
plt.show()

r, p = scistats.pearsonr(df["WHO5-Score"], df["MLTr"])
print ("WHO5-Score - Travel")
print("Correlation:", r, "P-Value", p)
plt.scatter(df["WHO5-Score"], df["MLTr"])
plt.title("Travel")
plt.ylabel("Emotional loading average")
plt.xlabel("WHO-5 Score sum")
plt.savefig("Travel_Histogram.png")
plt.show()

r, p = scistats.pearsonr(df["WHO5-Score"], df["MLRe"])
print ("WHO5-Score - Responsibility")
print("Correlation:", r, "P-Value", p)
plt.scatter(df["WHO5-Score"], df["MLRe"])
plt.title("Responsibility")
plt.ylabel("Emotional loading average")
plt.xlabel("WHO-5 Score sum")
plt.savefig("Responsibility_Histogram.png")
plt.show()

r, p = scistats.pearsonr(df["WHO5-Score"], df["MLDi"])
print ("WHO5-Score - Divorce")
print("Correlation:", r, "P-Value", p)
plt.scatter(df["WHO5-Score"], df["MLDi"])
plt.title("Divorce")
plt.ylabel("Emotional loading average")
plt.xlabel("WHO-5 Score sum")
plt.savefig("Divorce_Histogram.png")
plt.show()

r, p = scistats.pearsonr(df["WHO5-Score"], df["MLFr"])
print ("WHO5-Score - Freedom")
print("Correlation:", r, "P-Value", p)
plt.scatter(df["WHO5-Score"], df["MLFr"])
plt.title("Freedom")
plt.ylabel("Emotional loading average")
plt.xlabel("WHO-5 Score sum")
plt.savefig("Freedom_Histogram.png")
plt.show()

r, p = scistats.pearsonr(df["WHO5-Score"], df["MeanTotal"])
print ("WHO5-Score - Total")
print("Correlation:", r, "P-Value", p)
plt.scatter(df["WHO5-Score"], df["MeanTotal"])
plt.title("Total")
plt.ylabel("Emotional loading average")
plt.xlabel("WHO-5 Score sum")
plt.savefig("Total_Histogram.png")
plt.show()