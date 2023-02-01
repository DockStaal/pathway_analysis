from os import wait
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy.linalg import svd
import matplotlib.pyplot as plt
import sys
import random as rnd
sys.path.append("..") # Read loacal modules for tcga access and qvalue calculations
import tcga_read as tcga
import utils as ut

# Read data
brca = tcga.get_expression_data("../cb2030/data/brca.tsv.gz", 'http://download.cbioportal.org/brca_tcga_pub2015.tar.gz',"data_RNA_Seq_v2_expression_median.txt")
brca_clin = tcga.get_clinical_data("../cb2030/data/brca_clin.tsv.gz", 'http://download.cbioportal.org/brca_tcga_pub2015.tar.gz',"data_clinical_sample.txt")
brca.dropna(axis=0, how='any', inplace=True)
brca = brca.loc[~(brca<=0.0).any(axis=1)]
brca = pd.DataFrame(data=np.log2(brca),index=brca.index,columns=brca.columns)
brca_clin.loc["3N"]= (brca_clin.loc["PR status by ihc"]=="Negative") & (brca_clin.loc["ER Status By IHC"]=="Negative") & (brca_clin.loc["IHC-HER2"]=="Negative")
tripple_negative_bool = (brca_clin.loc["3N"] == True)
random_bool = pd.Series([rnd.choice([True, False]) for i in range(817)]) #Randomly selected as a test

# Give the Y_p matrix for the specific pathway p
pathway_genes = ut.return_pathway_genes()
Y = brca.loc[pathway_genes]

# Calculate U and V with the PMF method
model = ut.PMF(np.array(Y), 1)
U_PMF = model.map["U"]
V_PMF = model.map["V"]
U_PMF = U_PMF.reshape((U_PMF.shape[0],))
V_PMF = np.abs(V_PMF.reshape((V_PMF.shape[0],)))

# Calculate U and V with the SVD method
U_TEMP, S_TEMP, V_TEMP = svd(Y, full_matrices=False)
U_SVD = U_TEMP[:,0]
V_SVD = V_TEMP[0,:]
S_SVD = S_TEMP[0]

U_SVD = U_SVD.reshape((len(U_SVD),)) * np.sqrt(S_SVD)
V_SVD = np.abs(V_SVD.reshape((len(V_SVD),)) * np.sqrt(S_SVD))

print("PMF U: ", U_PMF.shape)
print("PMF V: ", V_PMF.shape)

print("SVD U: ", U_SVD.shape)
print("SVD V: ", V_SVD.shape)

# print the values of V only for the triple negative patients
# mu_trip = np.mean(V[tripple_negative_bool], axis=0)
# mu_non_trip = np.mean(V[~tripple_negative_bool], axis=0)
# 
# print("The mean of the first component of V for the triple negative patients is: ", mu_trip[0])
# print("The mean of the first component of V for the non triple negative patients is: ", mu_non_trip[0])
# 
# # Cacluclate the p-value using a t-test 
pval_PMF = ttest_ind(V_PMF[tripple_negative_bool], V_PMF[~tripple_negative_bool], axis=0)[1]
print("p-value PMF for pathway component is: ", pval_PMF)

pval_SVD = ttest_ind(V_SVD[tripple_negative_bool], V_SVD[~tripple_negative_bool], axis=0)[1]
print("p-value SVD for pathway component is: ", pval_SVD)

data = [V_PMF[tripple_negative_bool], V_PMF[~tripple_negative_bool], V_SVD[tripple_negative_bool], V_SVD[~tripple_negative_bool]]

fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)
# Creating axes instance
bp = ax.boxplot(data, patch_artist = True, notch ='True', vert = 0)
colors = ['#0000FF', '#00FF00','#FFFF00', '#FF00FF']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":")

# changing color and linewidth of
# caps
for cap in bp['caps']:
    cap.set(color ='#8B008B',
            linewidth = 2)

# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color ='red',
                linewidth = 3)
    
# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)

# x-axis labels
ax.set_yticklabels(['PMF TN', 'PMF non-TN', 'SVD TN', 'SVD non-TN'])
ax.set_xlabel("Pathway activity level")

# Adding title
plt.title("Box plot of pathway activity levels")

# Removing top axes and right axes
# ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# Save figure as png
plt.savefig("boxplot.png")

# show plot
plt.show()

# print("The check p-value is: ", ttest_ind(V[random_bool], V[~random_bool], axis=0)[1])
