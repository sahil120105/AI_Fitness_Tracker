import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# ----------------------------------------------------
# Load the data
# ----------------------------------------------------


df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictor_cols = list(df.columns[:6])

# plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# ----------------------------------------------------
# Dealing with missing values (imputation)
# ----------------------------------------------------


for col in predictor_cols:
    df[col] = df[col].interpolate()
    
df.info()


# ----------------------------------------------------
# Calculating set duration
# ----------------------------------------------------


# plotting individual set
df[df["set"] == 25]["acc_y"].plot()
df[df["set"] == 50]["acc_y"].plot()

# calculate duration
duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds

# loop over all the sets
for s in df["set"].unique():
    
    start = df[df["set"] == s].index[0]
    end = df[df["set"] == s].index[-1]
    
    duration = end - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds
    
# grouping by category (taking mean)
duration_df = df.groupby(["category"])["duration"].mean()    

duration_df.iloc[0] /5
duration_df.iloc[1] /10


# ----------------------------------------------------
# Butterworth lowpass filter (smoothening)
# ----------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000/200
cutoff = 1.3      # lower the cutoff more the smoothening

# applying the filter
df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

#checking a single set
subset = df_lowpass[df_lowpass["set"] == 12]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# loop over all the columns and overwrite orignal data
df_lowpass = df.copy()
for col in predictor_cols:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col+"_lowpass"]
    del df_lowpass[col+"_lowpass"]
    
    
# ----------------------------------------------------
# Principal Component Analysis (PCA)
# ----------------------------------------------------


df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

# pc_values: Percentage of variance captured by each principal component.
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_cols)


# Plot graph to apply elbow method
# Elbow method: determine the optimal number of components to use when conducting a PCA.
# Select the point at which the rate of change in variance diminishes (the "elbow")
# Elbow is the optimal number of Principal Components
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_cols) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()


# applying pca (last attiribute comes fromt the elbow method)
df_pca = PCA.apply_pca(df_pca, predictor_cols, 3)


# ----------------------------------------------------
# Sum of Squares Attribute
# ----------------------------------------------------


df_squared = df_pca.copy()

# calculating the vectors of acc and gyr
acc_r = df_squared["acc_x"]**2 + df_squared["acc_y"]**2 + df_squared["acc_z"]**2
gyr_r = df_squared["gyr_x"]**2 + df_squared["gyr_y"]**2 + df_squared["gyr_z"]**2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

# plotting acc_r and gyr_r
subset = df_squared[df_squared["set"] == 39]
subset[["acc_r", "gyr_r"]].plot(subplots=True)