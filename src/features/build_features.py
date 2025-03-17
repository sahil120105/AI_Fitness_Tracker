import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


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


# ----------------------------------------------------
# Temporal Abstraction
# ----------------------------------------------------


df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_cols = predictor_cols + ["acc_r","gyr_r"]

ws = int(1000/200)          # each entry has a 200ms gap


# loop to extract temporal features for each set
df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    
    for col in predictor_cols:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    
    df_temporal_list.append(subset)
    

# concatinate all the subsets
df_temporal = pd.concat(df_temporal_list)
    

# plot the new temporal features
subset[["acc_x","acc_x_temp_mean_ws_5", "acc_x_temp_std_ws_5"]].plot()


# ----------------------------------------------------
# Frequency Features
# ----------------------------------------------------


df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

fs = int(1000/200)
ws = int(2000/200)

# looping over all the sets
df_freq_list = []
for s in df_freq["set"].unique():
    
    print(f"Applying Fourier Transformation to Set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_cols, ws, fs)
    df_freq_list.append(subset)
    
# concatinating the subsets
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)


# ----------------------------------------------------
# Dealing with Overlapping Windows
# ----------------------------------------------------


df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]


# ----------------------------------------------------
# Clustering
# ----------------------------------------------------


df_cluster = df_freq.copy()

cluster_cols = ["acc_x","acc_y","acc_z"]
k_values = range(2,10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_cols]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)
    
    
#plot inertias for elbow method
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

# selected k from elbow method
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
susbset = df_cluster[cluster_cols]
df_cluster["cluster"] = kmeans.fit_predict(subset)


# Plot clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()


# Plot accelerometer data to compare
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

plt.legend()
plt.show()


# ----------------------------------------------------
# Export Data
# ----------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")