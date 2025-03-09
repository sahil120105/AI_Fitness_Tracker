import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
import numpy as np

# ----------------------------------------------------
# Load the data
# ----------------------------------------------------


df = pd.read_pickle("../../data/interim/01_data_processed.pkl")


# ----------------------------------------------------
# Plot single columns
# ----------------------------------------------------


set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"].reset_index(drop=True))


# ----------------------------------------------------
# Plot all exercises
# ----------------------------------------------------


for label in df["label"].unique():
    subset = df[df["label"] ==  label]
    
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()
    
    
# ----------------------------------------------------
# Adjust Plot Settings
# ----------------------------------------------------


mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20,5)
mpl.rcParams["figure.dpi"] = 100


# ----------------------------------------------------
# Compare medium vs heavy sets
# ----------------------------------------------------


category_df = df[(df["label"] == "squat") & (df["participant"] == "A")].reset_index()

fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("sample")
plt.legend()


# ----------------------------------------------------
# Compare participants
# ----------------------------------------------------


participants_df = df[df["label"] == "bench"].sort_values("participant").reset_index()

fig, ax = plt.subplots()
participants_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("sample")
plt.legend()


# ----------------------------------------------------
# Plot multiple axis
# ----------------------------------------------------


participant = "A"
label = "squat"

all_axis_df = df[ (df["participant"]==participant) & (df["label"] == label)].reset_index()

fig, ax = plt.subplots()
all_axis_df[["acc_x","acc_y","acc_z"]].plot(ax=ax)
ax.set_ylabel("acc")
ax.set_xlabel("sample")
plt.legend()


# ----------------------------------------------------
# Create loop to plot all combinations per sensor
# ----------------------------------------------------

# list unique labels and participants
labels = df["label"].unique()
participants = np.sort(df["participant"].unique())


# loop to plot all accelerometer data for each exercise 
for label in labels:
    for participant in participants:
        
        all_axis_df = df[ (df["label"]==label) & (df["participant"] == participant)].reset_index()
        
        # !check if the participant has performed the exercise
        if len(all_axis_df) >0:
                       
            fig, ax = plt.subplots()
            all_axis_df[["acc_x","acc_y","acc_z"]].plot(ax=ax)
            ax.set_ylabel("Acc")
            ax.set_xlabel("sample")
            plt.title(f"{label}({participant})".title())
            plt.legend()
            
            
# loop to plot all gyroscope data for each exercise 
for label in labels:
    for participant in participants:
        
        all_axis_df = df[ (df["label"]==label) & (df["participant"] == participant)].reset_index()
        
        # !check if the participant has performed the exercise
        if len(all_axis_df) >0:
                       
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax)
            ax.set_ylabel("Gyr")
            ax.set_xlabel("sample")
            plt.title(f"{label}({participant})".title())
            plt.legend()
            
            
# ----------------------------------------------------
# Combine both Acc and Gyr plots
# ----------------------------------------------------


label = "row"
participant = "A"

combined_plot_df = df[ (df["participant"]==participant) & (df["label"] == label)].reset_index(drop=True)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])


# styling the legend
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=3, fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=3, fancybox=True, shadow=True)
ax[1].set_xlabel("samples")


# ----------------------------------------------------
# Loop over all combinations and export
# ----------------------------------------------------


labels = df["label"].unique()
participants = np.sort(df["participant"].unique())

 
for label in labels:
    for participant in participants:
        
                
        combined_plot_df = df[ (df["participant"]==participant) & (df["label"] == label)].reset_index(drop=True)

        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
        combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
        combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])

        fig.suptitle(f"{label}({participant})".title())    
        # styling the legend
        ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=3, fancybox=True, shadow=True)
        ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=3, fancybox=True, shadow=True)
        ax[1].set_xlabel("samples")
        
        
        plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
        plt.show()