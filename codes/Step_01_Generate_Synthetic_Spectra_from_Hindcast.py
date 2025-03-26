# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:22:42 2025

@author: moritzw
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import os
from matplotlib import gridspec
import xarray as xr
import scipy.stats as stats
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import gridspec


def make_plot_and_calc_errors(data1,data2):
    # 1. QQ Plot
    plt.figure(figsize=(6, 6))
    stats.probplot(data2, dist="norm", sparams=(np.mean(data1), np.std(data1)), plot=plt)  
    plt.title("QQ Plot: Data1 vs Data2")
    plt.show()
    
    # 2. Error Statistics
    mae = mean_absolute_error(data1, data2)
    mse = mean_squared_error(data1, data2)
    rmse = np.sqrt(mse)
    r2 = r2_score(data1, data2)
    
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"R-squared (R²): {r2:.3f}")
    
    # Optional: Visualizing the distributions using histograms
    plt.figure(figsize=(10, 5))
    sns.histplot(data1, color='blue', label='Synthetic', kde=True)
    sns.histplot(data2, color='red', label='Original', kde=True)
    plt.legend()
    plt.title("Distribution Comparison: Data1 vs Data2")
    plt.show()
    return( mae, mse, rmse, r2)



def validate_synthetic_data(original, synthetic):
    """
    Compares a synthetic dataset to the 95th percentile range of an original dataset using:
    - Empirical CDF plot
    - Kernel Density Estimation (KDE) plot
    - KS statistic and Wasserstein distance

    Parameters:
    - original (array-like): The original dataset.
    - synthetic (array-like): The synthetic dataset.

    Returns:
    - metrics (dict): Dictionary containing KS statistic, KS p-value, and Wasserstein distance.
    """
    # Compute 5th and 95th percentile thresholds of the original dataset
    p5_threshold = np.percentile(original, 5)  
    p95_threshold = np.percentile(original, 95)  

    # Filter original dataset to retain only values within the 95th percentile range
    original_filtered = original[(original >= p5_threshold) & (original <= p95_threshold)]

    # Sort datasets for ECDF plot
    x_original, y_original = np.sort(original_filtered), np.linspace(0, 1, len(original_filtered))
    x_synthetic, y_synthetic = np.sort(synthetic), np.linspace(0, 1, len(synthetic))

    # ECDF Plot
    plt.figure(figsize=(8,5))
    plt.plot(x_original, y_original, label="Original ECDF", color='blue', linewidth=2)
    plt.plot(x_synthetic, y_synthetic, label="Synthetic ECDF", color='red', linestyle='dashed', linewidth=2)
    plt.axvline(p5_threshold, color='k', linestyle="dashed", label="5th Percentile")
    plt.axvline(p95_threshold, color='k', linestyle="dashed", label="95th Percentile")
    plt.xlabel("Energy Value")
    plt.ylabel("Cumulative Probability")
    plt.title("Empirical CDF: Original vs. Synthetic")
    plt.legend()
    plt.grid(True)
    plt.show()

    # KDE Plot (PDF Comparison)
    plt.figure(figsize=(8,5))
    sns.kdeplot(original_filtered, label="Original KDE", color='blue', linewidth=2, fill=True, alpha=0.3)
    sns.kdeplot(synthetic, label="Synthetic KDE", color='red', linestyle='dashed', linewidth=2, fill=True, alpha=0.3)
    plt.axvline(p5_threshold, color='k', linestyle="dashed", label="5th Percentile")
    plt.axvline(p95_threshold, color='k', linestyle="dashed", label="95th Percentile")
    plt.xlabel("Energy Value")
    plt.ylabel("Density")
    plt.title("KDE Plot: Original vs. Synthetic")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Statistical Metrics
    ks_stat, ks_pval = stats.ks_2samp(original_filtered, synthetic)
    wasserstein_dist = stats.wasserstein_distance(original_filtered, synthetic)

    metrics = {
        "KS Statistic": ks_stat,
        "KS p-value": ks_pval,
        "Wasserstein Distance": wasserstein_dist
    }

    return metrics

#ts=100000 #plotting times
num_sims=100000



# 1. Generate synthetic data (3D Matrix: Frequencies x Directions x Time)
# Here we simulate 100 frequencies, 100 directions, and 50 time points
path_data=r'D:\CISPac-5\CIS-PAC5-Niue-Inundation-ML\extras\spec_hindcast'
path_res = r'D:\CISPac-5\CIS-PAC5-Niue-Inundation-ML\extras\processed_data'
sp_regular=xr.open_dataset(os.path.join(path_data, 'Niue_SuperPoint.nc'))
sp_regular = sp_regular.rename({'frequency':'freq','theta':'dir'})

output_nc_file_name = os.path.join(path_res,'Synth_Spectra_95.nc')
threshold_prctl = 95


sp_regular=sp_regular.transpose("time", "freq", "dir")
# sp_regular = sp_regular['efth'][0:ts,:,:].to_dataset()
#sp_regular['efth']=(('time', 'freq', 'dir'),sp_regular.efth.values/(1025*9.81))
sp_regular['efth']=(('time', 'freq', 'dir'),sp_regular.efth.values)

sp_regular['dir']=np.where(sp_regular.dir.values<0, sp_regular.dir.values+360, sp_regular.dir.values)
sp_regular=sp_regular.sortby('dir')
sp_regular['efth']
min_hist = np.nanmin(sp_regular.efth.spec.hs())
max_hist = np.nanmax(sp_regular.efth.spec.hs())

#sp_regular = sp_regular_subset
sp = sp_regular
# Parameters for synthetic data
num_frequencies = len(sp.freq)
num_directions = len(sp.dir)
# time_steps = ts
# Generate random data
frequencies = sp.freq.values  # Frequencies from 1 to 100
directions = sp.dir.values  # Directions from 0 to 180 degrees
# Create a 3D matrix (Frequency x Direction x Time)
#data = np.random.randn(n_freq, n_dir, n_time)  # Normally distributed data

## Find spectra with large waves
# Compute the 90th percentile
threshold = np.percentile(sp_regular.efth.spec.hs(), threshold_prctl)
# Find indices where values are >= 90th percentile
indices = np.argwhere(sp_regular.efth.spec.hs().values >= threshold)

index_array = indices.flatten()
# Convert integer indices to actual datetime values
time_values = sp_regular.time[index_array]

# Extract data values above the threshold
sp_regular_subset = sp_regular.efth.isel(time=index_array).to_dataset()


ts=len(indices) #plotting times
fig = plt.figure(figsize=[20,15])
gs1=gridspec.GridSpec(3,1)
ax0=fig.add_subplot(gs1[0])

ax0.plot(sp_regular.time.values, sp_regular.efth.spec.hs(), color='firebrick')
ax0.plot(sp_regular_subset.isel(time=range(ts)).time.values, sp_regular_subset.isel(time=range(ts)).efth.spec.hs(), 'ok')
ax0.set_ylabel('Hs (m)', fontsize=16)
ax0.grid()
ax0.set_xlim(sp_regular.time.values[0], sp_regular.time.values[-1])

ax1=fig.add_subplot(gs1[1], sharex=ax0)
ax1.plot(sp_regular.time.values, sp_regular.efth.spec.tp(), color='royalblue')
ax1.set_ylabel('Tp (s)', fontsize=16)
ax1.grid()

ax2=fig.add_subplot(gs1[2], sharex=ax0)
ax2.plot(sp_regular.time.values, sp_regular.efth.spec.dp(),'.',  color='darkgreen')
ax2.set_ylabel('Dir (º)', fontsize=16)
plt.grid()


original_data = sp_regular_subset.efth.values
# Reshape the data: (10, 37*36) to capture spatial structure
reshaped_data = original_data.reshape(len(original_data[:,0,0]), -1)  # Shape: (10, 1332)


# Find the optimal number of PCA components for 95% variance
pca_full = PCA()
pca_full.fit(reshaped_data)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
optimal_pca_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Optimal number of PCA components: {optimal_pca_components}")

#optimal_pca_components = 40
# Apply PCA with optimal number of components
pca = PCA(n_components=optimal_pca_components)
low_dim_data = pca.fit_transform(reshaped_data)  # Shape: (10, optimal_pca_components)

# Find the best number of GMM components using BIC
bic_scores = []
aic_scores = []
component_range = range(1, 60)  # Test 1 to 100 components

for n in component_range:
    print(n)
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm.fit(low_dim_data)
    bic_scores.append(gmm.bic(low_dim_data))
    aic_scores.append(gmm.aic(low_dim_data))

# Select the optimal number of GMM components
optimal_gmm_components = component_range[np.argmin(bic_scores)]
#optimal_gmm_components = 45
print(f"Optimal number of GMM components: {optimal_gmm_components}")

# Fit GMM using the optimal number of components
gmm = GaussianMixture(n_components=optimal_gmm_components, covariance_type='full', random_state=None)
gmm.fit(low_dim_data)

# Generate a new synthetic sample in PCA space
synthetic_low_dim = gmm.sample(1)[0]

# Transform back to full 37x36 matrix
synthetic_sample = pca.inverse_transform(synthetic_low_dim).reshape(38, 36)

# Ensure non-negativity
synthetic_sample = np.maximum(synthetic_sample, 0)


synthetic_samples = []
for _ in range(num_sims):
    synthetic_low_dim = gmm.sample(1)[0]  # Sample from GMM
    synthetic_matrix = pca.inverse_transform(synthetic_low_dim).reshape(38, 36)
    synthetic_matrix = np.maximum(synthetic_matrix, 0)  # Ensure non-negativity
    synthetic_samples.append(synthetic_matrix)

# Plot BIC scores
plt.figure(figsize=(8, 5))
plt.plot(component_range, bic_scores, label="BIC Score", marker="o")
plt.plot(component_range, aic_scores, label="AIC Score", marker="s")
plt.xlabel("Number of GMM Components")
plt.ylabel("Score")
plt.legend()
plt.title("Model Selection for GMM: BIC vs AIC")
plt.show()



efth_max = sp['efth'].max().values.item()

# Plot original and synthetic clusters
fig = plt.figure(figsize=[15,10])
gs3=gridspec.GridSpec(5,5,hspace=0.01, wspace=0.01)
for b in range(25):
    
    ax=fig.add_subplot(gs3[b],projection='polar')
    ix = np.random.randint(0,len(sp_regular.time.values))
    
    sp.isel(time=ix).spec.plot(
        col="",
        row="",
        add_colorbar=False,
        show_theta_labels=False,
        show_radii_labels=True,
        as_period=True,
        normalised=False,
        cmap="Spectral_r",
        levels=np.logspace(np.log10(0.005), np.log(efth_max), 128,endpoint=True),
        cbar_ticks=[0.005,0.01, 0.1, efth_max],
        title=False
    )
fig.savefig(path_res+'/Historic_sample_spec.png')
plt.close(fig)

sp_synth = xr.Dataset()
sp_synth['efth'] = (('clusters', 'freq', 'dir'),synthetic_samples)
sp_synth.assign_coords({'freq':sp.freq,'dir':sp.dir})

fig = plt.figure(figsize=[15,10])
gs3=gridspec.GridSpec(10,10,hspace=0.01, wspace=0.01)
for b in range(100):
    
    ax=fig.add_subplot(gs3[b],projection='polar')
    ix = np.random.randint(0,len(sp_synth.clusters.values))
    
    sp_synth.isel(clusters=ix).assign_coords({'freq':sp.freq,'dir':sp.dir}).spec.plot(
        add_colorbar=False,
        show_theta_labels=False,
        show_radii_labels=True,
        as_period=True,
        normalised=False,
        cmap="Spectral_r",
        levels=np.logspace(np.log10(0.005), np.log(efth_max), 128,endpoint=True),
        cbar_ticks=[0.005,0.01, 0.1, efth_max],
        title=False
    )
fig.savefig(path_res+'/Synth_sample_spec.png')
plt.close(fig)




fig, axes = plt.subplots(3, 4, figsize=(15, 10))
for i in range(10):
    ix = np.random.randint(0,ts)
    ax = axes.flat[i]
    pcm = ax.pcolormesh(frequencies,directions,original_data[ix].T, cmap='viridis')
    fig.colorbar(pcm, ax=ax)
    ax.set_title(f'Original Cluster {ix+1}')
ax = axes.flat[10]
pcm = ax.pcolormesh(frequencies,directions,synthetic_samples[np.random.randint(0,num_sims)].T, cmap='plasma')
fig.colorbar(pcm, ax=ax)
ax.set_title('Synthetic Cluster')
axes.flat[11].axis('off')  # Empty subplot for layout purposes
plt.tight_layout()
plt.show()


sp['efth_synth']=(('clusters', 'freq', 'dir'),synthetic_samples)
sp.to_netcdf(output_nc_file_name)

hs_synth = sp.efth_synth.spec.hs()
hs_orig = sp.efth.spec.hs()
tp_synth = sp.efth_synth.spec.tp()
tp_orig = sp.efth.spec.tp()
dp_synth = sp.efth_synth.spec.dp()
dp_orig = sp.efth.spec.dp()




fig = plt.figure(figsize=[10,10])
gs3=gridspec.GridSpec(2,2,hspace=0.1, wspace=0.1)
ax=fig.add_subplot(gs3[0])
ax.plot(hs_orig,tp_orig,'.', label='Original Data',color='black',markersize=2)
ax.plot(sp_regular_subset.isel(time=range(ts)).efth.spec.hs(),sp_regular_subset.isel(time=range(ts)).efth.spec.tp(),'.',label='Extreme Data', color='coral',markersize=1)
ax.set_ylabel('Tp (s)', fontsize=14)
ax.set_xlabel('Hs (m)', fontsize=14)
ax.legend(fontsize=12)


ax1=fig.add_subplot(gs3[1])
ax1.plot(hs_orig,dp_orig,'.', label='Original Data',color='black',markersize=2)
ax1.plot(sp_regular_subset.isel(time=range(ts)).efth.spec.hs(),sp_regular_subset.isel(time=range(ts)).efth.spec.dp(),'.',label='Extreme Data', color='coral',markersize=1)
ax1.set_ylabel('Dp (deg)', fontsize=14)
ax1.set_xlabel('Hs (m)', fontsize=14)


ax3=fig.add_subplot(gs3[3])
ax3.plot(tp_orig,dp_orig,'.', label='Original Data',color='black',markersize=2)
ax3.plot(sp_regular_subset.isel(time=range(ts)).efth.spec.tp(),sp_regular_subset.isel(time=range(ts)).efth.spec.dp(),'.',label='Extreme Data', color='coral',markersize=1)
ax3.set_ylabel('Dp (deg)', fontsize=14)
ax3.set_xlabel('Tp (s)', fontsize=14)

fig.savefig(path_res+'/Parametric_extremes_used_to_train_MC.png')
plt.close(fig)



fig = plt.figure(figsize=[10,10])
gs3=gridspec.GridSpec(2,2,hspace=0.1, wspace=0.1)
ax=fig.add_subplot(gs3[0])
ax.plot(hs_orig,tp_orig,'.', label='Original Data',color='black',markersize=2)
ax.plot(hs_synth,tp_synth,'.',label='Synthetic Data', color='coral',markersize=1)
ax.set_ylabel('Tp (s)', fontsize=14)
ax.set_xlabel('Hs (m)', fontsize=14)
ax.legend(fontsize=12)



ax1=fig.add_subplot(gs3[1])
ax1.plot(hs_orig,dp_orig,'.', label='Original Data',color='black',markersize=2)
ax1.plot(hs_synth,dp_synth,'.',label='Synthetic Data', color='coral',markersize=1)
ax1.set_ylabel('Dp (deg)', fontsize=14)
ax1.set_xlabel('Hs (m)', fontsize=14)


ax3=fig.add_subplot(gs3[3])
ax3.plot(tp_orig,dp_orig,'.', label='Original Data',color='black',markersize=2)
ax3.plot(tp_synth,dp_synth,'.',label='Synthetic Data', color='coral',markersize=1)
ax3.set_ylabel('Dp (deg)', fontsize=14)
ax3.set_xlabel('Tp (s)', fontsize=14)

fig.savefig(path_res+'/Parametric_results_from_MC.png')
plt.close(fig)





# results = validate_synthetic_data(hs_orig, hs_synth)
# print(results)



# mae, mse, rmse, r2 = make_plot_and_calc_errors(hs_synth,hs_orig)
# mae, mse, rmse, r2 = make_plot_and_calc_errors(tp_synth,tp_orig)
