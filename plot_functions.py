import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from binomial_functions import *

def plot_confidence_intervals(TPR, TNR, prevalence, Ntot_min, Ntot_max, alpha=0.05, num_points=10):
    """
    Calculate and plot confidence intervals for TPR and TNR using matplotlib
    
    Args:
        TPR: True Positive Rate
        TNR: True Negative Rate
        prevalence: Prevalence of positive samples
        Ntot_min: Minimum number of total samples
        Ntot_max: Maximum number of total samples
        alpha: Significance level
        num_points: Number of points to use for the x-axis
    
    Returns:
        fig: Matplotlib figure object
    """
    # Calculate confidence intervals
    TPR_CI, TNR_CI, Ntot_vec = ConfidenceIntervalTXR_array(TPR, TNR, prevalence, Ntot_min, Ntot_max, alpha=alpha, num_points=num_points)
    
    # Calculate CI widths
    TPR_width = TPR_CI[1] - TPR_CI[0]
    TNR_width = TNR_CI[1] - TNR_CI[0]
    
    # Set up figure with GridSpec for better layout control
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(14, 10), dpi=100)
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    
    # Set a nice style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Define colors with better contrast and appeal
    tpr_color = '#1f77b4'  # Blue
    tnr_color = '#d62728'  # Red
    
    # Create TPR subplot (top left)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(Ntot_vec, [TPR] * len(Ntot_vec), color=tpr_color, linewidth=2.5, label='True TPR')
    ax1.plot(Ntot_vec, TPR_CI[0], color=tpr_color, linestyle='--', linewidth=1.5, 
             label=f'Lower CI ({(1-alpha)*100:.1f}%)')
    ax1.plot(Ntot_vec, TPR_CI[1], color=tpr_color, linestyle='--', linewidth=1.5,
             label=f'Upper CI ({(1-alpha)*100:.1f}%)')
    ax1.fill_between(Ntot_vec, TPR_CI[0], TPR_CI[1], color=tpr_color, alpha=0.15)
    
    # Configure TPR subplot
    ax1.set_title(f'TPR Confidence Intervals (α={alpha})', fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel('TPR Value', fontsize=12, fontweight='bold')
    ax1.set_xlim(Ntot_min, Ntot_max)
    ax1.set_ylim(max(0, min(TPR_CI[0])-0.05), min(1, max(TPR_CI[1])+0.05))
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9)
    
    # Format x-axis with thousands separator
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    
    # Create TNR subplot (top right)
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(Ntot_vec, [TNR] * len(Ntot_vec), color=tnr_color, linewidth=2.5, label='True TNR')
    ax2.plot(Ntot_vec, TNR_CI[0], color=tnr_color, linestyle='--', linewidth=1.5,
             label=f'Lower CI ({(1-alpha)*100:.1f}%)')
    ax2.plot(Ntot_vec, TNR_CI[1], color=tnr_color, linestyle='--', linewidth=1.5,
             label=f'Upper CI ({(1-alpha)*100:.1f}%)')
    ax2.fill_between(Ntot_vec, TNR_CI[0], TNR_CI[1], color=tnr_color, alpha=0.15)
    
    # Configure TNR subplot
    ax2.set_title(f'TNR Confidence Intervals (α={alpha})', fontsize=14, fontweight='bold', pad=10)
    ax2.set_ylabel('TNR Value', fontsize=12, fontweight='bold')
    ax2.set_xlim(Ntot_min, Ntot_max)
    ax2.set_ylim(max(0, min(TNR_CI[0])-0.05), min(1, max(TNR_CI[1])+0.05))
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9)
    
    # Format x-axis with thousands separator
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    
    # Create CI width subplot (bottom)
    ax3 = plt.subplot(gs[1, :])
    ax3.plot(Ntot_vec, TPR_width, color=tpr_color, linewidth=2.5, label='TPR CI Width')
    ax3.plot(Ntot_vec, TNR_width, color=tnr_color, linewidth=2.5, label='TNR CI Width')
    
    # Configure CI width subplot
    ax3.set_title('Confidence Interval Widths', fontsize=14, fontweight='bold', pad=10)
    ax3.set_xlabel('Total Sample Size', fontsize=12, fontweight='bold')
    ax3.set_ylabel('CI Width', fontsize=12, fontweight='bold')
    ax3.set_xlim(Ntot_min, Ntot_max)
    y_max = max(max(TPR_width), max(TNR_width)) * 1.1
    ax3.set_ylim(0, y_max)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    # Format x-axis with thousands separator
    ax3.xaxis.set_major_formatter(ScalarFormatter())
    
    # Add a main title
    fig.suptitle('Confidence Intervals for TPR and TNR', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig