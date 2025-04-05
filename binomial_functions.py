import numpy as np
import scipy.stats
import scipy
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

def CreateConfusionMatrix(TPR,TNR,Npos,Nneg ,Montecarlo_sample=10**4, ord="01"):
  """
  Create a confusion matrix for a given TPR, TNR, Npos, Nneg
  Args:
    TPR: True Positive Rate
    TNR: True Negative Rate
    Npos: Number of positive samples
    Nneg: Number of negative samples
    Montecarlo_sample: Number of Montecarlo samples
    ord: Order of the confusion matrix
  Returns:
    CM: Confusion matrix (2,2,Montecarlo_sample)
  """

  CM = np.zeros((2,2,Montecarlo_sample))

  if (ord != "01") and (ord!="10"):
    return print("Wrong ord format, acepted format '01' or '10' ")
  
  TP = scipy.stats.binom(Npos,TPR).rvs(size=Montecarlo_sample)
  TN = scipy.stats.binom(Nneg,TNR).rvs(size=Montecarlo_sample)
  FN = Npos-TP
  FP = Nneg-TN

  if ord =="01":
    # [TN, FP]
    # [FN, TP]
    CM[0,0,:] = TN
    CM[0,1,:] = FP
    CM[1,0,:] = FN
    CM[1,1,:] = TP

  if ord =="10":
    # [TP, FN]
    # [FP, TN]
    CM[0,0,:] = TP
    CM[0,1,:] = FN
    CM[1,0,:] = FP
    CM[1,1,:] = TN

  return CM

def ConfidenceIntervalTXR(TPR, TNR, Npos=None, Nneg=None, prevalence=None, Ntot=None, alpha=0.05):
  """
  Calculate the confidence interval of the TPR and TNR using statsmodels
  Args:
    TPR: True Positive Rate
    TNR: True Negative Rate
    Npos: Number of positive samples (optional if prevalence and Ntot are provided)
    Nneg: Number of negative samples (optional if prevalence and Ntot are provided)
    prevalence: Prevalence of positive samples (optional if Npos and Nneg are provided)
    Ntot: Total number of samples (optional if Npos and Nneg are provided)
    alpha: Significance level
  Returns:
    TPR_CI: Confidence interval of the TPR
    TNR_CI: Confidence interval of the TNR
  """
  # Check which parameters were provided and calculate Npos and Nneg if needed
  if Npos is None or Nneg is None:
    if prevalence is None or Ntot is None:
      raise ValueError("Either (Npos, Nneg) or (prevalence, Ntot) must be provided")
    
    Npos = (prevalence * Ntot).astype(int)
    Nneg = Ntot - Npos
  
  TPR_CI = proportion_confint(Npos * TPR, Npos, alpha=alpha, method='beta')
  TNR_CI = proportion_confint(Nneg * TNR, Nneg, alpha=alpha, method='beta')
  return TPR_CI, TNR_CI

def ConfidenceIntervalTXR_array(TPR, TNR, prevalence, Ntot_min, Ntot_max, alpha=0.05, num_points=10):
  """
  Calculate the confidence interval of the TPR and TNR using statsmodels
  Args:
    TPR: True Positive Rate
    TNR: True Negative Rate
    Ntot_min: Minimum number of total samples
    Ntot_max: Maximum number of total samples
    alpha: Significance level
  Returns:
    TPR_CI: Confidence interval of the TPR
    TNR_CI: Confidence interval of the TNR
  """
  Ntot_vec = np.linspace(Ntot_min, Ntot_max, num_points)
  TPR_CI, TNR_CI = ConfidenceIntervalTXR(TPR, TNR, prevalence=prevalence, Ntot=Ntot_vec, alpha=alpha)
  return TPR_CI, TNR_CI, Ntot_vec

def CalculateStatistic(CM,ord="01"):
  """
  Calculate the statistics of a confusion matrix
  Args:
    CM: Confusion matrix (2,2,Montecarlo_sample)
    ord: Order of the confusion matrix
  Returns:
    Res: Dictionary of statistics
  """
  if (ord != "01") and (ord!="10"):
    return print("Wrong ord format, acepted format '01' or '10' ")
  
  if ord == "01":
    # [TN, FP]
    # [FN, TP]

    TN = CM[0,0,:]
    FP = CM[0,1,:]
    FN = CM[1,0,:]
    TP = CM[1,1,:]

  if ord == "10":
    # [TP, FN]
    # [FP, TN]
    TP = CM[0,0,:]
    FN = CM[0,1,:]
    FP = CM[1,0,:]
    TN = CM[1,1,:]

  Res ={}
  N = TP+TN+FP+FN
  Res["Accuracy"] = (TP+TN)/(N)
  Res["TPR"] = TP / (TP+FN)
  Res["TNR"] = TN / (TN+FP)
  Res["PPV"] = TP / (TP+FP)
  Res["NPV"] = TN / (TN+FN)
  Res["Balanced accuracy"] = (Res["TPR"]+Res["TNR"])/2
  Res["Correlation Coefficient"] = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
  Res["Markedness"] = Res["PPV"]+Res["NPV"]-1
  Res["F1 score"] = (2*TP)/(2*TP+FP+FN)
  Res["Fowlkes-Mallows"] = np.sqrt(Res["PPV"]*Res["TPR"])
  Res["Younden"] = Res["TPR"] + Res["TNR"] -1
  Res["P4"] = (4*TP*TN)/(4*TP*TN+(TP+TN)*(FP+FN))

  return Res
def print_statistics(Res):
  """
  Print the statistics of a dictionary
  Args:
    Res: Dictionary of statistics
  """
  print("\n=== STATISTICS SUMMARY ===")
  for stat_name, stat_values in Res.items():
    mean_value = stat_values.mean()
    std_value = stat_values.std()
    print(f"{stat_name:25} | Mean: {mean_value:.6f} | Std: {std_value:.6f}")
  print("=" * 60)

def probability_greater_equal(v1, v2):
    """
    Calculate probability that a random element from v1 is >= random element from v2.
    Uses sorted arrays and linear scan without large memory allocation.
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    
    # Sort both arrays
    v1_sorted = np.sort(v1)
    v2_sorted = np.sort(v2)
    
    count = 0
    i = 0
    
    # Linear scan through both sorted arrays
    for x in v1_sorted:
        # Find position in v2 where elements are <= x
        while i < len(v2_sorted) and v2_sorted[i] <= x:
            i += 1
        count += i
    
    return count / (len(v1) * len(v2))