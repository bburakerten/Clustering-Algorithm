#!/usr/bin/env python
# coding: utf-8

# 

# In[40]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import pandas as pd
import csv


# ## Data Generation

# In[41]:


df1 = pd.read_csv(
    "Copy of Go Climb! - Difficulty Curve - Copy of Sheet3.csv",
    dtype=float
)

feature_list = []
for col in df1.columns:
    feature+col = df1[col].values.reshape(400, 1)
    feature_list.append(feature+col)

np.random.seed(421)
# sample size
N = 400


# =============================================================================
# Feature 1 (400 data points)
# =============================================================================
feature1 = , dtype=float).reshape(400,1)

# =============================================================================
# Feature 3 (400 data points)
# =============================================================================
feature3 = np.array([
    1.2, 3.1, 4.2, 3.3, 2.5, 4.6, 7.2, 8.8, 9.4, 8.5, 7.7, 13.7, 11.5, 8.0,
    8.6, 7.0, 12.2, 11.0, 15.2, 10.0, 8.8, 9.1, 12.7, 7.8, 14.4, 14.1, 19.1,
    10.9, 11.2, 5.8, 20.9, 9.8, 18.4, 11.4, 21.6, 15.8, 15.8, 11.3, 21.5,
    15.9, 15.8, 7.9, 12.4, 13.4, 31.8, 9.0, 16.8, 13.1, 16.7, 23.1, 27.8,
    7.8, 28.5, 6.8, 18.7, 15.1, 28.8, 16.4, 22.8, 9.3, 18.1, 9.0, 23.3,
    14.4, 30.0, 10.4, 17.7, 18.5, 16.5, 9.6, 35.0, 12.8, 31.0, 20.5, 17.8,
    10.7, 26.9, 13.4, 18.7, 8.2, 14.9, 6.8, 29.1, 18.1, 18.6, 6.9, 26.2,
    10.0, 15.4, 18.2, 19.1, 33.7, 26.6, 20.6, 17.8, 35.5, 35.9, 31.3, 12.7,
    22.1, 25.7, 21.3, 17.1, 34.1, 36.4, 35.7, 16.1, 19.4, 26.4, 34.8, 31.9,
    26.2, 16.8, 17.7, 28.6, 27.4, 20.6, 26.9, 42.6, 21.3, 28.9, 38.4, 22.6,
    21.1, 28.0, 47.3, 19.5, 24.9, 34.6, 35.5, 31.2, 24.8, 23.1, 22.6, 22.5,
    21.8, 27.7, 29.9, 22.3, 34.0, 26.1, 27.5, 38.5, 21.6, 35.3, 26.9, 28.1,
    29.6, 27.0, 21.5, 25.4, 16.3, 31.8, 22.7, 26.0, 19.4, 32.7, 26.5, 35.6,
    20.6, 28.8, 23.0, 32.9, 25.3, 29.5, 22.7, 32.3, 23.3, 23.0, 29.4, 32.0,
    18.2, 27.1, 27.2, 46.5, 35.7, 38.3, 55.0, 25.7, 32.6, 35.7, 32.2, 37.6,
    35.7, 53.1, 18.9, 18.9, 11.4, 22.3, 43.6, 28.6, 27.6, 22.3, 35.8, 29.4,
    27.5, 35.1, 21.4, 31.4, 21.7, 33.7, 36.1, 28.6, 22.2, 27.1, 38.2, 29.8,
    25.3, 33.9, 21.1, 36.5, 43.3, 33.6, 22.0, 38.0, 30.5, 36.9, 29.5, 26.5,
    32.5, 41.4, 40.0, 27.2, 26.7, 35.4, 36.8, 36.9, 31.9, 29.1, 43.0, 22.0,
    40.7, 40.4, 33.6, 35.0, 21.3, 31.2, 29.9, 44.3, 34.7, 41.9, 34.1, 33.7,
    31.4, 37.7, 34.4, 52.0, 40.4, 37.4, 38.3, 32.0, 29.3, 39.3, 20.3, 38.4,
    30.6, 36.5, 39.6, 31.3, 35.6, 36.0, 47.9, 39.6, 45.0, 40.1, 25.0, 42.0,
    32.2, 31.7, 29.2, 33.0, 38.3, 46.4, 32.9, 37.7, 37.7, 33.7, 34.8, 37.2,
    26.0, 48.2, 24.1, 52.5, 67.7, 40.0, 45.9, 35.5, 37.5, 42.6, 34.9, 31.3,
    30.8, 40.3, 42.1, 43.4, 58.4, 48.2, 43.3, 46.2, 37.0, 45.1, 33.0, 25.2,
    27.5, 33.0, 40.2, 43.2, 43.9, 45.6, 31.5, 40.4, 46.0, 39.7, 40.2, 37.9,
    50.8, 56.9, 42.7, 32.4, 34.8, 53.4, 35.6, 37.2, 40.5, 49.9, 35.2, 40.0,
    40.4, 41.1, 56.0, 40.1, 32.3, 38.1, 41.2, 46.4, 49.5, 40.3, 33.8, 40.6,
    48.5, 47.1, 46.5, 43.4, 38.6, 53.4, 31.5, 54.6, 56.0, 43.1, 25.5, 37.9,
    48.6, 30.2, 35.0, 41.5, 32.9, 49.2, 55.4, 40.1, 44.3, 49.1, 57.1, 45.6,
    61.7, 45.7, 46.8, 60.2, 61.2, 48.1, 46.0, 59.1, 32.3, 48.1, 48.5, 40.1,
    63.2, 35.7, 44.7, 53.9, 56.8, 40.4, 45.5, 31.0, 32.3, 40.7, 51.6, 47.1,
    51.5, 58.5, 47.2, 63.2, 51.9, 52.8, 48.7, 29.0, 51.9, 46.1, 69.5, 62.0,
    58.2
], dtype=float).reshape(400,1)

# =============================================================================
# Feature 4 (400 data points)
# =============================================================================
feature4 = np.array([
    5, 9, 11, 9, 7, 11, 19, 23, 23, 25, 20, 40, 26, 19, 20, 16, 30, 27, 35,
    26, 22, 19, 28, 15, 32, 33, 44, 23, 26, 13, 45, 22, 40, 24, 46, 40, 42,
    25, 53, 37, 37, 17, 25, 25, 71, 17, 35, 26, 33, 51, 50, 16, 54, 13, 39,
    28, 54, 30, 42, 17, 35, 17, 51, 27, 60, 18, 33, 33, 32, 17, 72, 26, 64,
    41, 38, 22, 52, 25, 39, 16, 31, 14, 68, 33, 36, 14, 54, 17, 32, 32, 31,
    62, 48, 36, 35, 69, 77, 66, 26, 40, 55, 41, 34, 64, 76, 67, 16.1, 19.4,
    26.4, 34.8, 31.9, 26.2, 16.8, 17.7, 28.6, 27.4, 20.6, 26.9, 42.6, 21.3,
    28.9, 38.4, 22.6, 21.1, 28.0, 47.3, 19.5, 24.9, 34.6, 35.5, 31.2, 24.8,
    23.1, 22.6, 22.5, 21.8, 27.7, 29.9, 22.3, 34.0, 26.1, 27.5, 38.5, 21.6,
    35.3, 26.9, 28.1, 29.6, 27.0, 21.5, 25.4, 16.3, 31.8, 22.7, 26.0, 19.4,
    32.7, 26.5, 35.6, 20.6, 28.8, 23.0, 32.9, 25.3, 29.5, 22.7, 32.3, 23.3,
    23.0, 29.4, 32.0, 18.2, 27.1, 27.2, 46.5, 35.7, 38.3, 55.0, 25.7, 32.6,
    35.7, 32.2, 37.6, 35.7, 53.1, 18.9, 18.9, 11.4, 22.3, 43.6, 28.6, 27.6,
    22.3, 35.8, 29.4, 27.5, 35.1, 21.4, 31.4, 21.7, 33.7, 36.1, 28.6, 22.2,
    27.1, 38.2, 29.8, 25.3, 33.9, 21.1, 36.5, 43.3, 33.6, 22.0, 38.0, 30.5,
    36.9, 29.5, 26.5, 32.5, 41.4, 40.0, 27.2, 26.7, 35.4, 36.8, 36.9, 31.9,
    29.1, 43.0, 22.0, 40.7, 40.4, 33.6, 35.0, 21.3, 31.2, 29.9, 44.3, 34.7,
    41.9, 34.1, 33.7, 31.4, 37.7, 34.4, 52.0, 40.4, 37.4, 38.3, 32.0, 29.3,
    39.3, 20.3, 38.4, 30.6, 36.5, 39.6, 31.3, 35.6, 36.0, 47.9, 39.6, 45.0,
    40.1, 25.0, 42.0, 32.2, 31.7, 29.2, 33.0, 38.3, 46.4, 32.9, 37.7, 37.7,
    33.7, 34.8, 37.2, 26.0, 48.2, 24.1, 52.5, 67.7, 40.0, 45.9, 35.5, 37.5,
    42.6, 34.9, 31.3, 30.8, 40.3, 42.1, 43.4, 58.4, 48.2, 43.3, 46.2, 37.0,
    45.1, 33.0, 25.2, 27.5, 33.0, 40.2, 43.2, 43.9, 45.6, 31.5, 40.4, 46.0,
    39.7, 40.2, 37.9, 50.8, 56.9, 42.7, 32.4, 34.8, 53.4, 35.6, 37.2, 40.5,
    49.9, 35.2, 40.0, 40.4, 41.1, 56.0, 40.1, 32.3, 38.1, 41.2, 46.4, 49.5,
    40.3, 33.8, 40.6, 48.5, 47.1, 46.5, 43.4, 38.6, 53.4, 31.5, 54.6, 56.0,
    43.1, 25.5, 37.9, 48.6, 30.2, 35.0, 41.5, 32.9, 49.2, 55.4, 40.1, 44.3,
    49.1, 57.1, 45.6, 61.7, 45.7, 46.8, 60.2, 61.2, 48.1, 46.0, 59.1, 32.3,
    48.1, 48.5, 40.1, 63.2, 35.7, 44.7, 53.9, 56.8, 40.4, 45.5, 31.0, 32.3,
    40.7, 51.6, 47.1, 51.5, 58.5, 47.2, 63.2, 51.9, 52.8, 48.7, 29.0, 51.9,
    46.1, 69.5, 62.0, 58.2
], dtype=float).reshape(400,1)

X = np.hstack((feature1, feature2, feature3, feature4))

point_names = [f"Point_{i+1}" for i in range(X.shape[0])]
# This yields ["Point_1", "Point_2", ..., "Point_400"]

# ===========================================================================
# 4) Put everything into a DataFrame, using names as the index
# ===========================================================================
df = pd.DataFrame(X, 
                  columns=["Feature1", "Feature2", "Feature3", "Feature4"],
                  index=point_names)


# ## Parameters

# In[42]:


# cluster count
K = 4


# In[20]:


df1 = pd.read_csv(
    "Copy of Go Climb! - Difficulty Curve - Copy of Sheet3 (3).csv",
    dtype=float
)
print((df1["LEVEL"].values).reshape(400,1))
#feature_list = []
#or col in df1.columns:
 #   feature+col = df1[col].values.reshape(400, 1)
  #  feature_list.append(feature+col)


# ## Algorithm Steps

# In[43]:


def update_centroids(memberships, X):
    if memberships is None:
        # initialize centroids
        centroids = X[np.random.choice(range(N), K, False), :]
    else:
        # update centroids
        centroids = np.vstack([np.mean(X[memberships == k, :], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)


# ## Visualization

# In[ ]:





# ## Iterations

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import pandas as pd

np.random.seed(421)
# sample size
N = 400


#df1 = pd.read_csv("Copy of Go Climb! - Difficulty Curve - Copy of Sheet3.csv")

#feature_list = []
#for col in df1.columns:
 #   arr = df1[col].values.reshape(400, 1)
 #   feature_list.append(arr)



#for i in rane(len(feature_list)):
#    X = np.hstack((X, feature_list(i)))


# This yields ["Point_1", "Point_2", ..., "Point_400"]

# ===========================================================================
# 4) Put everything into a DataFrame, using names as the index
# ===========================================================================
#column = []
#for i in range(len(feature_list)):
#    columns.append("Feature"+ i)
    


feature1 = np.array([
    1.01, 1.01, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.04, 1.00, 1.06, 1.07,
    1.06, 1.08, 1.06, 1.02, 1.27, 1.06, 1.19, 1.04, 1.13, 1.02, 1.21, 1.02,
    1.16, 1.18, 1.22, 1.04, 1.34, 1.04, 1.38, 1.09, 1.22, 1.05, 1.45, 1.50,
    1.61, 1.22, 1.54, 1.10, 4.35, 1.01, 2.68, 1.04, 2.32, 1.02, 3.16, 1.04,
    2.67, 1.24, 2.18, 1.01, 2.72, 1.01, 2.56, 1.04, 1.85, 1.03, 3.58, 1.01,
    2.76, 1.06, 2.04, 1.03, 3.60, 1.01, 2.68, 1.04, 2.46, 1.01, 2.36, 1.03,
    2.10, 1.16, 2.05, 1.03, 4.18, 1.02, 2.16, 1.01, 2.10, 1.04, 2.95, 1.02,
    1.86, 1.04, 1.79, 1.04, 3.54, 1.02, 2.60, 1.06, 2.03, 1.06, 3.11, 1.08,
    1.96, 1.25, 2.09, 1.07, 1.05, 1.04, 2.95, 1.06, 2.00, 1.13, 2.94, 1.01,
    2.21, 1.16, 2.20, 1.07, 5.12, 1.02, 1.93, 1.03, 1.97, 1.07, 3.03, 1.02,
    1.77, 1.18, 2.09, 1.05, 3.71, 1.11, 2.16, 1.04, 1.78, 1.16, 2.91, 1.06,
    1.94, 1.11, 1.78, 1.03, 2.59, 1.02, 1.86, 1.02, 1.72, 1.07, 3.06, 1.03,
    2.48, 1.10, 1.77, 1.09, 4.24, 1.04, 1.74, 1.00, 1.57, 1.05, 3.38, 1.05,
    1.89, 1.14, 1.61, 1.02, 3.88, 1.06, 1.54, 1.01, 1.63, 1.04, 3.48, 1.01,
    1.89, 1.16, 1.64, 1.06, 4.01, 1.04, 1.71, 1.05, 1.47, 1.23, 2.79, 1.04,
    1.67, 1.14, 1.67, 1.10, 2.29, 1.01, 2.20, 1.02, 1.71, 1.13, 3.56, 1.03,
    1.28, 1.16, 1.77, 1.07, 6.37, 1.01, 1.88, 1.02, 1.63, 1.05, 3.37, 1.02,
    2.13, 1.12, 1.45, 1.10, 3.10, 1.06, 1.64, 1.08, 1.64, 1.06, 2.32, 1.02,
    1.51, 1.13, 1.55, 1.08, 2.85, 1.00, 1.47, 1.08, 1.37, 1.12, 3.42, 1.01,
    1.16, 1.20, 2.36, 1.09, 2.78, 1.04, 1.46, 1.03, 1.26, 1.05, 2.42, 1.03,
    1.45, 1.14, 1.63, 1.02, 2.59, 1.06, 1.69, 1.05, 1.29, 1.10, 2.35, 1.04,
    1.78, 1.08, 1.86, 1.05, 3.27, 1.10, 1.83, 1.04, 1.51, 1.22, 2.55, 1.05,
    1.52, 1.10, 1.54, 1.08, 2.87, 1.10, 1.71, 1.08, 1.64, 1.05, 3.21, 1.04,
    1.34, 1.12, 1.51, 1.04, 2.73, 1.00, 1.62, 1.17, 1.93, 1.18, 3.24, 1.04,
    1.54, 1.16, 2.17, 1.02, 6.81, 1.01, 2.07, 1.07, 2.62, 1.07, 6.30, 1.01,
    3.09, 1.04, 1.76, 1.05, 11.89, 1.04, 1.54, 1.05, 2.22, 1.03, 4.92, 1.08,
    1.78, 1.09, 1.50, 1.14, 4.86, 1.06, 1.29, 1.04, 1.34, 1.08, 6.73, 1.02,
    1.55, 1.10, 1.66, 1.10, 5.47, 1.09, 1.23, 1.04, 1.89, 1.06, 3.77, 1.06,
    1.19, 1.14, 1.40, 1.14, 6.68, 1.01, 3.83, 1.07, 1.25, 1.05, 1.98, 1.05,
    1.32, 1.08, 1.28, 1.15, 5.39, 1.01, 1.18, 1.03, 1.45, 1.16, 2.67, 1.04,
    1.42, 1.18, 1.91, 1.20, 6.23, 1.00, 1.57, 1.08, 1.57, 1.09, 2.06, 1.00,
    1.38, 1.07, 1.26, 1.10, 2.05, 1.01, 1.18, 1.03, 1.31, 1.18, 4.03, 1.06,
    1.42, 1.15, 1.17, 1.05, 4.68, 1.01, 1.52, 1.01, 1.22, 1.17, 3.91, 1.00,
    1.35, 1.18, 1.48, 1.17
], dtype=float).reshape(400,1)

# =============================================================================
# Feature 2 (400 data points)
# =============================================================================
feature2 = np.array([
    0.987, 0.990, 0.992, 0.992, 0.981, 0.996, 0.994, 0.994, 0.983, 0.993,
    0.948, 0.977, 0.945, 0.970, 0.941, 0.987, 0.775, 0.943, 0.844, 0.972,
    0.885, 0.988, 0.840, 0.979, 0.874, 0.943, 0.823, 0.963, 0.759, 0.967,
    0.723, 0.929, 0.834, 0.963, 0.719, 0.000, 0.053, 0.886, 0.383, 0.900,
    0.042, 0.990, 0.227, 0.952, 0.372, 0.981, 0.097, 0.958, 0.199, 0.886,
    0.285, 0.984, 0.127, 0.989, 0.252, 0.958, 0.382, 0.966, 0.070, 0.989,
    0.194, 0.989, 0.306, 0.972, 0.107, 0.989, 0.209, 0.964, 0.224, 0.987,
    0.228, 0.981, 0.294, 0.956, 0.284, 0.965, 0.092, 0.970, 0.305, 0.976,
    0.230, 0.978, 0.154, 0.959, 0.306, 0.986, 0.415, 0.996, 0.117, 0.974,
    0.245, 0.948, 0.401, 0.976, 0.182, 0.882, 0.358, 0.912, 0.394, 0.935,
    0.911, 0.951, 0.214, 0.951, 0.359, 0.940, 0.124, 0.981, 0.419, 0.933,
    0.279, 0.980, 0.061, 0.975, 0.493, 0.966, 0.508, 0.956, 0.250, 0.963,
    0.455, 0.936, 0.360, 0.986, 0.108, 0.826, 0.356, 0.905, 0.438, 0.900,
    0.163, 0.922, 0.389, 0.971, 0.434, 0.965, 0.255, 0.961, 0.486, 0.949,
    0.459, 0.946, 0.165, 0.977, 0.348, 0.989, 0.473, 0.950, 0.124, 0.965,
    0.482, 0.975, 0.550, 0.974, 0.245, 0.942, 0.374, 0.936, 0.556, 0.994,
    0.232, 0.943, 0.593, 0.981, 0.547, 0.993, 0.221, 0.985, 0.336, 0.951,
    0.480, 0.931, 0.175, 0.958, 0.444, 0.931, 0.556, 0.849, 0.309, 0.940,
    0.504, 0.958, 0.500, 0.938, 0.209, 0.984, 0.311, 0.981, 0.444, 0.904,
    0.155, 0.947, 0.690, 0.888, 0.442, 0.969, 0.119, 0.978, 0.476, 0.972,
    0.601, 0.978, 0.246, 0.955, 0.396, 0.954, 0.534, 0.933, 0.202, 0.914,
    0.514, 0.917, 0.322, 0.957, 0.254, 0.966, 0.592, 0.931, 0.553, 0.942,
    0.271, 0.947, 0.542, 0.892, 0.718, 0.878, 0.117, 0.973, 0.752, 0.870,
    0.394, 0.938, 0.197, 0.951, 0.570, 0.950, 0.692, 0.975, 0.280, 0.931,
    0.557, 0.973, 0.518, 0.947, 0.206, 0.910, 0.520, 0.906, 0.667, 0.890,
    0.238, 0.940, 0.392, 0.975, 0.507, 0.960, 0.111, 0.869, 0.447, 0.942,
    0.518, 0.836, 0.308, 0.917, 0.531, 0.964, 0.504, 0.911, 0.200, 0.897,
    0.486, 0.905, 0.391, 0.931, 0.185, 0.919, 0.677, 0.911, 0.504, 0.968,
    0.212, 0.983, 0.424, 0.851, 0.427, 0.858, 0.266, 0.967, 0.576, 0.923,
    0.352, 0.977, 0.018, 0.965, 0.500, 0.947, 0.264, 0.975, 0.029, 0.978,
    0.167, 0.955, 0.483, 0.973, 0.053, 0.969, 0.602, 0.950, 0.377, 0.983,
    0.082, 0.917, 0.431, 0.991, 0.622, 0.929, 0.075, 0.909, 0.615, 0.959,
    0.619, 0.942, 0.025, 0.966, 0.508, 0.925, 0.343, 0.904, 0.057, 0.919,
    0.608, 0.941, 0.344, 0.962, 0.107, 0.948, 0.691, 0.944, 0.602, 0.923,
    0.022, 0.966, 0.128, 0.925, 0.664, 0.972, 0.216, 0.933, 0.547, 0.975,
    0.581, 0.843, 0.009, 0.974, 0.708, 0.947, 0.564, 0.913, 0.205, 0.964,
    0.527, 0.861, 0.333, 0.947, 0.011, 0.989, 0.475, 0.941, 0.551, 0.939,
    0.223, 0.971, 0.587, 0.990, 0.540, 0.970, 0.189, 1.000, 0.681, 0.943,
    0.612, 0.909, 0.155, 0.941, 0.609, 0.954, 0.727, 0.954, 0.029, 0.986,
    0.594, 0.971, 0.725, 0.929, 0.185, 0.923, 0.648, 0.930, 0.587, 0.935
], dtype=float).reshape(400,1)

feature3 = np.array([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00,
    0.05, 0.01, 0.05, 0.02, 0.06, 0.01, 0.22, 0.05, 0.16, 0.02,
    0.11, 0.00, 0.17, 0.02, 0.13, 0.04, 0.18, 0.03, 0.26, 0.03,
    0.27, 0.07, 0.18, 0.03, 0.28, 0.33, 0.36, 0.06, 0.34, 0.07,
    0.77, 0.00, 0.64, 0.03, 0.56, 0.01, 0.69, 0.03, 0.63, 0.07,
    0.54, 0.00, 0.63, 0.00, 0.60, 0.02, 0.46, 0.02, 0.71, 0.01,
    0.63, 0.01, 0.49, 0.02, 0.71, 0.00, 0.62, 0.02, 0.59, 0.01,
    0.55, 0.01, 0.52, 0.02, 0.51, 0.02, 0.74, 0.02, 0.52, 0.00,
    0.51, 0.01, 0.63, 0.01, 0.45, 0.00, 0.40, 0.00, 0.69, 0.01,
    0.59, 0.02, 0.46, 0.00, 0.66, 0.03, 0.46, 0.04, 0.50, 0.02,
    0.03, 0.03, 0.61, 0.01, 0.46, 0.03, 0.65, 0.00, 0.53, 0.02,
    0.50, 0.01, 0.78, 0.00, 0.46, 0.02, 0.49, 0.03, 0.64, 0.01,
    0.43, 0.02, 0.47, 0.01, 0.71, 0.07, 0.52, 0.03, 0.42, 0.04,
    0.62, 0.06, 0.46, 0.01, 0.42, 0.02, 0.58, 0.01, 0.44, 0.02,
    0.38, 0.02, 0.62, 0.01, 0.54, 0.00, 0.39, 0.04, 0.73, 0.02,
    0.42, 0.00, 0.28, 0.01, 0.66, 0.04, 0.44, 0.03, 0.36, 0.01,
    0.69, 0.05, 0.33, 0.00, 0.36, 0.00, 0.69, 0.00, 0.46, 0.00,
    0.32, 0.03, 0.70, 0.00, 0.34, 0.02, 0.32, 0.04, 0.62, 0.04,
    0.39, 0.02, 0.36, 0.03, 0.51, 0.00, 0.53, 0.00, 0.41, 0.05,
    0.68, 0.02, 0.22, 0.03, 0.40, 0.02, 0.77, 0.01, 0.44, 0.01,
    0.34, 0.01, 0.68, 0.01, 0.52, 0.03, 0.28, 0.04, 0.65, 0.04,
    0.32, 0.03, 0.38, 0.02, 0.54, 0.01, 0.30, 0.04, 0.33, 0.03,
    0.60, 0.00, 0.28, 0.06, 0.23, 0.03, 0.65, 0.01, 0.12, 0.07,
    0.55, 0.01, 0.59, 0.01, 0.29, 0.00, 0.20, 0.02, 0.55, 0.00,
    0.30, 0.01, 0.35, 0.01, 0.56, 0.06, 0.37, 0.04, 0.20, 0.04,
    0.55, 0.02, 0.40, 0.00, 0.40, 0.01, 0.65, 0.05, 0.42, 0.03,
    0.31, 0.06, 0.53, 0.03, 0.28, 0.01, 0.32, 0.03, 0.62, 0.08,
    0.39, 0.04, 0.36, 0.02, 0.59, 0.04, 0.24, 0.03, 0.30, 0.02,
    0.57, 0.00, 0.34, 0.08, 0.46, 0.08, 0.65, 0.00, 0.34, 0.06,
    0.49, 0.00, 0.80, 0.01, 0.45, 0.02, 0.57, 0.02, 0.77, 0.00,
    0.57, 0.01, 0.39, 0.01, 0.81, 0.01, 0.32, 0.03, 0.47, 0.00,
    0.72, 0.05, 0.34, 0.00, 0.30, 0.05, 0.70, 0.04, 0.20, 0.02,
    0.21, 0.02, 0.71, 0.00, 0.30, 0.05, 0.32, 0.03, 0.77, 0.02,
    0.18, 0.04, 0.44, 0.00, 0.67, 0.00, 0.16, 0.02, 0.24, 0.00,
    0.81, 0.01, 0.65, 0.04, 0.13, 0.01, 0.41, 0.01, 0.23, 0.00,
    0.18, 0.05, 0.76, 0.00, 0.14, 0.04, 0.29, 0.01, 0.58, 0.00,
    0.28, 0.01, 0.39, 0.01, 0.76, 0.00, 0.28, 0.01, 0.32, 0.01,
    0.46, 0.00, 0.24, 0.00, 0.17, 0.00, 0.47, 0.00, 0.15, 0.00,
    0.23, 0.00, 0.73, 0.03, 0.29, 0.03, 0.09, 0.01, 0.71, 0.00,
    0.28, 0.00, 0.13, 0.04, 0.72, 0.00, 0.23, 0.00, 0.26, 0.00
], dtype=float).reshape(400,1)
X = np.hstack((feature1, feature2, feature3))
point_names = [f"Point_{i+1}" for i in range(X.shape[0])]

K = 4


df = pd.DataFrame(X, 
                  columns = ["Feature1", "Feature2", "Feature3"],
                  index=point_names)
def update_centroids(memberships, X):
    if memberships is None:
        centroids = X[np.random.choice(range(N), K, False), :]
    else:
        centroids = np.vstack([np.mean(X[memberships == k, :], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)


cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])

centroids = None
memberships = None
iteration = 1
while True:
    if iteration == 100:
        break
    print("Iteration#{}:".format(iteration))

    old_centroids = centroids
    centroids = update_centroids(memberships, X)
    if np.alltrue(centroids == old_centroids):
        break

    old_memberships = memberships
    memberships = update_memberships(centroids, X)
    if np.alltrue(memberships == old_memberships):
       
        break
    iteration = iteration + 1
df["Cluster"] = memberships

# 2) Print each data point and its group
for i in range(len(df)):
    # 'df.index[i]' is something like "Point_1"
    # 'df["Cluster"].iloc[i]' is the cluster ID
    print(f"{df.index[i]} => Cluster: {df['Cluster'].iloc[i]}")
    
cluster_colors = np.array([
    "#1f78b4", "#33a02c", "#e31a1c", 
    "#ff7f00", "#6a3d9a", "#b15928",
    "#a6cee3", "#b2df8a", "#fb9a99", 
    "#fdbf6f", "#cab2d6", "#ffff99"
])

def plot_3d(elev=30, azim=45):
    """
    elev -> 'vertical' angle, 
    azim -> 'horizontal' angle
    """
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    for k in range(K):
        # Extract points in cluster k
        pts_k = X[memberships == k]
        ax.scatter(
            pts_k[:, 0], pts_k[:, 1], pts_k[:, 2],
            c=cluster_colors[k],
            label=f"Cluster {k}",
            alpha=0.7,
            s=40
        )
    # Plot the centroids if you want
    for k in range(K):
        cx, cy, cz = centroids[k]
        ax.scatter(
            cx, cy, cz, 
            c=cluster_colors[k], 
            marker='^', 
            s=200, 
            edgecolor='black',
            label=f"Centroid {k}"
        )

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("Feature1")
    ax.set_ylabel("Feature2")
    ax.set_zlabel("Feature3")
    ax.set_title("Interactive 3D K-Means Clusters")
    ax.legend()
    plt.show()

# Create interactive sliders for elevation & azimuth
from ipywidgets import interact, IntSlider

interact(
        plot_3d,
        elev=IntSlider(min=0, max=90, step=1, value=30),
        azim=IntSlider(min=0, max=360, step=1, value=45)
    )


# In[ ]:





# In[29]:


for k in range(K):
    print(centroids[k])
cluster0 = []
cluster1 = []
cluster2 = []
cluster3 = []
cluster4 = []
feature_1_1 = np.array([], dtype=float)
feature_2_1 = np.array([], dtype=float)
feature_3_1 = np.array([], dtype=float)
point_names_1 = []
for i in range(len(X)):
    k = memberships[i]
    if k == 0:
        cluster0.append(df.index[i])
        feature_1_1 = np.append(feature_1_1, feature1[i]) 
        feature_2_1 = np.append(feature_2_1, feature2[i])  #
        feature_3_1 = np.append(feature_3_1, feature3[i])  #
        point_names_1 = np.append(point_names_1, [f"Point_{i+1}"])
    elif k == 1:
        cluster1.append(df.index[i])
        
    elif k == 2:
        cluster2.append(df.index[i])
    elif k == 3:
        cluster3.append(df.index[i])
    elif k == 4:
        cluster4.append(df.index[i])
print("Cluster M: ")
print(cluster0)
print(len(cluster0))
print("-----------------------------------------------")

print("Cluster E: ")
print(cluster1)
print(len(cluster1))
print("-----------------------------------------------")

print("Cluster SH: ")
print(cluster2)
print(len(cluster2))
print("-----------------------------------------------")

print("Cluster H: ")
print(cluster3)
print(len(cluster3))
print("-----------------------------------------------")
feature_1_1 = feature_1_1.reshape(-1, 1)
feature_2_1 = feature_2_1.reshape(-1, 1)
feature_3_1 = feature_3_1.reshape(-1, 1)

print (feature_1_1)
print(point_names_1)
print (feature_2_1.shape)
print(len(feature_2_1))
print (feature_3_1)
print (feature_2_1)
print(len(feature_3_1))


# In[30]:


N_1 = 99
feature_1_1.reshape(99,1)
feature_2_1.reshape(99,1)
feature_3_1.reshape(99,1)

print(feature_1_1.shape)
print(feature_2_1.shape)
print(feature_3_1.shape)

X_1 = np.hstack((feature_1_1, feature_2_1, feature_3_1))
print(X_1.shape)

K_1 = 4


df_1 = pd.DataFrame(X_1, 
                  columns = ["Feature1", "Feature2", "Feature3"],
                  index=point_names_1)
def update_centroidss(memberships_1, X_1):
    if memberships_1 is None:
        centroids_1 = X_1[np.random.choice(range(N_1), K_1, False), :]
    else:
        centroids_1 = np.vstack([np.mean(X_1[memberships_1 == k, :], axis = 0) for k in range(K_1)])
    return(centroids_1)

def update_membershipss(centroids_1, X_1):
    # calculate distances between centroids and data points
    D_1 = spa.distance_matrix(centroids_1, X_1)
    # find the nearest centroid for each data point
    memberships_1 = np.argmin(D_1, axis = 0)
    return(memberships_1)


cluster_colors_1 = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])

centroids_1 = None
memberships_1 = None
iteration = 1
while True:
    if iteration == 100:
        break
    print("Iteration#{}:".format(iteration))

    old_centroids_1 = centroids_1
    centroids_1 = update_centroidss(memberships_1, X_1)
    if np.alltrue(centroids_1 == old_centroids_1):
        break

    old_memberships = memberships_1
    memberships_1 = update_membershipss(centroids_1, X_1)
    if np.alltrue(memberships_1 == old_memberships):
       
        break
    iteration = iteration + 1
df_1["Cluster"] = memberships_1

# 2) Print each data point and its group
for i in range(len(df_1)):
    # 'df.index[i]' is something like "Point_1"
    # 'df["Cluster"].iloc[i]' is the cluster ID
    print(f"{point_names_1[i]} => Cluster: {df_1['Cluster'].iloc[i]}")
    
cluster_colors_1 = np.array([
    "#1f78b4", "#33a02c", "#e31a1c", 
    "#ff7f00", "#6a3d9a", "#b15928",
    "#a6cee3", "#b2df8a", "#fb9a99", 
    "#fdbf6f", "#cab2d6", "#ffff99"
])

def plot_3d(elev=30, azim=45):
    """
    elev -> 'vertical' angle, 
    azim -> 'horizontal' angle
    """
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    for k in range(K_1):
        # Extract points in cluster k
        pts_k = X_1[memberships_1 == k]
        ax.scatter(
            pts_k[:, 0], pts_k[:, 1], pts_k[:, 2],
            c=cluster_colors_1[k],
            label=f"Cluster {k}",
            alpha=0.7,
            s=40
        )
    # Plot the centroids if you want
    for k in range(K_1):
        cx, cy, cz = centroids_1[k]
        ax.scatter(
            cx, cy, cz, 
            c=cluster_colors_1[k], 
            marker='^', 
            s=200, 
            edgecolor='black',
            label=f"Centroid {k}"
        )

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("Feature1")
    ax.set_ylabel("Feature2")
    ax.set_zlabel("Feature3")
    ax.set_title("Interactive 3D K-Means Clusters")
    ax.legend()
    plt.show()

# Create interactive sliders for elevation & azimuth
from ipywidgets import interact, IntSlider

interact(
        plot_3d,
        elev=IntSlider(min=0, max=90, step=1, value=30),
        azim=IntSlider(min=0, max=360, step=1, value=45)
    )


# In[31]:


for k in range(K_1):
    print(centroids_1[k])
cluster0 = []
cluster1 = []
cluster2 = []
cluster3 = []
cluster4 = []


for i in range(len(X_1)):
    k = memberships_1[i]
    if k == 0:
        cluster0.append(df_1.index[i])
        

    elif k == 1:
        cluster1.append(df_1.index[i])
    elif k == 2:
        cluster2.append(df_1.index[i])
    elif k == 3:
        cluster3.append(df_1.index[i])

print("Cluster sh: ")
print(cluster0)
print(len(cluster0))
print("-----------------------------------------------")

print("Cluster e: ")
print(cluster1)
print(len(cluster1))
print("-----------------------------------------------")

print("Cluster m: ")
print(cluster2)
print(len(cluster2))
print("-----------------------------------------------")

print("Cluster h: ")
print(cluster3)
print(len(cluster3))
print("-----------------------------------------------")




# 

# In[44]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import pandas as pd

np.random.seed(421)
# sample size
N = 400


#df1 = pd.read_csv("Copy of Go Climb! - Difficulty Curve - Copy of Sheet3.csv")

#feature_list = []
#for col in df1.columns:
 #   arr = df1[col].values.reshape(400, 1)
 #   feature_list.append(arr)



#for i in rane(len(feature_list)):
#    X = np.hstack((X, feature_list(i)))


# This yields ["Point_1", "Point_2", ..., "Point_400"]

# ===========================================================================
# 4) Put everything into a DataFrame, using names as the index
# ===========================================================================
#column = []
#for i in range(len(feature_list)):
#    columns.append("Feature"+ i)
    


feature1 = np.array([
    1.00, 1.01, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.04, 1.01, 1.06, 1.06, 1.06, 1.07, 1.06, 1.01, 1.27, 1.07, 1.19, 
    1.03, 1.13, 1.02, 1.20, 1.03, 1.15, 1.18, 1.21, 1.05, 1.35, 1.03, 1.35, 1.07, 1.23, 1.04, 1.43, 1.19, 1.54, 1.16, 
    1.58, 1.09, 4.34, 1.01, 2.84, 1.03, 2.18, 1.02, 2.98, 1.04, 2.66, 1.23, 2.23, 1.02, 2.73, 1.01, 2.44, 1.04, 1.92, 
    1.04, 3.49, 1.02, 2.77, 1.06, 2.17, 1.02, 3.19, 1.01, 2.53, 1.04, 2.42, 1.01, 2.28, 1.02, 2.11, 1.15, 2.10, 1.03, 
    3.55, 1.03, 2.15, 1.01, 2.07, 1.04, 2.99, 1.02, 1.90, 1.04, 1.73, 1.04, 3.57, 1.02, 2.59, 1.03, 2.18, 1.04, 3.49, 
    1.06, 2.06, 1.19, 2.14, 1.06, 1.06, 1.03, 3.50, 1.06, 1.97, 1.13, 3.23, 1.02, 2.24, 1.11, 2.14, 1.08, 4.83, 1.02, 
    1.95, 1.02, 1.94, 1.05, 3.06, 1.02, 1.85, 1.16, 2.30, 1.02, 3.74, 1.16, 2.05, 1.06, 1.90, 1.08, 2.58, 1.04, 1.95, 
    1.07, 1.87, 1.05, 2.94, 1.04, 1.68, 1.05, 1.77, 1.05, 2.55, 1.01, 2.14, 1.07, 1.71, 1.09, 4.01, 1.03, 1.69, 1.00, 
    1.55, 1.03, 3.40, 1.04, 2.20, 1.07, 1.59, 1.04, 3.81, 1.11, 1.60, 1.02, 1.52, 1.03, 3.43, 1.01, 1.86, 1.10, 1.77, 
    1.07, 4.23, 1.03, 1.63, 1.07, 1.75, 1.21, 2.42, 1.03, 1.71, 1.12, 1.75, 1.09, 2.51, 1.01, 2.13, 1.01, 1.83, 1.14, 
    3.39, 1.03, 1.40, 1.16, 1.83, 1.04, 5.38, 1.02, 1.90, 1.02, 1.65, 1.06, 3.01, 1.03, 2.05, 1.16, 1.47, 1.07, 3.71, 
    1.04, 1.70, 1.06, 1.66, 1.06, 3.50, 1.03, 1.42, 1.11, 1.52, 1.09, 2.78, 1.02, 1.53, 1.07, 1.32, 1.12, 3.19, 1.01, 
    1.17, 1.23, 2.35, 1.11, 2.78, 1.05, 1.41, 1.02, 1.40, 1.04, 2.69, 1.06, 1.56, 1.11, 1.92, 1.04, 2.85, 1.05, 1.84, 
    1.07, 1.38, 1.08, 2.53, 1.01, 1.57, 1.04, 1.53, 1.06, 3.81, 1.10, 1.87, 1.06, 1.49, 1.22, 3.09, 1.08, 1.56, 1.12, 
    1.73, 1.05, 2.66, 1.05, 1.85, 1.06, 1.65, 1.03, 3.43, 1.04, 1.36, 1.15, 1.66, 1.02, 2.85, 1.02, 1.98, 1.16, 1.63, 
    1.14, 2.72, 1.05, 1.48, 1.13, 1.94, 1.01, 5.55, 1.01, 1.89, 1.09, 2.67, 1.06, 5.15, 1.05, 3.26, 1.05, 1.73, 1.05, 
    8.53, 1.03, 1.47, 1.05, 2.38, 1.05, 5.12, 1.08, 1.94, 1.08, 1.45, 1.14, 4.96, 1.04, 1.32, 1.02, 1.32, 1.07, 5.90, 
    1.03, 1.52, 1.06, 1.70, 1.12, 6.45, 1.05, 1.17, 1.02, 1.85, 1.06, 3.87, 1.04, 1.18, 1.12, 1.28, 1.21, 5.98, 1.03, 
    2.67, 1.08, 1.19, 1.03, 1.81, 1.05, 1.31, 1.07, 1.33, 1.18, 6.27, 1.01, 1.31, 1.03, 1.60, 1.14, 2.97, 1.06, 1.41, 
    1.10, 2.27, 1.12, 6.78, 1.03, 1.54, 1.09, 1.82, 1.07, 3.28, 1.04, 1.40, 1.10, 1.60, 1.09, 3.30, 1.03, 1.19, 1.01, 
    1.45, 1.15, 3.32, 1.02, 1.34, 1.17, 1.27, 1.07, 4.53, 1.01, 1.43, 1.02, 1.21, 1.04, 3.44, 1.04, 1.37, 1.14, 1.41, 
    1.13
], dtype=float).reshape(400,1)

# =============================================================================
# Feature 2 (400 data points)
# =============================================================================
feature2 = np.array([
    0.992, 0.991, 0.994, 0.993, 0.982, 0.996, 0.993, 0.994, 0.985,
       0.992, 0.948, 0.981, 0.944, 0.974, 0.946, 0.987, 0.775, 0.942,
       0.842, 0.975, 0.887, 0.987, 0.843, 0.979, 0.873, 0.944, 0.829,
       0.961, 0.741, 0.969, 0.745, 0.942, 0.842, 0.965, 0.738, 0.   ,
       0.042, 0.891, 0.351, 0.897, 0.043, 0.992, 0.206, 0.964, 0.36 ,
       0.986, 0.106, 0.959, 0.177, 0.895, 0.273, 0.985, 0.104, 0.989,
       0.251, 0.965, 0.347, 0.964, 0.085, 0.983, 0.154, 0.986, 0.305,
       0.975, 0.121, 0.988, 0.178, 0.961, 0.21 , 0.99 , 0.242, 0.982,
       0.289, 0.952, 0.325, 0.973, 0.083, 0.967, 0.281, 0.982, 0.226,
       0.98 , 0.148, 0.961, 0.27 , 0.985, 0.391, 0.983, 0.115, 0.971,
       0.285, 0.969, 0.401, 0.976, 0.178, 0.916, 0.346, 0.934, 0.324,
       0.963, 0.916, 0.952, 0.159, 0.929, 0.376, 0.916, 0.086, 0.973,
       0.39 , 0.934, 0.328, 0.97 , 0.058, 0.986, 0.475, 0.98 , 0.497,
       0.967, 0.238, 0.957, 0.45 , 0.93 , 0.424, 0.996, 0.179, 0.858,
       0.393, 0.94 , 0.438, 0.891, 0.17 , 0.947, 0.398, 0.973, 0.448,
       0.958, 0.21 , 0.953, 0.51 , 0.955, 0.439, 0.954, 0.164, 0.974,
       0.359, 0.984, 0.481, 0.953, 0.152, 0.954, 0.498, 0.976, 0.52 ,
       0.984, 0.213, 0.95 , 0.33 , 0.97 , 0.538, 0.985, 0.169, 0.869,
       0.517, 0.966, 0.578, 0.96 , 0.224, 0.987, 0.358, 0.963, 0.481,
       0.961, 0.124, 0.973, 0.47 , 0.925, 0.471, 0.873, 0.314, 0.945,
       0.5  , 0.958, 0.424, 0.933, 0.204, 0.996, 0.31 , 0.996, 0.412,
       0.884, 0.165, 0.942, 0.592, 0.895, 0.388, 0.952, 0.091, 0.975,
       0.444, 0.972, 0.56 , 0.96 , 0.238, 0.953, 0.397, 0.912, 0.534,
       0.959, 0.188, 0.942, 0.537, 0.929, 0.377, 0.96 , 0.248, 0.952,
       0.568, 0.931, 0.585, 0.947, 0.316, 0.953, 0.497, 0.916, 0.589,
       0.903, 0.129, 0.979, 0.79 , 0.876, 0.35 , 0.887, 0.242, 0.931,
       0.634, 0.95 , 0.662, 0.962, 0.236, 0.932, 0.511, 0.949, 0.425,
       0.955, 0.213, 0.921, 0.474, 0.881, 0.584, 0.929, 0.255, 0.957,
       0.511, 0.977, 0.672, 0.928, 0.127, 0.875, 0.444, 0.913, 0.56 ,
       0.822, 0.24 , 0.9  , 0.534, 0.954, 0.463, 0.948, 0.257, 0.922,
       0.48 , 0.919, 0.473, 0.949, 0.207, 0.921, 0.678, 0.904, 0.487,
       0.973, 0.196, 0.978, 0.428, 0.843, 0.431, 0.856, 0.27 , 0.955,
       0.606, 0.919, 0.377, 0.955, 0.039, 0.966, 0.412, 0.919, 0.262,
       0.976, 0.066, 0.955, 0.232, 0.946, 0.441, 0.957, 0.032, 0.953,
       0.576, 0.942, 0.327, 0.964, 0.074, 0.907, 0.487, 0.959, 0.59 ,
       0.923, 0.102, 0.893, 0.656, 0.954, 0.608, 0.939, 0.026, 0.942,
       0.488, 0.939, 0.42 , 0.897, 0.059, 0.93 , 0.635, 0.937, 0.32 ,
       0.959, 0.13 , 0.969, 0.695, 0.925, 0.588, 0.858, 0.04 , 0.942,
       0.14 , 0.9  , 0.641, 0.947, 0.323, 0.918, 0.522, 0.971, 0.535,
       0.826, 0.006, 0.959, 0.667, 0.951, 0.535, 0.881, 0.229, 0.936,
       0.493, 0.943, 0.317, 0.949, 0.023, 0.954, 0.436, 0.907, 0.438,
       0.958, 0.147, 0.97 , 0.515, 0.982, 0.459, 0.958, 0.211, 0.973,
       0.667, 0.959, 0.551, 0.912, 0.126, 0.946, 0.593, 0.925, 0.692,
       0.957, 0.055, 0.991, 0.584, 0.955, 0.721, 0.935, 0.172, 0.939,
       0.693, 0.924, 0.655, 0.937
], dtype=float).reshape(400,1)


X = np.hstack((feature1, feature2))
point_names = [f"Point_{i+1}" for i in range(X.shape[0])]

K = 4


df = pd.DataFrame(X, 
                  columns = ["Feature1", "Feature2"],
                  index=point_names)
def update_centroids(memberships, X):
    if memberships is None:
        centroids = X[np.random.choice(range(N), K, False), :]
    else:
        centroids = np.vstack([np.mean(X[memberships == k, :], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)


cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])

centroids = None
memberships = None
iteration = 1
while True:
    if iteration == 100:
        break
    print("Iteration#{}:".format(iteration))

    old_centroids = centroids
    centroids = update_centroids(memberships, X)
    if np.alltrue(centroids == old_centroids):
        break

    old_memberships = memberships
    memberships = update_memberships(centroids, X)
    if np.alltrue(memberships == old_memberships):
       
        break
    iteration = iteration + 1
df["Cluster"] = memberships

# 2) Print each data point and its group
for i in range(len(df)):
    # 'df.index[i]' is something like "Point_1"
    # 'df["Cluster"].iloc[i]' is the cluster ID
    print(f"{df.index[i]} => Cluster: {df['Cluster'].iloc[i]}")
    
cluster_colors = np.array([
    "#1f78b4", "#33a02c", "#e31a1c", 
    "#ff7f00", "#6a3d9a", "#b15928",
    "#a6cee3", "#b2df8a", "#fb9a99", 
    "#fdbf6f", "#cab2d6", "#ffff99"
])

import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import numpy as np

def plot_2d(rotation=0):
    """
    rotation -> angle of rotation in degrees
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Rotate the coordinates of the clusters for visualization
    theta = np.radians(rotation)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    for k in range(K):
        # Extract points in cluster k and apply rotation
        pts_k = X[memberships == k] @ rotation_matrix.T
        ax.scatter(
            pts_k[:, 0], pts_k[:, 1],
            c=cluster_colors[k],
            label=f"Cluster {k}",
            alpha=0.7,
            s=40
        )

    # Plot the centroids if desired
    for k in range(K):
        centroid_rotated = centroids[k] @ rotation_matrix.T
        ax.scatter(
            centroid_rotated[0], centroid_rotated[1],
            c=cluster_colors[k],
            marker='^',
            s=200,
            edgecolor='black',
            label=f"Centroid {k}"
        )

    ax.set_xlabel("Feature1 (atempt)")
    ax.set_ylabel("Feature2 (success rate)")
    ax.set_title("Interactive 2D K-Means Clusters")
    ax.legend()
    ax.grid(True)
    plt.show()

# Create an interactive slider for rotation
interact(
    plot_2d,
    rotation=IntSlider(min=0, max=360, step=1, value=0)
)


# In[45]:


for k in range(K):
    print(centroids[k])
cluster0 = []
cluster1 = []
cluster2 = []
cluster3 = []
cluster4 = []
feature_1_1 = np.array([], dtype=float)
feature_2_1 = np.array([], dtype=float)

point_names_1 = []
for i in range(len(X)):
    k = memberships[i]
    if k == 0:
        cluster0.append(df.index[i])
        feature_1_1 = np.append(feature_1_1, feature1[i]) 
        feature_2_1 = np.append(feature_2_1, feature2[i])  #

        point_names_1 = np.append(point_names_1, [f"Point_{i+1}"])
        
    elif k == 1:
        cluster1.append(df.index[i])
        
    elif k == 2:
        cluster2.append(df.index[i])
    elif k == 3:
        cluster3.append(df.index[i])
        
    elif k == 4:
        cluster4.append(df.index[i])
print("Cluster M: ")
print(cluster0)
print(len(cluster0))
print("-----------------------------------------------")

print("Cluster E: ")
print(cluster1)
print(len(cluster1))
print("-----------------------------------------------")

print("Cluster SH: ")
print(cluster2)
print(len(cluster2))
print("-----------------------------------------------")

print("Cluster H: ")
print(cluster3)
print(len(cluster3))
print("-----------------------------------------------")
feature_1_1 = feature_1_1.reshape(-1, 1)
feature_2_1 = feature_2_1.reshape(-1, 1)
feature_3_1 = feature_3_1.reshape(-1, 1)

print (feature_1_1)
print(point_names_1)
print (feature_2_1.shape)
print(len(feature_2_1))
print (feature_3_1)
print (feature_2_1)
print(len(feature_3_1))


# In[46]:


N_1 = 95
feature_1_1.reshape(95,1)
feature_2_1.reshape(95,1)


print(feature_1_1.shape)
print(feature_2_1.shape)


X_1 = np.hstack((feature_1_1, feature_2_1))
print(X_1.shape)

K_1 = 4


df_1 = pd.DataFrame(X_1, 
                  columns = ["Feature1", "Feature2"],
                  index=point_names_1)
def update_centroidss(memberships_1, X_1):
    if memberships_1 is None:
        centroids_1 = X_1[np.random.choice(range(N_1), K_1, False), :]
    else:
        centroids_1 = np.vstack([np.mean(X_1[memberships_1 == k, :], axis = 0) for k in range(K_1)])
    return(centroids_1)

def update_membershipss(centroids_1, X_1):
    # calculate distances between centroids and data points
    D_1 = spa.distance_matrix(centroids_1, X_1)
    # find the nearest centroid for each data point
    memberships_1 = np.argmin(D_1, axis = 0)
    return(memberships_1)


cluster_colors_1 = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])

centroids_1 = None
memberships_1 = None
iteration = 1
while True:
    if iteration == 100:
        break
    print("Iteration#{}:".format(iteration))

    old_centroids_1 = centroids_1
    centroids_1 = update_centroidss(memberships_1, X_1)
    if np.alltrue(centroids_1 == old_centroids_1):
        break

    old_memberships = memberships_1
    memberships_1 = update_membershipss(centroids_1, X_1)
    if np.alltrue(memberships_1 == old_memberships):
       
        break
    iteration = iteration + 1
df_1["Cluster"] = memberships_1

# 2) Print each data point and its group
for i in range(len(df_1)):
    # 'df.index[i]' is something like "Point_1"
    # 'df["Cluster"].iloc[i]' is the cluster ID
    print(f"{point_names_1[i]} => Cluster: {df_1['Cluster'].iloc[i]}")
    
cluster_colors_1 = np.array([
    "#1f78b4", "#33a02c", "#e31a1c", 
    "#ff7f00", "#6a3d9a", "#b15928",
    "#a6cee3", "#b2df8a", "#fb9a99", 
    "#fdbf6f", "#cab2d6", "#ffff99"
])

def plot_2d(rotation=0):
    """
    rotation -> angle of rotation in degrees
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Rotate the coordinates of the clusters for visualization
    theta = np.radians(rotation)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    for k in range(K):
        # Extract points in cluster k and apply rotation
        pts_k = X[memberships == k] @ rotation_matrix.T
        ax.scatter(
            pts_k[:, 0], pts_k[:, 1],
            c=cluster_colors[k],
            label=f"Cluster {k}",
            alpha=0.7,
            s=40
        )

    # Plot the centroids if desired
    for k in range(K):
        centroid_rotated = centroids[k] @ rotation_matrix.T
        ax.scatter(
            centroid_rotated[0], centroid_rotated[1],
            c=cluster_colors[k],
            marker='^',
            s=200,
            edgecolor='black',
            label=f"Centroid {k}"
        )

    ax.set_xlabel("Feature1 (atempt)")
    ax.set_ylabel("Feature2 (success rate)")
    ax.set_title("Interactive 2D K-Means Clusters")
    ax.legend()
    ax.grid(True)
    plt.show()

# Create an interactive slider for rotation
interact(
    plot_2d,
    rotation=IntSlider(min=0, max=360, step=1, value=0)
)


# In[47]:


for k in range(K_1):
    print(centroids_1[k])
cluster0 = []
cluster1 = []
cluster2 = []
cluster3 = []
cluster4 = []


for i in range(len(X_1)):
    k = memberships_1[i]
    if k == 0:
        cluster0.append(df_1.index[i])
        

    elif k == 1:
        cluster1.append(df_1.index[i])
    elif k == 2:
        cluster2.append(df_1.index[i])
    elif k == 3:
        cluster3.append(df_1.index[i])

print("Cluster sh: ")
print(cluster0)
print(len(cluster0))
print("-----------------------------------------------")

print("Cluster e: ")
print(cluster1)
print(len(cluster1))
print("-----------------------------------------------")

print("Cluster m: ")
print(cluster2)
print(len(cluster2))
print("-----------------------------------------------")

print("Cluster h: ")
print(cluster3)
print(len(cluster3))
print("-----------------------------------------------")




# In[43]:





# In[ ]:




