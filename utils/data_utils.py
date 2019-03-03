# -*- coding: utf-8 -*-

"""Utility methods for Exploratory Data Analysis and Pre-processing"""

import os
import shutil
import operator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
from google.cloud import storage

import seaborn as sns
from sklearn import preprocessing
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import percentileofscore
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

TM_pal_categorical_3 = ("#ef4631", "#10b9ce", "#ff9138")
sns.set(
    style="white",
    font_scale=1.25,
    palette=TM_pal_categorical_3,
)

SEED = 42
np.random.seed(SEED)

#### Scoring Helper Functions ####
def pearsonr2(estimator, X, y_true):
    """Calculates r-squared score using pearsonr
    
    Parameters
    ----------
    estimator 
        The model or regressor to be evaluated
    X : pandas dataframe or a 2-D matrix
        The feature matrix
    y : list of pandas series
        The target vector
        
    Returns
    ----------
    float
        R2 using pearsonr
    """
    y_pred = estimator.predict(X)
    return pearsonr(y_true, y_pred)[0]**2

def mae(estimator, X, y_true): 
    """Calculates mean absolute error
    
    Parameters
    ----------
    estimator 
        The model or regressor to be evaluated
    X : pandas dataframe or a 2-D matrix
        The feature matrix
    y : list of pandas series
        The target vector
        
    Returns
    ----------
    float
        Mean absolute error
    """
    y_pred = estimator.predict(X)
    return mean_absolute_error(y_true, y_pred)
    
def rmse(estimator, X, y_true): 
    """Calculates root mean squared error
    
    Parameters
    ----------
    estimator 
        The model or regressor to be evaluated
    X : pandas dataframe or a 2-D matrix
        The feature matrix
    y : list of pandas series
        The target vector
        
    Returns
    ----------
    float
        Root mean squared error
    """
    y_pred = estimator.predict(X)
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2(estimator, X, y_true): 
    """Calculates r-squared score using python's r2_score function
    
    Parameters
    ----------
    estimator 
        The model or regressor to be evaluated
    X : pandas dataframe or a 2-D matrix
        The feature matrix
    y : list of pandas series
        The target vector
        
    Returns
    ----------
    float
        R-squared score using python's r2_score function
    """
    y_pred = estimator.predict(X)
    return r2_score(y_true, y_pred)  

def mape(estimator, X, y_true): 
    """Calculates mean average percentage error
    
    Parameters
    ----------
    estimator 
        The model or regressor to be evaluated
    X : pandas dataframe or a 2-D matrix
        The feature matrix
    y : list of pandas series
        The target vector
        
    Returns
    ----------
    float
        Mean average percentage error
    """
    y_pred = estimator.predict(X)
    return np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100

def adj_r2(estimator, X, y_true):
    """Calculates adjusted r-squared score
    
    Parameters
    ----------
    estimator 
        The model or regressor to be evaluated
    X : pandas dataframe or a 2-D matrix
        The feature matrix
    y : list of pandas series
        The target vector
        
    Returns
    ----------
    float
        Adjusted r-squared score
    """
    y_pred = estimator.predict(X)
    r2 = r2_score(y_true, y_pred)
    n = X.shape[0]
    k = X.shape[1]
    adj_r2 = 1 - (((1-r2)*(n-1))/(n - k - 1))
    
    return adj_r2

def percentile_ranking(series):
    """Converts list of numbers to percentile and ranking
    
    Parameters
    ----------
    series : pandas Series
        A series of numbers to be converted to percentile ranking
    
    Returns
    ----------
    list (of floats)
        A list of converted percentile values using scipy.stats percentileofscore()
    list (of ints)
        A list containing the ranks 
    """
    percentiles = []
    for index, value in series.iteritems():
        curr_index = series.index.isin([index])
        percentile = percentileofscore(series[~curr_index], value)
        percentiles.append(percentile)
    ranks = series.rank(axis=0, ascending=False)
    
    return percentiles, ranks

#### Plotting Helper Functions ####

def plot_hist(data, title, x_label, y_label, bins=30):
    """Plots histogram for the given data
    
    Parameters
    ----------
    data : pandas Series
        The data to plot histogram
    title : str
        The title of the figure
    x_label : str
        Label of the x axis
    y_label : str
        Label of the y-axis
    bins : int
        Number of bins for histogram
    """
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_regplot(
    data, 
    x_label='Wealth Index', 
    y_label='Average Nightlight Intensity',
    y_var='ntl2016'
):
    """Produces the regression plot for the given data
    
    Parameters
    ----------
    data : pandas Series
        The data to plot regression plot
    x_var : str
        The variable name of the x-axis
    y_var : str
        The variable name of the y-axis
    x_label : str
        Label of the x axis
    y_label : str
        Label of the y-axis
    """
    ax = sns.regplot(
        x=x_label,
        y=y_var,
        data=data,
        lowess=True,
        line_kws={"color": "black", "lw": 2},
        scatter_kws={"alpha": 0.3},
    )
    plt.title(
        "Relationship between {} \nand {}".format(
            x_label, y_label
        )
        + r" ($\rho$ = %.2f, $r$ =%.2f)"
        % (
            spearmanr(
                data[x_label].tolist(), data[y_var].tolist()
            )[0],
            pearsonr(
                data[x_label].tolist(), data[y_var].tolist()
            )[0],
        )
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_corr(
    data,
    features_cols,
    indicator="Wealth Index",
    figsize=(5, 6),
    max_n=30,
):
    """Produces a barplot of the Spearman rank correlation and Pearson's correlation 
    for a group of values in descending order
    
    Parameters
    ----------
    data : pandas DataFrame
        The dataframe containing the feature columns
    feature_cols : str
        The list of feature column names in the data
    indicator : str (default is "Wealth Index")
        The socioeconomic indicator to correlate each variable with
    figsize : tuple (default is (5,6))
        Size of the figure
    max_n : int
        Maximum number of variables to plot 
    """
    
    n = len(features_cols)
    spearman = []
    pearsons = []
    for feature in features_cols:
        spearman.append(
            (
                feature,
                spearmanr(data[feature], data[indicator])[
                    0
                ],
            )
        )
        pearsons.append(
            (
                feature,
                pearsonr(data[feature], data[indicator])[0],
            )
        )
    spearman = sorted(spearman, key=lambda x: abs(x[1]))
    pearsons = sorted(pearsons, key=lambda x: abs(x[1]))

    plt.figure(figsize=figsize)
    plt.title(
        "Spearman Correlation Coefficient for {}".format(
            indicator
        )
    )
    plt.barh(
        [x[0] for x in spearman[n - max_n :]],
        [x[1] for x in spearman[n - max_n :]],
    )
    plt.grid()

    plt.figure(figsize=figsize)
    plt.title(
        "Pearsons Correlation Coefficient for {}".format(
            indicator
        )
    )
    plt.barh(
        [x[0] for x in pearsons[n - max_n :]],
        [x[1] for x in pearsons[n - max_n :]],
    )
    plt.grid()


#### Nighttime Lights Pre-processing Helper Functions ####

def ntl_agg_fnc(data):
    agg = {}
    agg['mean'] = data['ntl2016'].mean()
    agg['max'] = data['ntl2016'].max()
    agg['min'] = data['ntl2016'].min()
    agg['median'] = data['ntl2016'].median()
    agg['cov'] = data['ntl2016'].cov(data['ntl2016'])
    agg['std'] = data['ntl2016'].std()
    agg['skewness'] =  data['ntl2016'].skew()
    agg['kurtosis'] =  data['ntl2016'].kurtosis()
    return pd.Series(agg, index=[
        'mean', 
        'max', 
        'min', 
        'median', 
        'cov', 
        'std', 
        'skewness', 
        'kurtosis'
    ])

def unstack_clusters(
    data,
    id_col='ID',
    dhs_col='DHSCLUST',
    lat_col='ntllat',
    lon_col='ntllon',
    ntl_col='ntl2016',
    file_col='filename',
    ph_prefix=True
):
    """ Unstacks nightlights data where certain pixels can belong to two or more clusters. 
    Makes it so that each row is a unique (cluster, id) pair.
    
    Parameters
    ----------
    data : pandas DataFrame
        The nightlights dataset to be unstacked
    
    Returns
    ----------
    pandas DataFrame
        A dataframe of unstacked rows
    """
    
    first_row = data.iloc[0, :]
    temp = {x: [] for x in [id_col, dhs_col, lat_col, lon_col, ntl_col, file_col] if x in first_row}
    for index, row in tqdm(
        data.iterrows(), total=len(data)
    ):
        clusters = [
            x.strip() for x in row[dhs_col].split(",")
        ]
        for cluster in clusters:
            if ph_prefix:
                cluster = cluster.replace("PH2017", "").lstrip("0")
            temp[dhs_col].append(int(cluster))
            if id_col in row:
                temp[id_col].append(row[id_col])
            if lon_col in row:
                temp[lon_col].append(row[lon_col])
            if lat_col in row:
                temp[lat_col].append(row[lat_col])
            if ntl_col in row:
                temp[ntl_col].append(row[ntl_col])
            if file_col in row:
                temp[file_col].append(row[file_col])
    data = pd.DataFrame(temp)
    
    return data


def gaussian_mixture_model(
    data, 
    ntl_col='ntl2016',
    n_components=3, 
    max_iter=1000,
    tol=1e-10,
    covariance_type='full',
    bin_labels=['low', 'medium', 'high']
):
    """ Implements Gaussian Mixture Model (GMM) on the nighttime light intensities
    
    Parameters
    ----------
    data : pandas DataFrame
        Contains the nightlights column to be binned 
    ntl_col : str
        Name of the column containing nightlight intensities
    n_components : int (default is 3)
        Number of components for the GMM
    max_iter : int (default is 1000)
        Maximum number of iterations for GMM
    tol : float (default is 1e-10)
        GMM tolerance
    covariance_type: str (default is 'full')
        GMM covariance type
        
    Returns
    ----------
    pandas DataFrame
        A dataframe containing an additional field 'label' indicating the nightlight intensity level
    """
    series = np.array(data[ntl_col]).reshape(-1, 1)

    # Instantiate GMM
    gmm = GaussianMixture(
        n_components=n_components,
        max_iter=max_iter,
        tol=tol,
        covariance_type=covariance_type,
        random_state=SEED,
    ).fit(series)

    # Predict
    intensities = gmm.predict(series)

    # Assign night light intensity levels
    data["label"] = intensities
    
    bin_caps = {}
    for x in range(n_components):
        bin_caps[x] = data[data['label'] == x][ntl_col].max()
    print(bin_caps)
    bin_caps = sorted(bin_caps.items(), key=operator.itemgetter(1))
    
    assign_labels = {}
    for val, label in zip(bin_caps, bin_labels):
        assign_labels[val[0]] = label
    print(assign_labels)
    
    data["label"] = data["label"].replace(
        assign_labels
    )
    
    return data


def ad_hoc_binning(
    intensity, 
    bin_caps=[2, 20],
    bin_labels=['low', 'medium', 'high']
):
    """Implements ad-hoc binning (3 bins) for nighttime lights
    
    Parameters
    ----------
    intensity : float
        The nighttime light intensity of a single pixel
    bin_caps : list
        Maximum values per bin (should contain n-1 bins as there is 
        no need to specify the maximum cap of the last bin)
    
    Returns
    ----------
    str
        Indicates nighttime light intensity (low, medium, or high)
    """
    bin_caps.append(1e100)
    for val, label in zip(bin_caps, bin_labels):
        if intensity <= val:
            return label


def train_val_split(data, train_size=0.9):
    """Splits the data into training and validation set. 
    
    Parameters
    ----------
    data : pandas DataFrame
        The data to be split
    train_size : float (default is 0.9)
        The size of the training set. Size of validation set is 1 - train_size
        
    Returns
    -------
    pandas DataFrame
        The training set
    pandas DataFrame
        The validation set
    """
    
    train = data.iloc[: int(len(data) * train_size), :]
    val = data.iloc[int(len(data) * train_size) :, :]
    
    return train, val

def balance_dataset(data, size=60000):
    """Implements upsampling and downsampling for the three classes (low, medium, and high)
    
    Parameters
    ----------
    data : pandas DataFrame
        A dataframe containing the labels indicating the different nightlight intensity bins
    size : int
        The number of samples per classes for upsampling and downsampling
    
    Returns
    -------
    pandas DataFrame
        The data with relabelled and balanced nightlight intensity classes
    """
    
    bin_labels = data.label.unique()
    
    classes = []
    for label in bin_labels:
        class_ = data[data.label == label].reset_index()
        if len(class_) >= size:
            sample = class_.sample(
                n=size, replace=False, random_state=SEED
            )
        elif len(class_) < size:
            sample = class_.sample(
                n=size, replace=True, random_state=SEED
            )
        classes.append(sample)

    data_balanced = pd.concat(classes)
    data_balanced = data_balanced.sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)
    data_balanced = data_balanced.iloc[:, 1:]
    return data_balanced


def train_val_split_images(data, report, dst_dir, phase="train"):
    """Splits the downloaded images into training and validation folders
        
    Parameters
    ----------
    data : pandas DataFrame
        Contains the class labels of each image, idetified by its lat lng
    report : pandas DataFrame
        Contains the file locations of each images based on the lat lngs
    phase : str
        Indicates whether training or validation set
    """
    
    for index, row in tqdm(
        data.iterrows(), total=len(data)
    ):
        label = row["label"]
        lat = row["ntllat"]
        lon = row["ntllon"]
        id_ = row["ID"]

        image = report[
            (report["lat"] == lat)
            & (report["lon"] == lon)
            & (report["id"] == id_)
        ]
        if len(image) != 0:
            src_dir = image["filename"].iloc[0]
            filename = os.path.basename(src_dir)
            filename = filename.split(".")
            dst_file = "{}{}/{}".format(dst_dir, phase, label)
            dst_file = "{}/{}_{}.{}".format(
                dst_file, 
                ".".join(x for x in filename[:-1]), 
                str(index),
                filename[-1]
            )
            os.makedirs(
                os.path.dirname(dst_file), exist_ok=True
            )
            shutil.copyfile(src_dir, dst_file)


#### Gooogle Cloud Storage Migration Helper Functions ####

def upload_to_bucket(
    blob_name, directory, path_to_file, bucket_name
):
    """ Upload data to a Google Cloud Storage bucket
    
    Parameters
    ----------
    blob_name : str
        Name of file in GCS once uploaded
    directory : str
        GCS directory
    path_to_file : str
        Local path to file
    bucket_name : str
        Name of GCS bucket
    
    Returns
    -------
    str
        The public url of the file
    """

    storage_client = storage.Client()
    buckets = list(storage_client.list_buckets())
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob("{}/{}".format(directory, blob_name))
    blob.upload_from_filename(path_to_file)

    print("{} successfully uploaded".format(path_to_file))
    return blob.public_url


def download_from_bucket(
    blob_name, directory, destination_file_name, bucket_name
):
    """ Download data from Gogle Cloud Storage bucket
    
    Parameters
    ----------
    blob_name : str
        Name of file in GCS once uploaded
    directory : str
        GCS directory
    path_to_file : str
        Local path to file
    bucket_name : str
        Name of GCS bucket
    
    Returns
    -------
    str
        The public url of the file
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob("{}/{}".format(directory, blob_name))

    destination_file_dir = os.path.dirname(
        destination_file_name
    )
    if not os.path.exists(destination_file_dir):
        os.makedirs(destination_file_dir)
    blob.download_to_filename(destination_file_name)

    print(
        "{} successfully downloaded".format(
            destination_file_name
        )
    )
    return blob.public_url
