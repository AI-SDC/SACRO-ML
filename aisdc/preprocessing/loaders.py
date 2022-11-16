"""
loaders.py
A set of useful handlers to pull in datasets common to the project and perform the appropriate
pre-processing
"""

# pylint: disable=import-error, invalid-name, consider-using-with, too-many-return-statements

import logging
import os
from collections import Counter
from typing import List, Tuple
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import pylab as plt
from sklearn.datasets import fetch_openml, load_iris
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Following is to stop pylint always sqwarking at pandas things
# pylint: disable = no-member, unsubscriptable-object


logging.basicConfig(level="DEBUG")

logger = logging.getLogger(__file__)


PROJECT_ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))
logger.info("ROOT PROJECT FOLDER = %s", PROJECT_ROOT_FOLDER)


class UnknownDataset(Exception):
    """Exception raised if the user passes a name that we don't recognise"""


class DataNotAvailable(Exception):
    """Exception raised if the user asks for a dataset that they do not have the data for. I.e.
    some datasets require a .csv file to have been downloaded.
    """


def get_data_sklearn(  # pylint: disable = too-many-branches
    dataset_name: str, data_folder: str = os.path.join(PROJECT_ROOT_FOLDER, "data")
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main entry method to return data in format sensible for sklearn. User passes a name and that
    dataset is returned as a tuple of pandas DataFrames (data, labels).

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to load
    data_folder: str
        The name of the local folder in which data is stored.

    Returns
    -------

    X: pd.DataFrame
        The input dataframe -- rows are examples, columns variables
    y: pd.DataFrame
        the target dataframe -- has a single column containing the target values

    Notes
    -----

    The following datasets are available:
    mimic2-iaccd (requires data download)
    in-hospital-mortality (requires data download)
    medical-mnist-ab-v-br-100 (requires data download)
    medical-mnist-ab-v-br-500 (requires data download)
    medical-mnist-all-100 (requires data download)
    indian liver (requires data download)
    texas hospitals 10 (requires data download)
    synth-ae (requires data download)
    synth-ae-small (requires data download)
    nursery (downloads automatically)
    iris (available out of the box via sklearn)

    Datasets can be normalised by adding the following prefixes:
    standard: standardises all columns to have zero mean and unit variance.
    minmax: standardises all columns to have values between 0 and 1.
    round: rounds continues features to have 3dp

    These can be nested.

    Examples
    --------
    >>> X, y = get_data_sklearn("mimic2-iaccd") # pull the mimic2-iaccd data
    >>> X, y = get_data_sklearn("minmax iris") # pull the iris data and round continuous features


    """
    logger.info("DATASET FOLDER = %s", data_folder)

    if dataset_name.startswith("standard"):
        sub_name = dataset_name.split("standard")[1].strip()
        feature_df, target_df = get_data_sklearn(sub_name)
        for column in feature_df.columns:
            col_mean = feature_df[column].mean()
            col_std = np.sqrt(feature_df[column].var())
            feature_df[column] = feature_df[column] - col_mean
            feature_df[column] = feature_df[column] / col_std
        return feature_df, target_df

    if dataset_name.startswith("minmax"):
        sub_name = dataset_name.split("minmax")[1].strip()
        feature_df, target_df = get_data_sklearn(sub_name)
        for column in feature_df.columns:
            col_min = feature_df[column].min()
            col_range = feature_df[column].max() - col_min
            feature_df[column] = feature_df[column] - col_min
            feature_df[column] = feature_df[column] / col_range
        return feature_df, target_df

    if dataset_name.startswith("round"):
        sub_name = dataset_name.split("round")[1].strip()
        logger.debug(sub_name)
        feature_df, target_df = get_data_sklearn(sub_name)
        column_dtype = feature_df.dtypes  # pylint: disable = no-member

        for i, column in enumerate(feature_df.columns):
            if column_dtype[i] == "float64":
                feature_df[column] = feature_df[column].round(decimals=3)
        return feature_df, target_df

    if dataset_name == "mimic2-iaccd":
        return _mimic_iaccd(data_folder)
    if dataset_name == "in-hospital-mortality":
        return _in_hospital_mortality(data_folder)
    if dataset_name == "medical-mnist-ab-v-br-100":
        return _medical_mnist_loader(data_folder, 100, ["AbdomenCT", "BreastMRI"])
    if dataset_name == "medical-mnist-ab-v-br-500":
        return _medical_mnist_loader(data_folder, 500, ["AbdomenCT", "BreastMRI"])
    if dataset_name == "medical-mnist-all-100":
        return _medical_mnist_loader(
            data_folder,
            100,
            ["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"],
        )
    if dataset_name == "indian liver":
        return _indian_liver(data_folder)
    if dataset_name == "texas hospitals 10":
        return _texas_hospitals(data_folder)
    if dataset_name == "synth-ae":
        return _synth_ae(data_folder)
    if dataset_name == "synth-ae-small":
        return _synth_ae(data_folder, 200)
    if dataset_name == "nursery":
        return _nursery()
    if dataset_name == "iris":
        return _iris()
    raise UnknownDataset(dataset_name)


def _iris() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    sklearn iris data - just first two classes
    """
    X, y = load_iris(return_X_y=True, as_frame=True)
    X = X[y < 2]
    y = y[y < 2]
    return X, pd.DataFrame(y)


def _nursery() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    The sklearn nursery dataset
    """

    data = fetch_openml(data_id=26, as_frame=True)

    target_encoder = LabelEncoder()
    target_vals = target_encoder.fit_transform(data["target"].values)
    target_dataframe = pd.DataFrame({"target": target_vals})

    feature_encoder = OneHotEncoder()
    x_encoded = feature_encoder.fit_transform(data["data"]).toarray()
    feature_dataframe = pd.DataFrame(
        x_encoded, columns=feature_encoder.get_feature_names_out()
    )

    return feature_dataframe, target_dataframe


# Patched to support non-flattened images. Same behaviour as before except if called with
# flatten=False explicitly.
def _images_to_ndarray(
    images_dir: str, number_to_load: int, label: int, flatten: bool = True
) -> Tuple[np.array, np.array]:
    """
    Grab number_to_load images from the images_dir and create a np array and label array
    """
    folder_path = images_dir + os.sep
    images_names = sorted(os.listdir(folder_path))
    images_names = images_names[:number_to_load]
    # fix f or macosx
    if ".DS_Store" in images_names:  # pragma: no cover
        images_names.remove(".DS_Store")

    if flatten:
        np_images = np.array(
            [plt.imread(folder_path + img).flatten() for img in images_names]
        )
    else:
        np_images = np.array([plt.imread(folder_path + img) for img in images_names])
    labels = np.ones((len(np_images), 1), int) * label
    return (np_images, labels)


def _medical_mnist_loader(  # pylint: disable = too-many-locals
    data_folder: str, n_per_class: int, classes: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Medical MNIST into pandas format
    borrows heavily from: https://www.kaggle.com/harelshattenstein/medical-mnist-knn
    Creates a binary classification
    """

    base_folder = os.path.join(
        data_folder,
        "kaggle-medical-mnist",
        "archive",
    )

    zip_file = os.path.join(data_folder, "kaggle-medical-mnist", "archive.zip")

    logger.debug(base_folder, data_folder)
    if not any([os.path.exists(base_folder), os.path.exists(zip_file)]):
        help_message = f"""
Data file {base_folder} does not exist. Please download fhe file from:
https://www.kaggle.com/andrewmvd/medical-mnist
and place it in the correct folder. It unzips the file first.
        """
        raise DataNotAvailable(help_message)

    if os.path.exists(base_folder):
        pass
    elif os.path.exists(zip_file):
        try:
            with ZipFile(zip_file) as zip_handle:
                zip_handle.extractall(base_folder)
                logger.debug("Extracted all")
        except BadZipFile:  # pragma: no cover
            logger.error("Encountered bad zip file")
            raise

    labels_dict = {
        0: "AbdomenCT",
        1: "BreastMRI",
        2: "CXR",
        3: "ChestCT",
        4: "Hand",
        5: "HeadCT",
    }

    reverse_labels_dict = {v: k for k, v in labels_dict.items()}

    for i, class_name in enumerate(classes):
        label = reverse_labels_dict[class_name]
        x_images, y_images = _images_to_ndarray(
            os.path.join(base_folder, class_name), n_per_class, label
        )

        if i == 0:
            all_x = x_images
            all_y = y_images
        else:
            all_x = np.vstack((all_x, x_images))
            all_y = np.vstack((all_y, y_images))

    return (pd.DataFrame(all_x), pd.DataFrame(all_y))


def _synth_ae(
    data_folder: str, n_rows: int = 5000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    First norws (default 5000) rows from the Synthetic A&E data from NHS England
    https://data.england.nhs.uk/dataset/a-e-synthetic-data/resource/81b068e5-6501-4840-a880-a8e7aa56890e # pylint: disable=line-too-long
    """

    file_path = os.path.join(
        data_folder, "AE_England_synthetic.csv"  #'A&E Synthetic Data.csv'
    )

    if not os.path.exists(file_path):
        help_message = f"""
Data file {file_path} does not exist. Please download the file from:
https://data.england.nhs.uk/dataset/a-e-synthetic-data/resource/81b068e5-6501-4840-a880-a8e7aa56890e
unzip it (7z) and then copy the .csv file into your data folder.
    """
        raise DataNotAvailable(help_message)

    input_data = pd.read_csv(file_path, nrows=n_rows)
    columns_to_drop = [
        "AE_Arrive_Date",
        "AE_Arrive_HourOfDay",
        "Admission_Method",
        "ICD10_Chapter_Code",
        "Treatment_Function_Code",
        "Length_Of_Stay_Days",
        "ProvID",
    ]
    input_data.drop(columns_to_drop, axis=1, inplace=True)

    # Remove any rows with NAs in the remaining columns
    input_data.dropna(axis=0, inplace=True)

    # One-hot encode some columns
    encode_columns = ["Age_Band", "AE_HRG"]
    encode_data = input_data[encode_columns]
    input_data.drop(encode_columns, axis=1, inplace=True)

    oh = OneHotEncoder()
    oh.fit(encode_data)
    onehot_df = pd.DataFrame(
        oh.transform(encode_data).toarray(),
        columns=oh.get_feature_names_out(),
        index=input_data.index,
    )

    input_data = pd.concat([input_data, onehot_df], axis=1)

    X = input_data.drop(["Admitted_Flag"], axis=1)
    y = input_data[["Admitted_Flag"]]

    return (X, y)


def _indian_liver(data_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Indian Liver Patient Dataset
    https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv # pylint: disable=line-too-long
    """
    # (https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)
    file_path = os.path.join(data_folder, "Indian Liver Patient Dataset (ILPD).csv")
    if not os.path.exists(file_path):
        help_message = f"""
Data file {file_path} does not exist. Please download fhe file from:
https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset
and place it in the correct folder.
        """
        raise DataNotAvailable(help_message)

    column_names = [
        "age",
        "gender",
        "total Bilirubin",
        "direct Bilirubin",
        "Alkphos",
        "SGPT",
        "SGOT",
        "total proteins",
        "albumin",
        "A/G ratio",
        "class",
    ]

    liver_data = pd.read_csv(file_path, names=column_names, index_col=False)

    liver_data.gender.replace("Male", 0, inplace=True)
    liver_data.gender.replace("Female", 1, inplace=True)

    liver_data.dropna(axis=0, inplace=True)

    liver_labels = liver_data["class"]
    liver_data.drop(["class"], axis=1, inplace=True)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(liver_labels.values)
    liver_labels = pd.DataFrame({"class": encoded_labels})
    return (liver_data, liver_labels)


def _in_hospital_mortality(data_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    In-hospital mortality data from this study:
        https://datadryad.org/stash/dataset/doi:10.5061/dryad.0p2ngf1zd
    """
    # Check the data has been downloaded. If not throw an exception with instructions on how to
    # download, and where to store
    files = ["data01.csv", "doi_10.5061_dryad.0p2ngf1zd__v5.zip"]
    file_path = [os.path.join(data_folder, f) for f in files]

    if not any(  # pylint: disable=use-a-generator
        [os.path.exists(fp) for fp in file_path]
    ):  # pylint: disable=use-a-generator
        help_message = f"""
Data file {file_path[0]} or {file_path[1]} does not exist. Please download the file from:
https://datadryad.org/stash/dataset/doi:10.5061/dryad.0p2ngf1zd
and place it in the correct folder. It works with either the zip file or uncompressed.
        """
        raise DataNotAvailable(help_message)

    if os.path.exists(file_path[1]):
        input_data = pd.read_csv(ZipFile(file_path[1]).open("data01.csv"))
    else:
        input_data = pd.read_csv(file_path[0])
    clean_data = input_data.dropna(axis=0, how="any").drop(columns=["group", "ID"])
    target = "outcome"
    labels = clean_data[target]
    features = clean_data.drop([target], axis=1)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels.values)
    labels = pd.DataFrame({"outcome": encoded_labels})

    return (features, labels)


def _mimic_iaccd(data_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the mimic_iaccd data and performs Alba's pre-processing
    """

    # Check the data has been downloaded. If not throw an exception with instructions on how to
    # download, and where to store
    file_path = os.path.join(data_folder, "mimic2-iaccd", "1.0", "full_cohort_data.csv")

    if not os.path.exists(file_path):
        help_message = f"""
        The MIMIC2-iaccd data is not available in {data_folder}.
        The following file should exist: {file_path}.
        Please download from https://physionet.org/content/mimic2-iaccd/1.0/full_cohort_data.csv
        """
        raise DataNotAvailable(help_message)

    # File exists, load and preprocess#
    logger.info("Loading mimic2-iaccd")
    input_data = pd.read_csv(file_path)

    logger.info("Preprocessing")
    # remove columns non-numerical and repetitive or uninformative data for the analysis
    col = [
        "service_unit",
        "day_icu_intime",
        "hosp_exp_flg",
        "icu_exp_flg",
        "day_28_flg",
    ]
    # service_num is the numerical version of service_unit
    # day_icu_intime_num is the numerical version of day_icu_intime
    # the other columns are to do with death and are somewhat repetitive with censor_flg
    input_data.drop(col, axis=1, inplace=True)

    # drop columns with only 1 value
    input_data.drop("sepsis_flg", axis=1, inplace=True)

    # drop NA by row
    input_data.dropna(axis=0, inplace=True)

    # extract target
    target = "censor_flg"
    y = input_data[target]
    X = input_data.drop([target, "mort_day_censored"], axis=1)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y.values)
    y = pd.DataFrame({"censor_flag": encoded_labels})

    return (X, y)


def _texas_hospitals(
    data_folder: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:  # pragma: no cover
    # pylint: disable=too-many-statements, too-many-locals
    """
    Texas Hospitals Dataset
    (https://www.dshs.texas.gov/THCIC/Hospitals/Download.shtm)

    Note: this has been tested repeated in the GRAIMatter project.
    However, for licensing reasons we cannot redistribute the data.
    Therefore it is omitted from CI test coverage and metrics.

    """
    file_list = [
        "PUDF-1Q2006-tab-delimited.zip",
        "PUDF-1Q2007-tab-delimited.zip",
        "PUDF-1Q2008-tab-delimited.zip",
        "PUDF-1Q2009-tab-delimited.zip",
        "PUDF-2Q2006-tab-delimited.zip",
        "PUDF-2Q2007-tab-delimited.zip",
        "PUDF-2Q2008-tab-delimited.zip",
        "PUDF-2Q2009-tab-delimited.zip",
        "PUDF-3Q2006-tab-delimited.zip",
        "PUDF-3Q2007-tab-delimited.zip",
        "PUDF-3Q2008-tab-delimited.zip",
        "PUDF-3Q2009-tab-delimited.zip",
        "PUDF-4Q2006-tab-delimited.zip",
        "PUDF-4Q2007-tab-delimited.zip",
        "PUDF-4Q2008-tab-delimited.zip",
        "PUDF-4Q2009-tab-delimited.zip",
    ]

    files_path = [os.path.join(data_folder, "TexasHospitals", f) for f in file_list]

    found = [os.path.exists(file_path) for file_path in files_path]
    not_found = [file_path for file_path in files_path if not os.path.exists(file_path)]

    processed_data_file = "texas_data10_rm_binary.csv"
    if not all(found):
        help_message = f"""
    Some or all data files do not exist. Please accept their terms & conditions,then download the
    tab delimited files from each quarter during 2006-2009 from:
    https://www.dshs.texas.gov/THCIC/Hospitals/Download.shtm
and place it in the correct folder.

    Missing files are:
    {not_found}
        """
        raise DataNotAvailable(help_message)

    if not os.path.exists(
        os.path.join(data_folder, "TexasHospitals", processed_data_file)
    ):

        logger.info("Processing Texas Hospitals data (2006-2009)")

        # Load data
        columns_names = [
            "THCIC_ID",  # Provider ID. Unique identifier assigned to the provider by DSHS.
            # Hospitals with fewer than 50 discharges have been aggregated into the
            # Provider ID '999999'
            "DISCHARGE_QTR",  # yyyyQm
            "TYPE_OF_ADMISSION",
            "SOURCE_OF_ADMISSION",
            "PAT_ZIP",  # Patient’s five-digit ZIP code
            "PUBLIC_HEALTH_REGION",  # Public Health Region of patient’s address
            "PAT_STATUS",  # Code indicating patient status as of the ending date of service for
            # the period of care reported
            "SEX_CODE",
            "RACE",
            "ETHNICITY",
            "LENGTH_OF_STAY",
            "PAT_AGE",  # Code indicating age of patient in days or years on date of discharge.
            "PRINC_DIAG_CODE",  # diagnosis code for the principal diagnosis
            "E_CODE_1",  # external cause of injury
            "PRINC_SURG_PROC_CODE",  # Code for the principal surgical or other procedure performed
            # during the period covered by the bill
            "RISK_MORTALITY",  # Assignment of a risk of mortality score from the All Patient
            # Refined (APR) Diagnosis Related Group (DRG)
            "ILLNESS_SEVERITY",  # Assignment of a severity of illness score from the All Patient
            # Refined (APR) Diagnosis RelatedGroup (DRG
            "RECORD_ID",
        ]
        # obtain the 100 most frequent procedures
        tmp = []
        for f in files_path:
            df = [
                pd.read_csv(
                    ZipFile(f).open(i), sep="\t", usecols=["PRINC_SURG_PROC_CODE"]
                )
                for i in ZipFile(f).namelist()
                if "base" in i
            ][0]
            df.dropna(inplace=True)
            tmp.extend(list(df.PRINC_SURG_PROC_CODE))
        princ_surg_proc_keep = [k for k, v in Counter(tmp).most_common(10)]
        # remove unnecessary variables
        del tmp

        # Load the data
        tx_data = pd.DataFrame()
        for f in files_path:
            df = [
                pd.read_csv(ZipFile(f).open(i), sep="\t", usecols=columns_names)
                for i in ZipFile(f).namelist()
                if "base" in i
            ][0]
            # keep only those rows with one of the 10 most common principal surgical procedure
            df = df[df["PRINC_SURG_PROC_CODE"].isin(princ_surg_proc_keep)]
            # clean up data
            df.dropna(inplace=True)
            df.replace("`", pd.NA, inplace=True)
            df.replace("*", pd.NA, inplace=True)
            # replace sex to numeric
            df.SEX_CODE.replace("M", 0, inplace=True)
            df.SEX_CODE.replace("F", 1, inplace=True)
            df.SEX_CODE.replace("U", 2, inplace=True)
            # set to numerical variable
            for d_code in set(list(df.DISCHARGE_QTR)):
                df.DISCHARGE_QTR.replace(
                    d_code, "".join(d_code.split("Q")), inplace=True
                )
            df.dropna(inplace=True)
            # merge data
            tx_data = pd.concat([tx_data, df])
        # remove uncessary variables
        del df

        # Risk moratality, make it binary
        # 1 Minor
        # 2 Moderate
        # 3 Major
        # 4 Extreme
        tx_data.RISK_MORTALITY.astype(int)
        tx_data.RISK_MORTALITY.replace(1, 0, inplace=True)
        tx_data.RISK_MORTALITY.replace(2, 0, inplace=True)
        tx_data.RISK_MORTALITY.replace(3, 1, inplace=True)
        tx_data.RISK_MORTALITY.replace(4, 1, inplace=True)

        # renumber non-numerical codes for cols
        cols = ["PRINC_DIAG_CODE", "SOURCE_OF_ADMISSION", "E_CODE_1"]
        for col in cols:
            tmp = list(
                {
                    x
                    for x in tx_data[col]
                    if not str(x).isdigit() and not isinstance(x, float)
                }  # pylint: disable=consider-using-set-comprehension
            )
            n = max(
                list(
                    {
                        int(x)
                        for x in tx_data[col]
                        if str(x).isdigit() or isinstance(x, float)
                    }  # pylint: disable=consider-using-set-comprehension
                )
            )
            for i, x in enumerate(tmp):
                tx_data[col].replace(x, n + i, inplace=True)
        del tmp, n
        # set index
        tx_data.set_index("RECORD_ID", inplace=True)
        # final check and drop of NAs
        tx_data.dropna(inplace=True)
        # convert all data to numerical
        tx_data = tx_data.astype(int)
        # save csv file
        tx_data.to_csv(os.path.join(data_folder, "TexasHospitals", processed_data_file))
    else:
        logger.info("Loading processed Texas Hospitals data (2006-2009) csv file.")
        # load texas data processed csv file
        tx_data = pd.read_csv(
            os.path.join(data_folder, "TexasHospitals", processed_data_file)
        )

    # extract target
    var = "RISK_MORTALITY"
    labels = tx_data[var]
    # Drop the column that contains the labels
    tx_data.drop([var], axis=1, inplace=True)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels.values)
    labels = pd.DataFrame({var: encoded_labels})

    return (tx_data, labels)
