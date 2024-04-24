"""Test_loaders.py
Series of functions to use with pytest to check the loaders classes
Most use just the truncated files with first five examples of each class for brevity.
Please access the datasets from the sources listed in preprocessing/loaders.py
Please acknowledge those sources in any publications.
Jim Smith 2022.
"""

from __future__ import annotations

import os
import shutil

import pytest

from aisdc.preprocessing import loaders
from aisdc.preprocessing.loaders import DataNotAvailable, UnknownDataset

PROJECT_ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(PROJECT_ROOT_FOLDER, "datasets")

datasets = (
    "mimic2-iaccd",
    "in-hospital-mortality",
    "medical-mnist-ab-v-br-100",
    "medical-mnist-ab-v-br-500",
    "medical-mnist-all-100",
    "indian liver",
    "synth-ae",
    "synth-ae-small",
    "synth-ae-large",
    "synth-ae-extra-large",
    "synth-ae-XXL-large",
    "RDMP",
    "nursery",
    "iris",
)

preprocessing = ("standard", "minmax", "round")


def test_get_sklearn_dataset():
    """Test ability to load some standard datasets
    These loaders only return binary versions.
    """
    # test preprocessing with iris for speed
    # just gets first two classes
    for routine in preprocessing:
        x_df, y_df = loaders.get_data_sklearn(routine + " iris")
        assert x_df.shape == (100, 4)
        assert y_df.shape == (100, 1)

    # iris without pre-processing
    x_df, y_df = loaders.get_data_sklearn("iris")
    assert x_df.shape == (100, 4)
    assert y_df.shape == (100, 1)

    # nursery
    x_df, y_df = loaders.get_data_sklearn("nursery")
    assert x_df.shape == (12960, 27)
    assert y_df.shape == (12960, 1)


def test_data_absent():
    """Tests exceptions raised when datasets have not been downloaded."""
    # mimic2
    with pytest.raises(DataNotAvailable):
        _, _ = loaders.get_data_sklearn("mimic2-iaccd")

    # in house mortality
    with pytest.raises(DataNotAvailable):
        _, _ = loaders.get_data_sklearn("in-hospital-mortality")

    # mnist 100
    with pytest.raises(DataNotAvailable):
        _, _ = loaders.get_data_sklearn("medical-mnist-ab-v-br-100")

    # mnist 500
    with pytest.raises(DataNotAvailable):
        _, _ = loaders.get_data_sklearn("medical-mnist-ab-v-br-500")

    # mnist all 100
    with pytest.raises(DataNotAvailable):
        _, _ = loaders.get_data_sklearn("medical-mnist-all-100")

    # indian liver
    with pytest.raises(DataNotAvailable):
        _, _ = loaders.get_data_sklearn("indian liver")

    # synth-ae
    with pytest.raises(DataNotAvailable):
        _, _ = loaders.get_data_sklearn("synth-ae")

    # synthae-small
    with pytest.raises(DataNotAvailable):
        _, _ = loaders.get_data_sklearn("synth-ae-small")

    # synthae-large
    with pytest.raises(DataNotAvailable):
        _, _ = loaders.get_data_sklearn("synth-ae-large")

    # synthae-large
    with pytest.raises(DataNotAvailable):
        _, _ = loaders.get_data_sklearn("synth-ae-extra-large")

    # synthae-large
    with pytest.raises(DataNotAvailable):
        _, _ = loaders.get_data_sklearn("synth-ae-XXL")

    # RDMP
    with pytest.raises(DataNotAvailable):
        _, _ = loaders.get_data_sklearn("RDMP")

    # unknown
    with pytest.raises(UnknownDataset):
        _, _ = loaders.get_data_sklearn("no such dataset")


def test_mimic():
    """Load the mimic2 dataset."""
    # try:
    x_df, y_df = loaders.get_data_sklearn("mimic2-iaccd", DATA_FOLDER)
    assert x_df.shape == (1064, 38), f"x_df shape is {x_df.shape}"
    assert y_df.shape == (1064, 1)
    # except DataNotAvailable:
    #     pass


def test_in_hospital():
    """Tests loading the in hospital mortality data
    in two different ways.
    """

    zip_file_name = os.path.join(DATA_FOLDER, "doi_10.5061_dryad.0p2ngf1zd__v5.zip")
    new_file_name = os.path.join(DATA_FOLDER, "doi_10.5061_dryad.0p2ngf1zd__v5.renamed")
    # first attempt reads from zip file

    x_df, y_df = loaders.get_data_sklearn("in-hospital-mortality", DATA_FOLDER)
    assert x_df.shape == (428, 48), f"x_df shape is {x_df.shape}"
    assert y_df.shape == (428, 1)

    # now move zip file and make sure the loader reads properly from "data01.csv"
    os.rename(zip_file_name, new_file_name)
    x_df, y_df = loaders.get_data_sklearn("in-hospital-mortality", DATA_FOLDER)
    assert x_df.shape == (428, 48), f"x_df shape is {x_df.shape}"
    assert y_df.shape == (428, 1)

    # then put the zip file back for next time we run this test
    os.rename(new_file_name, zip_file_name)


def test_mnist():
    """Tests loading medical mnist data from different sources."""
    # mnist 100
    # this ne assumes the zip file is present
    x_df, y_df = loaders.get_data_sklearn("medical-mnist-ab-v-br-100", DATA_FOLDER)
    assert x_df.shape == (10, 4096), f"x_df shape is {x_df.shape}"
    assert y_df.shape == (10, 1)

    # mnist 500
    # now the archive file is present
    x_df, y_df = loaders.get_data_sklearn("medical-mnist-ab-v-br-500", DATA_FOLDER)
    assert x_df.shape == (10, 4096), f"x_df shape is {x_df.shape}"
    assert y_df.shape == (10, 1)

    # mnist all 100
    x_df, y_df = loaders.get_data_sklearn("medical-mnist-all-100", DATA_FOLDER)
    assert x_df.shape == (30, 4096), f"x_df shape is {x_df.shape}"
    assert y_df.shape == (30, 1)

    # test unflattend images
    imgdir = os.path.join(DATA_FOLDER, "kaggle-medical-mnist", "archive", "CXR")
    x_df, y_df = loaders._images_to_ndarray(  # pylint:disable=protected-access
        imgdir, 5, 0, flatten=False
    )
    assert x_df.shape == (5, 64, 64), f"x_df shape is {x_df.shape}"
    assert y_df.shape == (5, 1)

    # remove archive folder
    name = os.path.join(DATA_FOLDER, "kaggle-medical-mnist", "archive")
    if os.path.exists(name) and os.path.isdir(name):
        shutil.rmtree(name)


def test_indian_liver():
    """The indian liver dataloader."""
    x_df, y_df = loaders.get_data_sklearn("indian liver", DATA_FOLDER)
    assert x_df.shape == (11, 10), f"x_df shape is {x_df.shape}"
    assert y_df.shape == (11, 1)


def test_synth_ae():
    """Tests different versions of the  synthetic A&E dataset."""
    x_df, y_df = loaders.get_data_sklearn("synth-ae", DATA_FOLDER)
    assert x_df.shape == (8, 16), f"x_df shape is {x_df.shape}"
    assert y_df.shape == (8, 1)

    x_df, y_df = loaders.get_data_sklearn("synth-ae-small", DATA_FOLDER)
    assert x_df.shape == (8, 16), f"x_df shape is {x_df.shape}"
    assert y_df.shape == (8, 1)

    x_df, y_df = loaders.get_data_sklearn("synth-ae-large", DATA_FOLDER)
    assert x_df.shape == (8, 16), f"x_df shape is {x_df.shape}"
    assert y_df.shape == (8, 1)

    x_df, y_df = loaders.get_data_sklearn("synth-ae-extra-large", DATA_FOLDER)
    assert x_df.shape == (8, 16), f"x_df shape is {x_df.shape}"
    assert y_df.shape == (8, 1)

    x_df, y_df = loaders.get_data_sklearn("synth-ae-XXL", DATA_FOLDER)
    assert x_df.shape == (8, 16), f"x_df shape is {x_df.shape}"
    assert y_df.shape == (8, 1)


# def test_rdmp():
#     """The RDMP dataloader."""
#     x_df, y_df = loaders.get_data_sklearn("RDMP", DATA_FOLDER)

#     assert 'death' not in x_df.columns
#     assert 'age' in x_df.columns

#     assert y_df.shape[1] == 1
