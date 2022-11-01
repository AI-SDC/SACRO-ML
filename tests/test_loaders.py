"""test_loaders.py
Series of functions to use with pytest to check the loaders classes
Jim Smith 2022
"""

from preprocessing import loaders

datasets = (
    "mimic2-iaccd",
    "in-hospital-mortality",
    "medical-mnist-ab-v-br-100",
    "medical-mnist-ab-v-br-500",
    "medical-mnist-all-100",
    "indian liver",
    "texas hospitals 10",
    "synth-ae",
    "synth-ae-small",
    "nursery",
    "iris",
)

preprocessing = ("standard", "minmax", "round")


def test_get_sklearn_dataset():
    """Test ability to load some standard datasets
    These loaders only return binary versions
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
