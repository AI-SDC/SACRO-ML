"""Test the data interface code."""

from __future__ import annotations

import unittest

import pandas as pd

from aisdc.preprocessing.loaders import UnknownDataset, get_data_sklearn


class TestLoaders(unittest.TestCase):
    """Test the data loaders."""

    def test_iris(self):
        """Nursery data."""
        feature_df, target_df = get_data_sklearn("iris")
        self.assertIsInstance(feature_df, pd.DataFrame)
        self.assertIsInstance(target_df, pd.DataFrame)

    def test_unknown(self):
        """Test that a nonsense string raises the correct exception."""
        with self.assertRaises(UnknownDataset):
            _, _ = get_data_sklearn("NONSENSE")

    def test_standard(self):
        """Test that standardisation creates standard features."""
        feature_df, _ = get_data_sklearn("standard iris")
        for column in feature_df.columns:
            temp = feature_df[column].mean()
            self.assertAlmostEqual(temp, 0.0)
            temp = feature_df[column].std()
            self.assertAlmostEqual(temp, 1.0)

    def test_minmax(self):
        """Test the minmax scaling."""
        feature_df, _ = get_data_sklearn("minmax iris")
        for column in feature_df.columns:
            temp = feature_df[column].min()
            self.assertAlmostEqual(temp, 0.0)
            temp = feature_df[column].max()
            self.assertAlmostEqual(temp, 1.0)
