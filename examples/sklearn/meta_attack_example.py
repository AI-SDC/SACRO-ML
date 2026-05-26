"""Example: run a MetaAttack combining QMIA and structural attacks.

Trains a RandomForest on synthetic data, wraps it in a Target, then
runs MetaAttack to produce a cross-attack vulnerability DataFrame.

Usage::

    python examples/sklearn/meta_attack_example.py
"""

import logging

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sacroml.attacks.meta_attack import MetaAttack
from sacroml.attacks.target import Target

logging.basicConfig(level=logging.INFO)

output_dir = "output_meta_attack"

if __name__ == "__main__":
    # --- Prepare target ---
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    target = Target(
        model=model,
        dataset_name="synthetic",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_orig=X_train,
        y_train_orig=y_train,
        X_test_orig=X_test,
        y_test_orig=y_test,
    )
    for idx in range(X.shape[1]):
        target.add_feature(f"feature_{idx}", [idx], "float")

    # --- Run MetaAttack ---
    meta = MetaAttack(
        attacks=[
            ("qmia", {}, 2),  # QMIA with 2 repetitions
            ("structural", {}),  # Structural (single run)
        ],
        behaviour="run_all",  # alternatives: "use_existing_only", "fill_missing"
        mia_threshold=0.5,
        output_dir=output_dir,
    )
    meta.attack(target)

    # --- Inspect results ---
    df = meta.vulnerability_df

    print("\n=== Vulnerability Matrix (first 10 records) ===")
    print(df.head(10).to_string())

    print("\n=== Summary Statistics ===")
    n_train = int(df["is_member"].sum())
    n_test = len(df) - n_train
    print(f"Training records:  {n_train}")
    print(f"Test records:      {n_test}")

    # MIA vulnerability
    if "qmia_vuln" in df.columns:
        n_qmia = int(df["qmia_vuln"].sum())
        print(f"QMIA vulnerable:   {n_qmia}")

    # Structural vulnerability (training records only)
    if "struct_vuln" in df.columns:
        train_df = df[df["is_member"] == 1]
        n_struct = int(train_df["struct_vuln"].sum())
        print(f"Struct vulnerable:  {n_struct} (of {n_train} training)")

    # Records vulnerable to all attacks
    max_attacks = int(df["n_vulnerable"].max())
    n_all = int((df["n_vulnerable"] == max_attacks).sum())
    print(f"Vulnerable to all:  {n_all} (flagged by {max_attacks} attacks)")

    # Top-10 most vulnerable training records by MIA mean
    if "mia_mean" in df.columns:
        top10 = df[df["is_member"] == 1].nlargest(10, "mia_mean")[
            ["mia_mean", "mia_gmean", "n_vulnerable"]
        ]
        print("\n=== Top 10 Most Vulnerable Training Records ===")
        print(top10.to_string())

    print(f"\nReport saved to: {output_dir}/")
    print(f"CSV saved to:    {output_dir}/vulnerability_matrix.csv")
