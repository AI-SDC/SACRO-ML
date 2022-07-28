"""
Differentially private SVC
"""
import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from attack_utilities.estimator_template import GenericEstimator

local_logger = logging.getLogger(__file__)
local_logger.setLevel(logging.WARNING)


# pylint: disable = invalid-name


class DPSVC(GenericEstimator):
    ## Wrapper for differentially private SVM
    ##
    ## James Liley
    ## 21/03/22

    """
    Wrapper for differentially private SVM, implemented according to the method in

    https://arxiv.org/pdf/0911.5708.pdf

    Essentially approximates an infinite-dimensional latent space (and corresponding kernel) with
    a finite dimensional latent space, and adds noise to the normal to the separating hyperplane
    in this latent space.

    Only currently implemented for a radial basis kernel, but could be extended.

    More specifically
    - draws a set of dhat random vectors from a probability measure induced by the Fourier
        transform of the kernel function
    - approximates the kernel with a 2*dhat dimensional latent space
    - computes the separating hyperplane in this latent space with normal w
    - then adds Laplacian noise to w and returns it along with the map to the latent space.

    The SKlearn SVM (see https://scikit-learn.org/stable/modules/svm.html#mathematical-formulation)
    minimises the function

    (1/2) ||w||_2 + C sum(zeta_i)

    where 1-zeta_i≤ y_i (w phi(x_i) + b), where phi maps x to the latent space and zeta_i ≥ 0.
    This is equivalent to minimising

    (1/2) ||w||_2 + C/n sum(l(y_i,f_w(x_i)))

    where l(x,y)=n*max(0,1- x.y), which is n-Lipschitz continuous in y (given x is in {-1,1})

    """

    def __init__(self, C=1.0, gamma="scale", dhat=1000, eps=10, **kwargs):
        self.svc = None
        self.gamma = gamma
        self.dpsvc_gamma = None
        self.dhat = dhat
        self.eps = eps
        self.C = C
        self.lambdaval = None
        self.rho = None
        self.support = None
        self.platt_transform = LogisticRegression()
        self.b = None
        self.classes_ = [0, 1]
        self.intercept = None
        self.noisy_weights = None

    def phi_hat(self, input_vector):
        """Project a single feature"""
        vt1 = (self.rho * input_vector).sum(axis=1)
        vt = (self.dhat ** (-0.5)) * np.column_stack((np.cos(vt1), np.sin(vt1)))
        return vt.reshape(2 * self.dhat)

    def phi_hat_multi(self, input_features):
        """
        Compute feature space for a matrix of inputs
        """
        # TODO: could this be vectorised?
        n_data, _ = input_features.shape
        phi_hat = np.zeros((n_data, 2 * self.dhat), float)
        for i in range(n_data):
            phi_hat[i, :] = self.phi_hat(input_features[i, :])
        return phi_hat

    def k_hat_svm(self, x, y=None):
        """
        Define the version which is sent to sklearn.svm. AFAICT python/numpy
        doesn't have an 'outer' for arbitrary functions.
        """
        phi_hat_x = self.phi_hat_multi(x)
        if y is None:
            phi_hat_y = phi_hat_x
        else:
            phi_hat_y = self.phi_hat_multi(y)
        return np.dot(phi_hat_x, phi_hat_y.T)

    def fit(self, train_features: Any, train_labels: Any) -> None:
        """
        Fit the model
        """

        # Check that the data passed is np.ndarray
        if not isinstance(train_features, np.ndarray) or not isinstance(
            train_labels, np.ndarray
        ):
            raise NotImplementedError("DPSCV needs np.ndarray inputs")

        n_data, n_features = train_features.shape

        # Check the data passed in train_labels
        unique_labels = np.unique(train_labels)
        local_logger.info(unique_labels)
        for label in unique_labels:
            if label not in [0, 1]:
                raise NotImplementedError(
                    (
                        "DP SVC can only handle binary classification with",
                        "labels = 0 and 1",
                    )
                )

        if self.eps > 0:
            self.lambdaval = (2**2.5) * self.C * np.sqrt(self.dhat) / self.eps
        else:
            self.lambdaval = 0

        # Mimic sklearn skale and auto params
        if self.gamma == "scale":
            self.gamma = 1.0 / (n_features * train_features.var())
        elif self.gamma == "auto":
            self.gamma = 1.0 / n_features

        self.dpsvc_gamma = 1.0 / np.sqrt(
            2.0 * self.gamma
        )  # alternative parameterisation
        local_logger.info(
            "Gamma = %f (dp parameterisation = %f)", self.gamma, self.dpsvc_gamma
        )

        # Draw dhat random vectors rho from Fourier transform of RBF
        # (which is Gaussian with SD 1/gamma)
        self.rho = np.random.normal(0, 1.0 / self.dpsvc_gamma, (self.dhat, n_features))
        local_logger.info("Sampled rho")

        # Fit support vector machine
        # Create the gram matrix to pass to SVC
        gram_matrix = self.k_hat_svm(train_features)
        local_logger.info("Fitting base SVM")
        self.svc = SVC(kernel="precomputed", C=self.C)
        self.svc.fit(gram_matrix, train_labels)

        # Get separating hyperplane and intercept
        alpha = (
            self.svc.dual_coef_
        )  # alpha from solved dual, multiplied by labels (-1,1)
        xi = train_features[self.svc.support_, :]  # support vectors x_i
        weights = np.zeros(2 * self.dhat)
        for i in range(alpha.shape[1]):
            weights = weights + alpha[0, i] * self.phi_hat(xi[i, :])

        self.intercept = self.svc.intercept_

        # Add Laplacian noise
        self.noisy_weights = weights + np.random.laplace(
            0, self.lambdaval, len(weights)
        )

        # Logistic transform for predict_proba (rough): generate predictions (DP) for training data
        ypredn = np.zeros(n_data)
        for i in range(n_data):
            ypredn[i] = (
                np.dot(self.phi_hat(train_features[i, :]), self.noisy_weights)
                + self.intercept
            )

        local_logger.info("Fitting Platt scaling")
        self.platt_transform.fit(
            ypredn.reshape(-1, 1), train_labels
        )  # was called ptransform

    def set_params(self, **kwargs) -> None:
        """
        Set params
        """
        for key, value in kwargs.items():
            if key == "gamma":
                self.gamma = value
            elif key == "eps":
                self.eps = value
            elif key == "dhat":
                self.dhat = value
            else:
                local_logger.warning("Unsupported parameter: %s", key)

    def _raw_outputs(self, test_features: Any) -> np.ndarray:
        """
        Get the raw output, used by predict and predict_proba
        """
        projected_features = self.phi_hat_multi(test_features)
        out = np.dot(projected_features, self.noisy_weights) + self.intercept
        return out

    def predict(self, test_features: Any) -> np.ndarray:
        """
        Make predictions
        """
        out = self._raw_outputs(test_features)
        out = 1 * (out > 0)
        return out  # Predictions

    def predict_proba(self, test_features: Any) -> np.ndarray:
        """
        Predictive probabilities
        """
        out = self._raw_outputs(test_features)
        pred_probs = self.platt_transform.predict_proba(out.reshape(-1, 1))
        return pred_probs
