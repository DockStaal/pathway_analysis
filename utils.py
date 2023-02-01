import numpy as np
import pandas as pd
import pymc3 as pm

from matplotlib import pyplot as plt

import logging
import time

import scipy as sp
import theano

# Enable on-the-fly graph computations, but ignore
# absence of intermediate test values.
theano.config.compute_test_value = "ignore"

# Set up logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class PMF:
    """Probabilistic Matrix Factorization model using pymc3."""

    def __init__(self, train, dim, alpha=2, std=0.01, bounds=(1, 5)):
        """Build the Probabilistic Matrix Factorization model using pymc3.

        :param np.ndarray train: The training data to use for learning the model.
        :param int dim: Dimensionality of the model; number of latent factors.
        :param int alpha: Fixed precision for the likelihood function.
        :param float std: Amount of noise to use for model initialization.
        :param (tuple of int) bounds: (lower, upper) bound of ratings.
            These bounds will simply be used to cap the estimates produced for R.

        """
        self.dim = dim
        self.alpha = alpha
        self.std = np.sqrt(1.0 / alpha)
        self.bounds = bounds
        self.data = train.copy()
        n, m = self.data.shape

        # Perform mean value imputation
        nan_mask = np.isnan(self.data)
        self.data[nan_mask] = self.data[~nan_mask].mean()

        # Low precision reflects uncertainty; prevents overfitting.
        # Set to the mean variance across users and items.
        self.alpha_u = 1 / self.data.var(axis=1).mean()
        self.alpha_v = 1 / self.data.var(axis=0).mean()

        # Specify the model.
        logging.info("building the PMF model")
        with pm.Model() as pmf:
            U = pm.MvNormal(
                "U",
                mu=0,
                tau=self.alpha_u * np.eye(dim),
                shape=(n, dim),
                testval=np.random.randn(n, dim) * std,
            )
            V = pm.MvNormal(
                "V",
                mu=0,
                tau=self.alpha_v * np.eye(dim),
                shape=(m, dim),
                testval=np.random.randn(m, dim) * std,
            )
            R = pm.Normal(
                "R", mu=(U @ V.T)[~nan_mask], tau=self.alpha, observed=self.data[~nan_mask]
            )

        logging.info("done building the PMF model")
        self.model = pmf

    def __str__(self):
        return self.name

def _find_map(self):
    """Find mode of posterior using L-BFGS-B optimization."""
    tstart = time.time()
    with self.model:
        logging.info("finding PMF MAP using L-BFGS-B optimization...")
        self._map = pm.find_MAP(method="L-BFGS-B")

    elapsed = int(time.time() - tstart)
    logging.info("found PMF MAP in %d seconds" % elapsed)
    return self._map

def _map(self):
    try:
        return self._map
    except:
        return self.find_map()

PMF.map = property(_map)
PMF.find_map = _find_map



def return_pathway_genes():
    unfilted_pathway_genes = """
        TRNT1
        RAN
        XPOT
        FAM98B
        RTRAF
        DDX1
        C2orf49
        RTCB
        ZBTB8OS
        ELAC2
        CLP1
        CPSF4
        CSTF2
        TSEN15
        CPSF1
        TSEN34
        TSEN2
        TSEN54
        RPP38
        POP4
        POP1
        POP7
        RPP40
        RPP25
        POP5
        RPP21
        RPP14
        RPP30
        NUP98
        SEH1L
        NUP107
        NUP43
        NUP160
        NUP37
        NUP85
        NUP133
        SEC13
        TPR
        NUP153
        NUP88
        NUP188
        NUP214
        NDC1
        NUP210
        POM121
        POM121C
        NUP35
        NUP93
        NUP155
        NUP205
        SEH1L
        RAE1
        NUP98
        NUP98
        NUP62
        NUP58
        NUP58
        NUP54
        NUP50
        AAAS
        RANBP2
    """
    # Make a list of the genes in the string
    pathway_genes = unfilted_pathway_genes.split()
    # Remove duplicates
    pathway_genes = list(set(pathway_genes))
    # Remove from the list:'C2orf49', 'NUP58', 'RTRAF', 'NUP42'
    pathway_genes = [x for x in pathway_genes if x not in ['C2orf49', 'NUP58', 'RTRAF', 'NUP42']]
    return pathway_genes

