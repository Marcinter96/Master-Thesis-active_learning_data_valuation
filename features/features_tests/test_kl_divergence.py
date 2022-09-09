"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import pytest
from _pytest.fixtures import SubRequest
from features.features_classes.kl_divergence import KLDivergence
import numpy as np
MIN_DELTA = 1e-15


@pytest.fixture
def feature(request: SubRequest):
    param = getattr(request, 'param', None)
    yield param[0](param[1])


@pytest.mark.parametrize('feature, logits',
                         [((KLDivergence, np.array([[0.1, 0.2, 0.7], [0.15, 0.2, 0.65]])),
                           np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.4, 0.1], [0.3, 0.3, 0.4]]).T)

                          ],
                         indirect=['feature'])
def test_augment_shapes(feature, logits):
    kl = feature.augment(logits)
    assert(kl.shape[0] == feature.mean_logits.shape[0])
    assert(kl.shape[1] == logits.shape[1])
