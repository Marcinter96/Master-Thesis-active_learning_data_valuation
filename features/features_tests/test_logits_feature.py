"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import pytest
from _pytest.fixtures import SubRequest
from features.features_classes.logits_feature import LogitsFeature
import numpy as np
MIN_DELTA = 1e-15


@pytest.fixture
def logit_batch(request: SubRequest):
    batch_size, n_classes = getattr(request, 'param', None)
    yield np.random.rand(batch_size, n_classes)


@pytest.fixture
def feature(request: SubRequest):
    param = getattr(request, 'param', None)
    yield param()


@pytest.mark.parametrize('feature, logit_batch',
                         [(LogitsFeature, (50, 3)), (LogitsFeature, (50, 3))],
                         indirect=['feature', 'logit_batch'])
def test_augment_not_empty(feature, logit_batch):
    feature.augment(logit_batch)
    for f in feature.get_features():
        assert(len(f) != 0)


@pytest.mark.parametrize('feature, logit_batch, expected',
                         [(LogitsFeature, (50, 3), (50, 3)), (LogitsFeature, (50, 3), (50, 3))],
                         indirect=['feature', 'logit_batch'])
def test_batch_shape(feature, logit_batch, expected):
    feature.augment(logit_batch)
    assert(feature.get_logits()[0].shape == expected)
    for f in feature.get_features()[1:]:
        assert(f[0].shape == (expected[0],))
