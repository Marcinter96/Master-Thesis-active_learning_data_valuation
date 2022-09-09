"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import pytest
from _pytest.fixtures import SubRequest
import numpy as np
from features.features_classes.logits_feature import LogitsFeature


@pytest.fixture
def feature(request: SubRequest):
    param = getattr(request, 'param', None)
    yield param()


@pytest.mark.parametrize('feature, expected',
                         [(LogitsFeature, None)],
                         indirect=['feature'])
def test_augment_return(feature, expected):
    assert(feature.augment(np.expand_dims(np.random.rand(10), axis=0)) == expected)
