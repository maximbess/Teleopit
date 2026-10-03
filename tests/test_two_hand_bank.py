"""Portable geometry checks must tolerate compiler noise, not changed ladders."""
from types import SimpleNamespace

import numpy as np
import pytest

from train_mimic.data.two_hand_bank import (
    MODEL_FIELDS, model_signature, model_snapshot, validate_model,
)


def fixture_model():
    return SimpleNamespace(**{field: np.zeros((2, 3), dtype=(
        np.int32 if field.endswith(('id', 'type')) else np.float64))
        for field in MODEL_FIELDS})


def test_roundoff_and_signed_zero_are_portable():
    model = fixture_model()
    bank = model_snapshot(model)
    metadata = {'model_signature': model_signature(model)}
    model.geom_pos[0, 0] = 1e-10
    model.body_pos[0, 0] = -0.0
    assert model_signature(model) != metadata['model_signature']
    validate_model(model, bank, metadata)


@pytest.mark.parametrize('field,value', [('geom_pos', .001), ('jnt_range', .01),
                                        ('body_parentid', 1), ('site_pos', np.nan)])
def test_real_changes_are_rejected_with_field_and_index(field, value):
    model = fixture_model()
    bank = model_snapshot(model)
    metadata = {'model_signature': model_signature(model)}
    getattr(model, field)[0, 0] = value
    with pytest.raises(ValueError, match=field + ': max_delta=.*first_index='):
        validate_model(model, bank, metadata)


def test_shape_changes_and_legacy_bank_fail_closed():
    model = fixture_model()
    bank = model_snapshot(model)
    metadata = {'model_signature': model_signature(model)}
    validate_model(model, {}, metadata)
    model.geom_pos = np.zeros((3, 3))
    with pytest.raises(ValueError, match='geom_pos: shape'):
        validate_model(model, bank, metadata)
    with pytest.raises(ValueError, match='Legacy pose bank'):
        validate_model(model, {}, metadata)
