"""Kinematic compatibility check for portable measured poses and reference clips."""
import hashlib
import numpy as np


MODEL_FIELDS = ('body_parentid','body_pos','body_quat','jnt_type','jnt_pos','jnt_axis',
                'jnt_range','site_bodyid','site_pos','site_quat','geom_bodyid',
                'geom_type','geom_pos','geom_quat','geom_size')


def model_snapshot(model):
    return {'model_' + field: np.asarray(getattr(model, field)).copy()
            for field in MODEL_FIELDS}


def validate_model(model, bank, metadata):
    """Reject geometry changes, allowing only sub-micrometre numerical noise."""
    actual_signature = model_signature(model)
    if actual_signature == metadata['model_signature']:
        return
    missing = [field for field in MODEL_FIELDS if 'model_' + field not in bank]
    if missing:
        raise ValueError(
            'Legacy pose bank has no geometry snapshot; rebuild it for detailed validation. '
            f"Bank signature={metadata['model_signature']}, scene={actual_signature}")
    differences = []
    for field in MODEL_FIELDS:
        expected = np.asarray(bank['model_' + field])
        actual = np.asarray(getattr(model, field))
        if actual.shape != expected.shape:
            differences.append(f'{field}: shape bank={expected.shape}, scene={actual.shape}')
            continue
        if expected.dtype.kind in 'iu':
            close = actual == expected
        else:
            # Absolute tolerance in metres/radians (or unit quaternion components).
            close = np.isclose(actual, expected, rtol=0., atol=1e-7)
        if not np.all(close):
            index = tuple(np.argwhere(~close)[0])
            delta = np.max(np.abs(actual.astype(float) - expected.astype(float)))
            differences.append(f'{field}: max_delta={delta:.9g}, first_index={index}, '
                               f'bank={expected[index]}, scene={actual[index]}')
    if differences:
        raise ValueError('Pose-bank geometry mismatch (float atol=1e-7, topology exact):\n' +
                         '\n'.join(differences) +
                         '\nCheck repository/assets and MuJoCo versions; rebuild and validate '
                         'the bank if the geometry intentionally changed.')


def model_signature(model):
    digest = hashlib.sha256()
    for field in MODEL_FIELDS:
        value = getattr(model,field)
        digest.update(field.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()
