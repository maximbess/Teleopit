"""Kinematic compatibility check for portable measured poses and reference clips."""
import hashlib


def model_signature(model):
    digest = hashlib.sha256()
    for field in ('body_parentid','body_pos','body_quat','jnt_type','jnt_pos','jnt_axis',
                  'jnt_range','site_bodyid','site_pos','site_quat','geom_bodyid',
                  'geom_type','geom_pos','geom_quat','geom_size'):
        value = getattr(model,field)
        digest.update(field.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()
