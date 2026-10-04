"""The default training scene must use the pose bank's tracked collision hulls."""
import json
import hashlib

import numpy as np
from mjlab.scene import Scene
from teleopit.runtime.assets import PROJECT_ROOT
from teleopit.runtime.g1_collision_assets import PINNED_ASSET_DIR
from train_mimic.data.two_hand_bank import validate_model
from train_mimic.tasks.tracking.config.two_hand import make_two_hand_env_cfg


def test_pinned_hulls_are_complete_and_match_measured_bank():
    manifest = json.loads((PINNED_ASSET_DIR / 'manifest.json').read_text())
    assert len(manifest['parts']) == 180
    for part in manifest['parts']:
        assert hashlib.sha256((PINNED_ASSET_DIR / part['file']).read_bytes()).hexdigest() == part['sha256']
    cfg = make_two_hand_env_cfg()
    cfg.scene.num_envs = 1
    model = Scene(cfg.scene, device='cpu').compile()
    assert model.ngeom == 229
    with np.load(PROJECT_ROOT / 'assets/motions/ladder_two_hand_bank.npz', allow_pickle=False) as bank:
        validate_model(model, bank, json.loads(str(bank['metadata'])))
