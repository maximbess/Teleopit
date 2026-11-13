"""Observe effective contacts, including friction below the geom mixing floor."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

SCENE = '''<mujoco><worldbody>
<geom name="left_ladder_rung_01" type="box" size="2 2 .1" friction="1.8 .02 .002" condim="4" solref=".005 1" solimp=".99 .999 .001" contype="1" conaffinity="0"/>
<body name="pelvis" pos="0 0 .19"><freejoint/>
 <geom name="foot" type="sphere" pos="-.5 0 0" size=".1" friction=".8 .001 .0001" contype="2" conaffinity="3"/>
 <body name="left_o6_finger"><geom name="left_finger" type="sphere" size=".1" friction=".8 .001 .0001" condim="4" solref=".006 1" contype="2" conaffinity="3"/></body>
 <body name="right_o6_finger"><geom name="right_finger" type="sphere" pos=".5 0 0" size=".1" friction=".8 .001 .0001" condim="4" contype="2" conaffinity="3"/>
 <geom name="visual" type="sphere" pos="0 .5 0" size=".1" contype="0" conaffinity="0"/>
 <geom name="masked" type="sphere" pos="0 -.5 0" size=".1" contype="4" conaffinity="4"/></body>
</body></worldbody></mujoco>'''


def contact_model(scope, mu):
    from teleopit.sim.g1_o6.material import Material, configure_material

    root = ET.fromstring(SCENE)
    before = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    configure_material(root, before, Material(scope, mu))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return before, model, data


@pytest.mark.parametrize("mu", [.4, .6, .8, 1.2, 1.8])
@pytest.mark.parametrize("scope", ["H", "M"])
def test_actual_contact_coefficients_and_unselected_properties(scope, mu):
    before, model, data = contact_model(scope, mu)
    original = mujoco.MjData(before)
    mujoco.mj_forward(before, original)
    originals = {frozenset((c.geom1, c.geom2)): c for c in original.contact}
    assert len(originals) == len(data.contact) == 3
    observed = set()
    for c in data.contact:
        names = {model.geom(g).name for g in (c.geom1, c.geom2)}
        changed = scope == "M" or any("finger" in n for n in names)
        expected = mu if changed else 1.8
        np.testing.assert_allclose(c.friction[:2], [expected, expected])
        old = originals[frozenset((c.geom1, c.geom2))]
        for field in ("solref", "solimp", "solreffriction", "includemargin", "dim"):
            np.testing.assert_allclose(getattr(c, field), getattr(old, field), err_msg=field)
        np.testing.assert_array_equal(c.friction[2:], old.friction[2:])
        observed.update(names)
    assert {"left_finger", "right_finger", "foot"} <= observed
    assert not {"visual", "masked"} & observed
    np.testing.assert_array_equal(model.geom_friction, before.geom_friction)


def test_material_boundary_rejects_ambiguous_or_invalid_values():
    from teleopit.sim.g1_o6.material import Material

    for scope, mu in (("global", .4), ("H", 0), ("M", float("nan"))):
        with pytest.raises(ValueError):
            Material(scope, mu)


def test_coupled_pairs_cover_all_front_and_rear_rungs_but_preserve_masks():
    from teleopit.sim.g1_o6.config import Config
    from teleopit.sim.g1_o6.model import build_model
    from teleopit.sim.g1_o6.material import Material
    from teleopit.sim.ladder_config import geometry_matrix

    model, _, _, audit = build_model(Config(), ladder=geometry_matrix()["G6"], material=Material("H", .4))
    pairs = audit["material"]["pairs"]
    assert {p["rung"] for p in pairs} == {
        f"{side}_ladder_rung_{i:02d}" for side in ("left", "right") for i in range(1, 9)}
    for p in pairs:
        a, b = (model.geom(p[k]).id for k in ("robot", "rung"))
        assert "_o6_" in model.body(model.geom_bodyid[a]).name
        assert model.geom_contype[a] & model.geom_conaffinity[b] or model.geom_contype[b] & model.geom_conaffinity[a]
    assert {p["robot"].split("_")[0] for p in pairs} == {"left", "right"}
