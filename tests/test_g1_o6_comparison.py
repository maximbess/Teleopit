"""XML cache metadata may vary; physical scene attributes may not."""

from scripts.validate_g1_o6_consolidation import scene_structure


def test_scene_comparison_preserves_physical_attributes(tmp_path):
    path = tmp_path / "scene.xml"
    original = '<mujoco><asset><mesh name="a" file="a.obj" content_type="model/obj"/></asset><worldbody><geom friction="1.8 .1 .1"/></worldbody></mujoco>'
    path.write_text(original)
    reference = scene_structure(path)
    path.write_text(original.replace(' content_type="model/obj"', ""))
    assert scene_structure(path) == reference
    path.write_text(original.replace("1.8 .1 .1", "1.9 .1 .1"))
    assert scene_structure(path) != reference
