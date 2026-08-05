#!/usr/bin/env python3
"""Генератор двух статических наклонных лестниц для MuJoCo.

Радиус непрерывных capsule-site зон задаётся аргументом --grip-radius.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


# Геометрия ранее созданной лестницы:
# высота 2.80 м, половина расстояния между нижними опорами 1.05 м,
# ширина 0.70 м и 9 перекладин.
PREVIOUS_HEIGHT = 2.80
PREVIOUS_HALF_BASE = 1.05
DEFAULT_NUM_RUNGS = 9
DEFAULT_ANGLE_DEG = math.degrees(
    2.0 * math.atan2(PREVIOUS_HALF_BASE, PREVIOUS_HEIGHT)
)
DEFAULT_WIDTH = 0.70
DEFAULT_RUNG_SPACING = (
    math.hypot(PREVIOUS_HALF_BASE, PREVIOUS_HEIGHT)
    / (DEFAULT_NUM_RUNGS + 1)
)

RAIL_RADIUS = 0.050
RUNG_RADIUS = 0.035
DEFAULT_GRIP_RADIUS = 0.075


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("значение должно быть больше нуля")
    return number


def ladder_angle(value: str) -> float:
    angle = float(value)
    if not 0.0 < angle < 180.0:
        raise argparse.ArgumentTypeError("угол должен находиться между 0 и 180°")
    return angle


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("количество должно быть не меньше 1")
    return number


def format_point(x: float, y: float, z: float) -> str:
    return f"{x:.6f} {y:.6f} {z:.6f}"


def capsule_geom(
    *,
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
) -> str:
    return (
        f'    <geom name="{name}"\n'
        f'          type="capsule"\n'
        f'          fromto="{format_point(*start)}  {format_point(*end)}"\n'
        f'          size="{radius:.3f}"/>'
    )


def capsule_site(
    *,
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
) -> str:
    """Создать одну непрерывную зону захвата вдоль всей перекладины."""
    return (
        f'    <site name="{name}"\n'
        f'          type="capsule"\n'
        f'          fromto="{format_point(*start)}  {format_point(*end)}"\n'
        f'          size="{radius:.3f}"\n'
        f'          group="3"\n'
        f'          rgba="0.15 0.95 0.25 0.25"/>'
    )


def generate_ladder_xml(
    *,
    angle_deg: float = DEFAULT_ANGLE_DEG,
    width: float = DEFAULT_WIDTH,
    rung_spacing: float = DEFAULT_RUNG_SPACING,
    num_rungs: int = DEFAULT_NUM_RUNGS,
    grip_radius: float = DEFAULT_GRIP_RADIUS,
) -> str:
    """Сформировать MJCF для статической А-образной пары лестниц.

    angle_deg — внутренний угол между лестницами в вершине.
    width — длина перекладины между осями боковых стоек, м.
    rung_spacing — расстояние между перекладинами вдоль стоек, м.
    num_rungs — количество перекладин на каждой лестнице.
    grip_radius — радиус непрерывной capsule-site зоны захвата, м.
    """
    if not 0.0 < angle_deg < 180.0:
        raise ValueError("angle_deg должен находиться между 0 и 180°")
    if width <= 0.0:
        raise ValueError("width должен быть больше нуля")
    if rung_spacing <= 0.0:
        raise ValueError("rung_spacing должен быть больше нуля")
    if num_rungs < 1:
        raise ValueError("num_rungs должен быть не меньше 1")
    if grip_radius <= 0.0:
        raise ValueError("grip_radius должен быть больше нуля")
    if grip_radius < RUNG_RADIUS:
        raise ValueError(
            f"grip_radius не должен быть меньше радиуса перекладины "
            f"({RUNG_RADIUS:.3f} м)"
        )

    half_angle = math.radians(angle_deg / 2.0)

    # Как и в исходной конструкции, от нижнего конца стоек до первой
    # перекладины и от последней перекладины до вершины оставлен один шаг.
    rail_length = (num_rungs + 1) * rung_spacing
    half_base = rail_length * math.sin(half_angle)
    height = rail_length * math.cos(half_angle)
    half_width = width / 2.0

    geoms: list[str] = []

    for side, base_x in (("left", -half_base), ("right", half_base)):
        geoms.append(f"    <!-- {side.capitalize()} inclined ladder. -->")

        for rail_index, y in enumerate((-half_width, half_width), start=1):
            geoms.append(
                capsule_geom(
                    name=f"{side}_ladder_rail_{rail_index}",
                    start=(base_x, y, 0.0),
                    end=(0.0, y, height),
                    radius=RAIL_RADIUS,
                )
            )

        for rung_index in range(1, num_rungs + 1):
            fraction = rung_index / (num_rungs + 1)
            x = base_x * (1.0 - fraction)
            z = height * fraction
            rung_start = (x, -half_width, z)
            rung_end = (x, half_width, z)
            rung_name = f"{side}_ladder_rung_{rung_index:02d}"

            geoms.append(
                capsule_geom(
                    name=rung_name,
                    start=rung_start,
                    end=rung_end,
                    radius=RUNG_RADIUS,
                )
            )
            geoms.append(
                capsule_site(
                    name=f"{rung_name}_grip",
                    start=rung_start,
                    end=rung_end,
                    radius=grip_radius,
                )
            )

    geom_block = "\n\n".join(geoms)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="a_frame_double_ladder">
  <compiler angle="degree"/>

  <default>
    <geom friction="1.2 0.01 0.001"
          condim="4"
          solref="0.005 1"
          solimp="0.95 0.99 0.001"
          rgba="0.55 0.55 0.58 1"/>
  </default>

  <worldbody>
{geom_block}
  </worldbody>
</mujoco>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Создаёт MJCF с двумя статическими наклонными лестницами, "
            "соприкасающимися в вершине."
        )
    )
    parser.add_argument(
        "--angle",
        type=ladder_angle,
        default=DEFAULT_ANGLE_DEG,
        help=(
            "угол между лестницами в градусах "
            f"(по умолчанию: {DEFAULT_ANGLE_DEG:.6f})"
        ),
    )
    parser.add_argument(
        "--width",
        type=positive_float,
        default=DEFAULT_WIDTH,
        help=f"ширина лестницы в метрах (по умолчанию: {DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "--rung-spacing",
        type=positive_float,
        default=DEFAULT_RUNG_SPACING,
        help=(
            "расстояние между перекладинами вдоль стоек в метрах "
            f"(по умолчанию: {DEFAULT_RUNG_SPACING:.6f})"
        ),
    )
    parser.add_argument(
        "--num-rungs",
        type=positive_int,
        default=DEFAULT_NUM_RUNGS,
        help=(
            "количество перекладин на каждой лестнице "
            f"(по умолчанию: {DEFAULT_NUM_RUNGS})"
        ),
    )
    parser.add_argument(
        "--grip-radius",
        type=positive_float,
        default=DEFAULT_GRIP_RADIUS,
        help=(
            "радиус непрерывной зоны захвата вокруг каждой перекладины "
            f"в метрах (по умолчанию: {DEFAULT_GRIP_RADIUS})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ladder.xml"),
        help="путь к создаваемому XML (по умолчанию: ladder.xml)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    xml = generate_ladder_xml(
        angle_deg=args.angle,
        width=args.width,
        rung_spacing=args.rung_spacing,
        num_rungs=args.num_rungs,
        grip_radius=args.grip_radius,
    )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml, encoding="utf-8")

    rail_length = (args.num_rungs + 1) * args.rung_spacing
    half_angle = math.radians(args.angle / 2.0)
    height = rail_length * math.cos(half_angle)
    base_distance = 2.0 * rail_length * math.sin(half_angle)

    print(f"Создан файл: {output_path}")
    print(f"Высота конструкции: {height:.3f} м")
    print(f"Расстояние между нижними опорами: {base_distance:.3f} м")
    print(
        "Зоны захвата: "
        f"{2 * args.num_rungs} capsule-site, радиус {args.grip_radius:.3f} м"
    )


if __name__ == "__main__":
    main()
