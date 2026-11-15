"""Nesting into hollows, and pieces cut short: small parts go into the middle of a ring rather than onto a sheet of
their own, and a band or a ring slice longer than `max_length`, or too big for the sheet, is cut across its wall into
pieces joined by puzzle tabs."""
import math
import pathlib

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

import core.nest as nest
from core.model import Piece, Slice
from core.plan import build

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"
CUBE = EXAMPLES / "cube.stl"                        # 180 mm, centred
BANDS = {"autofix": "off", "thickness": 4, "nx": 3, "ny": 3, "nz": 3, "ring_wall": 12, "labels": False}
RINGS = {"autofix": "off", "distribution": "count", "count": 4, "thickness": 6, "connect": "none", "rings": True,
         "ring_wall": 12, "labels": False}


def _pieces(geoms):
    parent = Slice("P", "Z", None, 3.0)
    return [Piece(f"P-{i}", g, parent) for i, g in enumerate(geoms)]


def _polys(plan):
    return [(pc["place"][0], unary_union([Polygon(r[0], r[1:]) for r in pc["placed"]])) for s in plan["slices"] for pc in s["pieces"]]


def test_small_parts_go_into_the_hollow_of_a_ring():
    # a square ring filling a 400 mm sheet, and more small squares than the polygon nest takes: they all fit inside it
    ring = box(0, 0, 390, 390).difference(box(10, 10, 380, 380))
    small = [box(0, 0, 20, 20)] * (nest.MANY + 50)
    pcs = _pieces([ring, *small])
    assert nest.nest(pcs, (400, 400), 3, 5) == 1
    placed = [pc.placed for pc in pcs]
    for i in range(len(placed)):                                      # the gap kept, and the margin
        assert all(placed[i].distance(placed[k]) >= 3 - 1e-6 for k in range(i + 1, len(placed)))
        x0, y0, x1, y1 = placed[i].bounds
        assert x0 >= 5 - 1e-6 and y0 >= 5 - 1e-6 and x1 <= 395 + 1e-6 and y1 <= 395 + 1e-6


def test_a_ring_goes_into_a_bigger_one():
    big = box(0, 0, 380, 380).difference(box(10, 10, 370, 370))
    mid = box(0, 0, 340, 340).difference(box(10, 10, 330, 330))
    pcs = _pieces([big, mid])
    assert nest.nest(pcs, (400, 400), 3, 5) == 1
    assert pcs[0].placed.distance(pcs[1].placed) >= 3 - 1e-6


def test_bands_are_cut_to_the_longest_piece_asked_for():
    whole = build(CUBE, "bands", BANDS)
    short = build(CUBE, "bands", {**BANDS, "max_length": 200})
    assert short["counts"]["errors"] == 0 and short["counts"]["warnings"] == 0
    assert short["counts"]["parts"] > whole["counts"]["parts"]
    for sl in short["slices"]:
        assert len(sl["pieces"]) >= math.ceil(4 * 180 / 200)          # the outline is 720 mm round
        for pc in sl["pieces"]:
            g = unary_union([Polygon(r[0], r[1:]) for f in pc["facets"] for r in f["rings"]])
            x0, y0, x1, y1 = g.bounds
            # along a square ring, 200 mm of outline spans at most 200 mm, plus the wall and a puzzle tab
            assert math.hypot(x1 - x0, y1 - y0) <= 200 + 12 + 2 * 8
    # every cut has its tab: no piece is a plain stretch of the ring
    assert all(len(f["rings"][0][0]) > 40 for sl in short["slices"] for pc in sl["pieces"] for f in pc["facets"])


def test_a_band_too_big_for_the_sheet_is_cut_across_its_wall():
    plan = build(CUBE, "bands", {**BANDS, "z": False, "sheet": [170, 170]})
    assert plan["counts"]["errors"] == 0
    first = [s for s in plan["slices"] if s["label"].startswith("X")]
    assert first and all(len(s["pieces"]) >= 2 for s in first)        # placed first, so whole until the sheet says no
    for sheet, g in _polys(plan):
        x0, y0, x1, y1 = g.bounds
        assert x1 <= 170 and y1 <= 170


def test_ring_slices_are_cut_to_the_longest_piece_asked_for():
    plan = build(CUBE, "stacked", {**RINGS, "max_length": 250})
    assert plan["counts"]["errors"] == 0 and plan["counts"]["warnings"] == 0
    assert all(len(s["pieces"]) == math.ceil(720 / 250) for s in plan["slices"])
    whole = build(CUBE, "stacked", RINGS)
    assert all(len(s["pieces"]) == 1 for s in whole["slices"])        # 0: no limit, and a ring that fits stays whole


def test_a_ring_slice_too_big_for_the_sheet_is_cut_across_its_wall_not_in_strips():
    plan = build(CUBE, "stacked", {**RINGS, "sheet": [170, 170]})
    assert plan["counts"]["errors"] == 0
    for sl in plan["slices"]:
        area = sum(pc["area"] for pc in sl["pieces"])
        assert abs(area - (180 ** 2 - 156 ** 2)) < 300                 # still the ring: nothing added, nothing lost
        assert len(sl["pieces"]) >= 2
    for sheet, g in _polys(plan):
        x0, y0, x1, y1 = g.bounds
        assert x1 <= 170 and y1 <= 170
