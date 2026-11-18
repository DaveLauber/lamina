"""The build as steps, and what each step's result depends on, so a change re-runs only the steps it reaches.

    model   load, size, turn, modify form, preview          (core.mesh keeps the last prepared model itself)
    slices  place slices and slots, section, check, auto-fix passes, coverage
    parts   split and cut the parts (ring cuts, oversize), orientation marks
    nest    nest on sheets, labels, identical parts

Each step's result is kept with the parameters it *read*, recorded as it reads them (`Tracked`) rather than listed
by hand: a technique that starts reading a new parameter is then covered without anyone remembering to add it. A
step is reused when every parameter it read still has the value it had, and the step before it is the very result
it was built on. A change therefore re-runs the first step that read it and every step after it — a new gap or font
re-nests (24 s of a 45 s head) instead of slicing the model again.
"""
from __future__ import annotations
import copy, itertools, threading
from collections import OrderedDict

STAGES = ("model", "slices", "parts", "nest")
_MISSING = object()
_tokens = itertools.count(1)


def _immutable(*types):
    """Shapely geometries never change once made: a copy of a kept step shares them instead of copying every
    polygon (most of what a step holds)."""
    for t in types:
        copy._deepcopy_dispatch[t] = lambda x, memo: x


def _register_shapely():
    import shapely.geometry as g
    _immutable(g.Point, g.MultiPoint, g.LineString, g.MultiLineString, g.LinearRing, g.Polygon, g.MultiPolygon, g.GeometryCollection)


_register_shapely()


def _same(a, b):
    try:
        return bool(a == b)
    except Exception:                     # an array compared element-wise: not worth a cache hit
        return False


class Tracked(dict):
    """The build's parameters, noting which ones the step under way reads (with the value it saw) and which it
    writes (auto-fix adds crossing slices to `extra_x` …). A key read after the step wrote it is the step's own
    value, not an input. Iterating it (`{**p}`) reads every key — that step then depends on all of them."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.on, self.reads, self.writes = False, {}, set()

    def begin(self):
        self.on, self.reads, self.writes = True, {}, set()

    def end(self):
        self.on = False
        return self.reads, {k: copy.deepcopy(dict.get(self, k)) for k in self.writes}

    def _read(self, k):
        if self.on and k not in self.writes and k not in self.reads:
            v = dict.get(self, k, _MISSING)
            self.reads[k] = v if v is _MISSING else copy.deepcopy(v)

    def __getitem__(self, k):
        self._read(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        self._read(k)
        return super().get(k, default)

    def __contains__(self, k):
        self._read(k)
        return super().__contains__(k)

    def _all(self):
        for k in list(dict.keys(self)):
            self._read(k)

    def __iter__(self):
        self._all()
        return super().__iter__()

    def keys(self):
        self._all()
        return super().keys()

    def items(self):
        self._all()
        return super().items()

    def values(self):
        self._all()
        return super().values()

    def __setitem__(self, k, v):
        if self.on:
            self.writes.add(k)
        super().__setitem__(k, v)

    def setdefault(self, k, v=None):
        if k not in self:
            self[k] = v
        return self[k]

    def update(self, *a, **kw):
        for k, v in dict(*a, **kw).items():
            self[k] = v

    def pop(self, k, *d):
        self._read(k)
        if self.on:
            self.writes.add(k)
        return super().pop(k, *d)

    def plain(self):
        return dict(dict.items(self))


class Entry:
    """A step's kept result: what it was built on (`upstream`: the step before it, or the model and technique for
    the first), the parameters it read and wrote, and its world (technique instance, slicing context, slices …)."""

    def __init__(self, upstream, reads, writes, world, refs):
        self.token = next(_tokens)
        self.upstream, self.reads, self.writes = upstream, reads, writes
        self.world, self.refs = world, refs      # refs: the build's mesh and params the kept world points at

    def fits(self, upstream, p):
        return (self.upstream == upstream
                and all(_same(dict.get(p, k, _MISSING), v) for k, v in self.reads.items()))


class StageCache:
    """One client's kept steps: the newest result of each. Not shared between clients — two people tuning two
    models would only push each other's steps out."""

    def __init__(self):
        self.entries: dict[str, Entry] = {}
        self.lock = threading.Lock()

    def get(self, stage, upstream, p):
        e = self.entries.get(stage)
        return e if e and e.fits(upstream, p) else None

    def put(self, stage, entry):
        with self.lock:
            self.entries[stage] = entry


def snapshot(world, mesh, p):
    """A copy of a step's result to keep: the model and the parameters are the build's own, not copied."""
    return copy.deepcopy(world, {id(mesh): mesh, id(p): p})


def restore(entry, mesh, p):
    """A kept result as this build's own: its copy, pointing at this build's model and parameters."""
    old_mesh, old_p = entry.refs
    return copy.deepcopy(entry.world, {id(old_mesh): mesh, id(old_p): p})


class Caches:
    """The kept steps of the last few clients (a server), least recently used dropped first."""

    def __init__(self, keep=4):
        self.keep, self.d, self.lock = keep, OrderedDict(), threading.Lock()

    def __call__(self, client) -> StageCache:
        with self.lock:
            c = self.d.pop(client, None) or StageCache()
            self.d[client] = c
            while len(self.d) > self.keep:
                self.d.popitem(last=False)
            return c

    def drop(self, client):
        with self.lock:
            self.d.pop(client, None)
