"""IEEE 13-bus and IEEE 34-bus radial test feeders.

Both feeders are reduced to balanced positive-sequence equivalents so they can be
solved by a single-phase backward/forward sweep (and by OpenDSS with an
equivalent circuit). The data is taken from the IEEE PES Distribution Test Feeder
reports:

* Line lengths (ft) and line configurations are the published values.
* Positive-sequence impedances are derived from the published phase-impedance
  matrices as Z1 = mean(Z_self) - mean(Z_mutual) (ohm/mile). Single-phase
  laterals use their self impedance.
* Spot loads and distributed loads are the published per-phase kW/kVAR summed
  over phases. Distributed loads are split half/half between the two end buses.
* Shunt capacitors (34-bus: 844, 848) are modelled as constant negative kVAR.
* Regulators are modelled as ideal fixed-tap voltage boosts (taps frozen at the
  published steady-state positions) so the post-fault voltage rise is not masked
  by tap action within the first seconds after the outage.

Everything is held in per-unit on a 1 MVA system base with a voltage base per
voltage zone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import networkx as nx
import numpy as np

FT_TO_KM = 0.0003048
MILE_TO_KM = 1.609344
S_BASE_MVA = 1.0

# Positive-sequence impedance per line configuration, ohm/mile (R1, X1).
IEEE13_LINE_CONFIGS = {
    "601": (0.1860, 0.5968),  # 3ph overhead, 556,500 26/7 ACSR
    "602": (0.5921, 0.7603),  # 3ph overhead, 4/0 6/1 ACSR
    "603": (1.1206, 0.8945),  # 2ph overhead (C-B), 1/0 ACSR
    "604": (1.1206, 0.8945),  # 2ph overhead (A-C), 1/0 ACSR
    "605": (1.3292, 1.3475),  # 1ph overhead (C), 1/0 ACSR
    "606": (0.4790, 0.4135),  # 3ph underground, 250 kcmil AA concentric neutral
    "607": (1.3425, 0.5124),  # 1ph underground (A), 1/0 AA tape shield
}

IEEE34_LINE_CONFIGS = {
    "300": (1.1203, 0.8330),  # 3ph overhead, 1/0 ACSR
    "301": (1.6920, 0.8510),  # 3ph overhead, #2 6/1 ACSR
    "302": (2.7995, 1.4855),  # 1ph overhead (A), #4 6/1 ACSR
    "303": (2.7995, 1.4855),  # 1ph overhead (B), #4 6/1 ACSR
    "304": (1.9217, 1.4212),  # 1ph overhead (B), #2 6/1 ACSR
}


@dataclass(frozen=True)
class Edge:
    """A switchable network section between parent bus ``u`` and child bus ``v``."""

    idx: int
    u: str
    v: str
    kind: str  # "line" | "transformer" | "regulator" | "switch"
    length_km: float
    r_pu: float
    x_pu: float
    tap: float = 1.0  # voltage ratio applied across the element (regulators)
    config: str = ""
    r_ohm_per_km: float = 0.0
    x_ohm_per_km: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.u}-{self.v}"

    @property
    def label(self) -> str:
        tag = {"line": "Line", "transformer": "Xfmr", "regulator": "Reg", "switch": "Switch"}[self.kind]
        return f"{tag} {self.u}→{self.v}"

    @property
    def patrol_km(self) -> float:
        """Length a crew must inspect; devices still cost a site visit."""
        return max(self.length_km, 0.05)


@dataclass
class Feeder:
    key: str
    name: str
    source: str
    buses: list[str]
    edges: list[Edge]
    load_kw: dict[str, float]
    load_kvar: dict[str, float]
    base_kv: dict[str, float]
    coords: dict[str, tuple[float, float]]
    source_pu: float
    trunk: set[str] = field(default_factory=set)  # edge names on the main trunk

    def __post_init__(self) -> None:
        self.bus_index = {b: i for i, b in enumerate(self.buses)}
        self.n_bus = len(self.buses)
        self.n_edge = len(self.edges)
        self.edge_by_name = {e.name: e for e in self.edges}
        self.parent_edge = {e.v: e for e in self.edges}
        self.children: dict[str, list[str]] = {b: [] for b in self.buses}
        for e in self.edges:
            self.children[e.u].append(e.v)

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.buses)
        for e in self.edges:
            self.graph.add_edge(e.u, e.v, idx=e.idx, length_km=e.length_km)
        if not nx.is_arborescence(self.graph):
            raise ValueError(f"{self.name} is not a radial tree rooted at {self.source}")

        # Breadth-first order from the source: parents always precede children.
        self.bfs_order = [self.source] + [v for _, v in nx.bfs_edges(self.graph, self.source)]
        self.depth = nx.shortest_path_length(self.graph, self.source)
        self.dist_km = nx.shortest_path_length(self.graph, self.source, weight="length_km")
        self.ancestors = {b: nx.shortest_path(self.graph, self.source, b)[:-1] for b in self.buses}

        # Downstream membership matrix D[e, i] = 1 if bus i is in the subtree cut off by edge e.
        self.downstream = np.zeros((self.n_edge, self.n_bus), dtype=bool)
        for e in self.edges:
            for b in nx.descendants(self.graph, e.v) | {e.v}:
                self.downstream[e.idx, self.bus_index[b]] = True

        self.p_kw = np.array([self.load_kw.get(b, 0.0) for b in self.buses])
        self.q_kvar = np.array([self.load_kvar.get(b, 0.0) for b in self.buses])
        self.edge_child_idx = np.array([self.bus_index[e.v] for e in self.edges])
        self.edge_parent_idx = np.array([self.bus_index[e.u] for e in self.edges])
        self.edge_len_km = np.array([e.length_km for e in self.edges])
        self.edge_patrol_km = np.array([e.patrol_km for e in self.edges])
        self.shed_kw = self.downstream.astype(float) @ self.p_kw
        self.total_km = float(self.edge_len_km.sum())

        undirected = self.graph.to_undirected()
        self._bus_hops = dict(nx.all_pairs_shortest_path_length(undirected))

    # ----------------------------------------------------------------- helpers
    def edge_hops(self, a: int, b: int) -> int:
        """Topological distance between two sections (hops between their child buses)."""
        return self._bus_hops[self.edges[a].v][self.edges[b].v]

    def edge_label(self, idx: int) -> str:
        e = self.edges[idx]
        role = "Main Trunk" if e.name in self.trunk else "Lateral"
        if e.kind != "line":
            role = e.kind.capitalize()
        return f"{e.label} ({role})"

    def dfs_edge_order(self) -> list[int]:
        """Order in which a conventional crew patrols sections walking out from the substation."""
        order: list[int] = []

        def walk(bus: str) -> None:
            # Visit the heavier (trunk) branch first, like a crew following the backbone.
            kids = sorted(self.children[bus], key=lambda c: -self.shed_kw[self.parent_edge[c].idx])
            for c in kids:
                order.append(self.parent_edge[c].idx)
                walk(c)

        walk(self.source)
        return order


def _line_pu(r_mi: float, x_mi: float, length_ft: float, kv: float) -> tuple[float, float, float, float, float]:
    length_km = length_ft * FT_TO_KM
    r_km, x_km = r_mi / MILE_TO_KM, x_mi / MILE_TO_KM
    z_base = kv**2 / S_BASE_MVA
    return length_km, r_km * length_km / z_base, x_km * length_km / z_base, r_km, x_km


def _xfmr_pu(r_pct: float, x_pct: float, kva: float) -> tuple[float, float]:
    scale = S_BASE_MVA / (kva / 1000.0)
    return r_pct / 100.0 * scale, x_pct / 100.0 * scale


def _assemble(key, name, source, source_pu, spec, configs, base_kv, spot, distributed, caps, coords, trunk):
    edges: list[Edge] = []
    for i, row in enumerate(spec):
        u, v, kind = row[0], row[1], row[2]
        if kind == "line":
            length_ft, cfg = row[3], row[4]
            length_km, r, x, r_km, x_km = _line_pu(*configs[cfg], length_ft, base_kv[u])
            edges.append(Edge(i, u, v, kind, length_km, r, x, 1.0, cfg, r_km, x_km))
        elif kind == "regulator":
            length_ft, cfg, tap = row[3], row[4], row[5]
            length_km, r, x, r_km, x_km = _line_pu(*configs[cfg], length_ft, base_kv[u])
            edges.append(Edge(i, u, v, kind, length_km, r, x, tap, cfg, r_km, x_km))
        elif kind == "transformer":
            r_pct, x_pct, kva = row[3], row[4], row[5]
            r, x = _xfmr_pu(r_pct, x_pct, kva)
            edges.append(Edge(i, u, v, kind, 0.0, r, x, 1.0, f"{kva:.0f} kVA"))
        elif kind == "switch":
            edges.append(Edge(i, u, v, kind, 0.0, 1e-5, 1e-5, 1.0, "switch"))
        else:
            raise ValueError(kind)

    load_kw: dict[str, float] = {}
    load_kvar: dict[str, float] = {}
    for bus, (p, q) in spot.items():
        load_kw[bus] = load_kw.get(bus, 0.0) + p
        load_kvar[bus] = load_kvar.get(bus, 0.0) + q
    for (a, b), (p, q) in distributed.items():
        for bus in (a, b):
            load_kw[bus] = load_kw.get(bus, 0.0) + p / 2
            load_kvar[bus] = load_kvar.get(bus, 0.0) + q / 2
    for bus, q in caps.items():
        load_kvar[bus] = load_kvar.get(bus, 0.0) - q

    buses = list(coords.keys())
    return Feeder(key, name, source, buses, edges, load_kw, load_kvar, base_kv, coords, source_pu, trunk)


@lru_cache(maxsize=None)
def ieee13() -> Feeder:
    kv = {b: 4.16 for b in ["650", "632", "633", "645", "646", "671", "680", "684", "611", "652", "692", "675"]}
    kv["634"] = 0.48
    spec = [
        ("650", "632", "line", 2000, "601"),
        ("632", "633", "line", 500, "602"),
        ("633", "634", "transformer", 1.1, 2.0, 500),
        ("632", "645", "line", 500, "603"),
        ("645", "646", "line", 300, "603"),
        ("632", "671", "line", 2000, "601"),
        ("671", "684", "line", 300, "604"),
        ("684", "611", "line", 300, "605"),
        ("684", "652", "line", 800, "607"),
        ("671", "680", "line", 1000, "601"),
        ("671", "692", "switch"),
        ("692", "675", "line", 500, "606"),
    ]
    spot = {
        "634": (400, 290),
        "645": (170, 125),
        "646": (230, 132),
        "652": (128, 86),
        "671": (1155, 660),
        "675": (843, 462),
        "692": (170, 151),
        "611": (170, 80),
    }
    distributed = {("632", "671"): (200, 116)}
    coords = {
        "650": (2.0, 4.0),
        "632": (2.0, 3.0),
        "633": (3.0, 3.0),
        "634": (4.0, 3.0),
        "645": (1.0, 3.0),
        "646": (0.0, 3.0),
        "671": (2.0, 2.0),
        "684": (1.0, 2.0),
        "611": (0.0, 2.0),
        "652": (1.0, 1.0),
        "680": (2.0, 1.0),
        "692": (3.0, 2.0),
        "675": (4.0, 2.0),
    }
    trunk = {"650-632", "632-671", "671-692", "692-675", "671-680"}
    # Source voltage reflects the substation regulator (RG60) boosting the 650 bus.
    return _assemble("ieee13", "IEEE 13-Bus Standard Feeder", "650", 1.0625, spec, IEEE13_LINE_CONFIGS,
                     kv, spot, distributed, {}, coords, trunk)


@lru_cache(maxsize=None)
def ieee34() -> Feeder:
    buses_hv = ["800", "802", "806", "808", "810", "812", "814", "850", "816", "818", "820", "822", "824", "826",
                "828", "830", "854", "856", "852", "832", "858", "864", "834", "842", "844", "846", "848", "860",
                "836", "840", "862", "838"]
    kv = {b: 24.9 for b in buses_hv}
    kv["888"] = kv["890"] = 4.16
    spec = [
        ("800", "802", "line", 2580, "300"),
        ("802", "806", "line", 1730, "300"),
        ("806", "808", "line", 32230, "300"),
        ("808", "810", "line", 5804, "303"),
        ("808", "812", "line", 37500, "300"),
        ("812", "814", "line", 29730, "300"),
        ("814", "850", "regulator", 10, "301", 1.075),
        ("850", "816", "line", 310, "301"),
        ("816", "818", "line", 1710, "302"),
        ("816", "824", "line", 10210, "301"),
        ("818", "820", "line", 48150, "302"),
        ("820", "822", "line", 13740, "302"),
        ("824", "826", "line", 3030, "303"),
        ("824", "828", "line", 840, "301"),
        ("828", "830", "line", 20440, "301"),
        ("830", "854", "line", 520, "301"),
        ("854", "856", "line", 23330, "303"),
        ("854", "852", "line", 36830, "301"),
        ("852", "832", "regulator", 10, "301", 1.08125),
        ("832", "888", "transformer", 1.9, 4.08, 500),
        ("888", "890", "line", 10560, "300"),
        ("832", "858", "line", 4900, "301"),
        ("858", "864", "line", 1620, "303"),
        ("858", "834", "line", 5830, "301"),
        ("834", "842", "line", 280, "301"),
        ("842", "844", "line", 1350, "301"),
        ("844", "846", "line", 3640, "301"),
        ("846", "848", "line", 530, "301"),
        ("834", "860", "line", 2020, "301"),
        ("860", "836", "line", 2680, "301"),
        ("836", "840", "line", 860, "301"),
        ("836", "862", "line", 280, "301"),
        ("862", "838", "line", 4860, "304"),
    ]
    spot = {
        "860": (60, 48),
        "840": (27, 21),
        "844": (405, 315),
        "848": (60, 48),
        "890": (450, 225),
        "830": (45, 20),
    }
    distributed = {
        ("802", "806"): (55, 29),
        ("808", "810"): (16, 8),
        ("818", "820"): (34, 17),
        ("820", "822"): (135, 70),
        ("816", "824"): (5, 2),
        ("824", "826"): (40, 20),
        ("824", "828"): (4, 2),
        ("828", "830"): (7, 3),
        ("854", "856"): (4, 2),
        ("832", "858"): (15, 7),
        ("858", "864"): (2, 1),
        ("858", "834"): (32, 17),
        ("834", "860"): (146, 73),
        ("860", "836"): (82, 43),
        ("836", "840"): (40, 20),
        ("862", "838"): (28, 14),
        ("842", "844"): (9, 5),
        ("844", "846"): (45, 23),
        ("846", "848"): (23, 11),
    }
    caps = {"844": 300, "848": 450}
    coords = {
        "800": (0.0, 6.0), "802": (1.0, 6.0), "806": (2.0, 6.0), "808": (3.2, 6.0), "810": (3.2, 7.2),
        "812": (4.6, 6.0), "814": (6.0, 6.0), "850": (7.0, 6.0), "816": (8.0, 6.0), "818": (8.0, 7.0),
        "820": (8.0, 8.0), "822": (8.0, 9.0), "824": (9.4, 6.0), "826": (9.4, 7.2), "828": (10.4, 6.0),
        "830": (11.8, 6.0), "854": (13.0, 6.0), "856": (13.0, 7.2), "852": (13.0, 4.6), "832": (13.0, 3.5),
        "888": (12.0, 2.4), "890": (11.0, 1.4), "858": (14.2, 3.5), "864": (14.2, 4.7), "834": (15.6, 3.5),
        "842": (16.6, 3.5), "844": (17.6, 3.5), "846": (18.6, 3.5), "848": (19.6, 3.5), "860": (15.6, 2.3),
        "836": (15.6, 1.2), "840": (16.8, 1.2), "862": (14.6, 1.2), "838": (14.6, 0.0),
    }
    trunk = {"800-802", "802-806", "806-808", "808-812", "812-814", "814-850", "850-816", "816-824", "824-828",
             "828-830", "830-854", "854-852", "852-832", "832-858", "858-834", "834-842", "842-844", "844-846",
             "846-848"}
    return _assemble("ieee34", "IEEE 34-Bus Distribution Feeder", "800", 1.05, spec, IEEE34_LINE_CONFIGS,
                     kv, spot, distributed, caps, coords, trunk)


FEEDERS = {
    "IEEE 13-Bus Standard Feeder": ieee13,
    "IEEE 34-Bus Distribution Feeder": ieee34,
}


def get_feeder(name: str) -> Feeder:
    return FEEDERS[name]()
