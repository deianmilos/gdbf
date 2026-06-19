"""
plot_parse.py — Parse simulation.res result files into ResultRow objects.

simulation.res columns (after header):
  alpha  NbEr(BER)  NbFer(FER)  Nbtested  IterAver(Itermax)
  SuccessIter(min/avg/max)   <- iteration stats for decoded frames
  FailedBits(min/avg/max)    <- residual bit errors in UNDECODABLE frames
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

_TRIPLET = r"(?:[0-9]*\.?[0-9]+|-)/(?:[0-9]*\.?[0-9]+|-)/(?:[0-9]*\.?[0-9]+|-)"
LINE_RE = re.compile(
    r"^\s*"
    r"(?P<alpha>[0-9]*\.?[0-9]+)\s+"
    r"(?P<nber>\d+)\s*\((?P<ber>[0-9eE+\-.]+)\)\s+"
    r"(?P<nbfer>\d+)\s*\((?P<fer>[0-9eE+\-.]+)\)\s+"
    r"(?P<nbtested>\d+)\s+"
    r"(?P<iter_avg>[0-9]*\.?[0-9]+)\((?P<iter_max>\d+)\)\s+"
    r"(?P<success_triplet>" + _TRIPLET + r")\s+"
    r"(?P<failed_triplet>" + _TRIPLET + r")"
)


@dataclass
class ResultRow:
    alpha: float
    ber: float
    fer: float
    iter_avg: float
    nbtested: int
    nbfer: int         # number of failed (undecodable) frames
    failed_min: float  # min residual bit errors in undecodable frames
    failed_avg: float  # avg residual bit errors in undecodable frames
    failed_max: float  # max residual bit errors in undecodable frames


def _parse_triplet(token: str) -> tuple[float, float, float]:
    parts = token.split("/")
    if len(parts) != 3:
        return (float("nan"), float("nan"), float("nan"))

    def _v(s: str) -> float:
        s = s.strip()
        if s == "-":
            return float("nan")
        try:
            return float(s)
        except ValueError:
            return float("nan")

    return (_v(parts[0]), _v(parts[1]), _v(parts[2]))


def parse_res_file(path: Path) -> List[ResultRow]:
    """
    Parse a simulation.res file and return one ResultRow per alpha value.
    When the same alpha appears multiple times (e.g. multiple runs appended),
    the *last* row for that alpha is kept.
    """
    rows_by_alpha: Dict[float, ResultRow] = {}

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.lower().startswith("alpha"):
                continue
            m = LINE_RE.match(line)
            if not m:
                continue

            alpha = float(m.group("alpha"))
            failed_min, failed_avg, failed_max = _parse_triplet(m.group("failed_triplet"))
            rows_by_alpha[alpha] = ResultRow(
                alpha=alpha,
                ber=float(m.group("ber")),
                fer=float(m.group("fer")),
                iter_avg=float(m.group("iter_avg")),
                nbtested=int(m.group("nbtested")),
                nbfer=int(m.group("nbfer")),
                failed_min=failed_min,
                failed_avg=failed_avg,
                failed_max=failed_max,
            )

    if not rows_by_alpha:
        raise ValueError(f"No valid rows parsed from: {path}")

    return [rows_by_alpha[a] for a in sorted(rows_by_alpha.keys(), reverse=True)]
