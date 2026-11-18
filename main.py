"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

make.fun bounty gta6-twitch-ccu-forecast-01 (Python 3.13, standard library only).

Method
------
The score is the mean pinball loss on the log scale, so the forecast must give
honest quantiles of each hour's viewer count, not just a good median.

What the kit's comparable launches show (leave-one-out, 37 real launches in
the kit's practice cases):

* Hour 1 (00:00-01:00 UTC) is a weak anchor on its own.  For a rolling
  midnight unlock, the first hour is only New Zealand / Australia playing,
  so later hours are driven by the game's total demand and by the time of
  day, not by hour 1.  Across all 37 launches the correlation of
  log(hour-1 viewers) with log(viewers at hour 18) is -0.91 for the *ratio*,
  i.e. the peak is almost independent of hour 1.  A persistence forecast
  (the house baseline) therefore misses the day's shape badly: on
  rolling-midnight launches it scores about 0.36-0.42, and on the full
  mixed set about 0.93, because simultaneous-global-unlock games have an
  hour 1 near zero and then jump 100-1000x.
* Per hour, log(viewers) is well described by a one-predictor regression
  on log(hour-1 viewers): the slope falls from ~1.0 at hour 2 (the count
  simply persists) to ~0.1 at hours 18-24 (the count sits at the game's
  own peak level, almost independent of hour 1).  Fitting that regression
  per hour captures the diurnal ramp (US East unlocks at 05:00 UTC, US
  West at 08:00 UTC, peak in the UTC evening) without assuming any shape.
* The regression must be trained on the *right* launches.  GTA VI is a
  console-only game with a midnight-local rolling unlock and no PC
  version.  Training only on comparables with a rolling midnight unlock,
  hour-1 viewers >= 5,000 and not free-to-play (12 launches in the kit:
  ELDEN RING, Resident Evil Requiem, Hogwarts Legacy, Cyberpunk 2077,
  Ghost of Yotei, Monster Hunter Wilds, three Call of Duty titles,
  Zelda: Tears of the Kingdom, Final Fantasy VII Rebirth, The Last of
  Us Part II) beats the baseline by a wide margin.  Training on all 37
  launches is worse for this population, because simultaneous-unlock
  games (Black Myth: Wukong, Silksong, Battlefield 6, ...) have a
  near-zero hour 1 and a completely different shape.

Final forecast, per hour h = 2..24, in log space:

    log q_tau = 0.8 * log q_reg_tau + 0.2 * log q_base_tau

  q_reg:  exp(a_h + b_h * log(p1) + z_tau * sd_h), where (a_h, b_h) is the
          OLS fit of log(viewers at hour h) on log(hour-1 viewers) over
          the GTA-like comparables in *this* case file, sd_h is the
          residual standard deviation floored per hour band (0.35 for
          hours 2-6, 0.45 for hours 7-12, 0.5 for hours 13-24 — the
          floor guards against over-confident narrow bands from a
          12-launch fit, and early hours are genuinely tighter), and
          z_tau is the standard normal quantile
          (-1.645, -0.674, 0, 0.674, 1.645).
  q_base: the house persistence baseline, p1 * exp(z_tau * 0.15 *
          sqrt(h - 1)).  The 20% blend is a hedge: if hour 1 turns out
          to be an unusually informative anchor for GTA VI, the forecast
          still moves with it.

Leave-one-out validation (same code path as this file, scoring each
comparable launch against a training set that excludes it):

* GTA-like launches (n=12): pinball 0.233, house baseline 0.357.
* Console-only launches (n=8): pinball 0.211, house baseline 0.323.
* All 37 launches: pinball 0.62 -- worse than baseline on the
  simultaneous-unlock games, which are out of distribution for a
  rolling-midnight console launch like GTA VI.  The house baseline
  scores about 0.387 on the four practice cases (per the rules); this
  entry is designed to beat it on the final's population, not on
  simultaneous-unlock games.

The program computes everything from the case file it is given
(`comparables` in the case), so in the final it automatically trains on
all 42 comparables, including the four practice launches (Wolverine,
Red Dead Redemption 2, God of War Ragnarok, Spider-Man 2 -- all rolling
midnight console games).  It uses no network, no randomness, no clock
and no model call, and it is deterministic: same input, same bytes.

Fallbacks: if fewer than 6 GTA-like comparables are present, the
regression uses all comparables with hour-1 viewers >= 1,000; if fewer
than 6 of those, it uses every comparable with a usable release day;
if there are no usable comparables at all, it falls back to the house
persistence baseline alone.  The GTA VI category's own pre-release
history entry (named "... category ...", day_start 2026-08-27) is not a
launch and is always excluded.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
BASELINE_SIGMA = 0.15
REGRESSION_WEIGHT = 0.8
# Residual-SD floor, banded by hour: early hours are tightly predicted by
# hour 1 (the count simply persists), so a flat 0.5 floor over-widens
# them; late hours sit at the game's own peak level and need the full
# floor.  LOO on GTA-like launches: banded 0.2331 vs flat 0.5 at 0.2334.
SD_FLOOR_EARLY = 0.35  # hours 2-6
SD_FLOOR_MID = 0.45  # hours 7-12
SD_FLOOR_LATE = 0.5  # hours 13-24
MIN_TRAIN = 6


def _sd_floor(h):
    """Residual-SD floor for forecast hour h."""
    if h <= 6:
        return SD_FLOOR_EARLY
    if h <= 12:
        return SD_FLOOR_MID
    return SD_FLOOR_LATE


def round2(x):
    """x rounded to 2 decimals, half up (as in the starter)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def _release_day_viewers(comp):
    """Hourly viewers for a comparable's release UTC day, hours 1..24.

    Returns a list of 24 numbers/None, or None if the entry is not a
    usable launch (no day_start, or no hour-1 value)."""
    day_start = comp.get("day_start")
    hourly = comp.get("hourly")
    if not day_start or not hourly:
        return None
    day = day_start[:10]
    by_from = {}
    for row in hourly:
        try:
            by_from[row["from"]] = row["viewers"]
        except (KeyError, TypeError):
            continue
    vals = []
    for h in range(24):
        v = by_from.get("%sT%02d:00:00Z" % (day, h))
        vals.append(v if isinstance(v, (int, float)) else None)
    if vals[0] is None or vals[0] <= 0:
        return None
    return vals


def _is_gta_like(name, platforms, unlock, hour1):
    """Rolling midnight unlock, a real hour-1 audience, not free-to-play."""
    unlock_l = (unlock or "").lower()
    platforms_l = (platforms or "").lower()
    if "grand theft auto vi category" in (name or "").lower():
        return False
    if "free-to-play" in platforms_l:
        return False
    if not ("midnight local" in unlock_l or "rolling" in unlock_l):
        return False
    return hour1 >= 5000


def _training_set(case):
    """Comparable launches as (hour1, [hour1..hour24 viewers]) rows."""
    rows = []
    for comp in case.get("comparables") or []:
        vals = _release_day_viewers(comp)
        if vals is None:
            continue
        rows.append(
            {
                "name": comp.get("name", ""),
                "platforms": comp.get("platforms", ""),
                "unlock": comp.get("unlock", ""),
                "y": vals,
            }
        )
    gta_like = [r for r in rows if _is_gta_like(r["name"], r["platforms"], r["unlock"], r["y"][0])]
    if len(gta_like) >= MIN_TRAIN:
        return gta_like
    big = [r for r in rows if r["y"][0] >= 1000]
    if len(big) >= MIN_TRAIN:
        return big
    return rows


def _fit_hour(rows, h):
    """OLS of log(viewers at hour h) on log(hour-1 viewers).

    Returns (intercept, slope, residual_sd) or None if unusable."""
    pts = []
    for r in rows:
        y = r["y"][h - 1]
        if y is not None and y > 0:
            pts.append((math.log(max(r["y"][0], 1)), math.log(y)))
    n = len(pts)
    if n < 3:
        return None
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    vx = sum((x - mx) ** 2 for x, _ in pts) / n
    if vx <= 0:
        return (my, 0.0, 0.5)
    b = sum((x - mx) * (y - my) for x, y in pts) / n / vx
    a = my - b * mx
    sd = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in pts) / max(1, n - 2))
    return (a, b, max(sd, _sd_floor(h)))


def _baseline_quantiles(p1, h):
    return [p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1)) for z in Z]


def forecast_case(case, name=""):
    if case.get("levels") != LEVELS:
        raise ValueError("%s asks for levels %s, not %s" % (name, case.get("levels"), LEVELS))
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("%s has no hour-1 print" % name)
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)
    rows = _training_set(case)

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        base = _baseline_quantiles(p1, h)
        fit = _fit_hour(rows, h) if rows else None
        if fit is None:
            reg = base
        else:
            a, b, sd = fit
            reg = [math.exp(a + b * log_p1 + z * sd) for z in Z]
        qs = []
        prev = 0.0
        for i in range(5):
            q = math.exp(
                REGRESSION_WEIGHT * math.log(max(reg[i], 1e-9))
                + (1.0 - REGRESSION_WEIGHT) * math.log(max(base[i], 1e-9))
            )
            q = round2(q)
            q = min(max(q, 0.01), 1_000_000_000)
            if q < prev:
                q = prev
            prev = q
            qs.append(q)
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / ("%s.out" % path.stem)).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
