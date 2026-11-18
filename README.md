[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/isnalestari09/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/isnalestari09/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewers — forecast entry

make.fun bounty: **Forecast GTA VI's launch-day Twitch viewers hour by hour**
(`gta6-twitch-ccu-forecast-01`) — $250, 20 places paid. Deadline
2026-11-19 08:00 WIB (01:00 UTC).

## What this is

`main.py` (Python 3.13, standard library only) reads one case
`/input/<id>.in` and writes `/output/<id>.out` with 23 rows (hours 2–24),
five quantiles each `[0.05, 0.25, 0.5, 0.75, 0.95]`, exactly as the
bounty specifies. It runs offline, deterministically, in ~0.1 s per case.

## Method (see the docstring in `main.py` for the full write-up)

1. **Hour 1 alone is a bad anchor.** For a rolling-midnight unlock,
   hour 1 (00:00–01:00 UTC) is only New Zealand/Australia playing, so
   the peak later in the day is almost independent of hour 1
   (correlation of log hour-1 with log ratio at hour 18 is −0.91).
   The house persistence baseline therefore misses the day's shape.
2. **Per-hour log regression.** For each hour h, fit
   `log(viewers_h) = a_h + b_h · log(hour-1 viewers)` over the
   comparable launches in the case file. The fitted slope falls from
   ~1.0 at hour 2 to ~0.1 at hours 18–24 — the count persists early,
   then sits at the game's own peak level. This captures the diurnal
   ramp (US East unlock 05:00 UTC, US West 08:00 UTC, peak UTC evening)
   with no assumed shape.
3. **Train on the right launches only.** GTA VI is console-only with a
   midnight-local rolling unlock. Training uses comparables with a
   rolling midnight unlock, hour-1 viewers ≥ 5,000, not free-to-play
   (12 launches in the kit). Simultaneous-global-unlock games
   (hour 1 ≈ 0, then a 100–1000× jump) are excluded — they are a
   different population.
4. **Quantiles + hedge.** Quantiles come from the regression's residual
   SD (floored at 0.5 log units, so a 12-launch fit can't be
   over-confident), blended 80/20 in log space with the house
   persistence baseline.

## Validation (leave-one-out, same code path as `main.py`)

Scoring each comparable launch against a training set that excludes it:

| Population | This entry | House baseline |
|---|---|---|
| GTA-like launches (n=12) | **0.233** | 0.357 |
| Console-only launches (n=8) | **0.211** | 0.323 |
| All 37 launches | 0.617 | 0.930 |

The house baseline scores ~0.387 on the four practice cases (per the
rules). This entry beats it by ~35% on the population GTA VI belongs
to. It is *worse* than the baseline would be on simultaneous-unlock
games — those are out of distribution for a rolling-midnight console
launch, and the final is not one of them. No claim is made about the
house AI-model forecast, whose score is unknown; the entry only needs
to beat the *lower* of the two house entries to be prize-eligible.

Caveats: the practice cases' true answers are not in the kit, so
validation is leave-one-out on the comparables, not the practice
leaderboard score. The final will train on all 42 comparables in the
final case file, including the four practice launches.

## Local test

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Verified: all 4 practice cases produce valid output (23 rows, hours
2–24, non-decreasing quantiles, 0 < q ≤ 1e9, ≤ 1,553 bytes each),
identical bytes on two runs, ~0.3 s for all 4 cases, single-case run
works, and edge cases (no comparables → baseline fallback; hour 1 =
1 / 100 / 1,000,000) do not crash.

## How to submit (NOT done here — needs the account owner)

1. Create a public GitHub repo owned by your own GitHub account,
   created after 2026-11-17 06:00 UTC, OSI licence (MIT `LICENSE`
   included), with this `main.py` at the root.
2. From the kit directory, follow the make.fun skill (`make.fun/skill.md`):
   fetch the kit, use its CLI to bundle the repo with the AI session
   transcripts that built it (`bundle`), and submit before
   2026-11-19 01:00 UTC. Max 10 submissions/day, 20 total; the last
   submission before the deadline is the final entry.
3. The $250 is locked in escrow on Solana, so a winning entry is
   claimed in USDC to a Solana wallet linked to the entrant's account.
   The transcript gate applies: sessions must be recorded/handed over
   as the skill describes, or the entry is ranked but takes no paid
   place.
