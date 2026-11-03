"""Everything that touches the CL1 for the trading experiment.

Input: the last 5 daily returns. Each day has an "up" channel and a "down"
channel; bigger moves get more pulses. Output: spikes on a BUY group and a
WAIT group of electrodes, plus the full spike-count vector for reservoir mode.
"""
import random
import numpy as np
import cl
from cl import ChannelSet, StimDesign, BurstDesign

LOOKBACK = 5
STIMMABLE = [c for c in range(1, 63) if c not in (4, 7, 56)]

UP_CHANNELS = STIMMABLE[0:LOOKBACK]                  # one per day, newest first
DOWN_CHANNELS = STIMMABLE[LOOKBACK:2 * LOOKBACK]
INPUT_CHANNELS = set(UP_CHANNELS + DOWN_CHANNELS)
_rest = [c for c in STIMMABLE if c not in INPUT_CHANNELS]
BUY_GROUP = _rest[0:8]
WAIT_GROUP = _rest[8:16]
FEEDBACK_CHANNELS = sorted(INPUT_CHANNELS | set(BUY_GROUP) | set(WAIT_GROUP))
# Reservoir features: spike counts on every channel we don't stimulate
# (stimulated channels mostly record stimulation artefacts).
FEATURE_CHANNELS = [c for c in range(64) if c not in INPUT_CHANNELS]

PULSE = StimDesign(160, -1.0, 160, 1.0)   # ask the hosts for safe settings
TICKS_PER_SECOND = 100
RESPONSE_TICKS = 30                        # 300 ms read-out
PCT_PER_PULSE = 0.005                      # one pulse per 0.5% move, max 5


def encode(returns):
    """returns: the last LOOKBACK daily returns, oldest first.
    Returns a list of (channel, pulse_count)."""
    stims = []
    for k, r in enumerate(reversed(list(returns))):   # k = 0 is the newest day
        pulses = min(5, max(1, round(abs(r) / PCT_PER_PULSE)))
        stims.append((UP_CHANNELS[k] if r >= 0 else DOWN_CHANNELS[k], pulses))
    return stims


def present(neurons, returns):
    """Stimulate with the encoded returns and return spike counts for all 64 channels."""
    stims = encode(returns)
    counts = np.zeros(64, dtype=int)
    for attempt in range(3):  # retry if the loop hiccups
        try:
            counts[:] = 0
            for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND, stop_after_ticks=RESPONSE_TICKS):
                if tick.iteration == 0:
                    for ch, n in stims:
                        neurons.stim(ChannelSet(ch), PULSE, BurstDesign(n, 50))
                if tick.iteration < 2:
                    continue  # skip stimulation artefacts
                for spike in tick.analysis.spikes:
                    counts[spike.channel] += 1
            return counts
        except TimeoutError:
            if attempt == 2:
                raise


def decide(counts):
    """Buy if the BUY electrodes fired more than the WAIT electrodes."""
    return counts[BUY_GROUP].sum() > counts[WAIT_GROUP].sum()


def give_feedback(neurons, good):
    try:
        _feedback(neurons, good)
    except TimeoutError:
        pass  # a missed feedback burst is better than a crashed run


def _feedback(neurons, good):
    """Same principle as the Pong study: predictable stimulation after a
    profitable trade, unpredictable noise after a losing one."""
    if good:
        for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND, stop_after_ticks=TICKS_PER_SECOND):
            if tick.iteration == 0:
                neurons.stim(ChannelSet(*FEEDBACK_CHANNELS), PULSE, BurstDesign(10, 100))
    else:
        for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND, stop_after_ticks=TICKS_PER_SECOND):
            if random.random() < 0.15:
                neurons.stim(ChannelSet(*random.sample(FEEDBACK_CHANNELS, 4)), PULSE)


def rest(neurons, seconds):
    try:
        for _ in neurons.loop(ticks_per_second=TICKS_PER_SECOND, stop_after_seconds=seconds):
            pass
    except TimeoutError:
        pass
