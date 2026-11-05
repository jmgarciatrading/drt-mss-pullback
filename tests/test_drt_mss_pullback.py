"""Sequence spec for DRT_MSS_Pullback.pine.

A bullish DRT runs from the swing low before a swing high to that high.
The bullish MSS is the close through that high, and it completes the range.
A bearish DRT is the mirror: the swing high before a swing low, down to the
low the bearish MSS closes through.

The long then needs a sweep of the DRT high, a bearish engulfing, a later
trade to the EQ or below, a bearish order block there, and a buy on the next
candle. The short mirrors that. The Pine script follows this file.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass


@dataclass(frozen=True)
class Bar:
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Event:
    index: int
    kind: str
    direction: int = 0
    high: float | None = None
    low: float | None = None
    eq: float | None = None


PHASE_NONE = 0
PHASE_SWEEP = 1
PHASE_ENGULF = 2
PHASE_PULL = 3
PHASE_DONE = 4
PHASE_CANCEL = 5


def bearish_engulfing(previous: Bar, bar: Bar) -> bool:
    return (
        bar.close < bar.open
        and previous.close > previous.open
        and bar.open >= previous.close
        and bar.close <= previous.open
    )


def bullish_engulfing(previous: Bar, bar: Bar) -> bool:
    return (
        bar.close > bar.open
        and previous.close < previous.open
        and bar.open <= previous.close
        and bar.close >= previous.open
    )


def pivot_high(bars: list[Bar], index: int, left: int, right: int) -> float | None:
    # TradingView: an older bar at the same price does not remove the swing.
    # A newer bar at the same price does. The swing is the newest of the equals.
    pivot = index - right
    if pivot < left or index < left + right or index >= len(bars):
        return None
    price = bars[pivot].high
    for other in range(pivot - left, index + 1):
        if other == pivot:
            continue
        if bars[other].high > price or (other > pivot and bars[other].high == price):
            return None
    return price


def pivot_low(bars: list[Bar], index: int, left: int, right: int) -> float | None:
    pivot = index - right
    if pivot < left or index < left + right or index >= len(bars):
        return None
    price = bars[pivot].low
    for other in range(pivot - left, index + 1):
        if other == pivot:
            continue
        if bars[other].low < price or (other > pivot and bars[other].low == price):
            return None
    return price


class Engine:
    def __init__(self, left: int = 1, right: int = 1, max_bars: int = 80,
                 cancel_beyond: bool = True, body_only: bool = False):
        self.left = left
        self.right = right
        self.max_bars = max_bars
        self.cancel_beyond = cancel_beyond
        self.body_only = body_only
        self.structure = 0
        self.active_high = None
        self.active_low = None
        self.last_high = None
        self.last_high_bar = None
        self.last_low = None
        self.last_low_bar = None
        self.low_before_high = None
        self.high_before_low = None
        self.phase = PHASE_NONE
        self.drt_dir = 0
        self.drt_high = None
        self.drt_low = None
        self.drt_eq = None
        self.drt_bar = None
        self.took = False
        self.sweep_bar = None
        self.engulf_bar = None
        self.eq_seen = False
        self.ob_set = False
        self.ob_bar = None
        self.ob_top = None
        self.ob_bot = None

    def run(self, bars: list[Bar]) -> list[Event]:
        events: list[Event] = []
        for index, bar in enumerate(bars):
            self._step(bars, index, bar, events)
        return events

    def _install(self, direction: int, low: float, high: float, index: int,
                 events: list[Event]) -> None:
        self.structure = direction
        self.drt_dir = direction
        self.drt_low = low
        self.drt_high = high
        self.drt_eq = (high + low) / 2.0
        self.drt_bar = index
        self.phase = PHASE_SWEEP
        self.took = False
        self.sweep_bar = None
        self.engulf_bar = None
        self.eq_seen = False
        self.ob_set = False
        self.ob_bar = None
        self.ob_top = None
        self.ob_bot = None
        events.append(Event(index, "mss", direction, high, low, self.drt_eq))

    def _cancel(self, index: int, events: list[Event]) -> None:
        if self.phase in (PHASE_SWEEP, PHASE_ENGULF, PHASE_PULL):
            self.phase = PHASE_CANCEL
            events.append(Event(index, "cancel", self.drt_dir))

    def _break_high(self, index: int, bar: Bar, events: list[Event]) -> bool:
        if self.active_high is None or bar.close <= self.active_high:
            return False
        valid = (
            self.structure == -1
            and self.low_before_high is not None
            and self.low_before_high < self.active_high
        )
        if valid:
            self._install(1, self.low_before_high, self.active_high, index, events)
        else:
            if self.structure == -1:
                self._cancel(index, events)
            self.structure = 1
        self.active_high = None
        return valid

    def _break_low(self, index: int, bar: Bar, events: list[Event]) -> bool:
        if self.active_low is None or bar.close >= self.active_low:
            return False
        valid = (
            self.structure == 1
            and self.high_before_low is not None
            and self.high_before_low > self.active_low
        )
        if valid:
            self._install(-1, self.active_low, self.high_before_low, index, events)
        else:
            if self.structure == 1:
                self._cancel(index, events)
            self.structure = -1
        self.active_low = None
        return valid

    def _bounds(self, bar: Bar) -> tuple[float, float]:
        if self.body_only:
            return max(bar.open, bar.close), min(bar.open, bar.close)
        return bar.high, bar.low

    def _set_ob(self, index: int, bar: Bar, events: list[Event]) -> None:
        top, bottom = self._bounds(bar)
        self.ob_set = True
        self.ob_bar = index
        self.ob_top = top
        self.ob_bot = bottom
        events.append(Event(index, "ob", self.drt_dir, top, bottom, self.drt_eq))

    def _advance(self, bars: list[Bar], index: int, bar: Bar, events: list[Event]) -> None:
        if self.phase not in (PHASE_SWEEP, PHASE_ENGULF, PHASE_PULL):
            return
        if self.max_bars > 0 and index - self.drt_bar > self.max_bars:
            self._cancel(index, events)
            return
        if self.cancel_beyond and self.drt_dir == 1 and bar.close < self.drt_low:
            self._cancel(index, events)
            return
        if self.cancel_beyond and self.drt_dir == -1 and bar.close > self.drt_high:
            self._cancel(index, events)
            return

        if self.phase == PHASE_SWEEP and index > self.drt_bar:
            if self.drt_dir == 1:
                if bar.high > self.drt_high:
                    self.took = True
                if self.took and bar.close < self.drt_high:
                    self.phase = PHASE_ENGULF
                    self.sweep_bar = index
                    events.append(Event(index, "sweep", 1, self.drt_high, self.drt_low, self.drt_eq))
            else:
                if bar.low < self.drt_low:
                    self.took = True
                if self.took and bar.close > self.drt_low:
                    self.phase = PHASE_ENGULF
                    self.sweep_bar = index
                    events.append(Event(index, "sweep", -1, self.drt_high, self.drt_low, self.drt_eq))

        previous = bars[index - 1] if index else None
        if (
            self.phase == PHASE_ENGULF
            and previous is not None
            and self.sweep_bar is not None
            and index >= self.sweep_bar
        ):
            if self.drt_dir == 1 and bearish_engulfing(previous, bar):
                self.phase = PHASE_PULL
                self.engulf_bar = index
                events.append(Event(index, "engulf", 1))
            elif self.drt_dir == -1 and bullish_engulfing(previous, bar):
                self.phase = PHASE_PULL
                self.engulf_bar = index
                events.append(Event(index, "engulf", -1))

        if self.phase != PHASE_PULL or self.engulf_bar is None or index <= self.engulf_bar:
            return
        if self.ob_set and index > self.ob_bar + 1:
            self.ob_set = False

        if self.drt_dir == 1:
            if bar.low <= self.drt_eq:
                self.eq_seen = True
            if self.eq_seen and self.ob_set and index == self.ob_bar + 1:
                overlaps = bar.low <= self.ob_top and bar.high >= self.ob_bot
                if bar.close > bar.open and overlaps:
                    self.phase = PHASE_DONE
                    events.append(Event(index, "buy", 1, self.ob_top, self.ob_bot, self.drt_eq))
                elif bar.close < bar.open and bar.low <= self.drt_eq:
                    self._set_ob(index, bar, events)
                else:
                    self.ob_set = False
            elif self.eq_seen and not self.ob_set and bar.close < bar.open and bar.low <= self.drt_eq:
                self._set_ob(index, bar, events)
            return

        if bar.high >= self.drt_eq:
            self.eq_seen = True
        if self.eq_seen and self.ob_set and index == self.ob_bar + 1:
            overlaps = bar.low <= self.ob_top and bar.high >= self.ob_bot
            if bar.close < bar.open and overlaps:
                self.phase = PHASE_DONE
                events.append(Event(index, "sell", -1, self.ob_top, self.ob_bot, self.drt_eq))
            elif bar.close > bar.open and bar.high >= self.drt_eq:
                self._set_ob(index, bar, events)
            else:
                self.ob_set = False
        elif self.eq_seen and not self.ob_set and bar.close > bar.open and bar.high >= self.drt_eq:
            self._set_ob(index, bar, events)

    def _step(self, bars: list[Bar], index: int, bar: Bar, events: list[Event]) -> None:
        bull_break = self.active_high is not None and bar.close > self.active_high
        bear_break = self.active_low is not None and bar.close < self.active_low
        installed = False
        if bull_break and bear_break:
            pass
        elif bull_break:
            installed = self._break_high(index, bar, events)
        elif bear_break:
            installed = self._break_low(index, bar, events)

        high = pivot_high(bars, index, self.left, self.right)
        low = pivot_low(bars, index, self.left, self.right)
        if high is not None:
            high_bar = index - self.right
            self.last_high = high
            self.last_high_bar = high_bar
            if self.last_low_bar is not None and self.last_low_bar < high_bar:
                self.low_before_high = self.last_low
            self.active_high = high
        if low is not None:
            low_bar = index - self.right
            self.last_low = low
            self.last_low_bar = low_bar
            if self.last_high_bar is not None and self.last_high_bar < low_bar:
                self.high_before_low = self.last_high
            self.active_low = low

        if not installed:
            self._advance(bars, index, bar, events)


def kinds(events: list[Event], kind: str) -> list[Event]:
    return [event for event in events if event.kind == kind]


# Swing length 1. The bullish MSS is bar 11: DRT low 95, high 109, EQ 102.
# Sweep 12, bearish engulfing 13, order block 14, buy 15.
LONG_BARS = [
    Bar(105, 106, 104, 105),
    Bar(105, 106, 100, 101),
    Bar(101, 108, 101, 107),
    Bar(107, 112, 106, 111),
    Bar(111, 111, 109, 110),
    Bar(110, 111, 98, 99),
    Bar(99, 103, 99, 102),
    Bar(102, 103, 96, 96.5),
    Bar(96.5, 104, 95, 103),
    Bar(103, 109, 102, 108),
    Bar(108, 108, 103, 104),
    Bar(104, 111, 104, 110.5),
    Bar(107, 112, 106, 108),
    Bar(108.5, 109, 104, 105),
    Bar(105, 106, 100, 101),
    Bar(101, 107, 100.5, 106.5),
]


def mirror(bar: Bar, around: float = 200.0) -> Bar:
    return Bar(around - bar.open, around - bar.low, around - bar.high, around - bar.close)


class DrtSequenceTests(unittest.TestCase):
    def test_bullish_drt_runs_from_the_low_before_the_broken_high(self):
        events = Engine().run(LONG_BARS)
        mss = kinds(events, "mss")
        self.assertEqual(len(mss), 1, events)
        self.assertEqual(mss[0].index, 11)
        self.assertEqual(mss[0].direction, 1)
        self.assertEqual(mss[0].low, 95)
        self.assertEqual(mss[0].high, 109)
        self.assertEqual(mss[0].eq, 102)

    def test_long_signals_on_the_candle_after_the_order_block(self):
        events = Engine().run(LONG_BARS)
        self.assertEqual([event.index for event in kinds(events, "sweep")], [12])
        self.assertEqual([event.index for event in kinds(events, "engulf")], [13])
        self.assertEqual([event.index for event in kinds(events, "ob")], [14])
        buys = kinds(events, "buy")
        self.assertEqual([event.index for event in buys], [15])
        self.assertEqual(buys[0].high, 106)
        self.assertEqual(buys[0].low, 100)
        self.assertEqual(kinds(events, "sell"), [])
        self.assertEqual(kinds(events, "cancel"), [])

    def test_same_candle_can_sweep_and_engulf(self):
        bars = list(LONG_BARS[:12])
        bars.append(Bar(110.5, 113, 103, 104))
        bars.append(Bar(105, 106, 100, 101))
        bars.append(Bar(101, 107, 100.5, 106.5))
        events = Engine().run(bars)
        self.assertEqual([event.index for event in kinds(events, "sweep")], [12])
        self.assertEqual([event.index for event in kinds(events, "engulf")], [12])
        self.assertEqual([event.index for event in kinds(events, "ob")], [13])
        self.assertEqual([event.index for event in kinds(events, "buy")], [14])

    def test_mss_candle_does_not_count_as_the_sweep(self):
        bars = list(LONG_BARS)
        bars[12] = Bar(107, 108.5, 106, 108)
        events = Engine().run(bars)
        self.assertEqual(kinds(events, "sweep"), [])
        self.assertEqual(kinds(events, "buy"), [])

    def test_missing_engulfing_blocks_the_buy(self):
        bars = list(LONG_BARS)
        bars[13] = Bar(107.5, 108, 106, 106.5)
        events = Engine().run(bars)
        self.assertEqual(kinds(events, "engulf"), [])
        self.assertEqual(kinds(events, "buy"), [])

    def test_eq_must_be_traded_after_the_engulfing(self):
        bars = list(LONG_BARS)
        bars[14] = Bar(105, 106, 103, 104)
        bars[15] = Bar(104, 106, 103.5, 105.5)
        events = Engine().run(bars)
        self.assertEqual(kinds(events, "engulf"), [Event(13, "engulf", 1)])
        self.assertEqual(kinds(events, "ob"), [])
        self.assertEqual(kinds(events, "buy"), [])

    def test_order_block_candle_is_not_the_entry(self):
        events = Engine().run(LONG_BARS)
        self.assertNotIn(14, [event.index for event in kinds(events, "buy")])

    def test_a_bearish_candle_after_the_block_rolls_the_block_forward(self):
        bars = list(LONG_BARS)
        bars[15] = Bar(101, 102, 99, 99.5)
        bars.append(Bar(99.5, 106, 99, 105))
        events = Engine().run(bars)
        self.assertEqual([event.index for event in kinds(events, "ob")], [14, 15])
        self.assertEqual([event.index for event in kinds(events, "buy")], [16])

    def _close_through_far_side(self):
        # A newer swing low at 90 is confirmed before the close through 95, so
        # that close breaks the DRT without being a new MSS.
        bars = list(LONG_BARS[:14])
        bars.append(Bar(105, 106, 90, 96))
        bars.append(Bar(96, 97, 92, 95.5))
        bars.append(Bar(90, 97, 89, 94))
        return bars

    def test_close_through_the_far_side_cancels_the_long(self):
        events = Engine().run(self._close_through_far_side())
        self.assertEqual([event.index for event in kinds(events, "cancel")], [16], events)
        self.assertEqual(kinds(events, "buy"), [])

    def test_turning_the_far_side_rule_off_allows_the_long(self):
        events = Engine(cancel_beyond=False).run(self._close_through_far_side())
        self.assertEqual(kinds(events, "cancel"), [])
        self.assertEqual([event.index for event in kinds(events, "buy")], [16], events)

    def test_continuation_break_does_not_complete_a_second_drt(self):
        bars = list(LONG_BARS)
        # After the buy, build a swing high and close through it. Structure is
        # already bullish, so this is a BOS and the original DRT stands.
        bars.extend([
            Bar(106, 114, 105, 113),
            Bar(113, 113.5, 110, 111),
            Bar(111, 116, 110.5, 115.5),
        ])
        events = Engine().run(bars)
        self.assertEqual(len(kinds(events, "mss")), 1, events)
        self.assertEqual([event.index for event in kinds(events, "buy")], [15])

    def test_bearish_drt_is_the_opposite_and_sells_on_the_next_candle(self):
        events = Engine().run([mirror(bar) for bar in LONG_BARS])
        mss = kinds(events, "mss")
        self.assertEqual(len(mss), 1, events)
        self.assertEqual(mss[0].index, 11)
        self.assertEqual(mss[0].direction, -1)
        self.assertEqual(mss[0].high, 105)
        self.assertEqual(mss[0].low, 91)
        self.assertEqual(mss[0].eq, 98)
        self.assertEqual([event.index for event in kinds(events, "sweep")], [12])
        self.assertEqual([event.index for event in kinds(events, "engulf")], [13])
        self.assertEqual([event.index for event in kinds(events, "ob")], [14])
        sells = kinds(events, "sell")
        self.assertEqual([event.index for event in sells], [15])
        self.assertEqual(sells[0].high, 100)
        self.assertEqual(sells[0].low, 94)
        self.assertEqual(kinds(events, "buy"), [])

    def test_body_order_block_uses_the_open_and_close(self):
        events = Engine(body_only=True).run(LONG_BARS)
        order_block = kinds(events, "ob")
        self.assertEqual(order_block[0].high, 105)
        self.assertEqual(order_block[0].low, 101)
        self.assertEqual([event.index for event in kinds(events, "buy")], [15])

    def test_expired_setup_does_not_signal(self):
        events = Engine(max_bars=2).run(LONG_BARS)
        self.assertEqual([event.kind for event in events if event.kind in ("buy", "cancel")], ["cancel"])
        self.assertEqual(kinds(events, "cancel")[0].index, 14)


if __name__ == "__main__":
    unittest.main()
