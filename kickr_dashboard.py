#!/usr/bin/env python3
"""
KICKR workout runner with a live terminal dashboard (Textual).

Runs a pre-programmed workout on a Wahoo KICKR (BLE FTMS), records heart rate
from a BLE HR source (e.g. a Garmin watch in broadcast mode), shows live
graphs, and logs everything to CSV.

Usage:
    python kickr_dashboard.py sample_workout.json
    python kickr_dashboard.py sample_workout.json --ftp 230 --hr-name Instinct
    python kickr_dashboard.py sample_workout.json --no-hr
    python kickr_dashboard.py sample_workout.json --demo      # no hardware needed

Workout file: each step has ONE length ('seconds', 'meters', 'km' or 'miles') and
at least one of 'watts', 'pct_ftp', 'grade', 'resistance'. A step with a power
target ('watts'/'pct_ftp') may also carry 'grade' and/or 'resistance'; for now
those are only shown in the Live panel (they are not sent to the trainer).

Keys:  n = next step    r = restart step (or previous step if <2 s in)
       p = pause/play   q (or Ctrl+Q) = stop workout and quit

Requires: pip install bleak textual textual-plotext      (Python 3.10+)
Close Zwift / the Wahoo app / anything else connected to the trainer first.
"""

import argparse
import asyncio
import csv
import json
import random
import struct
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bleak import BleakClient, BleakScanner
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, ProgressBar, RichLog, Static
from textual_plotext import PlotextPlot


def uuid16(n: int) -> str:
    """Expand a 16-bit Bluetooth UUID to its full 128-bit string form."""
    return f"0000{n:04x}-0000-1000-8000-00805f9b34fb"


HR_SERVICE = uuid16(0x180D)
HR_MEASUREMENT = uuid16(0x2A37)
CONTROL_POINT = uuid16(0x2AD9)   # FTMS Fitness Machine Control Point
BIKE_DATA = uuid16(0x2AD2)       # FTMS Indoor Bike Data

OP_REQUEST_CONTROL = 0x00
OP_SET_TARGET_RESISTANCE = 0x04
OP_SET_TARGET_POWER = 0x05
OP_START = 0x07
OP_STOP = 0x08
OP_SET_SIMULATION = 0x11
RESULT_SUCCESS = 0x01


# --------------------------------------------------------------------------
# Workout file handling
# --------------------------------------------------------------------------

@dataclass
class Step:
    name: str
    seconds: int | None = None         # time-based length, or...
    meters: float | None = None        # ...distance-based length
    watts: int | None = None           # ERG target power
    grade: float | None = None         # % grade
    resistance: float | None = None    # FTMS resistance level

    @property
    def is_distance(self) -> bool:
        return self.meters is not None

    @property
    def length(self) -> float:
        """Step length in its own unit: seconds, or meters for distance steps."""
        return self.meters if self.is_distance else self.seconds

    @property
    def length_str(self) -> str:
        return dist_str(self.meters) if self.is_distance else mmss(self.seconds)

    @property
    def target(self) -> str:
        """What is commanded to the trainer for this step."""
        if self.watts is not None:
            return f"{self.watts} W"
        if self.grade is not None:
            return f"{self.grade:g}% grade"
        return f"resistance {self.resistance:g}"


LENGTH_UNITS = {"seconds": None, "meters": 1.0, "km": 1000.0, "miles": 1609.344}


def load_workout(path: str, ftp_override: int | None) -> tuple[str, list[Step]]:
    """Read a workout JSON file.

    Each step needs exactly one length key ('seconds', 'meters', 'km' or
    'miles') and at least one of 'watts', 'pct_ftp' (relative to the FTP in the
    file/CLI), 'grade' or 'resistance'. Power can be combined with grade and/or
    resistance; a step without power is driven by its grade (else resistance).
    """
    spec = json.loads(Path(path).read_text())
    ftp = ftp_override or spec.get("ftp")
    steps = []
    for i, s in enumerate(spec["steps"], start=1):
        name = s.get("name", f"Step {i}")

        lengths = [k for k in LENGTH_UNITS if k in s]
        if len(lengths) != 1:
            raise SystemExit(f"Step {i} needs exactly one of 'seconds', 'meters', "
                             "'km' or 'miles'.")
        key = lengths[0]
        amount = float(s[key])
        if amount <= 0:
            raise SystemExit(f"Step {i}: '{key}' must be greater than zero.")
        length = ({"seconds": round(amount)} if key == "seconds"
                  else {"meters": amount * LENGTH_UNITS[key]})

        if "watts" in s and "pct_ftp" in s:
            raise SystemExit(f"Step {i} has both 'watts' and 'pct_ftp'; use one.")
        watts = None
        if "watts" in s:
            watts = int(s["watts"])
        elif "pct_ftp" in s:
            if not ftp:
                raise SystemExit(f"Step {i} uses pct_ftp but no FTP is set "
                                 "(add 'ftp' to the file or pass --ftp).")
            watts = round(ftp * s["pct_ftp"] / 100)
        grade = float(s["grade"]) if "grade" in s else None
        resistance = float(s["resistance"]) if "resistance" in s else None
        if watts is None and grade is None and resistance is None:
            raise SystemExit(f"Step {i} needs 'watts', 'pct_ftp', 'grade' or 'resistance'.")

        steps.append(Step(name, watts=watts, grade=grade, resistance=resistance, **length))
    return spec.get("name", Path(path).stem), steps


# --------------------------------------------------------------------------
# BLE data parsing
# --------------------------------------------------------------------------

def parse_bike_data(data: bytearray) -> dict:
    """Parse the FTMS Indoor Bike Data characteristic (speed, cadence, power)."""
    flags = int.from_bytes(data[0:2], "little")
    pos = 2
    out = {}

    def take(n: int, signed: bool = False) -> int:
        nonlocal pos
        value = int.from_bytes(data[pos:pos + n], "little", signed=signed)
        pos += n
        return value

    if not flags & 0x0001:        # bit 0 is inverted: 0 means speed IS present
        out["speed_kmh"] = take(2) * 0.01
    if flags & 0x0002:            # average speed
        take(2)
    if flags & 0x0004:            # instantaneous cadence (0.5 rpm units)
        out["cadence"] = take(2) * 0.5
    if flags & 0x0008:            # average cadence
        take(2)
    if flags & 0x0010:            # total distance (3 bytes)
        take(3)
    if flags & 0x0020:            # resistance level
        take(2, signed=True)
    if flags & 0x0040:            # instantaneous power (watts)
        out["power"] = take(2, signed=True)
    return out


def parse_heart_rate(data: bytearray) -> int:
    """Parse the standard Heart Rate Measurement characteristic."""
    if data[0] & 0x01:            # bit 0 set: HR is a uint16
        return int.from_bytes(data[1:3], "little")
    return data[1]                # otherwise a uint8


# --------------------------------------------------------------------------
# Trainer wrappers
# --------------------------------------------------------------------------

class Trainer:
    """Thin wrapper around the FTMS control point.

    Commands are written to the control point; the trainer answers with an
    *indication* ([0x80, request_opcode, result_code]). We turn each answer
    into an awaitable Future so `await command(...)` returns when it arrives.
    """

    def __init__(self, client: BleakClient, live: dict):
        self.client = client
        self.live = live
        self._pending: dict[int, asyncio.Future] = {}

    async def start_notifications(self) -> None:
        await self.client.start_notify(CONTROL_POINT, self._on_control_point)
        await self.client.start_notify(BIKE_DATA, self._on_bike_data)

    def _on_bike_data(self, _, data: bytearray) -> None:
        self.live.update(parse_bike_data(data))

    def _on_control_point(self, _, data: bytearray) -> None:
        if len(data) >= 3 and data[0] == 0x80:
            fut = self._pending.pop(data[1], None)
            if fut and not fut.done():
                fut.set_result(data[2])

    async def command(self, opcode: int, params: bytes = b"") -> None:
        fut = asyncio.get_running_loop().create_future()
        self._pending[opcode] = fut
        await self.client.write_gatt_char(
            CONTROL_POINT, bytes([opcode]) + params, response=True)
        result = await asyncio.wait_for(fut, timeout=5.0)
        if result != RESULT_SUCCESS:
            raise RuntimeError(
                f"Trainer rejected opcode 0x{opcode:02x} (result code {result})")

    async def take_control(self) -> None:
        await self.command(OP_REQUEST_CONTROL)
        await self.command(OP_START)

    async def set_power(self, watts: int) -> None:
        await self.command(OP_SET_TARGET_POWER, struct.pack("<h", watts))

    async def set_resistance_level(self, level: float) -> None:
        """FTMS target resistance (uint8, 0.1 units). Support varies by trainer."""
        await self.command(OP_SET_TARGET_RESISTANCE, struct.pack("<B", round(level * 10)))

    async def set_simulation(self, grade_pct: float, crr: float = 0.004,
                             cw: float = 0.51, wind_mps: float = 0.0) -> None:
        """Simulate a road grade (crr and cw are the usual FTMS defaults)."""
        params = struct.pack("<hhBB",
                             round(wind_mps * 1000),   # sint16, 0.001 m/s
                             round(grade_pct * 100),   # sint16, 0.01 %
                             round(crr * 10000),       # uint8, 0.0001
                             round(cw * 100))          # uint8, 0.01 kg/m
        await self.command(OP_SET_SIMULATION, params)

    async def apply_step(self, step: Step) -> None:
        if step.watts is not None:
            await self.set_power(step.watts)
        elif step.grade is not None:
            await self.set_simulation(step.grade)
        else:
            await self.set_resistance_level(step.resistance)

    async def stop(self) -> None:
        await self.command(OP_STOP, b"\x01")   # 0x01 = stop (0x02 = pause)

    async def pause(self) -> None:
        await self.command(OP_STOP, b"\x02")   # FTMS "pause": resistance is released

    async def resume(self) -> None:
        await self.command(OP_START)            # Start/Resume (the caller re-applies the step)


class FakeTrainer:
    """Stand-in used by --demo: same interface as Trainer, no Bluetooth."""

    def __init__(self):
        self.step: Step | None = None
        self.paused = False

    async def take_control(self) -> None: ...
    async def stop(self) -> None: ...

    async def pause(self) -> None:
        self.paused = True

    async def resume(self) -> None:
        self.paused = False

    async def apply_step(self, step: Step) -> None:
        self.step = step


async def demo_feed(live: dict, trainer: FakeTrainer) -> None:
    """Generate plausible-looking readings for --demo."""
    hr = 70.0
    while True:
        await asyncio.sleep(0.5)
        step = trainer.step
        if step is None or trainer.paused:
            target = 0.0
        elif step.watts is not None:
            target = step.watts
        elif step.grade is not None:
            target = 120 + 18 * step.grade
        else:
            target = 80 + 12 * step.resistance
        power = max(0.0, target + random.gauss(0, 5))
        hr += (60 + 0.45 * power - hr) * 0.04
        live.update(power=round(power), cadence=round(88 + random.gauss(0, 2)),
                    speed_kmh=power * 0.13, hr=round(hr))


# --------------------------------------------------------------------------
# Device discovery
# --------------------------------------------------------------------------

async def find_devices(args, say, stop: asyncio.Event):
    """Scan until the trainer and an HR source are found (or time runs out).

    A device advertising the Heart Rate service always wins. A device that only
    matches --hr-name is used as a fallback, because some watches don't
    include their service list in every advertisement.
    """
    found = {"trainer": None, "hr": None, "hr_by_name": None}
    announced: set[str] = set()

    def announce(key: str, text: str) -> None:
        if key not in announced:
            announced.add(key)
            say(text)

    def check(dev, adv) -> None:
        name = dev.name or adv.local_name or ""
        uuids = {u.lower() for u in adv.service_uuids}

        if found["trainer"] is None and args.trainer_name.lower() in name.lower():
            found["trainer"] = dev
            announce("trainer", f"Found trainer: {name} ({dev.address})")
        elif not args.no_hr:
            if HR_SERVICE in uuids:
                found["hr"] = dev
                announce("hr", f"Found HR source: {name or '?'} ({dev.address})")
            elif (args.hr_name and found["hr_by_name"] is None
                  and args.hr_name.lower() in name.lower()):
                found["hr_by_name"] = dev
                announce("hr_name", f"HR candidate by name only: {name} ({dev.address})")

    say(f"Scanning (up to {args.scan_time:.0f}s)...")
    start = time.monotonic()
    name_seen_at = None
    async with BleakScanner(detection_callback=check) as scanner:
        while time.monotonic() - start < args.scan_time and not stop.is_set():
            # BlueZ merges advertisement packets in this cache; poll it too.
            cache = getattr(scanner, "discovered_devices_and_advertisement_data", {})
            for dev, adv in list(cache.values()):
                check(dev, adv)

            if found["hr_by_name"] and name_seen_at is None:
                name_seen_at = time.monotonic()
            if found["trainer"] and (args.no_hr or found["hr"]):
                break
            # Name-only HR candidate: give the UUID match a few more seconds.
            if found["trainer"] and name_seen_at and time.monotonic() - name_seen_at > 6:
                break
            await asyncio.sleep(0.5)

    if stop.is_set():
        return None, None
    if found["trainer"] is None:
        raise RuntimeError(f"Couldn't find a trainer matching '{args.trainer_name}'. "
                           "Spin the pedals and make sure nothing else is connected to it.")
    hr = found["hr"] or found["hr_by_name"]
    if hr is None and not args.no_hr:
        say("No heart rate source found; continuing without HR.")
    return found["trainer"], hr


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------

def mmss(seconds: float) -> str:
    seconds = max(0, round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def dist_str(meters: float) -> str:
    meters = max(0.0, meters)
    return f"{meters / 1000:.2f} km" if meters >= 1000 else f"{meters:.0f} m"


def val(v, spec="{:.0f}") -> str:
    return "--" if v is None else spec.format(v)


class DashboardApp(App):
    TITLE = "KICKR Workout"

    BINDINGS = [
        Binding("n", "skip", "Next step"),
        Binding("r", "reverse", "Previous / restart step"),
        Binding("p", "pause", "Pause/play"),
        Binding("q", "quit", "Stop & quit"),
    ]

    CSS = """
    #main { height: 1fr; }
    #left { width: 46; }
    #stats { height: auto; border: round $primary; padding: 0 1; }
    #steps { height: 1fr; border: round $primary; padding: 0 1; }
    #charts { width: 1fr; }
    PlotextPlot { height: 1fr; }
    #overall { margin: 0 1; }
    #log { height: 8; border: round $primary; }
    """

    def __init__(self, cli, workout_name: str, plan: list[Step]):
        super().__init__()
        self.cli = cli
        self.workout_name = workout_name
        self.plan = plan

        # How progress is shown depends on the kinds of steps in the plan:
        #   time     -> step left / total left / bar in time
        #   distance -> step left / total left / bar in distance
        #   mixed    -> only "step left"; total left and the bar are dropped
        kinds = {s.is_distance for s in plan}
        self.mode = "distance" if kinds == {True} else "time" if kinds == {False} else "mixed"
        self.unit_start = []                  # where each step starts, in the mode's unit
        t = 0.0
        for s in plan:
            self.unit_start.append(t)
            t += s.length
        self.total_units = t                  # seconds or meters (not used when mixed)

        self.live = {"power": None, "cadence": None, "speed_kmh": None, "hr": None}
        self.elapsed_s = 0.0                  # unpaused workout time
        self.dist_m = 0.0                     # distance, integrated from trainer speed
        self.hist_t: list[float] = []         # minutes of elapsed workout time
        self.hist_target: list[float | None] = []   # commanded watts (None if not ERG)
        self.hist_power: list[float | None] = []
        self.hist_hr: list[float | None] = []
        self.hist_cad: list[float | None] = []

        self.stop_event = asyncio.Event()
        self.skip_requested = False
        self.reverse_requested = False
        self.pause_requested = False
        self.paused = False
        self.wake = asyncio.Event()           # set by keys so the loop reacts at once
        self.session_done = False
        self.closing = False
        self.csv_path: Path | None = None

    # ---- layout -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Static(id="stats")
                yield Static(id="steps")
            with Vertical(id="charts"):
                yield PlotextPlot(id="power_plot")
                yield PlotextPlot(id="hr_plot")
        yield ProgressBar(total=self.total_units if self.mode != "mixed" else 1,
                          show_eta=False, id="overall")
        yield RichLog(id="log", wrap=True, markup=False, highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"KICKR Workout: {self.workout_name}"
        self.sub_title = "Starting..."
        self.query_one("#stats").border_title = "Live"
        self.query_one("#steps").border_title = "Plan"
        self.query_one("#log").border_title = "Log"
        if self.mode == "mixed":
            self.query_one("#overall").display = False
        self.render_all(None, 0, 0, 0)
        self.run_worker(self.run_session(), exclusive=True)

    def msg(self, text: str) -> None:
        try:
            self.query_one("#log", RichLog).write(f"{datetime.now():%H:%M:%S}  {text}")
        except Exception:
            pass          # the UI may already be shutting down

    # ---- key actions ------------------------------------------------------

    def action_skip(self) -> None:
        if not self.session_done:
            self.skip_requested = True
            self.wake.set()

    def action_reverse(self) -> None:
        if not self.session_done:
            self.reverse_requested = True
            self.wake.set()

    def action_pause(self) -> None:
        if not self.session_done:
            self.pause_requested = True
            self.wake.set()

    async def action_quit(self) -> None:
        if self.session_done:
            self.exit()
        elif not self.stop_event.is_set():
            self.stop_event.set()
            self.wake.set()
            self.sub_title = "Stopping..."
            self.msg("Stopping: releasing the trainer...")

    # ---- rendering --------------------------------------------------------

    def render_all(self, step, idx, remaining, done) -> None:
        """remaining: left in the current step (its own unit). done: progress through
        the whole plan in the mode's unit (None when the plan mixes time and distance)."""
        self.render_stats(step, idx, remaining, done)
        self.render_steps(idx)
        self.render_charts(self.elapsed_s / 60, step, idx, remaining)
        if self.mode != "mixed":
            self.query_one("#overall", ProgressBar).update(progress=done)

    def render_stats(self, step, idx, remaining, done) -> None:
        d = self.live
        target_w = step.watts if step else None
        style = "bold"
        if d["power"] is not None and target_w:
            err = abs(d["power"] - target_w) / target_w
            style = "bold green" if err <= 0.05 else "bold yellow" if err <= 0.10 else "bold red"

        t = Text()
        t.append("POWER    ", "dim")
        t.append(f"{val(d['power']):>5} W\n", style)
        t.append("CADENCE  ", "dim")
        t.append(f"{val(d['cadence']):>5} rpm\n", "bold cyan")
        t.append("HEART    ", "dim")
        t.append(f"{val(d['hr']):>5} bpm\n", "bold red")
        t.append("SPEED    ", "dim")
        t.append(f"{val(d['speed_kmh'], '{:.1f}'):>5} km/h\n", "bold")
        t.append("DISTANCE ", "dim")
        t.append(f"{dist_str(self.dist_m):>8}\n", "bold")
        if self.paused:
            t.append("\n")
            t.append("** PAUSED ** (p to resume)\n", "bold reverse yellow")
        if step:
            t.append("\n")
            t.append(f"Step {idx + 1}/{len(self.plan)}: ", "dim")
            t.append(f"{step.name}\n", "bold")

            rows = [("Target     ", step.target, "bold orange1")]
            if step.watts is not None:
                # Display-only for now: grade/resistance alongside a power target.
                if step.grade is not None:
                    rows.append(("Grade      ", f"{step.grade:g} %", "bold"))
                if step.resistance is not None:
                    rows.append(("Resistance ", f"{step.resistance:g}", "bold"))
            rows.append(("Step left  ",
                         dist_str(remaining) if step.is_distance else mmss(remaining), ""))
            if self.mode == "time":
                rows.append(("Total left ", mmss(self.total_units - done), ""))
            elif self.mode == "distance":
                rows.append(("Total left ", dist_str(self.total_units - done), ""))
            for n, (label, text, sty) in enumerate(rows):
                t.append(label, "dim")
                t.append(text + ("\n" if n < len(rows) - 1 else ""), sty)
        self.query_one("#stats", Static).update(t)

    def render_steps(self, idx: int) -> None:
        t = Text()
        lo = max(0, idx - 1)
        for i in range(lo, min(len(self.plan), lo + 12)):
            s = self.plan[i]
            line = f"{'>' if i == idx else ' '}{i + 1:>2} {s.name[:12]:<12} {s.length_str:>8} {s.target}\n"
            t.append(line, "bold" if i == idx else "dim" if i < idx else "")
        self.query_one("#steps", Static).update(t)

    @staticmethod
    def _pairs(xs, ys):
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        return [p[0] for p in pts], [p[1] for p in pts]

    WINDOW_MIN = 5.0          # width of the scrolling chart window, in minutes

    def target_runs(self, step, idx: int, remaining: float):
        """Target power vs. elapsed workout time, as runs of (xs, ys).

        History: the target that was actually commanded, sample by sample (so
        skips, restarts and pauses show up as they happened). Ahead of "now" the
        line is projected through the remaining TIME-based steps; it stops at the
        first distance step, whose duration can't be known in advance.
        Grade/resistance-only steps (no watts) break the line into runs.
        """
        runs, xs, ys = [], [], []
        for x, w in zip(self.hist_t, self.hist_target):
            if w is None:
                if xs:
                    runs.append((xs, ys))
                xs, ys = [], []
            else:
                xs.append(x)
                ys.append(w)

        if xs and step is not None and step.watts is not None and not step.is_distance:
            ex = self.elapsed_s / 60 + max(remaining, 0.0) / 60
            xs.append(ex)
            ys.append(step.watts)
            for nxt in self.plan[idx + 1:]:
                if nxt.watts is None or nxt.is_distance:
                    break
                xs += [ex, ex + nxt.seconds / 60]
                ys += [nxt.watts, nxt.watts]
                ex += nxt.seconds / 60
        if xs:
            runs.append((xs, ys))
        return runs

    @staticmethod
    def _clip(xs, ys, lo: float, hi: float):
        """Clip a polyline to the x range [lo, hi], interpolating at the edges."""
        ox, oy = [], []
        for i, (x, y) in enumerate(zip(xs, ys)):
            if i:
                x0, y0 = xs[i - 1], ys[i - 1]
                for edge in (lo, hi):
                    if x0 < edge < x or x < edge < x0:
                        ox.append(edge)
                        oy.append(y0 + (y - y0) * (edge - x0) / (x - x0))
            if lo <= x <= hi:
                ox.append(x)
                oy.append(y)
        return ox, oy

    def render_charts(self, now_min: float, step=None, idx: int = 0, remaining: float = 0.0) -> None:
        # EKG-style window: it starts at 0 and the trace grows rightward until
        # it fills the window, then the window slides so "now" stays at the right edge.
        lo = max(0.0, now_min - self.WINDOW_MIN)
        hi = lo + self.WINDOW_MIN
        marker = "braille"      # 2x4 dots per cell: thinner lines, less overlap

        def visible(xs, ys):
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None and x >= lo - 0.05]
            return [p[0] for p in pts], [p[1] for p in pts]

        # Power chart: target (orange, drawn first) under live power.
        w = self.query_one("#power_plot", PlotextPlot)
        plt = w.plt
        plt.clear_data()
        plt.title("Power (W)")
        n = 0
        for rx, ry in self.target_runs(step, idx, remaining):
            rx, ry = self._clip(rx, ry, lo, hi)
            if rx:
                plt.plot(rx, ry, color="orange", marker=marker, label="Target" if n == 0 else None)
                n += 1
        xs, ys = visible(self.hist_t, self.hist_power)
        if xs:
            plt.plot(xs, ys, color="green", marker=marker, label="Power")
        peaks = [s.watts * 1.25 for s in self.plan if s.watts is not None]
        plt.xlim(lo, hi)
        plt.ylim(0, max([250] + peaks + [max(ys, default=0) * 1.1]))
        w.refresh()

        # Heart rate (left axis) and cadence (right axis) have separate scales,
        # so their lines no longer sit on top of each other.
        w = self.query_one("#hr_plot", PlotextPlot)
        plt = w.plt
        plt.clear_data()
        plt.title("Heart rate (bpm, left)  /  cadence (rpm, right)")
        hx, hy = visible(self.hist_t, self.hist_hr)
        cx, cy = visible(self.hist_t, self.hist_cad)
        if hx:
            plt.plot(hx, hy, color="red", marker=marker, label="HR", yside="left")
        if cx:
            plt.plot(cx, cy, color="cyan", marker=marker, label="Cadence", yside="right")
        plt.xlim(lo, hi)
        plt.ylim(40, max(180, max(hy, default=0) * 1.05), yside="left")
        plt.ylim(0, max(120, max(cy, default=0) * 1.1), yside="right")
        w.refresh()

    # ---- session ----------------------------------------------------------

    async def run_session(self) -> None:
        outcome = "complete"
        try:
            await self.session()
            if self.stop_event.is_set():
                outcome = "stopped"
        except Exception as exc:
            outcome = "error"
            self.msg(f"ERROR: {type(exc).__name__}: {exc}")
        finally:
            self.session_done = True
            if self.stop_event.is_set():
                self.exit()
            elif outcome == "error":
                self.sub_title = "ERROR: press q to exit"
                self.bell()
            else:
                self.sub_title = "Complete: press q to exit"
                self.msg("Workout complete.")

    async def session(self) -> None:
        async with AsyncExitStack() as stack:
            self.sub_title = "Connecting..."
            if self.cli.demo:
                trainer, lost = await self.connect_demo(stack)
            else:
                trainer, lost = await self.connect_real(stack)
            if trainer is None:                   # stopped during scanning
                return

            self.csv_path = Path(f"workout_{datetime.now():%Y%m%d_%H%M%S}.csv")
            csv_file = stack.enter_context(open(self.csv_path, "w", newline=""))
            writer = csv.writer(csv_file)
            writer.writerow(["timestamp", "elapsed_s", "step", "target",
                             "power_w", "cadence_rpm", "speed_kmh", "hr_bpm", "distance_m"])

            self.msg("Taking control of the trainer...")
            await trainer.take_control()
            self.msg(f"Logging to {self.csv_path}")
            self.sub_title = "Running"
            scale = self.cli.demo_speed if self.cli.demo else 1.0
            if self.mode == "mixed":
                self.msg("Mixed time/distance workout: showing step left only.")
            elif self.mode == "distance":
                self.msg("Distance workout: progress is measured from trainer speed.")

            try:
                t0 = time.monotonic()
                trainer_paused = False
                idx = 0
                while idx < len(self.plan):
                    if self.stop_event.is_set():
                        break
                    step = self.plan[idx]
                    self.msg(f"Step {idx + 1}/{len(self.plan)}: {step.name}, "
                             f"{step.target} for {step.length_str}")
                    self.skip_requested = False
                    self.reverse_requested = False
                    apply_pending = True          # send the target when not paused
                    move = 1                      # +1 next step, 0 restart, -1 previous
                    remaining = float(step.length)  # seconds left, or meters left
                    step_t = 0.0                  # real seconds spent in this step
                    last = time.monotonic()

                    while True:
                        now = time.monotonic()
                        if not self.paused:
                            step_t += now - last
                            dt = (now - last) * scale        # workout seconds
                            v = (self.live.get("speed_kmh") or 0.0) / 3.6   # m/s
                            self.elapsed_s += dt
                            self.dist_m += v * dt
                            remaining -= v * dt if step.is_distance else dt
                        last = now

                        # Pause / play toggle.
                        if self.pause_requested:
                            self.pause_requested = False
                            self.paused = not self.paused
                            if self.paused:
                                self.sub_title = "Paused"
                                self.msg("Paused.")
                                try:
                                    await trainer.pause()
                                    trainer_paused = True
                                except Exception as exc:
                                    self.msg(f"(Trainer didn't accept pause: {exc}. "
                                             "The timer is paused; resistance is unchanged.)")
                            else:
                                self.sub_title = "Running"
                                self.msg("Resumed.")
                                if trainer_paused:
                                    try:
                                        await trainer.resume()
                                    except Exception as exc:
                                        self.msg(f"(Trainer didn't accept resume: {exc})")
                                    trainer_paused = False
                                apply_pending = True

                        if apply_pending and not self.paused:
                            await trainer.apply_step(step)
                            apply_pending = False

                        left = max(remaining, 0.0)
                        done = (None if self.mode == "mixed"
                                else self.unit_start[idx] + step.length - left)

                        # Record one sample: CSV row, chart history, screen.
                        d = self.live
                        writer.writerow([datetime.now().isoformat(timespec="seconds"),
                                         f"{now - t0:.1f}", step.name,
                                         "PAUSED" if self.paused else step.target,
                                         d["power"], d["cadence"],
                                         None if d["speed_kmh"] is None else round(d["speed_kmh"], 1),
                                         d["hr"], round(self.dist_m, 1)])
                        csv_file.flush()
                        if not self.paused:
                            self.hist_t.append(self.elapsed_s / 60)
                            self.hist_target.append(step.watts)
                            self.hist_power.append(d["power"])
                            self.hist_hr.append(d["hr"])
                            self.hist_cad.append(d["cadence"])
                        self.render_all(step, idx, left, done)

                        if lost.is_set():
                            raise ConnectionError("Lost connection to the trainer.")
                        if self.stop_event.is_set():
                            break
                        if self.skip_requested:
                            self.msg("Skipping to the next step.")
                            break
                        if self.reverse_requested:
                            # Under 2 s into a step: go to the previous one.
                            # Otherwise: restart this one.
                            if step_t < 2.0 and idx > 0:
                                move = -1
                                self.msg("Back to the previous step.")
                            else:
                                move = 0
                                self.msg("Restarting this step.")
                            break
                        if remaining <= 0:
                            break

                        if self.paused:
                            timeout = 0.5
                        elif step.is_distance:
                            # Wake about when the distance should run out.
                            speed = max((self.live.get("speed_kmh") or 0.0) / 3.6, 0.5)
                            timeout = max(0.05, min(1.0, remaining / speed / scale))
                        else:
                            timeout = max(0.05, min(1.0, remaining / scale))
                        self.wake.clear()
                        try:
                            await asyncio.wait_for(self.wake.wait(), timeout)
                        except asyncio.TimeoutError:
                            pass

                    if self.stop_event.is_set():
                        break
                    idx += move
            finally:
                # Runs on completion, errors and quitting: release the trainer.
                self.closing = True
                try:
                    if self.cli.demo or trainer.client.is_connected:
                        await trainer.stop()
                        self.msg("Trainer released.")
                except Exception as exc:
                    self.msg(f"(Couldn't send stop command: {exc})")

    # ---- connections ------------------------------------------------------

    async def connect_demo(self, stack: AsyncExitStack):
        self.msg("DEMO MODE: simulated trainer and heart rate, no Bluetooth.")
        trainer = FakeTrainer()
        feed = asyncio.create_task(demo_feed(self.live, trainer))

        async def stop_feed() -> None:
            feed.cancel()
        stack.push_async_callback(stop_feed)
        return trainer, asyncio.Event()

    async def connect_real(self, stack: AsyncExitStack):
        trainer_dev, hr_dev = await find_devices(self.cli, self.msg, self.stop_event)
        if trainer_dev is None:
            return None, None

        lost = asyncio.Event()

        def on_trainer_disconnect(_client) -> None:
            if not self.closing:
                self.msg("Trainer disconnected!")
                lost.set()

        client = await stack.enter_async_context(
            BleakClient(trainer_dev, disconnected_callback=on_trainer_disconnect))
        trainer = Trainer(client, self.live)
        await trainer.start_notifications()
        self.msg("Trainer connected.")

        if hr_dev is not None:
            await self.connect_hr(stack, hr_dev)
        return trainer, lost

    async def connect_hr(self, stack: AsyncExitStack, dev) -> None:
        """Heart rate is optional: any failure here is logged, not fatal."""
        def on_hr(_, data: bytearray) -> None:
            self.live["hr"] = parse_heart_rate(data)

        def on_hr_disconnect(_client) -> None:
            if not self.closing:
                self.msg("HR source disconnected; continuing without HR.")
                self.live["hr"] = None

        client = BleakClient(dev, disconnected_callback=on_hr_disconnect)
        try:
            await client.connect()
            if client.services.get_characteristic(HR_MEASUREMENT) is None:
                offered = ", ".join(
                    s.uuid[4:8] if s.uuid.endswith(HR_SERVICE[8:]) else s.uuid
                    for s in client.services)
                raise RuntimeError(f"no Heart Rate characteristic; device offers: {offered}")
            await client.start_notify(HR_MEASUREMENT, on_hr)
        except Exception as exc:
            self.msg(f"HR unavailable: {exc}")
            try:
                await client.disconnect()
            except Exception:
                pass
            return
        stack.push_async_callback(client.disconnect)
        self.msg("Heart rate connected.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Run a workout on a KICKR with a live dashboard.")
    p.add_argument("workout", help="path to a workout JSON file")
    p.add_argument("--ftp", type=int, help="FTP in watts (overrides the file)")
    p.add_argument("--trainer-name", default="KICKR",
                   help="substring of the trainer's BLE name (default: KICKR)")
    p.add_argument("--hr-name", help="substring of the HR device's BLE name (fallback "
                                     "when its advertisement lacks the HR service UUID)")
    p.add_argument("--no-hr", action="store_true", help="skip heart rate entirely")
    p.add_argument("--scan-time", type=float, default=20.0, help="max scan seconds")
    p.add_argument("--demo", action="store_true",
                   help="simulate the trainer and HR instead of using Bluetooth")
    p.add_argument("--demo-speed", type=float, default=1.0,
                   help="time multiplier for --demo (e.g. 10 runs ten times faster)")
    args = p.parse_args()

    try:
        name, plan = load_workout(args.workout, args.ftp)
    except (OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"Couldn't load workout file: {type(exc).__name__}: {exc}")

    app = DashboardApp(args, name, plan)
    app.run()
    if app.csv_path:
        print(f"Data saved to {app.csv_path}")


if __name__ == "__main__":
    main()