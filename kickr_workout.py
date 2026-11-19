#!/usr/bin/env python3
"""
Run a pre-programmed ERG workout on a Wahoo KICKR (via BLE FTMS) while
recording heart rate from a Garmin watch (or any BLE HR source).

Usage:
    python kickr_workout.py sample_workout.json
    python kickr_workout.py sample_workout.json --ftp 230
    python kickr_workout.py sample_workout.json --hr-name Forerunner
    python kickr_workout.py sample_workout.json --no-hr

Requires: pip install bleak
Close Zwift / the Wahoo app / anything else connected to the trainer first.
"""

import argparse
import asyncio
import csv
import json
import struct
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bleak import BleakClient, BleakScanner


def uuid16(n: int) -> str:
    """Expand a 16-bit Bluetooth UUID to its full 128-bit string form."""
    return f"0000{n:04x}-0000-1000-8000-00805f9b34fb"


HR_SERVICE = uuid16(0x180D)
HR_MEASUREMENT = uuid16(0x2A37)
CONTROL_POINT = uuid16(0x2AD9)   # FTMS Fitness Machine Control Point
BIKE_DATA = uuid16(0x2AD2)       # FTMS Indoor Bike Data

OP_REQUEST_CONTROL = 0x00
OP_SET_TARGET_POWER = 0x05
OP_START = 0x07
OP_STOP = 0x08
RESULT_SUCCESS = 0x01
OP_SET_TARGET_RESISTANCE = 0x04
OP_SET_SIMULATION = 0x11

# --------------------------------------------------------------------------
# Workout file handling
# --------------------------------------------------------------------------

@dataclass
class Step:
    name: str
    seconds: int
    watts: int | None = None
    grade: float | None = None         # % grade (simulation mode)
    resistance: float | None = None    # FTMS resistance level

    @property
    def target(self) -> str:
        if self.watts is not None:
            return f"{self.watts} W"
        if self.grade is not None:
            return f"{self.grade:g}% grade"
        return f"resistance {self.resistance:g}"


def load_workout(path: str, ftp_override: int | None) -> tuple[str, list[Step]]:
    """Read a workout JSON file. Each step has 'seconds' plus either
    'watts' (absolute) or 'pct_ftp' (relative to the FTP in the file/CLI)."""
    spec = json.loads(Path(path).read_text())
    ftp = ftp_override or spec.get("ftp")
    steps = []
    for i, s in enumerate(spec["steps"], start=1):
        name, seconds = s.get("name", f"Step {i}"), int(s["seconds"])
        if "watts" in s:
            steps.append(Step(name, seconds, watts=int(s["watts"])))
        elif "pct_ftp" in s:
            if not ftp:
                raise SystemExit(f"Step {i} uses pct_ftp but no FTP is set "
                                 "(add 'ftp' to the file or pass --ftp).")
            steps.append(Step(name, seconds, watts=round(ftp * s["pct_ftp"] / 100)))
        elif "grade" in s:
            steps.append(Step(name, seconds, grade=float(s["grade"])))
        elif "resistance" in s:
            steps.append(Step(name, seconds, resistance=float(s["resistance"])))
        else:
            raise SystemExit(f"Step {i} needs 'watts', 'pct_ftp', 'grade' or 'resistance'.")
    return spec.get("name", Path(path).stem), steps


# --------------------------------------------------------------------------
# BLE data parsing
# --------------------------------------------------------------------------

def parse_bike_data(data: bytearray) -> dict:
    """Parse the FTMS Indoor Bike Data characteristic.

    A 16-bit flags field says which optional fields follow, in a fixed order.
    We only need speed, cadence and power, which come first, so we stop
    parsing once we have power.
    """
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
# Trainer wrapper
# --------------------------------------------------------------------------

class Trainer:
    """Thin wrapper around the FTMS control point.

    Commands are written to the control point; the trainer answers with an
    *indication* ([0x80, request_opcode, result_code]). We turn each answer
    into an awaitable Future so `await command(...)` returns when it arrives.
    """

    def __init__(self, client: BleakClient, state: dict):
        self.client = client
        self.state = state
        self._pending: dict[int, asyncio.Future] = {}

    async def start_notifications(self) -> None:
        await self.client.start_notify(CONTROL_POINT, self._on_control_point)
        await self.client.start_notify(BIKE_DATA, self._on_bike_data)

    def _on_bike_data(self, _, data: bytearray) -> None:
        self.state.update(parse_bike_data(data))

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

    async def stop(self) -> None:
        await self.command(OP_STOP, b"\x01")   # 0x01 = stop (0x02 = pause)

    async def set_resistance_level(self, level: float) -> None:
        """FTMS target resistance (uint8, 0.1 units). Support varies by trainer."""
        await self.command(OP_SET_TARGET_RESISTANCE, struct.pack("<B", round(level * 10)))

    async def set_simulation(self, grade_pct: float, crr: float = 0.004,
                             cw: float = 0.51, wind_mps: float = 0.0) -> None:
        """Simulate a road grade. crr and cw are the usual FTMS defaults
        (rolling resistance, wind resistance coefficient)."""
        params = struct.pack("<hhBB",
                             round(wind_mps * 1000),   # sint16, 0.001 m/s
                             round(grade_pct * 100),   # sint16, 0.01 %
                             round(crr * 10000),       # uint8, 0.0001
                             round(cw * 100))          # uint8, 0.01 kg/m
        await self.command(OP_SET_SIMULATION, params)

    async def apply_step(self, step: "Step") -> None:
        if step.watts is not None:
            await self.set_power(step.watts)
        elif step.grade is not None:
            await self.set_simulation(step.grade)
        else:
            await self.set_resistance_level(step.resistance)

# --------------------------------------------------------------------------
# Device discovery
# --------------------------------------------------------------------------

async def find_devices(args):
    found = {"trainer": None, "hr": None}
    done = asyncio.Event()

    def on_detect(dev, adv):
        name = dev.name or adv.local_name or ""
        uuids = {u.lower() for u in adv.service_uuids}
        
        if found["trainer"] is None and args.trainer_name.lower() in name.lower():
            found["trainer"] = dev
            print(f"  found trainer: {name} ({dev.address})")
        elif not args.no_hr and found["hr"] is None:
            if args.hr_name:
                match = args.hr_name.lower() in name.lower()
            else:
                match = HR_SERVICE in uuids
            if match:
                found["hr"] = dev
                print(f"  found HR source: {name or '?'} ({dev.address})")

        if found["trainer"] and (args.no_hr or found["hr"]):
            done.set()

    print(f"Scanning (up to {args.scan_time:.0f}s)...")
    async with BleakScanner(detection_callback=on_detect):
        try:
            await asyncio.wait_for(done.wait(), timeout=args.scan_time)
        except asyncio.TimeoutError:
            pass

    if found["trainer"] is None:
        raise SystemExit(f"Couldn't find a trainer matching '{args.trainer_name}'. "
                         "Spin the pedals, and make sure nothing else is connected to it.")
    if found["hr"] is None and not args.no_hr:
        print("No heart rate source found, continuing without HR. "
              "(Is the watch's broadcast screen still active?)")
    return found["trainer"], found["hr"]


# --------------------------------------------------------------------------
# Main workout loop
# --------------------------------------------------------------------------

def fmt(value, spec="{:.0f}") -> str:
    return "--" if value is None else spec.format(value)


async def run(args) -> None:
    name, steps = load_workout(args.workout, args.ftp)
    total = sum(s.seconds for s in steps)
    print(f"Workout: {name}, {len(steps)} steps, {total // 60}:{total % 60:02d} total")

    trainer_dev, hr_dev = await find_devices(args)

    # Latest readings, updated by the notification callbacks.
    state = {"power": None, "cadence": None, "speed_kmh": None, "hr": None}
    lost = asyncio.Event()

    def on_disconnect(_client) -> None:
        print("\nA device disconnected!")
        lost.set()

    def on_hr(_, data: bytearray) -> None:
        state["hr"] = parse_heart_rate(data)

    csv_path = Path(f"workout_{datetime.now():%Y%m%d_%H%M%S}.csv")

    async with AsyncExitStack() as stack:
        # AsyncExitStack lets us optionally connect to the HR device
        # and still get automatic disconnects on exit.
        trainer_client = await stack.enter_async_context(
            BleakClient(trainer_dev, disconnected_callback=on_disconnect))
        trainer = Trainer(trainer_client, state)
        await trainer.start_notifications()

        if hr_dev is not None:
            hr_client = await stack.enter_async_context(
                BleakClient(hr_dev, disconnected_callback=on_disconnect))
            await hr_client.start_notify(HR_MEASUREMENT, on_hr)

        print("Connected. Taking control of the trainer...")
        await trainer.take_control()

        csv_file = stack.enter_context(open(csv_path, "w", newline=""))
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "elapsed_s", "step", "target",
                         "power_w", "cadence_rpm", "speed_kmh", "hr_bpm"])

        try:
            t0 = time.monotonic()
            step_end = t0
            for idx, step in enumerate(steps, start=1):
                step_end += step.seconds          # cumulative, so no drift
                await trainer.apply_step(step)
                print(f"\n>>> Step {idx}/{len(steps)}: {step.name}, "
                    f"{step.target} for {step.seconds}s")

                while (now := time.monotonic()) < step_end:
                    if lost.is_set():
                        raise ConnectionError("Lost connection to a device.")

                    writer.writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        f"{now - t0:.1f}", step.name, step.target,
                        state["power"], state["cadence"],
                        state["speed_kmh"], state["hr"],
                    ])
                    csv_file.flush()

                    print(f"\r  {step_end - now:4.0f}s left | target {step.target:>12} | "
                          f"power {fmt(state['power']):>4} W | "
                          f"cadence {fmt(state['cadence']):>3} | "
                          f"HR {fmt(state['hr']):>3}   ", end="", flush=True)

                    await asyncio.sleep(min(1.0, step_end - now))
            print("\n\nWorkout complete!")
        finally:
            # Runs on normal finish, errors, and Ctrl+C: release the trainer.
            if trainer_client.is_connected:
                try:
                    await trainer.stop()
                except Exception as exc:
                    print(f"\n(Couldn't send stop command: {exc})")
            print(f"Data saved to {csv_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run an ERG workout on a KICKR.")
    p.add_argument("workout", help="path to a workout JSON file")
    p.add_argument("--ftp", type=int, help="FTP in watts (overrides the file)")
    p.add_argument("--trainer-name", default="KICKR",
                   help="substring of the trainer's BLE name (default: KICKR)")
    p.add_argument("--hr-name", help="substring of the HR device's BLE name "
                                     "(default: any device advertising the HR service)")
    p.add_argument("--no-hr", action="store_true", help="skip heart rate entirely")
    p.add_argument("--scan-time", type=float, default=20.0, help="scan seconds")
    args = p.parse_args()

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()