import asyncio
from bleak import BleakScanner

async def main():
    found = await BleakScanner.discover(timeout=10, return_adv=True)
    for dev, adv in sorted(found.values(), key=lambda x: x[1].rssi, reverse=True):
        name = dev.name or adv.local_name or "?"
        print(f"{adv.rssi:4d} dBm  {dev.address}  {name:<25} {adv.service_uuids}")

asyncio.run(main())