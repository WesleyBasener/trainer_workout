import asyncio
from bleak import BleakScanner, BleakClient

HR_SVC = "0000180d-0000-1000-8000-00805f9b34fb"
HR_CHR = "00002a37-0000-1000-8000-00805f9b34fb"

async def main():
    print("Waiting for an advertisement that includes the HR service...")
    dev = await BleakScanner.find_device_by_filter(
        lambda d, a: HR_SVC in [u.lower() for u in a.service_uuids], timeout=60)
    if dev is None:
        print("Never saw 0x180D in an advertisement.")
        return
    print("found:", dev.address, dev.name)
    async with BleakClient(dev) as c:
        print("services:")
        for svc in c.services:
            print("  ", svc.uuid)
        if c.services.get_characteristic(HR_CHR):
            await c.start_notify(HR_CHR, lambda _, d: print("HR packet:", d.hex()))
            await asyncio.sleep(30)
        else:
            print("No heart rate characteristic on this connection.")

asyncio.run(main())