import asyncio, pathlib, sys
from test_broker import FakeMiner
async def main():
    async with await asyncio.start_unix_server(FakeMiner().client,sys.argv[1]) as server:
        await server.serve_forever()
asyncio.run(main())
