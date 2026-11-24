"""Transparent loopback TCP relay. A drain signal stops incoming bytes and lets responses finish."""
import asyncio
import signal


class Relay:
    def __init__(self, upstream):
        self.upstream=upstream
        self.server=None
        self.draining=False
        self.clients=set()

    def drain(self):
        self.draining=True
        self.server.close()
        for writer in tuple(self.clients):
            writer.transport.pause_reading()

    async def client(self, incoming, outgoing):
        if self.draining:
            outgoing.close()
            return
        self.clients.add(outgoing)
        backend=None
        tasks=[]
        try:
            responses,backend=await asyncio.open_connection('127.0.0.1',self.upstream)
            async def transfer(reader,writer,requests):
                while data:=await reader.read(65536):
                    if requests and self.draining:
                        return
                    writer.write(data)
                    await writer.drain()
                if writer.can_write_eof():
                    writer.write_eof()
            to_model=asyncio.create_task(transfer(incoming,backend,True))
            to_client=asyncio.create_task(transfer(responses,outgoing,False))
            tasks=[to_model,to_client]
            done,_=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            if to_model in done and not to_client.done():
                await to_client
        except (OSError,ConnectionError):
            pass
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            self.clients.discard(outgoing)
            outgoing.close()
            if backend is not None:
                backend.close()


async def main():
    relay=Relay(18002)
    relay.server=await asyncio.start_server(relay.client,'127.0.0.1',8000,backlog=1024)
    loop=asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGUSR1,relay.drain)
    stop=asyncio.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):
        loop.add_signal_handler(sig,stop.set)
    async with relay.server:
        await stop.wait()


if __name__=='__main__':
    asyncio.run(main())
