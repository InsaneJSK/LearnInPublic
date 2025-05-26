# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Async Programming

# %%
from multiprocessing import Process


# %%
def myfunc():
    pass


# %%
procs = []
for _ in range(5):
    procs.append(Process(target=myfunc, args=()))

# %%
#For starting those functions
for proc in procs:
    proc.start()

# %%
for proc in procs:
    proc.join()

# %% [markdown]
# ## Multi Threading

# %%
from threading import Thread

# %%
f1 = Thread(target=myfunc)

# %%
f1.start() #Async method
f1.join() #Sync method


# %% [markdown]
# ## Coroutines

# %%
def sample(name):
    try:
        while True:
            what = (yield)
            print(f"{what} {name}")
    except GeneratorExit:
        print("Ok!")


# %%
co = sample("JSK")

# %%
next(co)

# %%
co.send("Get the hell lost")

# %%
co.close()

# %% [markdown]
# ## AsyncIO

# %%
import asyncio


# %%
async def main():
    print("Hello")
    await asyncio.sleep(1)
    print("World")


# %%
type(main())

# %%
if __name__ == "__main__":
    try:
        asyncio.run(main())  #The line that should work in a py-kernel but due to how main loop works in ipy, it doesn't work in jupyter
    except RuntimeError:
        await main()


# %%
async def waiter(n):
    await asyncio.sleep(n)
    print(f"Waited for {n} sec")


# %%
import time
async def main():
    print(time.strftime('%X'))
    await waiter(2)
    await waiter(3)
    print(time.strftime('%X'))


# %%
if __name__ == "__main__":
    await main()


# %%
async def main():
    task1 = asyncio.create_task(waiter(2))
    task2 = asyncio.create_task(waiter(3))

    print(time.strftime('%X'))
    await task1
    await task2
    print(time.strftime('%X'))


# %%
if __name__ == "__main__":
    await main()

# %%
import aiohttp


# %%
async def fetchfromgoogle():
    url = 'https://www.google.com'
    session = aiohttp.ClientSession()
    resp = await session.get(url)
    print(await resp.content.read())
    await session.close()


# %%
async def main():
    for _ in range(20):
        await fetchfromgoogle()


# %%
if __name__ == '__main__':
    await main()


# %%
async def main():
    await asyncio.gather(*[fetchfromgoogle() for _ in range(20)])


# %%
if __name__ == '__main__':
    await main()

# %%
