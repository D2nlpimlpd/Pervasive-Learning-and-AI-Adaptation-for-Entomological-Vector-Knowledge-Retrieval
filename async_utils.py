# async_utils.py
# 统一在一个后台线程里跑 asyncio 事件循环，并提供 run_async 工具给 Streamlit 主线程用。

import asyncio
import threading
from typing import Awaitable, Any

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _loop_runner(loop: asyncio.AbstractEventLoop) -> None:
    """
    在单独线程中启动事件循环，永久运行。
    """
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """
    确保全局事件循环已经在后台线程中启动。
    - 如果还没启动，就创建一个新的 loop 和线程；
    - 已经有的话，直接返回同一个 loop。
    """
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            _loop_thread = threading.Thread(
                target=_loop_runner,
                args=(_loop,),
                daemon=True,  # 后台线程，随主进程退出
            )
            _loop_thread.start()
        return _loop


def run_async(coro: Awaitable[Any]) -> Any:
    """
    在同步环境里执行一个协程：
    - 所有协程都被提交到同一个后台线程里的事件循环中执行；
    - 当前线程（例如 Streamlit 的主线程）同步等待结果返回；
    - 避免与当前线程中可能存在的 asyncio 事件循环发生冲突。
    """
    loop = _ensure_loop()
    # 用线程安全的方式把协程丢到后台 loop 里执行
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    # 在当前线程阻塞等待执行完成并返回结果（或抛出异常）
    return future.result()