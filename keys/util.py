import threading
from typing import Any, Set, Callable
from pynput.keyboard import Key, Listener, KeyCode

AUDIO_INPUT_START = {Key.ctrl, KeyCode.from_char("s")}
AUDIO_INPUT_END = {Key.ctrl, KeyCode.from_char("q")}


def monitor_keyboard_and_execute_func(
    expected_keys: Set[Any], stop_flag: threading.Event, func: Callable, **kwargs
) -> None:
    current_keys = set()

    def on_press(key: KeyCode):
        current_keys.add(key)
        # Check if the current_keys is contains "expected_keys".
        if expected_keys.issubset(current_keys):
            # Stop listener
            stop_flag.set()
            return False

    with Listener(on_press=on_press) as listener:
        func(**kwargs)
        listener.join()


if __name__ == "__main__":
    import time

    expected_keys = AUDIO_INPUT_END
    monitor_keyboard_and_execute_func(
        expected_keys=expected_keys, func=lambda: time.sleep(5)
    )
