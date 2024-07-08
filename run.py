import threading
import time
from audio.tts_service import TTSServiceType
from context.context_manager import ContextManager, start_services, stop_services
from keys.util import (
    CONVERSATION_INPUT_START_STR,
    CONVERSATION_INPUT_START,
    monitor_keyboard_and_execute_func,
)


def main():
    context_manager = ContextManager()
    services = start_services(
        context_manager=context_manager, tts_service_type=TTSServiceType.CHAT_TTS
    )
    start_conversation_flag = threading.Event()

    def _wait():
        while not start_conversation_flag.is_set():
            time.sleep(0.1)

    try:
        while True:
            print(f"print key '{CONVERSATION_INPUT_START_STR}' to start conversation!")
            monitor_keyboard_and_execute_func(
                expected_keys=CONVERSATION_INPUT_START,
                stop_flag=start_conversation_flag,
                func=_wait,
            )
            # Start conversation
            context_manager.start_conversation()
            start_conversation_flag.clear()
    finally:
        stop_services(services=services)
        context_manager.clear()


if __name__ == "__main__":
    main()
