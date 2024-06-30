from dataclasses import dataclass
import pyperclip

@dataclass(frozen=True)
class CopyFromClipboardTask:
    task_id: str

@dataclass
class CopyFromClipboardResult:
    task: CopyFromClipboardTask
    text: str

@dataclass
class TextManager:
    # Unique id for a conversation.
    conversation_id: str

    def copy_from_clipboard(self, task: CopyFromClipboardTask) -> CopyFromClipboardResult:
        # TODO: lost format when copying from clipboard.
        text = pyperclip.paste()
        return CopyFromClipboardResult(task=task, text=text)
    
if __name__ == "__main__":
    task = CopyFromClipboardTask(task_id="test")
    text_manager = TextManager(conversation_id="text")
    res = text_manager.copy_from_clipboard(task)
    print(f"copied from clipboard: {res.text}")