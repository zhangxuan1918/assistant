from dataclasses import dataclass, field
from typing import Dict
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

    copy_results: Dict[str, CopyFromClipboardResult] = field(default_factory=dict)

    def copy_from_clipboard(self, task: CopyFromClipboardTask) -> str:
        # TODO: lost format when copying from clipboard.
        text = pyperclip.paste()
        result = CopyFromClipboardResult(task=task, text=text)
        self.copy_results[task.task_id] = result
        return result.text
    
if __name__ == "__main__":
    task = CopyFromClipboardTask(task_id="test")
    text_manager = TextManager()
    text = text_manager.copy_from_clipboard(task)
    print(f"copied from clipboard: {text}")