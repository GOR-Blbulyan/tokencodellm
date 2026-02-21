"""Terminal UI helpers: banner and animated spinner."""

import itertools
import sys
import threading
import time

TRY_RICH = False
try:
    from rich.console import Console
    from rich.panel import Panel

    TRY_RICH = True
    console = Console()
except ImportError:
    console = None

_spinner_cycle = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])


def banner(app_name: str, copyright_line: str, device_label: str):
    txt = f"""
████████╗ ██████╗ ██╗  ██╗███████╗███╗   ██╗   {app_name}
╚══██╔══╝██╔═══██╗██║ ██╔╝██╔════╝████╗  ██║   {copyright_line}
   ██║   ██║   ██║█████╔╝ █████╗  ██╔██╗ ██║
   ██║   ██║   ██║██╔═██╗ ██╔══╝  ██║╚██╗██║
   ██║   ╚██████╔╝██║  ██╗███████╗██║ ╚████║
   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝
Hardware Backend: [ {device_label.upper()} ] | Tokenizer: BPE (cl100k_base)
"""
    if TRY_RICH:
        console.print(Panel(txt, title="System Init", style="bold magenta"))
    else:
        print(txt)


class AnimatedSpinner:
    def __init__(self, text="Working"):
        self.text = text
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            s = next(_spinner_cycle)
            sys.stdout.write(f"\r\033[95m{s}\033[0m {self.text} ")
            sys.stdout.flush()
            time.sleep(0.08)
        sys.stdout.write("\r" + " " * (len(self.text) + 20) + "\r")
        sys.stdout.flush()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
