"""Terminal UI helpers: banner, spinner, and chat-friendly styled output."""

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
Backend: [ {device_label.upper()} ] | Chat Engine: Gemini-first
"""
    if TRY_RICH:
        console.print(Panel(txt, title="TokenCode Boot", style="bold cyan"))
    else:
        print(txt)


def print_system(text: str):
    if TRY_RICH:
        console.print(f"[bold cyan]System:[/bold cyan] {text}")
    else:
        print(f"[System] {text}")


def print_user(text: str):
    if TRY_RICH:
        console.print(f"[bold green]You:[/bold green] {text}")
    else:
        print(f"You> {text}")


def print_ai(text: str):
    if TRY_RICH:
        console.print(Panel(text, title="AI", border_style="magenta"))
    else:
        print(f"AI> {text}")


def print_help_panel(help_text: str):
    if TRY_RICH:
        console.print(Panel(help_text, title="Commands", border_style="yellow"))
    else:
        print(help_text)


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
            sys.stdout.write(f"\r{s} {self.text} ")
            sys.stdout.flush()
            time.sleep(0.08)
        sys.stdout.write("\r" + " " * (len(self.text) + 20) + "\r")
        sys.stdout.flush()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
