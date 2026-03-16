from __future__ import annotations

import sys

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"


def use_colors() -> bool:
    return sys.stdout.isatty()


def colorize(text: str, *styles: str) -> str:
    if not use_colors():
        return text
    return f"{''.join(styles)}{text}{RESET}"


def print_section(title: str) -> None:
    banner = f"{'=' * 18} {title} {'=' * 18}"
    print(f"\n{colorize(banner, BOLD, CYAN)}")


def print_label(label: str, color: str) -> None:
    print(f"\n{colorize(label, BOLD, color)}\n")


def clear_screen() -> None:
    if not sys.stdout.isatty():
        return
    print("\033[2J\033[H", end="", flush=True)


def prompt_to_continue(message: str) -> None:
    input(colorize(f"\n{message}", BOLD, YELLOW) + " ")
    clear_screen()
