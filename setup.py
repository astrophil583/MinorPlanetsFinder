#!/usr/bin/env python3
"""
Interactive wizard to configure your observation sky window.

Guides you through setting your GPS location and the azimuth/altitude
range of your sky window (balcony, rooftop, open horizon, etc.),
then saves everything to config.json.
"""

import json
import math
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, FloatPrompt, Confirm
    from rich import box
    from rich.table import Table
except ImportError:
    print("Run: pip install rich")
    raise

console = Console()

COMPASS = """
                    N (0°/360°)
                        ↑
                        |
          NW (315°)  ───┼───  NE (45°)
                        |
    W (270°) ───────────+─────────── E (90°)
                        |
          SW (225°)  ───┼───  SE (135°)
                        |
                        ↓
                    S (180°)
"""


def draw_window(az_min: float, az_max: float, el_min: float, el_max: float):
    """Draw an ASCII preview of the current sky window."""
    dirs = [
        ("N",   0),  ("NE",  45), ("E",   90), ("SE", 135),
        ("S", 180),  ("SW", 225), ("W",  270), ("NW", 315),
    ]

    def in_window(az):
        if az_min <= az_max:
            return az_min <= az <= az_max
        else:   # wraps through North
            return az >= az_min or az <= az_max

    row = ""
    for name, az in dirs:
        marker = (f"[bold green][{name}][/bold green]" if in_window(az)
                  else f"[dim] {name} [/dim]")
        row += marker + "  "

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column()
    table.add_row(row)

    console.print("\n[bold]Current sky window:[/bold]")
    console.print(table)
    console.print(
        f"  Azimuth: [cyan]{az_min:.0f}°[/cyan] → [cyan]{az_max:.0f}°[/cyan]   "
        f"Altitude: [cyan]{el_min:.0f}°[/cyan] → [cyan]{el_max:.0f}°[/cyan]"
    )


def ask_location() -> dict:
    console.print(Panel(
        "[bold]Step 1 — GPS location[/bold]\n\n"
        "Open Google Maps, right-click your position\n"
        "and copy the coordinates that appear.\n\n"
        "Example: [cyan]45.4654, 9.1859[/cyan]",
        border_style="blue",
    ))

    name = Prompt.ask("Location name", default="My observation site")

    while True:
        coords = Prompt.ask("GPS coordinates (lat, lon)")
        try:
            parts = coords.replace(",", " ").split()
            lat = float(parts[0])
            lon = float(parts[1])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError
            break
        except (ValueError, IndexError):
            console.print("[red]Invalid format. Use: 45.4654, 9.1859[/red]")

    elevation = FloatPrompt.ask("Elevation above sea level (metres, approximate)", default=100)

    return {"name": name, "latitude": lat, "longitude": lon, "elevation_m": int(elevation)}


def ask_sky_window() -> dict:
    console.print(Panel(
        "[bold]Step 2 — Sky window[/bold]\n\n"
        "Define the azimuth and altitude range you can see from your\n"
        "observation spot (balcony, rooftop, garden, open field…).\n\n"
        "[bold yellow]How to measure azimuth:[/bold yellow]\n"
        "  1. Open a compass app on your phone\n"
        "  2. Stand at your spot and look at the LEFT edge of your view\n"
        "     → note the compass bearing  (this is azimuth_min)\n"
        "  3. Look at the RIGHT edge of your view\n"
        "     → note the bearing  (this is azimuth_max)\n"
        "  Tip: for open horizon set azimuth_min=0 and azimuth_max=360.\n\n"
        "[bold yellow]How to measure altitude:[/bold yellow]\n"
        "  • [cyan]altitude_min[/cyan]: degrees above horizon where your view starts.\n"
        "    Building in front → ~15–25°. Open horizon → 0–5°.\n"
        "  • [cyan]altitude_max[/cyan]: how high can you see? Roof overhead → ~60°.\n"
        "    Open sky → 85–90°.\n\n"
        + COMPASS,
        border_style="green",
    ))

    console.print("[dim]Azimuth reference: 0°=N, 90°=E, 180°=S, 270°=W[/dim]\n")

    while True:
        az_min = FloatPrompt.ask("Azimuth of the LEFT edge of your view (0–360°)")
        if 0 <= az_min <= 360:
            break
        console.print("[red]Value must be between 0 and 360[/red]")

    while True:
        az_max = FloatPrompt.ask("Azimuth of the RIGHT edge of your view (0–360°)")
        if 0 <= az_max <= 360:
            break
        console.print("[red]Value must be between 0 and 360[/red]")

    # Angular width
    width = (az_max - az_min) if az_max >= az_min else (360 - az_min) + az_max
    console.print(f"\n  → Angular width: [bold cyan]{width:.0f}°[/bold cyan]")
    if width < 30:
        console.print("  [yellow]⚠  Very narrow view — few objects will pass through.[/yellow]")
    elif width >= 180:
        console.print("  [green]✓  Wide view — great chances![/green]")

    while True:
        el_min = FloatPrompt.ask("\nMinimum altitude visible (degrees above horizon)", default=15)
        if 0 <= el_min < 90:
            break
        console.print("[red]Value must be between 0 and 89[/red]")

    while True:
        el_max = FloatPrompt.ask(
            "Maximum altitude visible (e.g. 85 for open sky, 60 if there is a roof above)",
            default=85,
        )
        if el_min < el_max <= 90:
            break
        console.print(f"[red]Value must be between {int(el_min)+1} and 90[/red]")

    draw_window(az_min, az_max, el_min, el_max)

    return {
        "azimuth_min":  az_min,
        "azimuth_max":  az_max,
        "altitude_min": el_min,
        "altitude_max": el_max,
        "note": "Azimuth: 0=N, 90=E, 180=S, 270=W. Altitude: 0=horizon, 90=zenith. "
                "Set altitude_min=0 and azimuth 0-360 for open horizon.",
    }


def ask_observation() -> dict:
    console.print(Panel(
        "[bold]Step 3 — Observation parameters[/bold]\n\n"
        "These are runtime defaults (overridable interactively each run).\n\n"
        "[bold yellow]Magnitude limit guide:[/bold yellow]\n"
        "  • Naked eye (dark sky):  ~6.5\n"
        "  • 10×50 binoculars:      ~9.0\n"
        "  • 80 mm refractor:       ~11.0\n"
        "  • 200 mm reflector:      ~13.0\n"
        "  • CCD / imaging:         ~18.0+",
        border_style="magenta",
    ))

    step    = FloatPrompt.ask("Orbit sampling step (minutes)", default=15)
    mag_lim = FloatPrompt.ask("Default magnitude limit", default=11.0)

    return {
        "step_minutes":    int(step),
        "magnitude_limit": mag_lim,
    }


def main():
    console.print(Panel(
        "[bold cyan]🔭  Minor Planets Finder — Setup[/bold cyan]\n\n"
        "This wizard guides you through configuring\n"
        "your location and sky window, then saves config.json.",
        border_style="cyan",
        expand=False,
    ))

    config_path = Path("config.json")
    if config_path.exists():
        if not Confirm.ask(f"\n[yellow]config.json already exists. Overwrite?[/yellow]", default=False):
            console.print("Cancelled.")
            return

    loc        = ask_location()
    console.print()
    sky_window = ask_sky_window()
    console.print()
    obs        = ask_observation()

    config = {
        "location":    loc,
        "sky_window":  sky_window,
        "observation": obs,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    console.print(Panel(
        f"[bold green]✓  config.json saved![/bold green]\n\n"
        f"Now run:\n"
        f"  [bold cyan].venv\\Scripts\\python minorplanetsfinder.py[/bold cyan]\n\n"
        f"Or for a specific date:\n"
        f"  [bold cyan].venv\\Scripts\\python minorplanetsfinder.py --date 2026-03-20[/bold cyan]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
