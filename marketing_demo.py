#!/usr/bin/env python3
"""Marketing Demo - Art Crop System"""
import time
import sys

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    console = Console()
except ImportError:
    print("Run: pip install rich")
    sys.exit(1)

def pause(seconds=2):
    time.sleep(seconds)

def clear():
    console.clear()

# SCENE 1: Hook
clear()
console.print("\n" * 5)
console.print("[bold yellow]           YOUR ARTWORK PHOTOS LOOK AMATEUR?[/bold yellow]", justify="center")
pause(2)

# SCENE 2: Problem
clear()
console.print("\n" * 3)
console.print(Panel("""
[bold red]WHAT BUYERS SEE:[/bold red]

   • Wall visible in background
   • Frame edges cut off
   • Shadows and reflections
   • No detail shots

[dim]Amateur photos = fewer sales.[/dim]
""", title="❌ Bad Product Photos Kill Sales", border_style="red", width=60), justify="center")
pause(3)

# SCENE 3: Solution
clear()
console.print("\n" * 3)
console.print(Panel("""
[bold green]DROP A PHOTO. GET 8 PERFECT CROPS.[/bold green]

   ✓ AI detects artwork boundaries
   ✓ Removes background perfectly
   ✓ Generates corner details
   ✓ Captures signature close-up

[bold]Pro-quality images in seconds.[/bold]
""", title="✅ Art Crop System", border_style="green", width=60), justify="center")
pause(3)

# SCENE 4: Before/After
clear()
console.print("\n\n")
console.print("[bold cyan]              🖼️  BEFORE → AFTER[/bold cyan]", justify="center")
console.print()
pause(1)

console.print(Panel("""
[red]BEFORE:[/red]
┌─────────────────────────────┐
│  ~wall~   ┌───────┐  ~wall~ │
│           │ ART   │         │
│  ~shadow~ │       │ ~glare~ │
│           └───────┘         │
│         ~floor~             │
└─────────────────────────────┘
""", border_style="red", width=35), justify="center")
pause(2)

console.print(Panel("""
[green]AFTER:[/green]
┌─────────────────┐
│                 │
│    PERFECT      │
│    ARTWORK      │
│    ONLY         │
│                 │
└─────────────────┘
+ 8 detail crops!
""", border_style="green", width=35), justify="center")
pause(2)

# SCENE 5: Output
clear()
console.print("\n\n")
console.print("[bold magenta]              📁 YOUR 8 DETAIL SHOTS[/bold magenta]", justify="center")
console.print()

crops = [
    ("full.jpg", "Complete artwork", "1000x1200"),
    ("top_left.jpg", "Corner detail", "500x600"),
    ("top_right.jpg", "Corner detail", "500x600"),
    ("center.jpg", "Central focus", "500x600"),
    ("signature.jpg", "Artist signature", "300x100"),
    ("texture.jpg", "Surface detail", "400x400"),
    ("bottom_left.jpg", "Corner detail", "500x600"),
    ("bottom_right.jpg", "Corner detail", "500x600"),
]

table = Table(box=box.ROUNDED, width=55)
table.add_column("File", style="cyan")
table.add_column("Content", style="white")
table.add_column("Size", style="dim")

for f, content, size in crops:
    table.add_row(f, content, size)

console.print(table, justify="center")
pause(3)

# SCENE 6: CTA
clear()
console.print("\n" * 4)
console.print("[bold yellow]           ⭐ PRO PHOTOS IN SECONDS ⭐[/bold yellow]", justify="center")
console.print()
console.print("[bold white]            github.com/jjshay/art-crop-system[/bold white]", justify="center")
console.print()
console.print("[dim]                      python demo.py[/dim]", justify="center")
pause(3)
