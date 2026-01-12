#!/usr/bin/env python3
"""Art Crop System - Marketing Demo"""
import time
import sys

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.align import Align
    from rich import box
except ImportError:
    print("Run: pip install rich")
    sys.exit(1)

console = Console()

def pause(s=1.5):
    time.sleep(s)

def step(text):
    console.print(f"\n[bold white on #1a1a2e]  {text}  [/]\n")
    pause(0.8)

# INTRO
console.clear()
console.print()
intro = Panel(
    Align.center("[bold yellow]ART CROP SYSTEM[/]\n\n[white]AI-Powered Artwork Detection & Cropping[/]"),
    border_style="cyan",
    width=60,
    padding=(1, 2)
)
console.print(intro)
pause(2)

# STEP 1
step("STEP 1: UPLOAD PHOTO OF ARTWORK")

console.print("[dim]$[/] python art_crop.py [cyan]./photos/room_painting.jpg[/]\n")
pause(1)

console.print("  Loading image............", end="")
pause(0.5)
console.print(" [green]5472x3648 (20MP)[/]")

console.print("  Analyzing scene..........", end="")
pause(0.6)
console.print(" [green]Artwork on wall[/]")

console.print("  Checking lighting........", end="")
pause(0.4)
console.print(" [green]Good[/]")

pause(0.8)

photo = Panel(
    "[bold]room_painting.jpg[/]\n\n"
    "[dim]Resolution:[/]  5472 x 3648 pixels\n"
    "[dim]Scene:[/]       Living room with artwork\n"
    "[dim]Angle:[/]       [yellow]Slight perspective[/] (correctable)",
    title="[cyan]Image Loaded[/]",
    border_style="cyan",
    width=50
)
console.print(photo)
pause(1.5)

# STEP 2
step("STEP 2: TIER 1 - FAST BACKGROUND REMOVAL")

console.print("  Running rembg U2-Net.....", end="")
pause(0.8)
console.print(" [green]Done[/]")

console.print("  Detecting foreground.....", end="")
pause(0.5)
console.print(" [green]Done[/]")

console.print("  Evaluating quality.......", end="")
pause(0.5)
console.print(" [yellow]Partial[/]")

pause(0.5)
console.print("\n  [yellow]![/] Frame edges detected - escalating to Tier 2")
pause(1)

# STEP 3
step("STEP 3: TIER 2 - MULTI-AI CONSENSUS")

console.print("  Querying 4 AI models for precise boundaries...\n")
pause(0.8)

models = [
    ("GPT-4 Vision", "Top: 234, Left: 456, Bottom: 1890, Right: 2134"),
    ("Claude Vision", "Top: 231, Left: 452, Bottom: 1887, Right: 2130"),
    ("Gemini Pro", "Top: 236, Left: 458, Bottom: 1892, Right: 2138"),
    ("Grok Vision", "Top: 233, Left: 455, Bottom: 1889, Right: 2133"),
]

for name, coords in models:
    console.print(f"  {name}...", end="")
    pause(0.6)
    console.print(f" [green]OK[/]")

pause(0.8)

consensus = Panel(
    "[bold green]CONSENSUS REACHED[/]\n\n"
    "  Agreement:    [green]98.7%[/]\n"
    "  Deviation:    2.3 pixels\n"
    "  Crop Size:    1679 x 1655 px",
    title="[yellow]AI Consensus[/]",
    border_style="green",
    width=45
)
console.print(consensus)
pause(1.5)

# STEP 4
step("STEP 4: PERSPECTIVE CORRECTION")

corrections = [
    ("Perspective", "2.3 degrees"),
    ("Distortion", "0.8% corrected"),
    ("Rotation", "0.5 degrees"),
]

for name, detail in corrections:
    console.print(f"  {name}:", end="")
    pause(0.3)
    console.print(f" [green]{detail}[/]")

pause(1)

# STEP 5
step("STEP 5: QUALITY CHECK")

quality = Table(box=box.ROUNDED, width=50)
quality.add_column("Check", style="white")
quality.add_column("Status", justify="center")
quality.add_column("Score", justify="center")

quality.add_row("Sharpness", "[green]Pass[/]", "[green]94%[/]")
quality.add_row("Color Accuracy", "[green]Pass[/]", "[green]97%[/]")
quality.add_row("Edge Cleanliness", "[green]Pass[/]", "[green]92%[/]")
quality.add_row("No Frame Visible", "[green]Pass[/]", "[green]100%[/]")

console.print(quality)
pause(1.5)

# STEP 6
step("STEP 6: GENERATE 8 DETAIL CROPS")

crops = [
    "Full Artwork (1679x1655)",
    "Top-Left Corner (500x500)",
    "Top-Right Corner (500x500)",
    "Center Detail (600x600)",
    "Signature Close-up (400x200)",
    "Texture Sample (300x300)",
]

for crop in crops:
    console.print(f"  [green]>[/] {crop}")
    pause(0.15)

console.print("\n  [green]>[/] 8 total crops generated")
pause(1)

# STEP 7
step("STEP 7: EXPORT")

output = Panel(
    Align.center(
        "[bold green]CROP COMPLETE[/]\n\n"
        "[bold]Output:[/] ./output/artwork_*.jpg\n"
        "[bold]Files:[/] 8 images + report\n"
        "[bold]Time:[/] 4.2 seconds"
    ),
    title="[bold yellow]COMPLETE[/]",
    border_style="green",
    width=45
)
console.print(output)
pause(2)

# FOOTER
console.print()
footer = Panel(
    Align.center(
        "[dim]rembg + GPT-4V + Claude + Gemini + Grok[/]\n"
        "[bold cyan]github.com/jjshay/art-crop-system[/]"
    ),
    title="[dim]Art Crop System v2.0[/]",
    border_style="dim",
    width=50
)
console.print(footer)
pause(3)
