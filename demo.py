#!/usr/bin/env python3
"""
Art Crop System Demo
Demonstrates artwork detection and cropping with rich visual output.

Run: python demo.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    print("Installing Pillow...")
    os.system("pip install Pillow")
    from PIL import Image, ImageDraw, ImageFilter

# Try to import rich for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.columns import Columns
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


def print_header(text: str) -> None:
    """Print a formatted section header."""
    if RICH_AVAILABLE:
        console.print()
        console.rule(f"[bold gold1]{text}[/bold gold1]", style="gold1")
        console.print()
    else:
        print(f"\n{'='*60}")
        print(f" {text}")
        print(f"{'='*60}\n")


def show_banner() -> None:
    """Display the application banner."""
    if RICH_AVAILABLE:
        banner = """
[bold cyan]╔════════════════════════════════════════════════════════════════╗
║[/bold cyan] [bold gold1]    _         _      ____                                     [/bold gold1][bold cyan]║
║[/bold cyan] [bold gold1]   / \   _ __| |_   / ___|_ __ ___  _ __                      [/bold gold1][bold cyan]║
║[/bold cyan] [bold gold1]  / _ \ | '__| __| | |   | '__/ _ \| '_ \                     [/bold gold1][bold cyan]║
║[/bold cyan] [bold gold1] / ___ \| |  | |_  | |___| | | (_) | |_) |                    [/bold gold1][bold cyan]║
║[/bold cyan] [bold gold1]/_/   \_\_|   \__|  \____|_|  \___/| .__/                     [/bold gold1][bold cyan]║
║[/bold cyan] [bold gold1]                                   |_|       [bold white]System[/bold white]           [/bold gold1][bold cyan]║
║[/bold cyan]                                                                [bold cyan]║
║[/bold cyan]         [bold white]AI-Powered Artwork Detection & Detail Extraction[/bold white]      [bold cyan]║
╚════════════════════════════════════════════════════════════════╝[/bold cyan]
"""
        console.print(banner)
    else:
        print("\n" + "="*60)
        print("  ART CROP SYSTEM - AI-Powered Artwork Detection")
        print("="*60 + "\n")


def create_sample_photo() -> Tuple[Image.Image, Image.Image, Tuple[int, int, Tuple[int, int]]]:
    """Create a simulated photo of artwork on a wall."""
    if RICH_AVAILABLE:
        console.print("[dim]Creating simulated artwork photo...[/dim]")
    else:
        print("Creating simulated artwork photo...")

    # Create a "wall" background
    wall = Image.new('RGB', (1200, 900), '#E8E4E0')
    draw = ImageDraw.Draw(wall)

    # Add wall texture
    for i in range(0, 900, 50):
        draw.line([(0, i), (1200, i)], fill='#E0DCD8', width=1)

    # Create the "artwork" (abstract art)
    artwork_size = (500, 650)
    artwork = Image.new('RGB', artwork_size, '#FAFAFA')
    art_draw = ImageDraw.Draw(artwork)

    # Add abstract shapes to artwork
    art_draw.ellipse([50, 100, 250, 300], fill='#E74C3C')
    art_draw.rectangle([200, 250, 450, 550], fill='#3498DB')
    art_draw.polygon([(250, 50), (400, 200), (150, 250)], fill='#F1C40F')

    # Add signature area
    art_draw.text((380, 600), "Artist '24", fill='#2C3E50')

    # Add frame around artwork
    frame_width = 30
    framed_size = (artwork_size[0] + frame_width*2, artwork_size[1] + frame_width*2)
    framed = Image.new('RGB', framed_size, '#1a1a1a')
    framed.paste(artwork, (frame_width, frame_width))

    # Add mat
    mat_width = 40
    with_mat = Image.new('RGB', (framed_size[0] + mat_width*2, framed_size[1] + mat_width*2), '#FFFFFF')
    with_mat.paste(framed, (mat_width, mat_width))

    # Outer frame
    outer_frame = 20
    final_frame = Image.new('RGB', (with_mat.size[0] + outer_frame*2, with_mat.size[1] + outer_frame*2), '#2C2C2C')
    final_frame.paste(with_mat, (outer_frame, outer_frame))

    # Place on wall (centered, slightly up)
    x = (1200 - final_frame.size[0]) // 2
    y = (900 - final_frame.size[1]) // 2 - 50
    wall.paste(final_frame, (x, y))

    if RICH_AVAILABLE:
        table = Table(box=box.SIMPLE)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="gold1")
        table.add_row("Wall photo size", f"{wall.size[0]} x {wall.size[1]} px")
        table.add_row("Artwork location", f"({x}, {y})")
        table.add_row("Frame size", f"{final_frame.size[0]} x {final_frame.size[1]} px")
        console.print(table)
    else:
        print(f"   Wall photo size: {wall.size}")
        print(f"   Artwork location: ({x}, {y})")

    return wall, artwork, (x, y, final_frame.size)


def simulate_detection(wall_photo: Image.Image, actual_bounds: Tuple[int, int, Tuple[int, int]]) -> Dict[str, Any]:
    """Simulate AI detection of artwork boundaries."""
    print_header("TIER 1: BACKGROUND REMOVAL (rembg)")

    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Detecting foreground objects...", total=100)
            for i in range(100):
                time.sleep(0.01)
                progress.update(task, advance=1)

    x, y, size = actual_bounds
    detected = {
        'x': x + 5,
        'y': y - 3,
        'width': size[0] - 10,
        'height': size[1] + 5,
        'confidence': 0.87
    }

    if RICH_AVAILABLE:
        result_panel = Panel(
            f"""[bold]Detection Results[/bold]

[cyan]Bounds:[/cyan] ({detected['x']}, {detected['y']}) {detected['width']}x{detected['height']}
[cyan]Confidence:[/cyan] [{'green' if detected['confidence'] > 0.8 else 'yellow'}]{detected['confidence']:.0%}[/]
[cyan]Objects found:[/cyan] 1 rectangular object (likely framed artwork)""",
            title="🔍 rembg Analysis",
            border_style="cyan",
            box=box.ROUNDED
        )
        console.print(result_panel)
    else:
        print(f"   Initial bounds: ({detected['x']}, {detected['y']}) {detected['width']}x{detected['height']}")
        print(f"   Confidence: {detected['confidence']:.0%}")

    return detected


def simulate_ai_analysis(detected_bounds: Dict[str, Any], actual_bounds: Tuple[int, int, Tuple[int, int]]) -> Dict[str, Any]:
    """Simulate multi-AI consensus analysis."""
    print_header("TIER 2: MULTI-AI VISION ANALYSIS")

    x, y, size = actual_bounds

    ai_results = {
        'GPT-4V': {'bounds': (x + 2, y + 1, size[0] - 4, size[1] - 2), 'confidence': 0.94, 'color': 'green'},
        'Claude': {'bounds': (x + 1, y + 2, size[0] - 2, size[1] - 3), 'confidence': 0.92, 'color': 'magenta'},
        'Gemini': {'bounds': (x + 3, y + 1, size[0] - 5, size[1] - 2), 'confidence': 0.91, 'color': 'blue'},
        'Grok': {'bounds': (x + 2, y + 2, size[0] - 3, size[1] - 3), 'confidence': 0.89, 'color': 'red'},
    }

    if RICH_AVAILABLE:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            for model in ai_results:
                task = progress.add_task(f"[cyan]Analyzing with {model}...", total=None)
                time.sleep(0.3)
                progress.remove_task(task)

        table = Table(title="🤖 AI Vision Analysis Results", box=box.ROUNDED)
        table.add_column("AI Model", style="cyan")
        table.add_column("Confidence", justify="center")
        table.add_column("Signature", justify="center")
        table.add_column("Notes", style="dim")

        notes = {
            'GPT-4V': 'Abstract art, black frame, white mat',
            'Claude': 'Contemporary abstract, well-lit',
            'Gemini': 'Geometric shapes, primary colors',
            'Grok': 'Museum-quality framing detected',
        }

        for model, result in ai_results.items():
            conf = result['confidence']
            conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
            table.add_row(
                model,
                f"[{result['color']}]{conf_bar}[/{result['color']}] {conf:.0%}",
                "[green]✓ Found[/green]",
                notes[model]
            )

        console.print(table)

        avg_confidence = sum(r['confidence'] for r in ai_results.values()) / len(ai_results)
        consensus_panel = Panel(
            f"""[bold green]✓ CONSENSUS REACHED[/bold green]

Average confidence: [bold]{avg_confidence:.0%}[/bold]
All AIs agree on signature location: [green]YES[/green]
Recommended crop boundaries: [cyan]({x}, {y}) {size[0]}x{size[1]}[/cyan]""",
            title="📊 Consensus",
            border_style="green",
            box=box.ROUNDED
        )
        console.print(consensus_panel)
    else:
        for model, result in ai_results.items():
            print(f"   {model}: {result['confidence']:.0%} confidence")
        print("\n   Consensus reached: YES")

    return {
        'x': x, 'y': y, 'width': size[0], 'height': size[1],
        'confidence': sum(r['confidence'] for r in ai_results.values()) / len(ai_results),
        'signature_area': (380, 600, 120, 30)
    }


def simulate_quality_check(bounds: Dict[str, Any]) -> bool:
    """Simulate quality validation."""
    print_header("TIER 4: QUALITY VALIDATION")

    checks = [
        ('Blur Detection', 'PASS', 'green', 'Image is sharp (score: 0.92)'),
        ('Lighting', 'PASS', 'green', 'Even lighting, no harsh shadows'),
        ('Angle', 'PASS', 'green', 'Perpendicular (< 2° deviation)'),
        ('Color Balance', 'PASS', 'green', 'White balance correct'),
        ('Resolution', 'PASS', 'green', 'Sufficient for detail crops (> 500px)'),
    ]

    if RICH_AVAILABLE:
        table = Table(title="✅ Quality Checks", box=box.ROUNDED)
        table.add_column("Check", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Details", style="dim")

        for check, status, color, detail in checks:
            table.add_row(check, f"[{color}]{status}[/{color}]", detail)

        console.print(table)
    else:
        for check, status, _, detail in checks:
            print(f"   [{status}] {check}: {detail}")

    return True


def generate_detail_crops(artwork: Image.Image) -> Dict[str, Image.Image]:
    """Generate 8 detail crop shots."""
    print_header("GENERATING DETAIL SHOTS")

    w, h = artwork.size

    crops_config = {
        'full': (0, 0, w, h),
        'top_left': (0, 0, w//2, h//2),
        'top_right': (w//2, 0, w, h//2),
        'bottom_left': (0, h//2, w//2, h),
        'bottom_right': (w//2, h//2, w, h),
        'center': (w//4, h//4, 3*w//4, 3*h//4),
        'signature': (int(w*0.6), int(h*0.85), w, h),
        'texture': (int(w*0.3), int(h*0.3), int(w*0.7), int(h*0.6)),
    }

    results = {}

    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[gold1]Generating crops...", total=len(crops_config))
            for name, bounds in crops_config.items():
                crop = artwork.crop(bounds)
                results[name] = crop
                time.sleep(0.1)
                progress.update(task, advance=1, description=f"[gold1]Cropped: {name}")

        table = Table(title="🖼️ Generated Crops", box=box.ROUNDED)
        table.add_column("Crop Name", style="cyan")
        table.add_column("Dimensions", justify="center", style="gold1")
        table.add_column("Purpose", style="dim")

        purposes = {
            'full': 'Complete artwork',
            'top_left': 'Corner detail',
            'top_right': 'Corner detail',
            'bottom_left': 'Corner detail',
            'bottom_right': 'Corner detail',
            'center': 'Central composition',
            'signature': 'Artist signature',
            'texture': 'Surface texture',
        }

        for name, crop in results.items():
            table.add_row(name, f"{crop.size[0]} x {crop.size[1]} px", purposes[name])

        console.print(table)
    else:
        for name, bounds in crops_config.items():
            crop = artwork.crop(bounds)
            results[name] = crop
            print(f"   {name:15s} : {crop.size[0]:4d} x {crop.size[1]:4d} px")

    return results


def save_outputs(wall: Image.Image, artwork: Image.Image, crops: Dict[str, Image.Image]) -> Path:
    """Save all outputs."""
    print_header("SAVING OUTPUT FILES")

    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    files_saved = []

    wall.save(output_dir / "original_photo.jpg", quality=95)
    files_saved.append(("original_photo.jpg", "Input photo", wall.size))

    artwork.save(output_dir / "extracted_artwork.jpg", quality=95)
    files_saved.append(("extracted_artwork.jpg", "Clean extraction", artwork.size))

    for name, crop in crops.items():
        filename = f"detail_{name}.jpg"
        crop.save(output_dir / filename, quality=95)
        files_saved.append((filename, f"Detail: {name}", crop.size))

    if RICH_AVAILABLE:
        table = Table(title="💾 Saved Files", box=box.ROUNDED)
        table.add_column("Filename", style="cyan")
        table.add_column("Description", style="dim")
        table.add_column("Size", justify="right", style="gold1")

        for filename, desc, size in files_saved:
            table.add_row(filename, desc, f"{size[0]}x{size[1]}")

        console.print(table)
        console.print(f"\n[bold green]✓[/bold green] All files saved to: [cyan]{output_dir}/[/cyan]")
    else:
        for filename, desc, _ in files_saved:
            print(f"   Saved: {filename}")

    return output_dir


def main() -> None:
    """Main entry point for the demo."""
    show_banner()

    if RICH_AVAILABLE:
        console.print("[dim]This demo shows how the multi-tier crop system works")
        console.print("without requiring any API keys.[/dim]\n")
    else:
        print("This demo shows how the multi-tier crop system works")
        print("without requiring any API keys.\n")

    # Create sample photo
    print_header("CREATING SAMPLE ARTWORK PHOTO")
    wall_photo, artwork, bounds = create_sample_photo()

    # Tier 1: Background removal
    detected = simulate_detection(wall_photo, bounds)

    # Tier 2: AI Analysis
    consensus = simulate_ai_analysis(detected, bounds)

    # Skip Tier 3 (CV fallback) - not needed with good consensus
    print_header("TIER 3: CV EDGE DETECTION")
    if RICH_AVAILABLE:
        console.print("[dim]Skipped - AI consensus confidence > 90%[/dim]")
    else:
        print("   Skipped - AI consensus confidence > 90%")

    # Tier 4: Quality check
    simulate_quality_check(consensus)

    # Generate crops
    crops = generate_detail_crops(artwork)

    # Save outputs
    output_dir = save_outputs(wall_photo, artwork, crops)

    # Summary
    print_header("SUMMARY")

    if RICH_AVAILABLE:
        summary = f"""
[cyan]Input:[/cyan]  1 photo of artwork on wall
[cyan]Output:[/cyan] 1 extracted artwork + 8 detail crops
[cyan]Location:[/cyan] {output_dir}/

[bold]Files generated:[/bold]
  • original_photo.jpg [dim](simulated input)[/dim]
  • extracted_artwork.jpg [dim](clean crop)[/dim]
  • detail_*.jpg [dim](8 detail shots)[/dim]

[bold gold1]Next Steps:[/bold gold1]
  1. Install rembg: [cyan]pip install rembg[/cyan]
  2. Add API keys to .env file
  3. Run: [cyan]python ai_art_crop_system.py[/cyan]
"""
        console.print(Panel(summary, title="📋 Results", border_style="cyan", box=box.ROUNDED))
    else:
        print(f"Input: 1 photo of artwork on wall")
        print(f"Output: 1 extracted artwork + 8 detail crops")
        print(f"Location: {output_dir}/")

    print_header("DEMO COMPLETE")


if __name__ == "__main__":
    main()
