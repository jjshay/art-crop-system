#!/usr/bin/env python3
"""
🎨 GALLERY QUALITY CROP AUTOMATION
Simple, fast, NO background removal, perfect rotation
"""

import os
import json
import time
from pathlib import Path
from PIL import Image
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG_FILE = "gdrive_config.json"
PROCESSED_FILE = "processed_files.json"

# ============================================================
# LOAD CONFIG
# ============================================================

def load_config():
    """Load configuration from JSON file"""
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ Configuration file not found: {CONFIG_FILE}")
        print("   Please run the setup script first")
        exit(1)

    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    print(f"✅ Loaded configuration from {CONFIG_FILE}\n")
    return config

# ============================================================
# GOOGLE DRIVE SETUP
# ============================================================

def setup_drive():
    """Initialize Google Drive API"""
    print("🔄 Initializing Google Drive...")

    if not os.path.exists('oauth_credentials.json'):
        print("❌ OAuth credentials not found!")
        print("   Please run setup_oauth.py first")
        exit(1)

    print("📋 Found oauth_credentials.json - will use OAuth authentication")

    with open('oauth_credentials.json', 'r') as f:
        creds_data = json.load(f)

    creds = Credentials(
        token=creds_data['token'],
        refresh_token=creds_data.get('refresh_token'),
        token_uri=creds_data['token_uri'],
        client_id=creds_data['client_id'],
        client_secret=creds_data['client_secret'],
        scopes=creds_data['scopes']
    )

    service = build('drive', 'v3', credentials=creds)
    print("✅ Connected to Google Drive")
    return service

# ============================================================
# IMAGE PROCESSING - GALLERY QUALITY
# ============================================================

def detect_orientation(image: Image.Image) -> str:
    """Simple orientation detection based on dimensions"""
    width, height = image.size

    if height > width:
        return "PORTRAIT"
    else:
        return "LANDSCAPE"

def apply_simple_rotation(image: Image.Image) -> Image.Image:
    """
    Simple rotation logic:
    - Portrait images: Rotate 180° (they come upside down from camera)
    - Landscape images: No rotation needed
    """
    orientation = detect_orientation(image)

    if orientation == "PORTRAIT":
        print(f"   📐 Portrait detected - rotating 180°")
        return image.rotate(180, expand=True)
    else:
        print(f"   📐 Landscape detected - no rotation needed")
        return image

def create_gallery_crops(image_path: str, output_folder: Path):
    """
    Create GALLERY QUALITY crops:
    - NO background removal
    - High quality LANCZOS resampling
    - 4:3 aspect ratio for main crops
    - Square corners for signatures/edition
    """
    print(f"\n🎨 Creating gallery quality crops...")

    # Load original image
    img = Image.open(image_path).convert('RGB')
    print(f"   📐 Original: {img.size[0]}x{img.size[1]}")

    # Apply simple rotation
    img = apply_simple_rotation(img)
    print(f"   📐 After rotation: {img.size[0]}x{img.size[1]}")

    width, height = img.size
    output_folder.mkdir(parents=True, exist_ok=True)

    # Calculate square corner size (20% of shortest dimension)
    corner_size = int(min(width, height) * 0.20)

    crops = {}

    # ============================================================
    # 1. THUMBNAIL (400x400 square)
    # ============================================================
    print("\n   🔲 Creating thumbnail...")
    thumb = img.copy()
    thumb.thumbnail((400, 400), Image.Resampling.LANCZOS)

    # Center on white background
    canvas = Image.new('RGB', (400, 400), 'white')
    offset = ((400 - thumb.width) // 2, (400 - thumb.height) // 2)
    canvas.paste(thumb, offset)

    thumb_path = output_folder / "0_THUMBNAIL.png"
    canvas.save(thumb_path, 'PNG', quality=95)
    crops['0_THUMBNAIL'] = thumb_path
    print(f"      ✅ Saved: 400x400")

    # ============================================================
    # 2. MAIN PORTRAIT (4:3 aspect ratio - 1600x1200)
    # ============================================================
    print("\n   🔲 Creating main portrait (4:3 ratio)...")

    # Center crop to 4:3
    target_ratio = 4/3
    current_ratio = width / height

    if current_ratio > target_ratio:
        # Too wide - crop width
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        crop_box = (left, 0, left + new_width, height)
    else:
        # Too tall - crop height
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        crop_box = (0, top, width, top + new_height)

    main = img.crop(crop_box)
    main = main.resize((1600, 1200), Image.Resampling.LANCZOS)

    main_path = output_folder / "2_MAIN_PORTRAIT.png"
    main.save(main_path, 'PNG', quality=95)
    crops['2_MAIN_PORTRAIT'] = main_path
    print(f"      ✅ Saved: 1600x1200 (4:3 ratio)")

    # ============================================================
    # 3. BOTTOM LEFT - SIGNATURE (square)
    # ============================================================
    print("\n   🔲 Creating bottom left signature crop...")
    crop_region = (0, height - corner_size, corner_size, height)
    bl = img.crop(crop_region)
    bl = bl.resize((1600, 1600), Image.Resampling.LANCZOS)

    bl_path = output_folder / "3_BOTTOM_LEFT_SIGNATURE.png"
    bl.save(bl_path, 'PNG', quality=95)
    crops['3_BOTTOM_LEFT_SIGNATURE'] = bl_path
    print(f"      ✅ Saved: 1600x1600 (signature & date)")

    # ============================================================
    # 4. BOTTOM RIGHT - EDITION NUMBER (square)
    # ============================================================
    print("\n   🔲 Creating bottom right edition crop...")
    crop_region = (width - corner_size, height - corner_size, width, height)
    br = img.crop(crop_region)
    br = br.resize((1600, 1600), Image.Resampling.LANCZOS)

    br_path = output_folder / "4_BOTTOM_RIGHT_EDITION.png"
    br.save(br_path, 'PNG', quality=95)
    crops['4_BOTTOM_RIGHT_EDITION'] = br_path
    print(f"      ✅ Saved: 1600x1600 (edition #)")

    # ============================================================
    # 5. TOP LEFT - CONDITION CHECK (square)
    # ============================================================
    print("\n   🔲 Creating top left corner...")
    crop_region = (0, 0, corner_size, corner_size)
    tl = img.crop(crop_region)
    tl = tl.resize((1600, 1600), Image.Resampling.LANCZOS)

    tl_path = output_folder / "7_TOP_LEFT_TEXTURE.png"
    tl.save(tl_path, 'PNG', quality=95)
    crops['7_TOP_LEFT_TEXTURE'] = tl_path
    print(f"      ✅ Saved: 1600x1600 (condition check)")

    # ============================================================
    # 6. TOP RIGHT - CONDITION CHECK (square)
    # ============================================================
    print("\n   🔲 Creating top right corner...")
    crop_region = (width - corner_size, 0, width, corner_size)
    tr = img.crop(crop_region)
    tr = tr.resize((1600, 1600), Image.Resampling.LANCZOS)

    tr_path = output_folder / "8_TOP_RIGHT_TEXTURE.png"
    tr.save(tr_path, 'PNG', quality=95)
    crops['8_TOP_RIGHT_TEXTURE'] = tr_path
    print(f"      ✅ Saved: 1600x1600 (condition check)")

    # ============================================================
    # 7. CENTER DETAIL (4:3 ratio from center)
    # ============================================================
    print("\n   🔲 Creating center detail crop...")

    # Center crop 50% of image
    center_width = width // 2
    center_height = height // 2
    left = (width - center_width) // 2
    top = (height - center_height) // 2

    center = img.crop((left, top, left + center_width, top + center_height))
    center = center.resize((1600, 1200), Image.Resampling.LANCZOS)

    center_path = output_folder / "5_DETAILED_CENTER.png"
    center.save(center_path, 'PNG', quality=95)
    crops['5_DETAILED_CENTER'] = center_path
    print(f"      ✅ Saved: 1600x1200 (center detail)")

    # ============================================================
    # 8. TIGHT CROP (square, 90% of art)
    # ============================================================
    print("\n   🔲 Creating tight crop...")

    # Crop to 90% of shortest dimension
    crop_dim = int(min(width, height) * 0.90)
    left = (width - crop_dim) // 2
    top = (height - crop_dim) // 2

    tight = img.crop((left, top, left + crop_dim, top + crop_dim))
    tight = tight.resize((1600, 1600), Image.Resampling.LANCZOS)

    tight_path = output_folder / "1_TIGHT_CROP.png"
    tight.save(tight_path, 'PNG', quality=95)
    crops['1_TIGHT_CROP'] = tight_path
    print(f"      ✅ Saved: 1600x1600 (tight crop)")

    # ============================================================
    # 9. AI TEXTURE/SUBJECT (4:3 ratio from upper center)
    # ============================================================
    print("\n   🔲 Creating AI texture crop...")

    # Upper third of image
    ai_height = height // 3
    ai_width = int(ai_height * 4/3)
    left = (width - ai_width) // 2

    ai = img.crop((left, 0, left + ai_width, ai_height))
    ai = ai.resize((1600, 1200), Image.Resampling.LANCZOS)

    ai_path = output_folder / "6_AI_TEXTURE_SUBJECT.png"
    ai.save(ai_path, 'PNG', quality=95)
    crops['6_AI_TEXTURE_SUBJECT'] = ai_path
    print(f"      ✅ Saved: 1600x1200 (AI texture)")

    print(f"\n✅ Created {len(crops)} gallery-quality crops")
    return crops

# ============================================================
# GOOGLE DRIVE UPLOAD
# ============================================================

def upload_crops_to_drive(service, crops: dict, output_folder_id: str, image_number: int):
    """Upload crops to numbered subfolder"""
    print(f"\n⬆️  Uploading to Google Drive...")

    # Create subfolder
    folder_name = f"{image_number}_IMG"
    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [output_folder_id]
    }

    folder = service.files().create(body=folder_metadata, fields='id').execute()
    folder_id = folder['id']
    print(f"   📁 Created folder: {folder_name}")

    # Upload each crop
    for crop_name, crop_path in sorted(crops.items()):
        filename = f"{image_number}_{crop_name}.png"

        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }

        media = MediaFileUpload(str(crop_path), mimetype='image/png')

        try:
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, webViewLink'
            ).execute()

            print(f"   ✅ Uploaded: {filename}")
        except Exception as e:
            print(f"   ❌ Error uploading {filename}: {e}")

    print(f"✅ All crops uploaded to folder: {folder_name}\n")

# ============================================================
# PROCESSED FILES TRACKING
# ============================================================

def load_processed_files():
    """Load list of processed file IDs"""
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_files(processed):
    """Save list of processed file IDs"""
    with open(PROCESSED_FILE, 'w') as f:
        json.dump(list(processed), f, indent=2)

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print("=" * 60)
    print("🎨 GALLERY QUALITY CROP AUTOMATION")
    print("   - NO background removal")
    print("   - Simple rotation (portrait=180°, landscape=0°)")
    print("   - Gallery quality crops")
    print("=" * 60)
    print()

    # Load config
    config = load_config()

    input_folder_id = config['input_folder_id']
    output_folder_id = config['output_folder_id']
    watch_mode = config.get('watch_mode', False)

    print(f"📋 Configuration:")
    print(f"   Input folder ID: {input_folder_id}")
    print(f"   Output folder ID: {output_folder_id}")
    print(f"   Watch mode: {watch_mode}")
    print()

    # Setup Google Drive
    service = setup_drive()

    # Load processed files
    processed_files = load_processed_files()
    print(f"📋 Loaded {len(processed_files)} previously processed files\n")

    # Get next image number
    image_counter = len(processed_files) + 1

    check_count = 0

    while True:
        check_count += 1
        print("=" * 60)
        print(f"🔍 Check #{check_count} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # List files in input folder
        query = f"'{input_folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType)"
        ).execute()

        files = results.get('files', [])

        # Filter for new image files
        new_images = [
            f for f in files
            if f['id'] not in processed_files
            and f['mimeType'].startswith('image/')
        ]

        if not new_images:
            if watch_mode:
                print("😴 No new images. Waiting 30 seconds...\n")
                time.sleep(30)
                continue
            else:
                print("✅ No new images to process. Exiting.\n")
                break

        print(f"📸 Found {len(new_images)} new image(s) to process\n")

        for img_file in new_images:
            file_id = img_file['id']
            filename = img_file['name']

            print(f"⬇️  Downloading: {filename}")

            # Download file
            request = service.files().get_media(fileId=file_id)
            temp_path = Path('/tmp') / filename

            with open(temp_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()

            # Process image
            output_folder = Path('processed_crops') / filename.replace('.', '')
            crops = create_gallery_crops(str(temp_path), output_folder)

            # Upload to Drive
            upload_crops_to_drive(service, crops, output_folder_id, image_counter)

            # Mark as processed
            processed_files.add(file_id)
            save_processed_files(processed_files)

            # Increment counter
            image_counter += 1

            # Cleanup
            temp_path.unlink()
            print(f"🗑️  Cleaned up temporary file\n")

        if not watch_mode:
            break

        print("😴 Waiting 30 seconds before next check...\n")
        time.sleep(30)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopped by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
