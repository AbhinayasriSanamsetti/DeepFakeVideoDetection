import os
import random
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path

def add_noise(image, noise_factor=0.1):
    """Add random noise to an image."""
    image = np.array(image) / 255.0  # Normalize to [0,1]
    noise = np.random.normal(0, noise_factor, image.shape)  # Gaussian noise
    noisy_image = np.clip(image + noise, 0, 1)  # Clip to valid [0,1] range
    noisy_image = (noisy_image * 255).astype(np.uint8)  # Convert back to uint8
    return Image.fromarray(noisy_image)

def augment_image(image):
    """Apply augmentations such as brightness, contrast, and flipping."""
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))  # Random brightness
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))  # Random contrast
    
    # Random horizontal flip
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    return image

def process_and_augment_frames(labels_file, frames_labels_folder):
    """
    Process frames by augmenting and adding noise to real video frames.
    Saves augmented frames in `frames_labels/augmented_frames/` 
    and stores metadata in `labels_aug.txt` without modifying `labels.txt`.
    """
    augmented_entries = []
    labels_aug_file = os.path.join(frames_labels_folder, "labels_aug.txt")  

    # Ensure the labels file exists
    if not os.path.exists(labels_file):
        print(f"❌ Error: Labels file {labels_file} not found!")
        return

    # Read labels file
    with open(labels_file, 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:  # Skip header
        parts = line.strip().split()
        
        # Ensure the line contains enough parts
        if len(parts) < 6:
            print(f"⚠️ Skipping malformed entry: {line.strip()}")
            continue
        
        try:
            # Extract image path (it might contain spaces, so join until the label)
            image_path = ' '.join(parts[:-5])  # All but the last 5 elements
            label = int(parts[-5])             # 5th last element = label
            x1, y1, x2, y2 = map(int, parts[-4:])  # Last 4 elements = bounding box
            
            if not os.path.exists(image_path):
                print(f"⚠️ Warning: Image file {image_path} not found, skipping...")
                continue

            # Load image
            image = Image.open(image_path)
            
            # Augment image
            augmented_image = augment_image(image)
            
            # Add noise
            noisy_image = add_noise(augmented_image)
            
            # Get video name from the image path
            video_name = Path(image_path).parent.name  
            
            # Create directory for augmented frames
            augmented_folder = os.path.join(frames_labels_folder, "augmented_frames", video_name)
            os.makedirs(augmented_folder, exist_ok=True)

            # Create augmented filename
            augmented_frame_filename = f"{Path(image_path).stem}_augmented_noisy{Path(image_path).suffix}"
            augmented_frame_path = os.path.join(augmented_folder, augmented_frame_filename)
            
            # Save augmented frame
            noisy_image.save(augmented_frame_path)

            # Store augmented frame entry for labels_aug.txt
            augmented_entries.append(f"{augmented_frame_path} {label} {x1} {y1} {x2} {y2}\n")

            # Print progress
            print(f"✅ Saved: {augmented_frame_filename} -> {augmented_folder}")

        except ValueError as e:
            print(f"❌ Error parsing line: {line.strip()} -> {e}")
        except Exception as e:
            print(f"❌ Unexpected error processing {image_path}: {e}")

    # Write augmented frame details to labels_aug.txt
    if augmented_entries:
        try:
            with open(labels_aug_file, 'w') as f:  # Overwrite or create new file
                f.writelines(augmented_entries)
            print(f"✅ Augmented frame entries saved to {labels_aug_file}")
        except Exception as e:
            print(f"❌ Error writing to {labels_aug_file}: {e}")
    else:
        print("⚠️ No augmented frames to save.")

# Paths
labels_file = "C:/Users/HP/Desktop/deepfake/frames_labels/labels.txt"
frames_labels_folder = "C:/Users/HP/Desktop/deepfake/frames_labels"

# Process and augment frames
process_and_augment_frames(labels_file, frames_labels_folder)
