import os
import glob
from PIL import Image
import statistics

def check_image_resolutions(directory_path):
    """
    Checks all JPEG images in a directory and reports their resolutions and file sizes
    """
    # Find all JPEG images in the directory
    image_paths = glob.glob(os.path.join(directory_path, "*.jpeg"))
    
    if not image_paths:
        print(f"No JPEG images found in {directory_path}")
        return
        
    print(f"Found {len(image_paths)} JPEG images in {directory_path}")
    
    # Collect statistics
    widths = []
    heights = []
    sizes_kb = []
    
    # Sample the first 10 images (or fewer if less are available)
    sample_size = min(10, len(image_paths))
    sample_images = sorted(image_paths)[:sample_size]
    
    print("\nSample Image Details:")
    print("-" * 60)
    print(f"{'Filename':<15} {'Resolution':<15} {'Size (KB)':<10} {'Aspect Ratio':<12}")
    print("-" * 60)
    
    for img_path in sample_images:
        filename = os.path.basename(img_path)
        # Get file size in KB
        file_size = os.path.getsize(img_path) / 1024
        
        # Open image and get dimensions
        with Image.open(img_path) as img:
            width, height = img.size
            aspect_ratio = width / height
            
        print(f"{filename:<15} {width}x{height:<15} {file_size:.2f}KB{'':<4} {aspect_ratio:.2f}")
        
        widths.append(width)
        heights.append(height)
        sizes_kb.append(file_size)
    
    # Calculate averages
    avg_width = statistics.mean(widths)
    avg_height = statistics.mean(heights)
    avg_size = statistics.mean(sizes_kb)
    
    # Calculate estimated uncompressed size in memory (RGB, 3 bytes per pixel)
    avg_uncompressed_kb = (avg_width * avg_height * 3) / 1024
    
    print("\nSummary Statistics:")
    print(f"Average Resolution: {avg_width:.1f}x{avg_height:.1f} pixels")
    print(f"Average File Size: {avg_size:.2f}KB (compressed JPEG)")
    print(f"Est. Uncompressed Size: {avg_uncompressed_kb:.2f}KB (~{avg_uncompressed_kb/1024:.2f}MB) in memory as RGB")
    
    print("\nEstimated State Size:")
    bbox_pct = 0.15  # Assumption: average bounding box is 15% of image area
    objects_per_frame = 5  # Assumption: average 5 objects per frame
    history_frames = 10  # For max_age=10
    
    bbox_kb = avg_uncompressed_kb * bbox_pct
    frame_state_kb = (bbox_kb * objects_per_frame * history_frames)
    print(f"Estimated state size with just bboxes: ~{frame_state_kb:.2f}KB (~{frame_state_kb/1024:.2f}MB)")
    
    full_frame_state_kb = (avg_uncompressed_kb * objects_per_frame * history_frames)
    print(f"Estimated state size with full frames: ~{full_frame_state_kb:.2f}KB (~{full_frame_state_kb/1024:.2f}MB)")

# Run the function with your directory
check_image_resolutions("/usr/local/src/Coral-Pie-FaaS/camera_stream_data")