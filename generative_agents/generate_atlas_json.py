#!/usr/bin/env python3
"""
Generate Phaser-compatible Atlas JSON files for agent sprites.
"""
import json
import os
import subprocess
import re

def get_image_size(texture_path):
    """Get image dimensions using file command."""
    try:
        result = subprocess.run(['file', texture_path], capture_output=True, text=True)
        match = re.search(r'(\d+) x (\d+)', result.stdout)
        if match:
            return int(match.group(1)), int(match.group(2))
    except:
        pass
    # Fallback: known size for this project
    return 832, 3456

def generate_atlas_json(texture_path, output_path, sprite_size=48):
    """Generate Phaser Atlas JSON from texture image and sprite config.
    
    Args:
        texture_path: Path to the texture PNG file
        output_path: Path to save the generated atlas JSON
        sprite_size: Size of each sprite frame (default: 48)
    """
    
    # Read sprite config
    sprite_config_path = os.path.join(os.path.dirname(texture_path), "..", "sprite.json")
    with open(sprite_config_path, "r") as f:
        config = json.load(f)
    
    frames_per_dir = config.get("framesPerDir", 3)
    dirs = config.get("dirs", ["down", "left", "right", "up"])
    
    # Get image dimensions
    texture_width, texture_height = get_image_size(texture_path)
    
    frames = {}
    
    # Generate frames for each direction
    # Layout: Each direction occupies one row, with frames arranged horizontally
    # - Each direction row: y = dir_idx * sprite_size
    # - Each frame: x = frame_idx * sprite_size
    
    for dir_idx, direction in enumerate(dirs):
        y = dir_idx * sprite_size
        
        # Static frame (use first frame as static)
        frames[direction] = {
            "frame": {"x": 0, "y": y, "w": sprite_size, "h": sprite_size},
            "rotated": False,
            "trimmed": False,
            "spriteSourceSize": {"x": 0, "y": 0, "w": sprite_size, "h": sprite_size},
            "sourceSize": {"w": sprite_size, "h": sprite_size}
        }
        
        # Walk animation frames
        for frame_idx in range(frames_per_dir):
            x = frame_idx * sprite_size
            frame_name = f"{direction}-walk.{frame_idx:03d}"
            frames[frame_name] = {
                "frame": {"x": x, "y": y, "w": sprite_size, "h": sprite_size},
                "rotated": False,
                "trimmed": False,
                "spriteSourceSize": {"x": 0, "y": 0, "w": sprite_size, "h": sprite_size},
                "sourceSize": {"w": sprite_size, "h": sprite_size}
            }
    
    atlas = {
        "frames": frames,
        "meta": {
            "app": "Phaser Atlas Generator",
            "version": "1.0",
            "image": os.path.basename(texture_path),
            "size": {"w": texture_width, "h": texture_height},
            "scale": "1"
        }
    }
    
    # Write output
    with open(output_path, "w") as f:
        json.dump(atlas, f, indent=2)
    
    print(f"Generated {output_path}")
    return atlas

if __name__ == "__main__":
    import sys
    
    # Allow sprite size to be specified via command line argument
    sprite_size = 48  # default
    if len(sys.argv) > 1:
        try:
            sprite_size = int(sys.argv[1])
            print(f"Using sprite size: {sprite_size}x{sprite_size}")
        except ValueError:
            print(f"Invalid sprite size: {sys.argv[1]}, using default: {sprite_size}x{sprite_size}")
    
    base_dir = "frontend/static/assets/village/agents"
    
    # Get all agent directories
    agents_dir = os.path.join(base_dir)
    if not os.path.exists(agents_dir):
        print(f"Error: {agents_dir} not found")
        sys.exit(1)
    
    # Generate atlas JSON for each agent
    for agent_name in os.listdir(agents_dir):
        agent_dir = os.path.join(agents_dir, agent_name)
        if not os.path.isdir(agent_dir) or agent_name == "sprite.json":
            continue
        
        texture_path = os.path.join(agent_dir, "texture.png")
        if not os.path.exists(texture_path):
            print(f"Warning: {texture_path} not found, skipping {agent_name}")
            continue
        
        output_path = os.path.join(agent_dir, "atlas.json")
        try:
            generate_atlas_json(texture_path, output_path, sprite_size)
        except Exception as e:
            print(f"Error generating atlas for {agent_name}: {e}")

