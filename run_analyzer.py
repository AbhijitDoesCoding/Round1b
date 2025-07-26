#!/usr/bin/env python3
"""
Main runner script for the optimized Round 1B analyzer
Handles configuration and execution in Docker environment
"""

import os
import sys
import json
from pathlib import Path
from optimized_persona_analyzer import process_round1b_optimized

def main():
    """Main execution function for Docker container"""
    input_dir = "/app/input"
    output_dir = "/app/output"
    config_file = "/app/input/config.json"
    
    # Ensure directories exist
    Path(input_dir).mkdir(exist_ok=True)
    Path(output_dir).mkdir(exist_ok=True)
    
    # Check for config file
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found at {config_file}")
        print("Expected config.json format:")
        print(json.dumps({
            "persona": "Your persona here",
            "job_to_be_done": "Your job description here"
        }, indent=2))
        sys.exit(1)
    
    # Load configuration
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        required_keys = ["persona", "job_to_be_done"]
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required key '{key}' in config.json")
                sys.exit(1)
                
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        sys.exit(1)
    
    # Find PDF files
    pdf_files = [str(p) for p in Path(input_dir).glob("*.pdf")]
    if not pdf_files:
        print(f"❌ No PDF files found in {input_dir}")
        print("Please mount PDF files to /app/input directory")
        sys.exit(1)
    
    print(f"🔍 Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  - {os.path.basename(pdf_file)}")
    
    # Process documents
    try:
        process_round1b_optimized(
            documents=pdf_files,
            persona=config['persona'],
            job=config['job_to_be_done'],
            output_dir=output_dir
        )
        print("\n🎉 Processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
