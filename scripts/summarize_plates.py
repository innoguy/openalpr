import re
import pandas as pd
from collections import defaultdict

def parse_license_plate_log(file_path):
    """
    Parse license plate detection log file and extract plate information.
    
    Args:
        file_path (str): Path to the log file
        
    Returns:
        pandas.DataFrame: Table with plate data
    """
    
    # Dictionary to store plate information
    # Key: plate_number, Value: {'confidences': [list], 'frames': [list]}
    plate_data = defaultdict(lambda: {'confidences': [], 'frames': []})
    
    current_frame = None
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                
                # Extract frame number
                frame_match = re.match(r'Frame:\s*(\d+)', line)
                if frame_match:
                    current_frame = int(frame_match.group(1))
                    continue
                
                # Extract plate information
                # Pattern matches lines like: "    - AI3NRU	 confidence: 82.6628"
                plate_match = re.match(r'\s*-\s*([A-Z0-9]+)\s+confidence:\s*(\d+\.?\d*)', line)
                if plate_match and current_frame is not None:
                    plate_number = plate_match.group(1)
                    confidence = float(plate_match.group(2))
                    
                    plate_data[plate_number]['confidences'].append(confidence)
                    plate_data[plate_number]['frames'].append(current_frame)
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame()
    
    # Convert to list of dictionaries for DataFrame creation
    results = []
    for plate, data in plate_data.items():
        if data['frames']:  # Only include plates that were actually found
            results.append({
                'plate_found': plate,
                'confidence_level': round(sum(data['confidences']) / len(data['confidences']), 2),  # Average confidence
                'first_frame': min(data['frames']),
                'last_frame': max(data['frames']),
                'total_detections': len(data['frames'])
            })
    
    # Create DataFrame and sort by first frame appearance
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('first_frame').reset_index(drop=True)
    
    return df

def save_results(df, output_file='license_plate_summary.csv'):
    """Save results to CSV file."""
    try:
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

def main():
    # Configuration
    input_file = 'license_plate_log.txt'  # Change this to your log file path
    output_file = 'license_plate_summary.csv'
    
    print("Parsing license plate detection log...")
    
    # Parse the log file
    results_df = parse_license_plate_log(input_file)
    
    if results_df.empty:
        print("No data found or error occurred.")
        return
    
    # Display results
    print("\n" + "="*80)
    print("LICENSE PLATE DETECTION SUMMARY")
    print("="*80)
    print(f"Total unique plates found: {len(results_df)}")
    print("\nDetailed Results:")
    print(results_df.to_string(index=False))
    
    # Save to CSV
    save_results(results_df, output_file)
    
    # Additional statistics
    print(f"\nStatistics:")
    print(f"- Average confidence: {results_df['confidence_level'].mean():.2f}")
    print(f"- Highest confidence: {results_df['confidence_level'].max():.2f}")
    print(f"- Lowest confidence: {results_df['confidence_level'].min():.2f}")
    print(f"- Total detections across all plates: {results_df['total_detections'].sum()}")

if __name__ == "__main__":
    main()
