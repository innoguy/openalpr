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
    
    # Create DataFrame and sort by confidence level (highest first)
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('confidence_level', ascending=False).reset_index(drop=True)
    
    return df

def save_results(df, output_file='license_plate_summary.csv'):
    """Save results to CSV file."""
    try:
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

def save_report(df, input_file, report_file):
    """Save detailed report to text file."""
    try:
        with open(report_file, 'w') as f:
            f.write(f"Parsing license plate detection log: {input_file}\n\n")
            f.write("="*80 + "\n")
            f.write("LICENSE PLATE DETECTION SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Total unique plates found: {len(df)}\n\n")
            f.write("Detailed Results:\n")
            f.write(df.to_string(index=False) + "\n\n")
            
            if not df.empty:
                f.write("Statistics:\n")
                f.write(f"- Average confidence: {df['confidence_level'].mean():.2f}\n")
                f.write(f"- Highest confidence: {df['confidence_level'].max():.2f}\n")
                f.write(f"- Lowest confidence: {df['confidence_level'].min():.2f}\n")
                f.write(f"- Total detections across all plates: {df['total_detections'].sum()}\n")
        
        print(f"Detailed report saved to {report_file}")
    except Exception as e:
        print(f"Error saving report: {e}")

def main():
    import sys
    
    # Get input file from command line argument or prompt user
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("Enter the path to your license plate log file: ").strip()
        if not input_file:
            print("No file path provided. Exiting.")
            return
    
    # Generate output filename based on input filename
    if input_file.endswith('.txt'):
        output_file = input_file.replace('.txt', '_summary.csv')
        report_file = input_file.replace('.txt', '_report.txt')
    else:
        output_file = input_file + '_summary.csv'
        report_file = input_file + '_report.txt'
    
    print(f"Parsing license plate detection log: {input_file}")
    
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
    
    # Save detailed report to text file
    save_report(results_df, input_file, report_file)
    
    # Additional statistics
    print(f"\nStatistics:")
    print(f"- Average confidence: {results_df['confidence_level'].mean():.2f}")
    print(f"- Highest confidence: {results_df['confidence_level'].max():.2f}")
    print(f"- Lowest confidence: {results_df['confidence_level'].min():.2f}")
    print(f"- Total detections across all plates: {results_df['total_detections'].sum()}")

if __name__ == "__main__":
    main()
