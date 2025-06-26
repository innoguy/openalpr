import re
import pandas as pd
from collections import defaultdict
from difflib import SequenceMatcher

def calculate_similarity(plate1, plate2):
    """Calculate similarity percentage between two license plates."""
    return SequenceMatcher(None, plate1, plate2).ratio() * 100

def group_similar_plates(plate_data, similarity_threshold=80.0):
    """
    Group similar license plates based on similarity threshold.
    
    Args:
        plate_data (dict): Dictionary with plate information
        similarity_threshold (float): Minimum similarity percentage (0-100)
        
    Returns:
        list: List of grouped plates with combined information
    """
    plates = list(plate_data.keys())
    used_plates = set()
    grouped_results = []
    
    for i, main_plate in enumerate(plates):
        if main_plate in used_plates:
            continue
            
        # Find similar plates
        similar_group = [main_plate]
        similar_data = {
            'confidences': plate_data[main_plate]['confidences'].copy(),
            'frames': plate_data[main_plate]['frames'].copy()
        }
        
        for j, other_plate in enumerate(plates[i+1:], i+1):
            if other_plate in used_plates:
                continue
                
            similarity = calculate_similarity(main_plate, other_plate)
            if similarity >= similarity_threshold:
                similar_group.append(other_plate)
                similar_data['confidences'].extend(plate_data[other_plate]['confidences'])
                similar_data['frames'].extend(plate_data[other_plate]['frames'])
                used_plates.add(other_plate)
        
        used_plates.add(main_plate)
        
        # Find the plate with highest average confidence in the group
        best_plate = main_plate
        best_avg_confidence = sum(plate_data[main_plate]['confidences']) / len(plate_data[main_plate]['confidences'])
        
        for plate in similar_group[1:]:
            avg_conf = sum(plate_data[plate]['confidences']) / len(plate_data[plate]['confidences'])
            if avg_conf > best_avg_confidence:
                best_plate = plate
                best_avg_confidence = avg_conf
        
        # Create group entry
        other_plates = [p for p in similar_group if p != best_plate]
        grouped_results.append({
            'main_plate': best_plate,
            'similar_plates': other_plates,
            'all_confidences': similar_data['confidences'],
            'all_frames': similar_data['frames']
        })
    
    return grouped_results

def parse_license_plate_log(file_path, similarity_threshold=80.0):
    """
    Parse license plate detection log file and extract plate information.
    
    Args:
        file_path (str): Path to the log file
        similarity_threshold (float): Minimum similarity percentage to group plates (0-100)
        
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
    
    # Group similar plates
    grouped_plates = group_similar_plates(plate_data, similarity_threshold)
    
    # Convert to list of dictionaries for DataFrame creation
    results = []
    for group in grouped_plates:
        if group['all_frames']:  # Only include plates that were actually found
            similar_plates_str = ', '.join(group['similar_plates']) if group['similar_plates'] else ''
            results.append({
                'plate_found': group['main_plate'],
                'similar_plates': similar_plates_str,
                'confidence_level': round(sum(group['all_confidences']) / len(group['all_confidences']), 2),
                'first_frame': min(group['all_frames']),
                'last_frame': max(group['all_frames']),
                'total_detections': len(group['all_frames'])
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

def save_report(df, input_file, report_file, similarity_threshold):
    """Save detailed report to text file."""
    try:
        with open(report_file, 'w') as f:
            f.write(f"Parsing license plate detection log: {input_file}\n")
            f.write(f"Using similarity threshold: {similarity_threshold}%\n\n")
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
    
    # Default similarity threshold
    similarity_threshold = 80.0
    
    # Get input file and optional similarity threshold from command line
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if len(sys.argv) > 2:
            try:
                similarity_threshold = float(sys.argv[2])
                if not 0 <= similarity_threshold <= 100:
                    print("Warning: Similarity threshold should be between 0 and 100. Using default 80.0")
                    similarity_threshold = 80.0
            except ValueError:
                print("Warning: Invalid similarity threshold. Using default 80.0")
                similarity_threshold = 80.0
    else:
        input_file = input("Enter the path to your license plate log file: ").strip()
        if not input_file:
            print("No file path provided. Exiting.")
            return
        
        threshold_input = input(f"Enter similarity threshold percentage (0-100, default {similarity_threshold}): ").strip()
        if threshold_input:
            try:
                similarity_threshold = float(threshold_input)
                if not 0 <= similarity_threshold <= 100:
                    print("Warning: Similarity threshold should be between 0 and 100. Using default 80.0")
                    similarity_threshold = 80.0
            except ValueError:
                print("Warning: Invalid similarity threshold. Using default 80.0")
                similarity_threshold = 80.0
    
    # Generate output filename based on input filename
    if input_file.endswith('.txt'):
        output_file = input_file.replace('.txt', '_summary.csv')
        report_file = input_file.replace('.txt', '_report.txt')
    else:
        output_file = input_file + '_summary.csv'
        report_file = input_file + '_report.txt'
    
    print(f"Parsing license plate detection log: {input_file}")
    print(f"Using similarity threshold: {similarity_threshold}%")
    
    # Parse the log file
    results_df = parse_license_plate_log(input_file, similarity_threshold)
    
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
    save_report(results_df, input_file, report_file, similarity_threshold)
    
    # Additional statistics
    print(f"\nStatistics:")
    print(f"- Average confidence: {results_df['confidence_level'].mean():.2f}")
    print(f"- Highest confidence: {results_df['confidence_level'].max():.2f}")
    print(f"- Lowest confidence: {results_df['confidence_level'].min():.2f}")
    print(f"- Total detections across all plates: {results_df['total_detections'].sum()}")

if __name__ == "__main__":
    main()

