import numpy as np
import re
from typing import Dict, List, Tuple, Union, Any


def read_gex(file_gex: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Read GEX file and return GENERAL section and CHANNELS list.
    
    Parameters:
    -----------
    file_gex : str
        Path to the GEX file
        
    Returns:
    --------
    GENERAL : dict
        Dictionary containing all key-value pairs from [General] section
    CHANNELS : list
        List of dictionaries, one per channel section
    """
    
    def parse_value(value_str: str) -> Union[str, int, float, np.ndarray]:
        """Parse a value string into appropriate data type."""
        value_str = value_str.strip()
        
        # Try to split into multiple values (space-separated)
        parts = value_str.split()
        
        if len(parts) == 1:
            # Single value
            try:
                # Try integer first
                if '.' not in value_str and 'E' not in value_str.upper() and 'e' not in value_str:
                    return int(value_str)
                else:
                    # Try float
                    return float(value_str)
            except ValueError:
                # Return as string
                return value_str
        else:
            # Multiple values - try to convert to numpy array of floats
            try:
                return np.array([float(p) for p in parts])
            except ValueError:
                # If conversion fails, return as string
                return value_str
    
    def group_numbered_keys(section_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Group numbered keys into 2D numpy arrays."""
        grouped = {}
        numbered_groups = {}
        
        for key, value in section_dict.items():
            # Check if key ends with a number
            match = re.match(r'^(.+?)(\d+)$', key)
            if match:
                base_name = match.group(1)
                number = int(match.group(2))
                
                if base_name not in numbered_groups:
                    numbered_groups[base_name] = {}
                numbered_groups[base_name][number] = value
            else:
                grouped[key] = value
        
        # Convert numbered groups to 2D arrays
        for base_name, numbered_dict in numbered_groups.items():
            # Sort by number and collect values
            sorted_items = sorted(numbered_dict.items())
            values = [item[1] for item in sorted_items]
            
            # If all values are numpy arrays of the same length, stack them
            if all(isinstance(v, np.ndarray) for v in values):
                try:
                    grouped[base_name] = np.array(values)
                except:
                    # If stacking fails, keep as list
                    grouped[base_name] = values
            else:
                # Keep as list if not all numpy arrays
                grouped[base_name] = values
                
        return grouped
    
    # Read the file
    with open(file_gex, 'r') as f:
        lines = f.readlines()
    
    GENERAL = {}
    CHANNELS = []
    current_section = None
    current_dict = None
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('/'):
            continue
            
        # Check for section headers
        if line.startswith('[') and line.endswith(']'):
            # Save previous section if it exists
            if current_section is not None and current_dict is not None:
                if current_section.lower() == 'general':
                    GENERAL = group_numbered_keys(current_dict)
                elif current_section.lower().startswith('channel'):
                    CHANNELS.append(group_numbered_keys(current_dict))
            
            # Start new section
            current_section = line[1:-1]  # Remove brackets
            current_dict = {}
            continue
        
        # Parse key-value pairs
        if '=' in line and current_dict is not None:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            if value:  # Only process non-empty values
                current_dict[key] = parse_value(value)
    
    # Handle the last section
    if current_section is not None and current_dict is not None:
        if current_section.lower() == 'general':
            GENERAL = group_numbered_keys(current_dict)
        elif current_section.lower().startswith('channel'):
            CHANNELS.append(group_numbered_keys(current_dict))
    
    return GENERAL, CHANNELS


def describe_gex(input_data, channels_data=None):
    """
    Describe a GEX system configuration in a human-readable format.
    
    Parameters:
    -----------
    input_data : str or dict
        Either a GEX filename (str) or a GENERAL dictionary
    channels_data : list of dict, optional
        List of channel dictionaries. Required if input_data is a dict.
        
    Returns:
    --------
    None
        Prints description to screen
    """
    
    # Handle input - either filename or dictionaries
    if isinstance(input_data, str):
        # It's a filename
        filename = input_data
        general, channels = read_gex(filename)
        print(f"GEX System Description: {filename}")
    else:
        # It's a dictionary
        general = input_data
        channels = channels_data or []
        print("GEX System Description")
    
    print("=" * 60)
    
    # System identification
    description = general.get('Description', 'Unknown')
    data_type = general.get('DataType', 'Unknown')
    print(f"System: {description}")
    if data_type != 'Unknown':
        print(f"Data Type: {data_type}")
    
    print()
    
    # Transmitter loop information
    loop_type = general.get('LoopType', 'Unknown')
    tx_area = general.get('TxLoopArea', 'Unknown')
    turns_lm = general.get('NumberOfTurnsLM', 'Unknown')
    turns_hm = general.get('NumberOfTurnsHM', 'Unknown')
    
    print("TRANSMITTER CONFIGURATION:")
    print(f"  Loop Type: {loop_type}")
    print(f"  Loop Area: {tx_area} m²")
    print(f"  Turns (Low Moment): {turns_lm}")
    print(f"  Turns (High Moment): {turns_hm}")
    
    # Transmitter loop geometry
    if 'TxLoopPoint' in general:
        tx_points = general['TxLoopPoint']
        if isinstance(tx_points, np.ndarray):
            n_points = len(tx_points)
            if n_points == 4:
                loop_shape = "Rectangular"
            elif n_points == 8:
                loop_shape = "Octagonal"
            else:
                loop_shape = f"{n_points}-sided polygon"
            
            print(f"  Loop Shape: {loop_shape} ({n_points} points)")
            
            # Calculate approximate dimensions
            x_coords = tx_points[:, 0]
            y_coords = tx_points[:, 1]
            x_range = np.max(x_coords) - np.min(x_coords)
            y_range = np.max(y_coords) - np.min(y_coords)
            print(f"  Dimensions: {x_range:.2f} x {y_range:.2f} m")
    
    print()
    
    # Receiver configuration
    print("RECEIVER CONFIGURATION:")
    if 'RxCoilPosition' in general:
        rx_pos = general['RxCoilPosition']
        if isinstance(rx_pos, np.ndarray) and len(rx_pos) > 0:
            if len(rx_pos) == 1:
                x, y, z = rx_pos[0]
                print(f"  Receiver Position: ({x:.2f}, {y:.2f}, {z:.2f}) m")
                # Calculate offset from transmitter
                if 'TxCoilPosition' in general:
                    tx_pos = general['TxCoilPosition'][0]
                    offset = np.sqrt((x - tx_pos[0])**2 + (y - tx_pos[1])**2)
                    print(f"  Tx-Rx Offset: {offset:.2f} m")
            else:
                print(f"  Multiple Receivers: {len(rx_pos)} positions")
    
    print()
    
    # Timing configuration
    print("TIMING CONFIGURATION:")
    front_gate_delay = general.get('FrontGateDelay', 'Unknown')
    print(f"  Front Gate Delay: {front_gate_delay}")
    
    if 'GateTime' in general:
        gate_times = general['GateTime']
        if isinstance(gate_times, np.ndarray):
            n_gates = len(gate_times)
            if gate_times.shape[1] >= 3:
                # Assume format: [center, start, end]
                earliest = np.min(gate_times[:, 1])  # earliest start time
                latest = np.max(gate_times[:, 2])    # latest end time
                print(f"  Number of Gates: {n_gates}")
                print(f"  Time Window: {earliest:.2e} to {latest:.2e} seconds")
                
                # Show early and late time examples
                print(f"  Early Gates: {gate_times[0, 0]:.2e}, {gate_times[1, 0]:.2e}, {gate_times[2, 0]:.2e} s")
                if n_gates > 3:
                    print(f"  Late Gates: {gate_times[-3, 0]:.2e}, {gate_times[-2, 0]:.2e}, {gate_times[-1, 0]:.2e} s")
    
    print()
    
    # Waveform information
    print("WAVEFORM CONFIGURATION:")
    waveform_types = []
    if 'WaveformLMPoint' in general:
        wf_lm = general['WaveformLMPoint']
        if isinstance(wf_lm, np.ndarray):
            n_points_lm = len(wf_lm)
            waveform_types.append(f"Low Moment ({n_points_lm} points)")
    
    if 'WaveformHMPoint' in general:
        wf_hm = general['WaveformHMPoint']
        if isinstance(wf_hm, np.ndarray):
            n_points_hm = len(wf_hm)
            waveform_types.append(f"High Moment ({n_points_hm} points)")
    
    if waveform_types:
        print(f"  Available Waveforms: {', '.join(waveform_types)}")
    else:
        print("  Waveform Data: Not available")
    
    print()
    
    # Channel information
    print("CHANNEL CONFIGURATION:")
    n_channels = len(channels)
    print(f"  Number of Channels: {n_channels}")
    
    if n_channels > 0:
        for i, channel in enumerate(channels):
            print(f"\n  Channel {i+1}:")
            
            # Channel-specific parameters
            rx_coil_num = channel.get('RxCoilNumber', 'Unknown')
            print(f"    Receiver Coil: {rx_coil_num}")
            
            no_gates = channel.get('NoGates', 'Unknown')
            if no_gates != 'Unknown':
                print(f"    Number of Gates: {no_gates}")
            
            rep_freq = channel.get('RepFreq', 'Unknown')
            if rep_freq != 'Unknown':
                print(f"    Repetition Frequency: {rep_freq} Hz")
            
            front_gate_time = channel.get('FrontGateTime', 'Unknown')
            if front_gate_time != 'Unknown':
                print(f"    Front Gate Time: {front_gate_time} s")
            
            # Data processing parameters
            uniform_std = channel.get('UniformDataSTD', 'Unknown')
            if uniform_std != 'Unknown':
                print(f"    Uniform Data STD: {uniform_std}")
            
            remove_gates = channel.get('RemoveInitialGates', 'Unknown')
            if remove_gates != 'Unknown':
                print(f"    Remove Initial Gates: {remove_gates}")
    
    print()
    
    # Position information
    if any(key in general for key in ['GPSPosition', 'TxCoilPosition', 'AltimeterPosition']):
        print("POSITION INFORMATION:")
        
        if 'GPSPosition' in general:
            gps_pos = general['GPSPosition']
            if isinstance(gps_pos, np.ndarray) and len(gps_pos) > 0:
                if len(gps_pos) == 1:
                    x, y, z = gps_pos[0]
                    print(f"  GPS Position: ({x:.2f}, {y:.2f}, {z:.2f})")
                else:
                    print(f"  GPS Positions: {len(gps_pos)} locations")
        
        if 'TxCoilPosition' in general:
            tx_pos = general['TxCoilPosition']
            if isinstance(tx_pos, np.ndarray) and len(tx_pos) > 0:
                x, y, z = tx_pos[0]
                print(f"  Transmitter Position: ({x:.2f}, {y:.2f}, {z:.2f}) m")
        
        if 'AltimeterPosition' in general:
            alt_pos = general['AltimeterPosition']
            if isinstance(alt_pos, np.ndarray) and len(alt_pos) > 0:
                if not np.allclose(alt_pos, 0):  # Only show if not all zeros
                    x, y, z = alt_pos[0]
                    print(f"  Altimeter Position: ({x:.2f}, {y:.2f}, {z:.2f}) m")
        
        print()
    
    # Summary
    print("SYSTEM SUMMARY:")
    summary_points = []
    
    if data_type != 'Unknown':
        summary_points.append(f"{data_type} system")
    
    if 'TxLoopPoint' in general:
        tx_points = general['TxLoopPoint']
        if isinstance(tx_points, np.ndarray):
            n_points = len(tx_points)
            if n_points == 4:
                summary_points.append("rectangular transmitter loop")
            elif n_points == 8:
                summary_points.append("octagonal transmitter loop")
            else:
                summary_points.append(f"{n_points}-sided transmitter loop")
    
    if tx_area != 'Unknown':
        summary_points.append(f"{tx_area} m² loop area")
    
    summary_points.append(f"{n_channels} channel{'s' if n_channels != 1 else ''}")
    
    if 'GateTime' in general:
        gate_times = general['GateTime']
        if isinstance(gate_times, np.ndarray):
            n_gates = len(gate_times)
            summary_points.append(f"{n_gates} time gates")
    
    print(f"  • {', '.join(summary_points).capitalize()}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test the functions
    print("Testing read_gex() and describe_gex() functions...")
    
    # Test files
    test_files = [
        '202500404_DEN_Diamond_Soeballe_mergedGates_SR.gex',
        'TX07_20230731_2x4_RC20-33.gex'
    ]
    
    for filename in test_files:
        print(f"\n{'='*80}")
        print(f"Testing describe_gex() with filename: {filename}")
        print(f"{'='*80}")
        
        try:
            # Test with filename
            describe_gex(filename)
            
            print(f"\n{'='*80}")
            print(f"Testing describe_gex() with dictionaries from: {filename}")
            print(f"{'='*80}")
            
            # Test with dictionaries
            general, channels = read_gex(filename)
            describe_gex(general, channels)
            
        except Exception as e:
            print(f"Error testing {filename}: {e}")
            import traceback
            traceback.print_exc()