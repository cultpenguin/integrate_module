import os
import numpy as np

def read_gex(name=None):
    """
    This function gets information from a given gex file.
    """
    # Find local system file if "name" is not provided
    if name is None:
        local_dir = os.listdir()
        gex_files = [file for file in local_dir if file.endswith('.gex')]
        
        if len(gex_files) == 0:
            print('No GEX file found in local directory')
            return
        else:
            name = gex_files[0]
            print(f'Found GEX file in local directory: {name}')

    print(f'{__file__}: Reading \'{name}\'')

    # Read File
    with open(name, 'r') as file:
        lines = file.readlines()

    S = {}
    temporary_array = []
    CheckWaveForm = CheckGateArray = CheckTxArray = False

    for line in lines:
        line = line.strip()

        # Skip if Line is empty
        if len(line) > 0:
            # If a square bracket is encountered make new struct field
            newsection = line[0] == '['
            if newsection:
                CurStruct = line[1:-1]
                S[CurStruct] = {}

            # Check if the Line contains an '=' sign
            if '=' in line:
                # Split line at equal sign
                fieldname, Right_Handside = map(str.strip, line.split('='))

                # Split right handside part by spaces to separate multiple values
                SplitLine2 = Right_Handside.split()
                numericvalue = np.array([float(x) if x.replace('.', '', 1).isdigit() else np.nan for x in SplitLine2])

                # Determine if right hand side is numeric or string
                right_handside_string = np.isnan(numericvalue).any()

                if right_handside_string:
                    fieldvalue = SplitLine2
                else:
                    fieldvalue = numericvalue

                # Check for specific array types
                CheckWaveForm = fieldname.startswith('Wave')
                CheckGateArray = fieldname.startswith('Gate')
                CheckTxArray = fieldname.startswith('TxLo') and fieldname.endswith('int')

                # If any is true then store the information
                if any([CheckWaveForm, CheckGateArray, CheckTxArray]):
                    temporary_array.append(fieldvalue)
                else:
                    S[CurStruct][fieldname] = fieldvalue

        else:
            # Empty line! Clear temporary array if it exists after placing it inside field!
            if temporary_array:
                if CheckWaveForm:
                    S[CurStruct][fieldname[:10]] = np.array(temporary_array)
                elif CheckGateArray:
                    S[CurStruct]['GateArray'] = np.array(temporary_array)
                elif CheckTxArray:
                    S[CurStruct]['TxArray'] = np.array(temporary_array)
                temporary_array = []

    return S
