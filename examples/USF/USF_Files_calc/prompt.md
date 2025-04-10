I need to implement a Ṕython function that read ths USF file attacched. 

Lines starting with '//' can be comments, but lines starting with "//XXX: YYY" should be read as variables wih name 'XXX' and value 'YYY'. 

'//DUMMY: YYYYY' defines the variables that should be converted NaN or "empty'

'//END' and '/END' define the end of a block of data


Lines staring with a single '/' contains information that should be read as variables "/XXX: YYY ". YYY can be both a string, a number, a list of number


"/SWEEPS: YYY" determnies the number of 'sweeps' 

For each SWEEP, a specific data section follows starting with "/SWEEP_NUMBER: 1" and ending with "/END" for the first sweep, and  starting with "/SWEEP_NUMBER: 2" and ending with "/END" for the 2nd sweep and so forth.


Following "/END" for each "Sweep" comes som data.
A first line containes a HEADER , "e.g. "TIME, 
VOLTAGE,,..."
and then the follwing "/POINTS" lines contains the corresponding data.
These data should be read as part of each SWEEP



I would like to read all the parameters unto Python structure I can access the values as e.g.
USF['DYMMY'] (=99990)
USF['SWEEPS']  (=2)
USF['SWEEP'][0]['CURRENT'] (=1)
USF['SWEEP'][0]['FREQUENCY'] (=2109.70)
--
USF['SWEEP'][1]['CURRENT'] (=10.11)
USF['SWEEP'][1]['FREQUENCY'] (=217.39)

I would like the 'data' for each SWEEP to be avalable as 
usf['SWEEP'][0]['TIME']
usf['SWEEP'][0]['VOLTAGE']
usf['SWEEP'][0]['ERRORBAR']


I would like to be able to call the function using 

>> USF = read_usf(file_usf)



_____________
Implementing a python function called 'read_usf_mul(exp='.usf') that read files in a specific folder as USF file and returns a list of USF stcrures

