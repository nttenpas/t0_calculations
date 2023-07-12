# t0_calculations
Recreating Dave E's work on calculating t0 via iteration.


The main software is the Recreate_t0_Calculations.ipynb
	- This is run in Jupyter Notebook and the paths/constants need to be changed there before running. 
	- The paths and constants are all globally defined in the 3rd cell

Alternatively, you can run the analysis via bash by simply running the following on the command line:

python t0_Analysis.py <path-to-sim-text-files> <nResets> <nChiSquared> <nEvents>
Where:
	- <path-to-sim-text-files> is the path to the directory storing resets_output.txt and g4_output.txt
	- <nResets> will be an integer number describing the min number of resest for each pixel to be considered
	- <nChiSquared> will be an integer number describing the nu,ber of chi squared needed to define an outlier
	- <nEvents> will be an integer number for total number of events in the sample

Using either software will result in the same outputs, each stored in the parent directory (respect to the t0_calculations directory)
under Analysis_Results/Analysis_MM_DD_YYY_HHMMSS with the date-time corresponding to the time the software was run.

If adjustments are made to the t0_Analysis.ipynb file, you can run the following to convert it to a python file named t0_Analysis.py
to run via bash:
jupyter nbconvert --to notebook t0_Analysis.ipynb

Recreate_t0_Calculations.ipynb and t0_Analysis.ipynb are identical files, with different print statements. t0_Analysis.ipynb
has more print statements used to make sure the program doesn't get stuck when running in SSH client. 
