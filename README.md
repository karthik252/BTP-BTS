
Third Party modules required
	*Tensorflow 2.4
	*numpy
	*nibabel
	*PIL
	*Antspyx
	*shutil
	*matplotlib
	*SimpleITK
	*intensity_normalization

Download the BRATS Dataset from https://sites.google.com/site/braintumorsegmentation/home/brats2015

Executions
Run following only if Dataset is not available in NII format
> python3 mhatonii.py
Do Preprocessing on the Data
> python3 PreProcessing.py
Configure the variables in main.py as required
>python3 main.py
