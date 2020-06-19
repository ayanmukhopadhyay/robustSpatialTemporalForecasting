# robustSpatialTemporalForecasting
Robust Incident Prediction (based on Mukhopadhyay et al. "Robust Spatial-Temporal Forecasting (UAI 2020)"

The code repo contains the following files -- 

1. main.py -- the parent file for running robust forecasting algorithms
2. gradientBased.py -- implementation of AdGrad
3. dualBased.py -- implementation of RSALA
4. utils.py -- data processing and helper methods


The data used for the paper are bounded by confidentiality agreements, and hence cannot be released. Therefore, data-processing methods in the code are marked with #TODO. 

Code run parameters -- 

To run AdGrad, use "python main.py --algorithm=gradient --model=poisson --neighbors=1". The parameter neighbors specifies the attacker budget. Refer to the paper (experiments section) for explanation. The model parameter can currently accept "poisson", "logisitc" and "survival". To run RSALA, use "python main.py --algorithm=dual --model=poisson --neighbors=1". Our implementation for the paper had several other arguements (e.g. data source, gradient step etc.). This section can be easily modified and parameters can be added through the main file. File names and data sources can be added through the params.conf file.

Version and Dependencies --
The code has been implemented in Python 3.7. It expects the following dependencies.
1. cvxpy
2. pandas
3. numpy
4. configparser (Can be replaced by a different config parsing library)
5. pickle


