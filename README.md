# TARNet
This is the official Pytorch implementation for KDD 2022 paper "TARNet : Task-Aware Reconstruction for Time-Series Transformer."


## Quick Start
```
git clone https://github.com/ranakroychowdhury/TARNet.git
```

## Datasets
The classification datasets can be found at [UEA Archive](https://www.timeseriesclassification.com/dataset.php), [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php), and from [Towards automatic spatial verification of sensor placement in buildings](https://cseweb.ucsd.edu/~dehong/pdf/buildsys13-paper.pdf).

The regression datasets are available in [Monash University, UEA, UCR Time Series Regression Archive](http://tseregression.org/).

The data directory contains an example of a preprocessed classification dataset: [Atrial Fibrillation (AF)](https://www.timeseriesclassification.com/description.php?Dataset=AtrialFibrillation) and a preprocessed regression dataset: [Appliances Energy (AE)](https://zenodo.org/record/3902637), along with their corresponding preprocessing files. 

The preprocessing file under `/data/AF/` can be used to preprocess any classification dataset from [UEA Archive](https://www.timeseriesclassification.com/dataset.php) by changing the filename parameter. 

The preprocessing file under `/data/AE/` can be used to preprocess any regression dataset from [Monash University, UEA, UCR Time Series Regression Archive](http://tseregression.org/) by changing the train_file and test_file parameter.

