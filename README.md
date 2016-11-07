# Learning Diverse Models: The Coulomb Structured Support Vector Machine

This repository provides an implementation for 

M. Schiegg, F. Diego, F. A. Hamprecht. 
   [**Learning Diverse Models: The Coulomb Structured Support Vector Machine**](https://hci.iwr.uni-heidelberg.de/node/6081). Proceedings of the the European 
   Conference on Computer Vision (ECCV 2016), 2016. 

```
@inproceedings{schiegg2016learning,
  title={Learning Diverse Models: The Coulomb Structured Support Vector Machine},
  author={Schiegg, Martin and Diego, Ferran and Hamprecht, Fred A},
  booktitle={European Conference on Computer Vision},
  pages={585--599},
  year={2016},
  organization={Springer}
}
```

The **Coulomb Structured Support Vector Machine** (CSSVM) algorithm is implemented as an extension to the [**pystruct**](https://github.com/pystruct/pystruct) framework. 

## Installation instruction

1. Install our version of pystruct (which includes the CSSVM extension) and its dependencies (tested on Ubuntu 16.04.1):
	```
	sudo apt-get install python-cvxopt python-scipy python-sklearn python-numpy
	cd ./pystruct
	sudo python setup.py install
	```
	If you experience problems while installing pystruct (or one of its dependencies), please check the [installation instructions of pystruct](http://pystruct.github.io/installation.html).

2. Run a CSSVM example
	```
	python cssvm_example.py 
	```
