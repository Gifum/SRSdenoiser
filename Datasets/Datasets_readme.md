# SRS simulated Datasets

Snoise and Sclean contain samples of simulated spectroscopic data as those that are obtained in lab by the SRS technique. Each sample is a vector of size (N<sub>omega</sub>,1), which represents a spectrum, the spectroscopic value measured at each detected frequency of the light. Here N<sub>omega</sub>=801. Each dataset contains 5000 spectra: Sclean and Snoise are matrices of size (N<sub>omega</sub>, 5000). Samples are orderd in columns, so each of the 5000 columns in Sclean and Snoise contains a different spectrum. 

Two independent datasets (HighNoise and LowNoise) have been prepared changing the maximum level of noise (for each point in the spectrum noise is random but lower than a fixed level). See the main text and SI of the associated paper for further details.


## Training and Test set

Sclean contains the clean SRS data that one would like to isolate from the raw data contained in Snoise which are noisy and superimposed to a unwanted baseline. Thus, Snoise is the raw data that feed the NN with the corresponding ground truth stored in Sclean.

SRS spectra can be of two different kinds: spectra acquired at lower wavelengths, usually called *blue side* spectra, and at higher wavelengths, named *red side* spectra. For the experimental conditions reproduced in this simulation, ground truths features in the red side are always shaped as positive (Lorentian) peaks while in the blue side the ground truth can be positive peaks, deeps (negative peaks) or dispersive lineshapes. The data in Snoise and Sclean are organized so that the first 2500 samples are red side spectra and the second half (from 2501 to 5000) are blue side spectra, so shuffling is needed before training.