#!/bin/bash

mkdir data
cd /data
kaggle competitions download -c bathymetry-estimation
unzip bathymetry-estimation.zip
