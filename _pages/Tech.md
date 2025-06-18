---
layout: archive
title: "Technology Stack and Data Pipeline of my PhD"
permalink: /Stack/
author_profile: true

---
<p align="justify">
The following sketch shows my data processing workflow (data pipeline). The 3D boxes are hardware involved, and the green boxes are the state in which the data exists at that point in the pipeline. The red arrows indicate the flow of data in the pipeline. The data at each stage is stored in the succeeding hardware within the pipeline.
</p> 

## Schematic of my data workflow
<img src="/images/Research/PhD data flow.svg" alt="PhD data flow" >

## Data Collection
<p align="justify">
As mentioned before, I observed millions of colloidal particles, and these particles move very fast, given their size of a few microns ... almost in the order of 1mm/s. Thus, high-speed imaging is necessary to study them (up to 1k frames per sec). I use <a href = 'https://www.ximea.com/'> XIMEA</a> CMOS cameras coupled to a PCIe reader for transferring images to a computer memory.  
</p>

## Data sizes
<p align="justify">
On a good day, when all aspects of my experiments are working. I can generate up to <b> 1-2 TeraBytes</b> of data.
In my Ph.D. so far, I have generated more than <b>30 TB of publication-quality data</b> and <b>80+ TB</b> in total. Yes, it is a lot of data ... about the size of what a small to mid-sized tech company stores. This also means I inevitably need robust data management and processing pipelines for successfully navigating through this stockpile of data and performing crazy large-scale statistical analyses. 
</p>

## Data wrangling and statistical analysis.
<p align="justify">
I use state-of-the-art statistical methods and advanced computational workflows to extract meaningful correlations in inter-particle dynamics from TeraBytes of experimental data.  I primarily use MATLAB, Python (Pandas, Matplotlib, Pyspark, Pytorch, and some other subsidiary packages), and ImageJ (and PyImageJ) to process the data and perform statistical analyses. A large amount of the computation effort involves matrix manipulations (to build correlation fields), hence better done on the GPUs. (I prefer using Pytorch for these than CuPy, although Pytorch is natively built for deep learning.) I use sklearn for particle tracking using the KNN algorithm enhanced using a "Kalman Filter", you can read about it(without the Kalman filter) in a blog on the <b>portfolio</b> section of this website.
</p>

## Deep learning:
<p align="justify">
In the second Year of my PhD, I realised that I needed to develop better strategies to identify the particles, and the traditional methods for particle detection would not suffice for fine measurements to extract weak correlation and Fourier modes. Hence, I developed Convolutional Neural Network architectures that are lightweight and fast, as well as accurately detect highly overlapping particles with unprecedented accuracy. Based on this success, I have started devloping an open-source project to develop fast, all-purpose models, particularly catering to my research community who commonly deal with such systems and regularly face such issues. Please check out this <a href = 'https://github.com/Samadarshi-Maity/CNN-Particle-Detection'> project</a>
</p>

## Debugging and error handling
<p align="justify">
This is a very important aspect. I use simple Python classes and associated methods to break down repeating processing steps. The above data workflow was built particularly to handle unexpected crashes with enough room for backup, as well as systematically searching for errors in the data pipeline. I developed simple unit tests for computations that I need to benchmark before I build upon them. For example, checking if the geometrical transformations from Cartesian to polar coordinates are correctly computed for each data set before I start performing correlation analyses. 
</p>

