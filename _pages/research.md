---
layout: archive
title: "My Doctoral Research"
permalink: /research/
author_profile: true

---
<p align="justify">
My research work primarily focuses on the experimental investigation of active polar fluids that are driven out of equilibrium through an 'active' drive. I build these so-called 'fluids' using micron-sized spherical particles suspended in oil that 'flow' together, hence termed 'active fluid'. 
</p>
<p align="justify">The particles are activated using a D.C. electric field (around a range of 3V/micron), causing them to exhibit a persistent rolling motion. This electro-rotation was first observed by a scientist named 'Quincke' and hence called as Quincke rollers. As the rollers move/propel, they generate hydrodynamamic perturbations, which cause the neighbouring rollers to reorient and coordinate their mutual direction of motion. This leads to the swarming of rollers that show macroscopic direction motion, which is popularly called 'flocking' behavior. Yes, the same flocking behavior that is seen in cattle, birds, and fish. When millions of such rollers are assembled, the flocks turn in spontaneous flows with broken rotational symmetry, which we call <I>active flows</I>. 
</p> 

<p align="justify">
If two or more of these active fluids are mixed, they can homogeneously mix or phase-separate based on how we confine them, causing the formation of rich multicomponent phases unique to such non-equilibrium systems. For my doctoral research, I have been studying the formation of these phases by conducting these experiments and also developing a generalized n-component hydrodynamic model to describe this fluid mixture behavior. 
</p>

<p align="justify">
If we confine the rollers in a circular well, we observe that a mixture of active fluids spontaneously demixes. I explain this phase-separation behavior in detail in my <a href = 'https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.131.178304'>article</a> published in the <b>Physical Review Letters</b>, which is one of the most reputed (Q1) physics journals. 
</p>

# Tech stack that I use for my Ph.D.
The following sketch show my data processing workflow (data pipeline). The 3D boxes are hardware involved and the green boxes are the state in which the data exists at that point in the pipeline. the red arrows indicate the flow of data in the pipeline.

## Schematic of my data workflow
<img src="/images/Research/PhD data flow.svg" alt="Price_m2_vs_year.png" >

## Data Collection
<p align="justify">
As mentioned before, I observed millions of colloidal particles, and these particles move very fast given their size of a few microns ... almost in the order of 1mm/s. Thus, high-speed imaging is necessary to study them (up to 1k frames per sec). I use <a href = 'https://www.ximea.com/'> XIMEA</a> CMOS cameras coupled to a PCIe reader for transferring images to a computer memory.  
</p>

## Data sizes
<p align="justify">
On a good day, when all aspects of my experiments are working. I can generate up to <b> 1-2 TeraBytes</b> of data.
In my Ph.D. so far, I have generated more than <b>30 TB of publication-quality data</b> and <b>80+ TB</b> in total. Yes, it is a lot of data ... about the size of what a small to mid-sized tech company stores. This also means I inevitably need robust data management and processing pipelines for successfully navigating through this stockpile of data and perform crazy large-scale satistical analyses. 
</p>

## Data wrangling and statistical analysis.
<p align="justify">
I use state-of-the-art statistical methods and advanced computational workflows to extract meaningful correlations in inter-particle dynamics from TeraBytes of experimental data.  I primarily use MATLAB, Python (pandas, Matplotlib, Pyspark Pytorch, and some other subsidiary packages), and ImageJ (and PyImageJ) to process the data and perform statistical analyses. A large amount of the computation effort involve matrix maniputions (to build correlations fields) hence better done on the GPUs. (I prefer using Pytorch for these, than CuPy although Pytorch is natively build for deep learning.) I use sklearn for particle tracking using the KNN algorithm, you can read more about it on a blog on the portfolio section. I am also currently build a CNN that caters for indentification of overlapping particles. 
</p>

## Debugging and error handling
<p align="justify">
This is very important aspect. I use simple Python classes and associated methods to break down repeating processing steps. The above data workflow was build particlularly to handle unexpected crashes with enough room for backup as well as systematically searching for errors in the data pipeline. I developed simple unit tests for computations that I need to benchmark before I build upon them. For example, checking if the geometrical transformations from cartesian to polar coordinates are correctly computed for each data set before I start peforming correlation analyses. 
</p>
