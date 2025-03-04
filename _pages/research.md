---
layout: archive
title: "Research"
permalink: /research/
author_profile: true

---

{% include base_path %}

This page includes details about my academic profile.  

# My Doctoral Research

My research work primarily focuses on the experimental investigation of active polar fluids that are driven out of equilibrium through an 'active' drive. I build these so-called 'fluids' using micron-sized spherical particles suspended in oil that 'flow' together, hence termed 'active fluid'. 
<br>
<p>The particles are activated using a D.C. electric field (around a range of 3V/micron), causing them to exhibit a persistent rolling motion. This electro-rotation was first observed by a scientist named 'Quincke' and hence called as Quincke rollers. As the rollers move/propel, they generate hydrodynamamic perturbations, which cause the neighbouring rollers to reorient and coordinate their mutual direction of motion. This leads to the swarming of rollers that show macroscopic direction motion, which is popularly called 'flocking' behavior. Yes, the same flocking behavior that is seen in cattle, birds, and fish. When millions of such rollers are assembled, the flocks turn in spontaneous flows with broken rotational symmetry, which we call <I>active flows</I>. 
</p> 
<!..... Add some of the videos here ..... ask alex .... >

<p>
If two or more of these active fluids are mixed, they can homogeneously mix or phase-separate based on how we confine them, causing the formation of rich multicomponent phases unique to such non-equilibrium systems. For my doctoral research, I have been studying the formation of these phases by conducting these experiments and also developing a generalized n-component hydrodynamic model to describe this fluid mixture behavior. 
</p>

<p>
If we confine the rollers in a circular well, we observe that a mixture of active fluids spontaneously demixes. I explain this phase-separation behavior in detail in my <a href = 'https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.131.178304'>article</a> published in the <b>Physical Review Letters</b>, which is one of the most reputed (Q1) physics journals. 
</p>

# Tech stack that I use for my Ph.D.

## Data Collection
As mentioned before, I observed millions of colloidal particles, and these particles move very fast given their size of a few microns ... almost in the order of 1mm/s. Thus, high-speed imaging is necessary to study them (up to 1k frames per sec). I use <a href = 'https://www.ximea.com/'> XIMEA</a> CMOS cameras coupled to a PCIe reader for transferring images to a computer memory.  

## Data sizes
On a good day, when all aspects of my experiments are working. I can generate up to <b> 1-2 TeraBytes</b> of data.
In my Ph.D. so far, I have generated more than <b>30 TB of publication-quality data</b> and <b>80+ TB</b> in total. Yes, it is a lot of data ... about the size of what a small to mid-sized tech company stores. This also means I inevitably need robust data management and processing pipelines for successfully navigating through this stockpile of data and perform crazy large-scale satistical analyses. 

## Data Type.
I use state-of-the-art statistical methods and advanced computational workflows to extract meaningful correlations in inter-particle dynamics from TeraBytes of experimental data.  I primarily use MATLAB, Python (pandas, Matplotlib, Pyspark Pytorch, and some other subsidiary packages), and ImageJ to process the data and perform statistical analyses. 

