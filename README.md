# OIBFBEM
## Claude de Rijke-Thomas
## Tuning FBEM (by Dr Jack Landy+) to Simulate Airborne Observations

This branch of the Facet-Based Echo Model (FBEM) is for qualitative evaluations made using airborne data from NASA's Operation IceBridge Spring 2016 sea ice campaign. This work corresponds to my (Claude de Rijke-Thomas') final research chapter of my doctoral thesis.

For this to be possible, FBEM was modified to account for unfocused SAR processing (instead of Delay-Doppler SAR). The air-snow interface topography inputted into the FBEM used a Generalised Additive Model (GAM) surface created from NASA's ATM lidar. Either the air-snow
interface GAM was used as a proxy for the snow-ice interface topography (within the OIB.m script), or the snow-ice interface topography
was individually estimated by creating a GAM surface of combined lidar+coincident $in\ situ$ snow depths (within OIB_indep_interfaces.m).

These modifications to FBEM are not particularly review-ready (nor in my primary language), but I thought I would keep it open source for potential future studies:) Please let me know if you have any questions by contacting me @ claude[at]derijke.org , or Jack at jack.c.landy[at]uit.no . Please refer to the other README file for copyright details.

There are some hardcoded modifications that I made within the files themselves (within snow_backscatter.m) that I haven't included in this repository. This included modifying eps_ds within snow_backscatter.m so that brine-wetted snow dielectrics (calculated using the Snow Microwave Radiative Transfer model (SMRT)'s Python implementation of Geldsetzer-et-al-2009's equations) could be used instead of dry snow dielectrics. Be careful of any hardcoded settings, such as dielectrics, snow depths, or discrepancies in function inputs!; it's probably best to run diffs between this repository and the repository that it forked to look for these:) Happy simulating!


References:

A Facet-Based Numerical Model for Simulating SAR Altimeter Echoes From Heterogeneous Sea Ice Surfaces
Jack C. Landy; Michel Tsamados; Randall K. Scharien
10.1109/TGRS.2018.2889763
https://github.com/jclandy/FBEM (which this repository is forked from!)


SMRT : Snow Microwave Radiative Transfer model
© 2023 Ghislain Picard, Melody Sandells and Henning Löwe 
https://smrt-model.science/
https://doi.org/10.5194/gmd-11-2763-2018


Dielectric properties of brine-wetted snow on first-year sea ice (2009)
Geldsetzer, Torsten and Langlois, Alexandre and Yackel, John
Cold Regions Science and Technology
https://doi.org/10.1016/j.coldregions.2009.03.009



