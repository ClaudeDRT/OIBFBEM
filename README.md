# OIBFBEM
## Claude de Rijke-Thomas
## Tuning FBEM (by Dr Jack Landy) to simulate Airborne Observations

This branch of the Facet-Based Echo Model (FBEM) is for qualitative evaluations made using airborne data from Operation IceBridge Spring 2016. This work corresponds to my (Claude de Rijke-Thomas') final research chapter of my doctoral thesis.

For this to be possible, FBEM was modified to account for unfocused SAR processing (instead of Delay-Doppler SAR). The air-snow interface topography inputted into the FBEM used a Generalised Additive Model (GAM) surface created from NASA's ATM lidar. Either the air-snow
interface GAM was used as a proxy for the snow-ice interface topography (within the OIB.m script), or the snow-ice interface topography
was individually estimated by creating a GAM surface of combined lidar+coincident $in\ situ$ snow depths (within OIB_indep_interfaces.m).

These modifications to FBEM are not particularly review-ready (nor in my primary language), but I thought I would keep it open source for potential future studies:) Please let me know if you have any questions by contacting me @ claude[at]derijke.org , or Jack at jack.c.landy[at]uit.no . Please refer to the other README file for copyright details.

There are some hardcoded modifications that I made within the files themselves (within snow_backscatter.m) that I haven't included in this repository. This included modifying eps_ds within snow_backscatter.m so that brine-wetted snow dielectrics (calculated using SMRT's Python implementation of Geldsetzer-et-al-2009's equations) could be used instead of dry snow dielectrics.






