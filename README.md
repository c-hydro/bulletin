# bulletin

**BULLETIN - Boîte à oUtils opérationneLLe pour la prÉvision des impacTs des rIsques Naturels** is a toolbox for the operational impact-based forecasting (IBF) of natural hazards.

It consists of different tools for dealing with:
- **Meteorological hazards**: natively supporting several global meteorological models (NOAA-GFS, DWD-ICON, CMC-GEM, ECMWF)  and easy to configure with local deterministic models in netcdf formats allow to assess the impact of forecasted meteorological variables like rainfall, wind and temperature.
- **Hydrological hazards**: analyse the output of JRC-GLOFAS global open hydrological model or of local implementations of the FloodPROOFS CIMA flood monitoring system to assess the impact-based flood hazard
- **Drought hazards**: analyse drought indexes provided as geotif files, to assess the impact of droughts

The tools are consistent with the risk definiton provided by UNDRR and allow to perform an impact-based forecast by considering different exposure layers provided as geotiff files.

![IBF_CIMA](https://github.com/c-hydro/bulletin/assets/57633516/a60ef6f1-8179-4147-a6f0-7936d06fe76c)
