# bulletin
*Multihazard bulletin warning tools*

A set of tools for operational impact-based forecasting of climate-related hazards.


It consists of different functions for dealing with:
- **Meteorological hazards**: supporting many meteorological models (NOAA-GFS, DWD-ICON, CMC-GEM, WRF) and easy to configure with local deterministic models in netcdf formats allow to assess the impact of forecasted meteorological variables like rainfall, wind and temperature.
- **Hydrological hazards**: analyse the output of JRC-GLOFAS global open hydrological model or of local implementations of the FloodPROOFS CIMA flood monitoring system to assess the impact-based flood hazard
- **Drought hazards**: analyse drought indexes provided as geotif files, to assess the impact of droughts

The tools are consistent with the risk definiton provided by UNDRR and allow to perform an impact-based forecast by considering different exposure layers provided as geotiff files.

![](/home/andrea/Pictures/Screenshots/IBF_CIMA.png)