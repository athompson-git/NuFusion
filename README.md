# NuFusion
### Calculate reaction rates and fluxes for neutrinos at a fusion reactor
Tokamak, Stellarators, and Pulsed machines in various temperatures and environments.

### Dependencies
* pynucastro
* openmc
* numpy
* scipy
* matplotlib


Installation notes go here:


### Directories

* ```/openmc/``` : python files for openmc API to simulate things
* ```/plots/``` : where we keep any plots of data
* ```/data/``` : to organize any simulated spectra (not including openmc temp files)
* ```/pynuc/``` : notebook explorers for Reaclib rates through ```pynucastro```
* ```/src/``` : Source code for special functions and classes for data analysis, including beta decay spectrum simulation and reaction tables



## Tokamak calculation

Geometry for steel, breeder, shield mix, and concrete bioshield.


## Pulsed devices

Geometry and energies go here.




#### Docker
```docker run -it --name=goofy_joliot openmc/openmc:latest```
