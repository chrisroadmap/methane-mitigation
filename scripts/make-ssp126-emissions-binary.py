from fair import FAIR, __version__
from fair.io import read_properties

print("Making SSP emissions binary...")

scenarios = ["ssp126"]

species, properties = read_properties()

f = FAIR(ch4_method="thornhill2021")
f.define_time(1750, 2101, 1)
f.define_configs(["unspecified"])
f.define_scenarios(scenarios)
f.define_species(species, properties)

f.allocate()
f.fill_from_rcmip()

f.emissions.to_netcdf("../data/ssp126_emissions_1750-2100.nc")
