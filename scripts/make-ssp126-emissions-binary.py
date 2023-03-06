from fair import FAIR, __version__
from fair.io import read_properties
import pandas as pd
import pooch

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

# Fix NOx.
rcmip_emissions_file = pooch.retrieve(
    url="doi:10.5281/zenodo.4589756/rcmip-emissions-annual-means-v5-1-0.csv",
    known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
)
df_emis = pd.read_csv(rcmip_emissions_file)
gfed_sectors = [
    "Emissions|NOx|MAGICC AFOLU|Agricultural Waste Burning",
    "Emissions|NOx|MAGICC AFOLU|Forest Burning",
    "Emissions|NOx|MAGICC AFOLU|Grassland Burning",
    "Emissions|NOx|MAGICC AFOLU|Peat Burning"
]
for scenario in scenarios:
    f.emissions.loc[dict(specie="NOx", scenario=scenario, config="unspecified")] = (
        df_emis.loc[
            (df_emis["Scenario"] == scenario)
            & (df_emis["Region"] == "World")
            & (df_emis["Variable"].isin(gfed_sectors)),
            "1750":"2500",
        ].interpolate(axis=1).values.squeeze().sum(axis=0) * 46.006/30.006 + df_emis.loc[
            (df_emis["Scenario"] == scenario)
            & (df_emis["Region"] == "World")
            & (df_emis["Variable"] == "Emissions|NOx|MAGICC AFOLU|Agriculture"),
            "1750":"2500",
        ].interpolate(axis=1).values.squeeze() + df_emis.loc[
            (df_emis["Scenario"] == scenario)
            & (df_emis["Region"] == "World")
            & (df_emis["Variable"] == "Emissions|NOx|MAGICC Fossil and Industrial"),
            "1750":"2500",
        ].interpolate(axis=1).values.squeeze()
    )[:351]


f.emissions.to_netcdf("../data/ssp126_emissions_1750-2100.nc")
