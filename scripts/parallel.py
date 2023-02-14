# put imports outside: we don't have a lot of overhead here, and it looks nicer.
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

def single_fair_run(cfg):
    model = cfg["model"]
    scenario_base = cfg["scenario_base"]
    scenario_variants = ['Baseline_CLE', 'MFR_tech', 'MFR_explore', 'MFR_struc',
       'MFR_behavior', 'MFR_develop']
    scenarios = []
    for variant in scenario_variants:
        scenarios.append(scenario_base + "|" + variant)

    df_configs = pd.read_csv("../data/calibrated_constrained_parameters_v1.0.csv", index_col=0)
    configs = np.array(list(df_configs.index))

    species, properties = read_properties()

    f = FAIR(ch4_method="Thornhill2021")
    f.define_time(1750, 2101, 1)
    f.define_scenarios(scenarios)
    f.define_configs(configs)
    f.define_species(species, properties)
    f.allocate()

    trend_shape = np.ones(352)
    trend_shape[:271] = np.linspace(0, 1, 271)

    df_solar = pd.read_csv(
        "../data/solar_erf_timebounds.csv", index_col="year"
    )
    df_volcanic = pd.read_csv(
        "../data/volcanic_ERF_monthly_-950001-201912.csv"
    )

    volcanic_forcing = np.zeros(352)
    for i, year in enumerate(np.arange(1750, 2021)):
        volcanic_forcing[i] = np.mean(
            df_volcanic.loc[
                ((year - 1) <= df_volcanic["year"]) & (df_volcanic["year"] < year)
            ].erf
        )
    volcanic_forcing[271:281] = np.linspace(1, 0, 10) * volcanic_forcing[270]

    solar_forcing = df_solar["erf"].loc[1750:2101].values

    # grab default emissions first, then we override relevant species
    da_emissions = xr.load_dataarray("../data/ssp126_emissions_1750-2100.nc")
    da = da_emissions.loc[dict(config="unspecified", scenario="ssp126")]
    fe = da.expand_dims(dim=["scenario", "config"], axis=(1, 2))
    fe = fe.drop(["scenario", "config"]) * np.ones((1, 6, 1001, 1))
    f.emissions = fe.assign_coords({"scenario": scenarios, "config": configs})

    df_scenarios = pd.read_csv("../data/gains_scenarios_harmonized.csv")
    species_mapping = {
        "CO2 FFI": "Emissions|CO2|Energy and Industrial Processes",
        "CO2 AFOLU": "Emissions|CO2|AFOLU",
        "CH4": "Emissions|CH4",
        "N2O": "Emissions|N2O",
        "Sulfur": "Emissions|Sulfur",
        "CO": "Emissions|CO",
        "VOC": "Emissions|VOC",
        "NOx": "Emissions|NOx",
        "BC": "Emissions|BC",
        "OC": "Emissions|OC",
        "NH3": "Emissions|NH3"
    }
    unit_convert = {specie: 1 for specie in species_mapping}
    unit_convert["CO2 FFI"] = 0.001
    unit_convert["CO2 AFOLU"] = 0.001
    unit_convert["N2O"] = 0.001

    for scenario in scenarios:
        for specie in species_mapping:
            f.emissions.loc[
                dict(
                    scenario=scenario,
                    timepoints=slice(2014.5, 2101),
                    specie=specie
                )
            ] = df_scenarios.loc[
                (df_scenarios['model']==model)&
                (df_scenarios['scenario']==scenario)&
                (df_scenarios['variable']==species_mapping[specie]),
                '2014':'2100'
            ].values.T * unit_convert[specie]

    # solar and volcanic forcing
    fill(
        f.forcing,
        volcanic_forcing[:, None, None] * df_configs["scale Volcanic"].values[None, None, :],
        specie="Volcanic",
    )
    fill(
        f.forcing,
        solar_forcing[:, None, None] * df_configs["solar_amplitude"].values[None, None, :]
        + trend_shape[:, None, None] * df_configs["solar_trend"].values[None, None, :],
        specie="Solar",
    )

    # climate response
    fill(
        f.climate_configs["ocean_heat_capacity"],
        np.array([df_configs["c1"], df_configs["c2"], df_configs["c3"]]).T,
    )
    fill(
        f.climate_configs["ocean_heat_transfer"],
        np.array([df_configs["kappa1"], df_configs["kappa2"], df_configs["kappa3"]]).T,
    )
    fill(f.climate_configs["deep_ocean_efficacy"], df_configs["epsilon"])
    fill(f.climate_configs["gamma_autocorrelation"], df_configs["gamma"])
    fill(f.climate_configs["sigma_eta"], df_configs["sigma_eta"])
    fill(f.climate_configs["sigma_xi"], df_configs["sigma_xi"])
    fill(f.climate_configs["seed"], df_configs["seed"])
    fill(f.climate_configs["stochastic_run"], True)
    fill(f.climate_configs["use_seed"], True)
    fill(f.climate_configs["forcing_4co2"], df_configs["F_4xCO2"])

    # species level
    f.fill_species_configs()

    # carbon cycle
    fill(f.species_configs["iirf_0"], df_configs["r0"], specie="CO2")
    fill(f.species_configs["iirf_airborne"], df_configs["rA"], specie="CO2")
    fill(f.species_configs["iirf_uptake"], df_configs["rU"], specie="CO2")
    fill(f.species_configs["iirf_temperature"], df_configs["rT"], specie="CO2")

    # methane lifetime baseline - should be imported from calibration
    fill(f.species_configs["unperturbed_lifetime"], 10.11702748, specie="CH4")

    # emissions adjustments for N2O and CH4
    fill(f.species_configs["baseline_emissions"], 19.019783117809567, specie="CH4")
    fill(f.species_configs["baseline_emissions"], 0.08602230754, specie="N2O")

    # aerosol indirect
    fill(f.species_configs["aci_scale"], df_configs["beta"])
    fill(f.species_configs["aci_shape"], df_configs["shape Sulfur"], specie="Sulfur")
    fill(f.species_configs["aci_shape"], df_configs["shape BC"], specie="BC")
    fill(f.species_configs["aci_shape"], df_configs["shape OC"], specie="OC")

    # forcing scaling
    fill(f.species_configs["forcing_scale"], df_configs["scale CO2"], specie="CO2")
    fill(f.species_configs["forcing_scale"], df_configs["scale CH4"], specie="CH4")
    fill(f.species_configs["forcing_scale"], df_configs["scale N2O"], specie="N2O")
    # fill(f.species_configs['forcing_scale'], cfg['scaling_minorGHG'], specie='CO2')
    fill(
        f.species_configs["forcing_scale"],
        df_configs["scale Stratospheric water vapour"],
        specie="Stratospheric water vapour",
    )
    fill(
        f.species_configs["forcing_scale"], df_configs["scale Contrails"], specie="Contrails"
    )
    fill(
        f.species_configs["forcing_scale"],
        df_configs["scale Light absorbing particles on snow and ice"],
        specie="Light absorbing particles on snow and ice",
    )
    fill(f.species_configs["forcing_scale"], df_configs["scale Land use"], specie="Land use")

    for specie in [
        "CFC-11",
        "CFC-12",
        "CFC-113",
        "CFC-114",
        "CFC-115",
        "HCFC-22",
        "HCFC-141b",
        "HCFC-142b",
        "CCl4",
        "CHCl3",
        "CH2Cl2",
        "CH3Cl",
        "CH3CCl3",
        "CH3Br",
        "Halon-1211",
        "Halon-1202",
        "Halon-1301",
        "Halon-2402",
        "CF4",
        "C2F6",
        "C3F8",
        "c-C4F8",
        "C4F10",
        "C5F12",
        "C6F14",
        "C7F16",
        "C8F18",
        "NF3",
        "SF6",
        "SO2F2",
        "HFC-125",
        "HFC-134a",
        "HFC-143a",
        "HFC-152a",
        "HFC-227ea",
        "HFC-23",
        "HFC-236fa",
        "HFC-245fa",
        "HFC-32",
        "HFC-365mfc",
        "HFC-4310mee",
    ]:
        fill(f.species_configs["forcing_scale"], df_configs["scale minorGHG"], specie=specie)

    # aerosol radiation interactions
    fill(f.species_configs["erfari_radiative_efficiency"], df_configs["ari BC"], specie="BC")
    fill(f.species_configs["erfari_radiative_efficiency"], df_configs["ari CH4"], specie="CH4")
    fill(f.species_configs["erfari_radiative_efficiency"], df_configs["ari N2O"], specie="N2O")
    fill(f.species_configs["erfari_radiative_efficiency"], df_configs["ari NH3"], specie="NH3")
    fill(f.species_configs["erfari_radiative_efficiency"], df_configs["ari NOx"], specie="NOx")
    fill(f.species_configs["erfari_radiative_efficiency"], df_configs["ari OC"], specie="OC")
    fill(
        f.species_configs["erfari_radiative_efficiency"],
        df_configs["ari Sulfur"],
        specie="Sulfur",
    )
    fill(f.species_configs["erfari_radiative_efficiency"], df_configs["ari VOC"], specie="VOC")
    fill(
        f.species_configs["erfari_radiative_efficiency"],
        df_configs["ari Equivalent effective stratospheric chlorine"],
        specie="Equivalent effective stratospheric chlorine",
    )

    # Ozone
    fill(
        f.species_configs["ozone_radiative_efficiency"], df_configs["o3 CH4"], specie="CH4"
    )
    fill(
        f.species_configs["ozone_radiative_efficiency"], df_configs["o3 N2O"], specie="N2O"
    )
    fill(f.species_configs["ozone_radiative_efficiency"], df_configs["o3 CO"], specie="CO")
    fill(
        f.species_configs["ozone_radiative_efficiency"], df_configs["o3 VOC"], specie="VOC"
    )
    fill(
        f.species_configs["ozone_radiative_efficiency"], df_configs["o3 NOx"], specie="NOx"
    )
    fill(
        f.species_configs["ozone_radiative_efficiency"],
        df_configs["o3 Equivalent effective stratospheric chlorine"],
        specie="Equivalent effective stratospheric chlorine",
    )

    # tune down volcanic efficacy
    fill(f.species_configs["forcing_efficacy"], 0.6, specie="Volcanic")

    # CO2 in 1750
    fill(f.species_configs["baseline_concentration"], df_configs["co2_concentration_1750"], specie="CO2")

    # initial conditions
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    f.run(progress=False)

    average_51yr = np.ones(52)
    average_51yr[0] = 0.5
    average_51yr[-1] = 0.5

    # at this point dump out some batch output
    temp_out = f.temperature[:, :, :, 0].data
    ohc_out = f.ocean_heat_content_change[:, :, :].data
    erf_out = f.forcing_sum[:, :, :].data
    ch4_out = f.concentration[:, :, :, 3].data

    idx0 = 0 if cfg['i_scen']==0 else 265
    year0 = 1750 if cfg['i_scen']==0 else 2015

    ds = xr.Dataset(
        {
            "temperature": (
                ["year", "scenario", "run"],
                temp_out[idx0:, :, :]
                - np.average(temp_out[100:152, :], weights=average_51yr, axis=0),
            ),
            "effective_radiative_forcing": (["year", "scenario", "run"], erf_out[idx0:, :, :]),
            "ocean_heat_content_change": (["year", "scenario", "run"], ohc_out[idx0:, :, :]),
            "ch4_concentration": (["year", "scenario", "run"], ch4_out[idx0:, :, :]),
        },
        coords={"year": (np.arange(year0, 2101.5)), "scenario": scenario_variants, "run": configs},
    )
    ds.to_netcdf("../results/%s_%s.nc" % (model, scenario_base))
    ds.close()
