#!/usr/bin/env python
# coding: utf-8

"""This script downloads the data from GFED we need to add to GAINS emissions."""


import os
from pathlib import PurePath

from calendar import monthrange
import h5py
import numpy as np
import pooch
import pandas as pd
from tqdm.auto import tqdm
import xarray as xr

print("Getting CMIP6 biomass burning data...")
urls_hashes = [
    (
        "areacella",
        "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/CMIP/VUA/VUA-CMIP-BB4CMIP6-1-2/atmos/fx/gridcellarea/gn/v20161213/gridcellarea-em-biomassburning_input4MIPs_emissions_CMIP_VUA-CMIP-BB4CMIP6-1-2_gn.nc",
        "c9da1b6f715dfeb4cb706a4946fe874e251023103beeb884f4b6b51c034f24e3"
    ), (
        "emissions",
        "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/CMIP/VUA/VUA-CMIP-BB4CMIP6-1-2/atmos/mon/CH4/gn/v20161213/CH4-em-biomassburning_input4MIPs_emissions_CMIP_VUA-CMIP-BB4CMIP6-1-2_gn_185001-201512.nc",
        "75957f8b7379a339989a425a3432164b6f533f3cab83b495ce8a1830eb30f2e2"
    ), (
        "DEFO",
        "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/CMIP/VUA/VUA-CMIP-BB4CMIP6-1-2/atmos/mon/CH4-percentage-DEFO/gn/v20161213/CH4-percentage-DEFO-em-biomassburning_input4MIPs_emissions_CMIP_VUA-CMIP-BB4CMIP6-1-2_gn_175001-201512.nc",
        "8bb61cf08c2487c9b0f6f5f642f00bd46f2be82e7c1f94df239b6b24f5841974"
    ), (
        "TEMF",
        "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/CMIP/VUA/VUA-CMIP-BB4CMIP6-1-2/atmos/mon/CH4-percentage-TEMF/gn/v20161213/CH4-percentage-TEMF-em-biomassburning_input4MIPs_emissions_CMIP_VUA-CMIP-BB4CMIP6-1-2_gn_175001-201512.nc",
        "03c98bbbbf8a82c533516cff56e02980f92ea0d8d3fe60cb9f7821cdfd5593f3"
    ), (
        "BORF",
        "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/CMIP/VUA/VUA-CMIP-BB4CMIP6-1-2/atmos/mon/CH4-percentage-BORF/gn/v20161213/CH4-percentage-BORF-em-biomassburning_input4MIPs_emissions_CMIP_VUA-CMIP-BB4CMIP6-1-2_gn_175001-201512.nc",
        "657377aa184c60003a86aa49123a1074b7c856c266d5a33797f991d83342136a"
    ), (
        "SAVA",
        "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/CMIP/VUA/VUA-CMIP-BB4CMIP6-1-2/atmos/mon/CH4-percentage-SAVA/gn/v20161213/CH4-percentage-SAVA-em-biomassburning_input4MIPs_emissions_CMIP_VUA-CMIP-BB4CMIP6-1-2_gn_175001-201512.nc",
        "796c9e59a77cc4b31821935b3b21808ee74d05ee85c6017281e7aeee8c4aac39"
    ), (
        "PEAT",
        "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/CMIP/VUA/VUA-CMIP-BB4CMIP6-1-2/atmos/mon/CH4-percentage-PEAT/gn/v20161213/CH4-percentage-PEAT-em-biomassburning_input4MIPs_emissions_CMIP_VUA-CMIP-BB4CMIP6-1-2_gn_175001-201512.nc",
        "72b19eddb517b195ab30fbb3f8ffcd9d2d59ac75e4b61111545e52fc32ee86c4"
    )
]
files = {}
for key, url, hash in urls_hashes:
    files[key] = pooch.retrieve(url, hash)

gridcellarea = xr.open_dataset(files["areacella"]).gridcellarea

frac = {}
fracvarname = {}
for part in ['DEFO', 'TEMF', 'BORF', 'SAVA', 'PEAT']:
    frac[part] = xr.open_dataset(files[part])
    fracvarname[part] = frac[part].variable_id

def calculate_annual_series(files):
    """Return aggregated, annual series"""
    ds = xr.open_dataset(files["emissions"])
    name = ds.variable_id
    assert ds[name].units == "kg m-2 s-1"
    series = pd.Series()
    series.name = name

    years = np.arange(1990, 1997)#np.unique(ds.time.dt.year)
    for year in tqdm(years):
        sel = ds[name].sel(time=str(year))
        selpart = {}
        for part in ['DEFO', 'TEMF', 'BORF', 'SAVA', 'PEAT']:
            selpart[part] = frac[part][fracvarname[part]].sel(time=str(year))
        days_in_year = [
            v[1]
            for v in [
                monthrange(int(i.dt.year.values), int(i.dt.month.values))
                for i in sel.time
            ]
        ]
        sel_kg_s = sel * gridcellarea * (
            selpart['DEFO'] + selpart['TEMF'] + selpart['BORF'] + selpart['SAVA'] + selpart['PEAT']
        )/100
        sel_monthly_kg = (
            sel_kg_s.sum(dim=["latitude", "longitude"]).to_series()
            * days_in_year
            * 24
            * 3600
        )
        sel_annual_Tg = sel_monthly_kg.sum() / 10 ** 9
        series.at[year] = sel_annual_Tg

    return series

bb4cmip = calculate_annual_series(files)

print("Getting GFED data...")

hashes = {
 'emissions_factors': '5f68c5c4ffdb7d81d3d2fefa662dcad9dd66f2b4097350a08a045523626383b2',
 1997: '997f54a532cae524757c3b35808c10ae0f71ce231c213617cb34ba4b72968bb9',
 1998: '36c13cdcec4f4698f3ab9f05bc83d2307252d89b81da5a14efd8e171148a6dc0',
 1999: '5d0d18b09d9a76e305522c5b46a97bf3180d9301d1d3c6bfa5a4c838fb0fa452',
 2000: 'ddbeff2326dded0e2248afd85c3ec7c84a36c6919711632e717d00985cd4ad6d',
 2001: '1b684bf0b348e92a5d63ea660564f01439f69c4eb88eacd46280237d51ce5815',
 2002: 'dcf624961512dbb93759248bc2b75d404b3be68f1f6fdcb01f0c7dc7f11a517a',
 2003: '91d61b67d04b4a32d534f5d68ae1de7929f7ea75bb9d25d3273c4d5d75bda4d3',
 2004: '931e063f796bf1f7d391d3f03342d2dd2ad1b234cb317f826adfab201003f4cd',
 2005: '159e7704d14089496d051546c20b644a443308eeb7d79bf338226af2b4bdc2b7',
 2006: 'a69d5bf6b8fa3324c2922aac07306ec6e488a850ca4f42d09a397cee30eebd4c',
 2007: '1d7f77e6f7b13cc2a8ef9d26ecb9ea3d18e70cfeb8a47e7ecb26f9613888f937',
 2008: 'bd3771b9b3032d459a79c0da449fdb497cd3400e0e07a0da6b41e930fc5d3e14',
 2009: '36ea9b6036cd0ff3672502c3c04180bd209ddb192f86a2e791a2b896308bc5ff',
 2010: '5b2d30b5ddc3e20c38c7971faf6791b313b1bbff22e8bc2b14ca7ea9079aa12c',
 2011: 'fb19c001bef26ca23d07dd8978fd998f4692bdecdec5eb86b91d4b1ffb4a9aa7',
 2012: '08033c90295bbc208fac426e01809b68cef62997668085b1e096d8a61ab43e9b',
 2013: 'cf5249811af4b7099f886e61125dcd15c1127b6125392fe8358d3f0bf8ddb064',
 2014: 'a293b4c6e03898a0dc184a082a37435673916a02ff02c06668152dcc4d4b8405',
 2015: 'c043e96a421247afbeb6580fca0bcddf8160180b14d37b13122fc3110534b309',
 2016: '2f3b54ff5698ba7f7aa2bb1d4b5e5f95124c0e0db32830ed94aa04bea2cbc2a6',
 2017: '35281b25654d6e2995c7a2d1ba673a2ec2381c5144fb900f307166d0aec76f49',
 2018: '6fca43abad4ca43627641f0ce8c759685ecdfe5ba4b15684e139cc4a59572f81',
 2019: 'e8f38c56a7d66f65de2bbd457885227fb830aee24ece1863ca5eb63bde16ce6f',
 2020: '395912f42e5e922e024dff779b436b21582e6a2868ddec2dc4e65ac90b233a11',
}

files = {}
for year in range(1997, 2017):
    files[year] = pooch.retrieve(
        f"https://www.geo.vu.nl/~gwerf/GFED/GFED4/GFED4.1s_{year}.hdf5",
        f"{hashes[year]}"
    )
for year in range(2017, 2021):
    files[year] = pooch.retrieve(
        f"https://www.geo.vu.nl/~gwerf/GFED/GFED4/GFED4.1s_{year}_beta.hdf5",
        f"{hashes[year]}"
    )

files['emissions_factors'] = pooch.retrieve(
    "https://www.geo.vu.nl/~gwerf/GFED/GFED4/ancill/GFED4_Emission_Factors.txt",
    hashes['emissions_factors']
)

efs = pd.read_csv(files['emissions_factors'], comment='#', delim_whitespace=True, index_col=0, header=None)
efs.columns = ['SAVA', 'BORF', 'TEMF', 'DEFO', 'PEAT', 'AGRI']
efs.index.rename('SPECIE', inplace=True)


sources=['SAVA', 'BORF', 'TEMF', 'DEFO', 'PEAT']
# we do not want agricultural waste burning. This is part of GAINS.
species=['CH4']

months       = '01','02','03','04','05','06','07','08','09','10','11','12'

start_year = 1997
end_year   = 2020


"""
make table with summed DM emissions for each region, year, and source
"""
table = np.zeros((1, end_year - start_year + 1)) # region, year

for year in range(start_year, end_year+1):
    print(year)
    f = h5py.File(files[year], 'r')


    if year == start_year: # these are time invariable
        grid_area     = f['/ancill/grid_cell_area'][:]

    emissions = np.zeros((1, 720, 1440))
    for month in range(12):
        # read in DM emissions
        string = '/emissions/'+months[month]+'/DM'
        DM_emissions = f[string][:]
        for ispec, specie in enumerate(species):
            for isrc, source in enumerate(sources):
                # read in the fractional contribution of each source
                string = '/emissions/'+months[month]+'/partitioning/DM_'+source
                contribution = f[string][:]
                # calculate emissions as the product of DM emissions (kg DM per
                # m2 per month), the fraction the specific source contributes to
                # this (unitless), and the emission factor (g per kg DM burned)
                emissions[ispec, ...] += DM_emissions * contribution * efs.loc[specie, source]
                #print(emissions[:, 88, 0])


    # fill table with total values for the globe (row 15) or basisregion (1-14)
    #mask = np.ones((720, 1440))
    table[:, year-start_year] = np.sum(grid_area[None, ...] * emissions, axis=(1,2))

table = table / 1E12

consol = np.concatenate((np.array([bb4cmip.values]), table), axis=1)

df_out = pd.DataFrame(consol, columns=range(1990, 2021), index=species)

os.makedirs(
    f"../data/",
    exist_ok=True,
)

df_out.to_csv('../data/gfed4.1s_ch4_biomass_burning_1997-2020.csv')
