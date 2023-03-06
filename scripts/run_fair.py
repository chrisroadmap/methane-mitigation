import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from parallel import single_fair_run
from utils import _parallel_process

# reduce N_SCEN and N_ENS for testing
N_SCEN = 113  # 113 total: how many baseline scenarios. Times by 9 for number of runs
N_CONF = 1001  # 1001 total: make a smaller number for testing
N_WORKERS = 19 # parallel workers
FRONT_SERIAL = 0    # non-zero if testing
FRONT_PARALLEL = 0  # non-zero if testing

if __name__ == "__main__":
    print("Running ensemble members...")

    df_scenarios = pd.read_csv("../data/gains_scenarios_harmonized.csv")
    df_scenarios[['scenario', 'variant']] = df_scenarios['scenario'].str.split("|", expand=True)

    conf = []
    i_scen = 0
    for ms, data in df_scenarios.groupby(["model", "scenario"]):
        conf.append(
            {
                'model': ms[0],
                'scenario_base': ms[1],
                'i_scen': i_scen
#                'configs': N_CONF
            }
        )
        i_scen = i_scen + 1
        if i_scen >= N_SCEN:
            break

    parallel_process_kwargs = dict(
        func=single_fair_run,
        configuration=conf,
        config_are_kwargs=False,
        front_serial=FRONT_SERIAL,
        front_parallel=FRONT_PARALLEL
    )

    os.makedirs("../results", exist_ok=True)

    print("Running FaIR in parallel...")
    with ProcessPoolExecutor(N_WORKERS) as pool:
        _parallel_process(
            **parallel_process_kwargs,
            pool=pool,
        )

    #print(df_scenarios["scenario"].unique()[:n_scen_variant])

