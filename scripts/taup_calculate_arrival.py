import glob
import os

import geopy.distance as geo
import numpy as np
import obspy
import pandas as pd
import utm
from joblib import Parallel, delayed
from obspy.taup import TauPyModel
from tqdm import tqdm

nterra = 8531
nkkfls = 8531
nkkfln = 8531
cha_spac = 9.571428

model = TauPyModel(model="iasp91")

#############################################################################
# interpolate channel location

kkfls = pd.read_csv(
    "../data/KKFLS_geom.xy",
    header=None,
    names=["lon", "lat"],
    delim_whitespace=True,
)
kkfln = pd.read_csv(
    "../data/KKFLN_geom.xy",
    header=None,
    names=["lon", "lat"],
    delim_whitespace=True,
)
terra = pd.read_csv(
    "../data/TERRA_geom.xy",
    header=None,
    names=["lon", "lat"],
    delim_whitespace=True,
)

#############################################################################

for idx, i in kkfls.iterrows():
    kkfls.loc[idx, "x"] = utm.from_latlon(i["lat"], i["lon"])[0]
    kkfls.loc[idx, "y"] = utm.from_latlon(i["lat"], i["lon"])[1]

dt = 0
kkfls.loc[0, "dist"] = 0
for i in range(len(kkfls) - 1):
    dx = kkfls.loc[i + 1]["x"] - kkfls.loc[i]["x"]
    dy = kkfls.loc[i + 1]["y"] - kkfls.loc[i]["y"]
    dt += np.sqrt(dx**2 + dy**2)
    kkfls.loc[i + 1, "dist"] = dt

#############################################################################

for idx, i in kkfln.iterrows():
    kkfln.loc[idx, "x"] = utm.from_latlon(i["lat"], i["lon"])[0]
    kkfln.loc[idx, "y"] = utm.from_latlon(i["lat"], i["lon"])[1]

dt = 0
kkfln.loc[0, "dist"] = 0
for i in range(len(kkfln) - 1):
    dx = kkfln.loc[i + 1]["x"] - kkfln.loc[i]["x"]
    dy = kkfln.loc[i + 1]["y"] - kkfln.loc[i]["y"]
    dt += np.sqrt(dx**2 + dy**2)
    kkfln.loc[i + 1, "dist"] = dt

#############################################################################

for idx, i in terra.iterrows():
    terra.loc[idx, "x"] = utm.from_latlon(i["lat"], i["lon"])[0]
    terra.loc[idx, "y"] = utm.from_latlon(i["lat"], i["lon"])[1]

dt = 0
terra.loc[0, "dist"] = 0
for i in range(len(terra) - 1):
    dx = terra.loc[i + 1]["x"] - terra.loc[i]["x"]
    dy = terra.loc[i + 1]["y"] - terra.loc[i]["y"]
    dt += np.sqrt(dx**2 + dy**2)
    terra.loc[i + 1, "dist"] = dt

#############################################################################

kkfls = kkfls[kkfls["dist"] <= nkkfls * cha_spac]
kkfln = kkfln[kkfln["dist"] <= nkkfln * cha_spac]
terra = terra[terra["dist"] <= nterra * cha_spac]

kkfls_lon = np.interp(
    np.arange(0, nkkfls * cha_spac, cha_spac), kkfls["dist"], kkfls["lon"]
)
kkfls_lat = np.interp(
    np.arange(0, nkkfls * cha_spac, cha_spac), kkfls["dist"], kkfls["lat"]
)

kkfln_lon = np.interp(
    np.arange(0, nkkfln * cha_spac, cha_spac), kkfln["dist"], kkfln["lon"]
)
kkfln_lat = np.interp(
    np.arange(0, nkkfln * cha_spac, cha_spac), kkfln["dist"], kkfln["lat"]
)

terra_lon = np.interp(
    np.arange(0, nterra * cha_spac, cha_spac), terra["dist"], terra["lon"]
)
terra_lat = np.interp(
    np.arange(0, nterra * cha_spac, cha_spac), terra["dist"], terra["lat"]
)

#############################################################################
def calculate_ps(i, rlat, rlon, olat, olon, depth):
    dist = geo.great_circle((olat, olon), (rlat[i], rlon[i]))
    dist_deg = dist.km * 180.0 / (np.pi * dist.RADIUS)
    _p = model.get_travel_times(
        source_depth_in_km=depth, distance_in_degree=dist_deg, phase_list=["P", "p"]
    )[0].time
    _s = model.get_travel_times(
        source_depth_in_km=depth, distance_in_degree=dist_deg, phase_list=["S", "s"]
    )[0].time
    return [_p, _s]

for event in tqdm(sorted(glob.glob("../data/QuakeML/*.qml"))):
    eid = event.split("/")[-1].split(".")[0]
    if os.path.exists(f"../data/arrivals/{eid}.csv"):
        continue
    e = obspy.read_events(f"../data/QuakeML/{eid}.qml")[0]
    pmag = e.preferred_magnitude()
    porig = e.preferred_origin()

    depth = porig.depth / 1e3
    olat = porig.latitude
    olon = porig.longitude

    try:
        t_kkfls = np.array(
            Parallel(n_jobs=40)(
                delayed(calculate_ps)(i, kkfls_lat, kkfls_lon, olat, olon, depth)
                for i in range(nkkfls)
            )
        )
        t_kkfln = np.array(
            Parallel(n_jobs=40)(
                delayed(calculate_ps)(i, kkfln_lat, kkfln_lon, olat, olon, depth)
                for i in range(nkkfln)
            )
        )
        t_terra = np.array(
            Parallel(n_jobs=40)(
                delayed(calculate_ps)(i, terra_lat, terra_lon, olat, olon, depth)
                for i in range(nterra)
            )
        )

        df = pd.DataFrame(
            {
                "cable": ["KKFLS"] * nkkfls + ["KKFLN"] * nkkfln + ["TERRA"] * nterra,
                "channel_index": np.concatenate([np.arange(nkkfls), np.arange(nkkfln), np.arange(nterra)]),
                "t_p": np.concatenate([t_kkfls[:, 0], t_kkfln[:, 0], t_terra[:, 0]]),
                "t_s": np.concatenate([t_kkfls[:, 1], t_kkfln[:, 1], t_terra[:, 1]]),
            }
        )
        df.to_csv(
            f"../data/arrivals/{eid}.csv", index=False, float_format="%.3f"
        )
    except:
        pass
