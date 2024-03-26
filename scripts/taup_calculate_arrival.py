import numpy as np
import pandas as pd
import utm
import time
import glob
import os
import obspy
from tqdm import tqdm
import geopy.distance as geo
from obspy.taup import TauPyModel
from obspy.core.utcdatetime import UTCDateTime
from joblib import Parallel, delayed

nterra = 8531
nkkfls = 8531
cha_spac = 9.571428

model = TauPyModel(model="iasp91")

#############################################################################
# interpolate channel location

kkfls = pd.read_csv("../../datasets/cables/KKFLS_geom.xy",
                    header=None, names=["lon", "lat"], delim_whitespace=True)
terra = pd.read_csv("../../datasets/cables/TERRA_geom.xy",
                    header=None, names=["lon", "lat"], delim_whitespace=True)

for idx, i in kkfls.iterrows():
    kkfls.loc[idx, 'x'] = utm.from_latlon(i['lat'], i['lon'])[0]
    kkfls.loc[idx, 'y'] = utm.from_latlon(i['lat'], i['lon'])[1]

dt = 0
kkfls.loc[0, 'dist'] = 0
for i in range(len(kkfls)-1):
    dx = kkfls.loc[i+1]['x'] - kkfls.loc[i]['x']
    dy = kkfls.loc[i+1]['y'] - kkfls.loc[i]['y']
    dt += np.sqrt(dx ** 2 + dy ** 2)
    kkfls.loc[i+1, 'dist'] = dt

for idx, i in terra.iterrows():
    terra.loc[idx, 'x'] = utm.from_latlon(i['lat'], i['lon'])[0]
    terra.loc[idx, 'y'] = utm.from_latlon(i['lat'], i['lon'])[1]

dt = 0
terra.loc[0, 'dist'] = 0
for i in range(len(terra)-1):
    dx = terra.loc[i+1]['x'] - terra.loc[i]['x']
    dy = terra.loc[i+1]['y'] - terra.loc[i]['y']
    dt += np.sqrt(dx ** 2 + dy ** 2)
    terra.loc[i+1, 'dist'] = dt

kkfls = kkfls[kkfls['dist'] <= nkkfls * cha_spac]
terra = terra[terra['dist'] <= nterra * cha_spac]

kkfls_lon = np.interp(np.arange(0, nkkfls*cha_spac, cha_spac), kkfls['dist'], kkfls['lon'])
kkfls_lat = np.interp(np.arange(0, nkkfls*cha_spac, cha_spac), kkfls['dist'], kkfls['lat'])

terra_lon = np.interp(np.arange(0, nterra*cha_spac, cha_spac), terra['dist'], terra['lon'])
terra_lat = np.interp(np.arange(0, nterra*cha_spac, cha_spac), terra['dist'], terra['lat'])

#############################################################################
def calculate_ps(i, rlat, rlon, olat, lon, depth):
    dist = geo.great_circle((olat, olon), (rlat[i], rlon[i]))
    dist_deg = dist.km * 180. / (np.pi * dist.RADIUS)
    _p = model.get_travel_times(source_depth_in_km=depth,
                        distance_in_degree=dist_deg,
                        phase_list=['P','p'])[0].time
    _s = model.get_travel_times(source_depth_in_km=depth,
                        distance_in_degree=dist_deg,
                        phase_list=['S','s'])[0].time
    return [_p, _s]

for event in tqdm(sorted(glob.glob("../../datasets/earthquakes/*.h5"))):
    eid = event.split("/")[-1].split(".")[0]
    if os.path.exists(f"../../datasets/arrivals/{eid}.csv"):
        continue
    e = obspy.read_events(f"../../datasets/quakeml/{eid}.qml")[0]
    pmag = e.preferred_magnitude()
    porig = e.preferred_origin()

    depth = porig.depth / 1e3
    olat = porig.latitude
    olon = porig.longitude
    
    try:
        t_kkfls = np.array(Parallel(n_jobs=40)(delayed(calculate_ps)(i, kkfls_lat, kkfls_lon, olat, olon, depth)
                                       for i in range(nkkfls)))
        t_terra = np.array(Parallel(n_jobs=40)(delayed(calculate_ps)(i, terra_lat, terra_lon, olat, olon, depth)
                                       for i in range(nterra)))

        df = pd.DataFrame({
            "cable": ["KKFLS"]*nkkfls + ['TERRA']*nterra,
            "channel_index": np.concatenate([np.arange(nkkfls), np.arange(nterra)]),
            "t_p": np.concatenate([t_kkfls[:, 0], t_terra[:, 0]]),
            "t_s": np.concatenate([t_kkfls[:, 1], t_terra[:, 1]]),
        })
        df.to_csv(f"../../datasets/arrivals/{eid}.csv", index=False, float_format="%.3f")
    except:
        pass
