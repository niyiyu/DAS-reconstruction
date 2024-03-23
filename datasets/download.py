# Download data from Cook Inlet DAS Experiment
# Yiyu Ni (niyiyu@uw.edu)

import pandas as pd
import os
import wget

# download data for TERRA?
TERRA = False
if TERRA:
    print("INFO: will download TERRA data.")
    os.makedirs("./TERRA", exist_ok=True)

# download data for KKFLS?
KKFLS = False
if KKFLS:
    print("INFO: will download KKFL-S data.")
    os.makedirs("./KKFLS", exist_ok=True)

# download data for QuakeML?
QUAKEML = True
if QUAKEML:
    print("INFO: will download QuakeML data.")
    os.makedirs("./QuakeML", exist_ok=True)
#######################################################


# base url for download
base_url = "https://dasway.ess.washington.edu/gci/events"

# load event list
df = pd.read_csv("./event_list.csv")
print(f"INFO: find {len(df)} events in the list.")

# download data for each event
for idx, i in df.iterrows():
    date = i['date']
    eid = i['IRIS ID']
    print("=============================================")
    print(f"INFO: processing event {eid} on {date}.")
    if TERRA:
        url = f"{base_url}/{date}/{eid}/TERRA.h5"
        wget.download(url, out=f"./TERRA/{eid}.h5")
        print(f"\nINFO: get data on TERRA to ./TERRA/{eid}.h5")
    if KKFLS:
        url = f"{base_url}/{date}/{eid}/KKFLS.h5"
        wget.download(url, out=f"./KKFLS/{eid}.h5")
        print(f"\nINFO: get data on KKFLS to ./KKFLS/{eid}.h5")
    if QUAKEML:
        url = f"{base_url}/{date}/{eid}/{eid}.qml"
        wget.download(url, out=f"./QuakeML/{eid}.qml")
        print(f"\nINFO: get data on QuakeML to ./QuakeML/{eid}.qml")