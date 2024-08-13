from datetime import datetime

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import glob
import os

import numpy as np

with open("./log.log", "a") as f:
    f.write(f"{rank}: ==============={datetime.now()}===========\n")

flist = sorted(
    glob.glob(
        "/home/niyiyu/Research/DAS-NIR/datasets/KKFL-S_FiberA_2p5Hz/decimator3_2023-07-16_*_UTC.h5"
    )
)

for f in flist:
    for ic, c in enumerate(np.linspace(1000, 6000, 6).astype("int")):
        if ic % size == rank:
            gpuid = rank % 4
            command = f"/home/niyiyu/anaconda3/envs/dasnir/bin/python /home/niyiyu/Research/DAS-NIR/gci-summary/tests/rff/run.py -f {f} -c {c} -g 3"
            os.system(command)

with open("./log.log", "w") as f:
    f.write(f"{rank}: ==============={datetime.now()}===========\n")
