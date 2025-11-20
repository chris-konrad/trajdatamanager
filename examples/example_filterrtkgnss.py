# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:52:19 2024

Load a RTKlib GNSS tracks, rotate and extract only those going from left to right.

@author: Christoph M. Konrad
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from trajdatamanager.datamanager import RTKLibGNSSManager


data_dir = 'data'
pos_file_name = '0001-00020.pos'

#create a datamanager instance
dataman = RTKLibGNSSManager(os.path.join(data_dir, pos_file_name))

#load the track
trk = dataman.load_tracks()[0]

#plot the full track
trk.plot_xy()

#rotate the track for convenience and plot again
trk.rotate_xy(-43, deg=True)
trk.plot_xy()

#segment the track 
seqs = trk.segment_by_geofencing([-18, 14])

# Verify segmentation by plotting
fig, axes = plt.subplots(1,3)
seqs[0].plot_xy(axes[0])
seqs[1].plot_xy(axes[1])
seqs[2].plot_xy(axes[2])

# filter by orientation to extract only the tracks towards the lights (from left to right)
seqs[1] = seqs[1].filter_by_feature('psi', np.pi/2, -np.pi/2)
fig, axes2 = plt.subplots(1,1)
seqs[1].plot_xy(axes2)