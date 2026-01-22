# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:10:17 2024

@author: Christoph M. Konrad
"""
import warnings
import os
import json
import pandas as pd
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D


import copy

from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from trajdatamanager.utils import cart2polar, limitAngle, to_finite


def sample_yaw(yaw_keys, t_keys, t_sampled):
    rots = Rotation.from_euler("Z", yaw_keys[:,np.newaxis])

    slerp = Slerp(t_keys, rots)
    rots_sampled = slerp(t_sampled)

    yaw_sampled = rots_sampled.as_euler("ZYX")[:, 0]

    return yaw_sampled


def interpolate(t, data, t_interp):
    cs = CubicSpline(t, data)
    return cs(t_interp)


def absolute_difference(data1, data2):
    """Default difference function to calculate relative tracks.

    Parameters
    ----------
    data1 : numpy.ndarray
        First data array.
    data2 : numpy.ndarray
        Second data array.

    Returns
    -------
    ddata. : numpy.ndarray
        Absolute difference between data1 and data2/

    """

    return np.abs(data1 - data2)


def difference_features_v1(data1, data2):
    """Constructs the features characterizing the relative data between
    the data of a Track1 and a Track2. To be used in Track.__sub__()


    Parameters
    ----------
    data1 : numpy.ndarray
        First data array.
    data2 : numpy.ndarray
        Second data array.

    Returns
    -------
    ddata. : numpy.ndarray
        Difference Features between data1 and data2 of shape (n_samples,
        n_features) with the features (radial distance, relative orientation,
        speed1, speed2)
    yaw_feature_index : int
        Index of the yaw features in ddata.

    """

    ddata = np.zeros((data1.shape[0], 4))

    r, phi = cart2polar(data2[:, 0] - data1[:, 0], data2[:, 1] - data1[:, 1])

    ddata[:, 0] = r
    ddata[:, 1] = phi
    ddata[:, 2] = data1[:, 4]
    ddata[:, 3] = data2[:, 4]

    yaw_feature_index = 1

    return ddata, yaw_feature_index


class DataManager:
    def __init__(self, directory):
        
        assert os.path.isdir(directory), (f"The path {directory} does not "
                                          "point to an existing directory!")
        
        self.dir = directory
        self.track_type = Track 


class ImptcManager(DataManager):
    FNAME_SAMPLES_OVERVIEW = "_overview.csv"
    FNAME_TRACKS = "track.json"
    DNAME_VEHICLE_TRACKS = "vehicles"
    DNAME_VRU_TRACKS = "vrus"

    def __init__(self, directory, setname):
        super().__init__(os.path.join(directory, "imptc_" + setname))

        self.overview = pd.read_csv(
            os.path.join(self.dir, setname + self.FNAME_SAMPLES_OVERVIEW),
            skipfooter=1,
        )

    def get_sequence_names(self):
        return self.overview["sequence name"]

    def load_all_sequences(
        self, load_vehicle_tracks=True, load_vru_tracks=True
    ):
        seq = Sequence([])
        for seqname in self.overview["sequence name"]:
            seq += self.load_sequence(
                seqname, load_vehicle_tracks, load_vru_tracks
            )

        return seq

    def load_sequence(
        self, seqname, load_vehicle_tracks=True, load_vru_tracks=True
    ):
        tracks = []
        # load vehicle tracks
        if load_vehicle_tracks:
            for path, tkfolders, files in os.walk(
                os.path.join(self.dir, seqname, self.DNAME_VEHICLE_TRACKS)
            ):
                tracks += self.load_tracks(path, tkfolders)

        if load_vru_tracks:
            for path, tkfolders, files in os.walk(
                os.path.join(self.dir, seqname, self.DNAME_VRU_TRACKS)
            ):
                tracks += self.load_tracks(path, tkfolders)

            return Sequence(tracks)

    def load_tracks(self, path, tkfolders):
        tracks = []
        for tkname in tkfolders:
            with open(
                os.path.join(path, tkname, self.FNAME_TRACKS)
            ) as track_file:
                data = json.load(track_file)

                class_id = int(data["overview"]["class_id"])
                t = [None] * len(data["track_data"].keys())

                for i, j in zip(
                    range(0, len(t)),
                    np.array(list(data["track_data"].keys()), dtype=int),
                ):
                    ts = data["track_data"][str(j)]["ts"]
                    t[i] = dt.datetime.fromtimestamp(int(ts[:10]), tz=dt.UTC)
                    t[i] += dt.timedelta(microseconds=int(ts[10:]))

                pos = np.array(
                    [
                        data["track_data"][ts]["coordinates"]
                        for ts in data["track_data"]
                    ]
                )

                v = [
                    data["track_data"][ts]["velocity"] / 3.6
                    for ts in data["track_data"]
                ]

                # add cuboid if vehicle
                if class_id == 7 or class_id == 8:
                    footprints = np.zeros((len(t), 4, 3))
                    i = 0

                    for ts in data["track_data"]:
                        cuboid = np.array(data["track_data"][ts]["cuboid"])
                        footprints[i, :, :] = self.get_floating_footprints(
                            cuboid, tkname
                        )
                        i += 1

                    orientations, vsize = self.estimate_car_orient(footprints)
                    orientations = self.estimate_vru_orient(pos, v)
                    metadata = {
                        "length": vsize[0],
                        "width": vsize[1],
                        "floating-footprints": footprints,
                        "static-footprint": np.array(
                            (
                                (vsize[0] / 2, vsize[1] / 2),
                                (-vsize[0] / 2, vsize[1] / 2),
                                (-vsize[0] / 2, -vsize[1] / 2),
                                (vsize[0] / 2, -vsize[1] / 2),
                            )
                        ),
                    }
                else:
                    if np.all(np.array(v) < 1):
                        continue
                    metadata = {}
                    orientations = self.estimate_vru_orient(pos, v)

                if np.isnan(np.sum(orientations)):
                    continue

                data = np.array(
                    (
                        pos[:, 0],
                        pos[:, 1],
                        pos[:, 2],
                        limitAngle(orientations),
                        v,
                    )
                )

                tname1, tname2 = os.path.split(path)
                tname0, tname1 = os.path.split(tname1)
                tname0 = os.path.split(tname0)[1]
                tname=f'{tname0}/{tname1}/{tname2}/{tkname}'
                
                tracks.append(
                    Track(
                        tname,
                        class_id,
                        t,
                        data,
                        metadata,
                        diff_func=difference_features_v1,
                        yaw_feature_index=3,
                        data_feature_keys=['x', 'y', 'z', 'psi', 'v']
                    )
                )

        return tracks

    def get_floating_footprints(self, cuboid, tkname):
        """Recover the floating footprint from the cuboid of a vehicle
        detection.

        Parameters
        ----------

        cuboid : array-like
            Array describing the 12 edges of the vehicle cuboid with shape
            (n_edges=12, n_points_per_edge=2, n_coords=3).

        """

        # indices of the four lowest edges
        fpi = np.sum(cuboid[:, :, 2], 1).argsort()[:4]

        # extract
        footprint_i = np.zeros((8, 3))
        footprint_i[0:4, :] = cuboid[fpi, 0, :]
        footprint_i[4:8, :] = cuboid[fpi, 1, :]

        footprint_i = np.unique(footprint_i, axis=0)

        assert (
            footprint_i.shape[0] == 4
        ), f"Error in cuboid \
            data! Cuboid of track {tkname} does not have four \
            unique lower corners."

        return footprint_i

    def estimate_vru_orient(self, pos, v):
        """Estimate the vru orientation from the tangent.

        Only update orientation when moving.

        Parameters
        ----------
        pos : array-like
            Position time series of the vru.
        v : array-like
            Speed time series of the vru.

        Returns
        -------
        orientations : array-like
            Estimated orientations
        """

        n = len(v)
        orientations = np.nan * np.zeros_like(v)
        first = None
        i_first = None

        for i in range(1, n - 1):
            if v[i] > 1:
                orientations[i] = cart2polar(
                    pos[i + 1, 0] - pos[i - 1, 0],
                    pos[i + 1, 1] - pos[i - 1, 1],
                )[1]
                if first is None:
                    first = orientations[i]
                    i_first = i
            else:
                orientations[i] = orientations[i - 1]

        if i_first is None:
            warnings.warn("Cant estimate orientation of stationary road user")
            return orientations

        for i in range(i_first):
            orientations[i] = first

        orientations[-1] = orientations[-2]

        return orientations

    def estimate_car_orient(self, footprints):
        """Estimate the vehicle orientation based on the footprint using
        principal component analysis.

        Algorithm:
            For all footprints:
                (1) Center around mean
                (2) Perform PCA and transform to principal component frame
                (3) Overlay all frames and calc width/high as mean of the
                    x/y coordinates
                (4) Recover orientations from the principal components

        Considers the slope.

        Parameters
        ----------
        footprints : array-like
            Array of footprints with shape (n_samples, n_corners=4, n_coords=3).

        Returns
        -------
        orientations : array-like
            Array of vehicle orientations.
        vsize : array-like
            Vehicle size given as (length, width).

        """

        n = footprints.shape[0]

        pca = [None] * n
        points = np.zeros((4 * n, 2))
        orientations = np.zeros(n)

        for i in range(0, n):
            footprint_i = footprints[i, :, :]
            footprint_i = footprint_i - np.mean(footprint_i)

            pca[i] = PCA(n_components=2)
            points[i * 4 : (i * 4 + 4), :] = pca[i].fit_transform(footprint_i)
            orientations[i] = np.arctan(
                pca[i].components_[0, 1] / pca[i].components_[0, 0]
            )

        vsize = 2 * np.mean(np.abs(points), axis=0)

        return orientations, vsize


class Sequence:
    def __init__(self, tracks, sequence_id=None):
        
        self.tracks = tracks
        self.n = len(tracks)
        self.sequence_id = sequence_id

        if len(tracks) > 0:
            tz = tracks[0].t_begin.tzinfo
        else:
            tz = None

        self.t_begin = dt.datetime(
            year=dt.MAXYEAR, month=12, day=31, tzinfo=tz
        )
        self.t_end = dt.datetime(year=dt.MINYEAR, month=1, day=1, tzinfo=tz)

        for trk in tracks:
            # find begin and end time stamp
            if trk.t_begin < self.t_begin:
                self.t_begin = trk.t_begin
            if trk.t_end > self.t_end:
                self.t_end = trk.t_end

        self.find_time_overlap()
        
    def serialize(self):
        """
        Serialize this sequence into a single track. This is only possible if
         - the tracks do not overlap in time. 
         - all tracks have identical properties 'class_id', 'diff_func', 
           'yaw_feature_index', 'data_feature_keys'

        Returns
        -------
        datamanager.Track

        """
        
        #verify that timeseris can be serialized to a track
        tb = [(trk.t_begin-self.t_begin).total_seconds() for trk in self.tracks]
        
        index = np.argsort(tb)
        
        data = self.tracks[index[0]].data
        t = self.tracks[index[0]].t
        class_id = self.tracks[index[0]].class_id
        diff_func = self.tracks[index[0]].diff_func
        yaw_feature_index = self.tracks[index[0]].yaw_feature_index
        data_feature_keys = self.tracks[index[0]].data_feature_keys
        metadata_i = self.tracks[index[0]].metadata
        metadata_i['t_begin'] = self.tracks[index[0]].t_begin
        metadata_i['t_end'] = self.tracks[index[0]].t_end
        
        metadata = {"track_type": 'serialized'}
        seq_elements_metadata = {}
        seq_elements_metadata[self.tracks[index[0]].track_id] = metadata_i
        
        name0 = self.tracks[index[0]].track_id
        name1 = self.tracks[index[-1]].track_id
        
        def msg(prop):
            return ('Sequence is not serialzable into a single track! The ',
                    f'track properties "{prop}" are not compatible.')
        
        for i in range(1, len(index)):
            
            trki = self.tracks[index[i]]
            
            if self.tracks[index[i-1]].t_end > trki.t_begin:
                raise ValueError('Sequence is not serialzable into a single '
                                 'track! The tracks of this sequence overlap '
                                 'in time.')
            elif self.tracks[index[i-1]].t_end == trki.t_begin:
                # first/last value is double -> drop one
                data = data[:-1,:] 
                t = t[:-1]
 
            data = np.r_[data, trki.data]
            t = np.r_[t, trki.t]
            
            assert class_id == trki.class_id, msg('class_id')
            assert diff_func == trki.diff_func, msg('diff_func')
            assert yaw_feature_index == trki.yaw_feature_index, msg('yaw_feature_index')
            assert data_feature_keys == trki.data_feature_keys, msg('data_feature_keys')
        
            metadata_i = self.tracks[index[i]].metadata
            metadata_i['t_begin'] = self.tracks[index[i]].t_begin
            metadata_i['t_end'] = self.tracks[index[i]].t_end
            
            seq_elements_metadata[self.tracks[index[i]].track_id] = metadata_i
            
        track_type = type(self.tracks[0])
        return track_type(name0 + '-' + name1, 
                     class_id,
                     t, 
                     data,
                     metadata=metadata,
                     diff_func=diff_func,
                     yaw_feature_index=yaw_feature_index,
                     data_feature_keys=data_feature_keys)
    
    def apply_timeshift(self, timeshift, data_feature_keys=None):
        """
        Apply a timeshift to individual features of all tracks in this 
        sequence.

        The signal will be shifted to t+dt. Handles unevenly spaced signals
        by interpolation to the original sample times. 

        Parameters
        ----------
        timeshift : datetime.timedelta
            Timeshift to be applied.
        data_feature_keys : list, optional
            A list of features to which the timeshift should be applied. If 
            None, the shift is applied to all features. The default is None.

        Returns
        -------
        None.

        """
        for trk in self.tracks:
            trk.apply_timeshift(timeshift, data_feature_keys=data_feature_keys)
        
    def rotate_xy(self, alpha, deg=False):
        """Rotate the tracks in this sequence in the xy plane

        This requires the tracks of this sequence to have the features 'x' and 
        'y'. Any orientation feature should be named 'psi' to be rotated 
        accordingly.        

        Parameters
        ----------
        alpha : float
            Rotation angle in rad (deg=False) or deg (deg=True).
        deg : bool, optional
            If true, alpha is interpreted in degrees. The default is False.

        Returns
        -------
        None.

        """
        
        for trk in self.tracks:
            trk.rotate_xy(alpha, deg=deg)

    def shift_xy(self, dx, dy):
        """Shift the tracks in this sequence in the xy plane

        This requires the tracks of this sequence to have the features 'x' and 
        'y'.       

        Parameters
        ----------
        dx : float
            X-shift so that x <- x + dx
        dy : float
            Y-shift so that y <- y + dy

        Returns
        -------
        None.

        """
        for trk in self:
            trk.shift_xy(dx, dy)

    def plot(self, track_ids=None, axes=None, features=None, plot_over_timestamps=False, **plot_kwargs):
        """Plot all features of this track

        Parameters
        ----------
        track_ids : list, optional
            List of track IDs to be plotted. 
        axes : list of axes, optional
            Axes to be plotted in. Must be the same number of axes as the
            track has features. Creates a new figure per default.
        features : list, optional
            List of feature names 
        plot_over_timestamps : bool, optional
            If true, the data is plotted over timestamps. If false, the data
            is plotted over sample number. The default is False.
        plot_kwargs : dict
            Keyword arguments passed to matplotlib.pyplot.plot(_,_,**plot_kwargs).

        Returns
        -------
        axes : list of axes
            The axes that the plot was created in.
        """
        
        if track_ids is None:
            tracks = self.tracks
        else:
            tracks = self.get_tracks_by_id(track_ids)

        if axes is None:
            fig, axes = plt.subplots(tracks[0].data.shape[1], 1, sharex=True)

        for trk in tracks:
            trk.plot(axes=axes, features = features, 
                     plot_over_timestamps=plot_over_timestamps, **plot_kwargs)
            
        return axes
            
    def partition(self, n_seq, shares, random_seed=None):
        """
        Partition a sequence in subsequences. 

        Parameters
        ----------
        n_seq : int
            Number of subsequences.
        shares : array_like
            Number of items in each subset given as a share of the total number
            of tracks in this sequence.
        random_seed : int, optional
            Random seed for random partitioning. The default is None.

        Returns
        -------
        subsequences : seq1, seq2, ..., seqn
            Randomly partitioned subsequences.
        """
        
        assert np.sum(shares) == 1, "The sum of shares must be 1!"
        
        n = len(self.tracks)
        
        n_tracks_per_seq = np.floor(np.array(shares)*n)
        
        if np.sum(n_tracks_per_seq) < n:
            n_tracks_per_seq[0] += 1
            
        n_tracks_per_seq = np.cumsum(n_tracks_per_seq) 
        n_tracks_per_seq = np.concatenate(([0], n_tracks_per_seq))
        n_tracks_per_seq = n_tracks_per_seq.astype(int)
            
        track_ids = np.arange(n)
        
        rng = np.random.default_rng(seed=random_seed)
        rng.shuffle(track_ids)
        
        seq = []
        for i in range(n_seq):
            
            tracks_ids_i = track_ids[n_tracks_per_seq[i]:n_tracks_per_seq[i+1]]
            
            tracks_i = [self.tracks[j] for j in tracks_ids_i]
            seq.append(Sequence(tracks_i))
            
        return seq
        
        
    def filter_by_metadata(self, metadata_key, test_func):
        """
        Filter this sequence by it's metadata.

        Parameters
        ----------
        metadata_key : str
            The metadata key to filter by
        test_func : function
            A function that if given the metadata value returns a boolean. Must have the signature test_func(metadata_val) -> bool.
            If the functions returns true, the track is included in the output sequence.
        """
        tracks = []
        for trk in self:
            if test_func(trk.metadata[metadata_key]):
                tracks.append(trk) 
        return Sequence(tracks)


    def filter_by_feature(self, key, minval, maxval, ret='inside'):
        """
        Return a new sequence holding only the tracks whose features 'key' all
        lie in minval < track[key] < maxval.
        

        Parameters
        ----------
        key : String
            Feature to be filtered by. E.g. 'psi'.
        maxval : float
            Upper limit.
        minval : float
            Lower limit.
        ret : string, optional.
            Must be 'inside' or 'outside'. For 'inside' tracks satisfying minval < trk[key] < maxval 
            are returned. For 'outside', tracks satisfying trk >= maxval or trk <= minval. The 
            default is 'inside'
        Returns
        -------
        Sequence
            Filtered sequence.

        """
        if not ret in ('inside', 'outside'):
            raise ValueError(f"ret must be either 'inside' or 'outside'. Instead it was {ret}.")
        
        inside = ret == 'inside'
        
        tracks = []
        for trk in self.tracks:
            if inside:
                if np.all((minval < trk[key]) & (trk[key] < maxval)):
                    tracks.append(trk)
            else:
                if np.all((minval >= trk[key]) | (trk[key] >= maxval)):
                    tracks.append(trk)
                
        return Sequence(tracks)

    def __getitem__(self, track_id):
        """Subcript to a sequence object returns the track given either by an
        track_id string or its running index in self.tracks.

        Sclicing or any other fancy subscription is not supported.

        Parameters
        ----------
        track_id : str or int
            track_id or index i of the requested track.

        Returns
        -------
        trajdatamanager.Track
            Requested Track.

        """
        if isinstance(track_id, int):
            return self.tracks[track_id]
        if isinstance(track_id, str):
            return self.get_tracks_by_id((track_id,))[0]

    def get_tracks_by_id(self, track_ids):
        """Return a list of tracks given by track_ids

        Parameters
        ----------
        track_ids : list
            List of track ids.

        Returns
        -------
        tracks : list
            List of tracks.

        """
        tracks = []
        for trk in self.tracks:
            if trk.track_id in track_ids:
                tracks.append(trk)
        return tracks

    def get_interaction_difference_sequence(
        self,
        ru1_class_ids=None,
        ru2_class_ids=None,
        dt_thresh=None,
        diff_thres=None,
    ):
        """Derive a sequence consisting of the difference tracks of all
        interactions between road users in this sequece.

        Interactions may be limited to interactions of road users of
        specific classes and filtered by a minium duration and thresholds for
        the difference thresholds.

        For every intraction, only one difference track is included.

        Parameters
        ----------
        ru1_class_ids : list
            Allowed class ids for the first interaction partner. If None,
            all classes are allowed. Default is None.
        ru2_class_ids : list
            Allowed class ids for the second interaction partner. If None,
            all classes are allowed. Default is None.
        dt_thresh : datetime.timedelft
            Minium time overlap for an interaction to be included. If None,
            all overlaps are included.
        diff_thresholds : list(list)
            Nested list of metrics function handles and threshold values to
            filter the difference tracks by. Example:
                ((np.amax, 10), (np.amin, 5), (metric_func, thresh), ...)
            A difference track is included in the sequence if
            metric_func(feature) < thresh for all features. If an entry of
            the outer list is None, no threshold is applied to this feature.
            If the whole list is None, no threshod is applied at all. Default
            is None.

        Returns
        -------
        seq : datamanager.Sequence
            Sequence of difference tracks.

        """

        if dt_thresh is None:
            dt_thresh = dt.timedelta(seconds=0)

        tracks = []

        self.find_time_overlap(dt_thresh)
        time_overlap = self.time_overlap

        for i in range(time_overlap.shape[0]):
            for j in range(time_overlap.shape[1]):
                if time_overlap[i, j]:
                    trki = self.tracks[i]
                    trkj = self.tracks[j]

                    if ru1_class_ids is not None:
                        if trki.class_id not in ru1_class_ids:
                            continue
                    if ru2_class_ids is not None:
                        if trkj.class_id not in ru2_class_ids:
                            continue

                    if trki.duration < dt_thresh or trkj.duration < dt_thresh:
                        continue

                    assert trki.has_time_overlap(
                        trkj, dt_thresh
                    ), "Tracks do not overlap!"
                    time_overlap[j, i] = False

                    dtrack = trki - trkj

                    include = True
                    if diff_thres is not None:
                        for k in range(dtrack.data.shape[1]):
                            if diff_thres[k] is None:
                                continue
                            if (
                                diff_thres[k][0](dtrack.data[:, k])
                                > diff_thres[k][1]
                            ):
                                include = False

                    if include:
                        tracks.append(dtrack)

        return Sequence(tracks)

    def find_time_overlap(self, dtmin=dt.timedelta(seconds=0)):
        n = len(self.tracks)
        self.time_overlap = np.zeros((n, n), dtype=bool)

        for i in range(n):
            for j in range(n):
                if not i == j:
                    trki = self.tracks[i]
                    trkj = self.tracks[j]

                    self.time_overlap[i, j] = trki.has_time_overlap(
                        trkj, dtmin
                    )

    def reduce_to_interactions_of_track(self, trackid):
        trk = self.tracks[trackid]
        return self.reduce_to_timespan(trk.t_begin, trk.t_end)

    def get_trackids_for_class(self, class_id):
        trackids = []
        for i in range(0, len(self.tracks)):
            if self.tracks[i].class_id == class_id:
                trackids.append(i)
        return trackids

    def reduce_to_timespan(self, t_begin=None, t_end=None, criterion='overlap'):
        """
        Reduce a sequence to a given timespan [t_begin, t_end].

        Parameters
        ----------
        t_begin : datetime
            Begin of the timespan.
        t_end : datetime
            End of the timespan.
        criterion : TYPE, optional
            Inclusion criterion. If 'include', tracks that fully lie in the 
            requested timespan are included in the reduced sequence. If 
            'overlap', all tracks overlapping the requested timespan are
            included. The default is 'overlap'.

        Returns
        -------
        Sequence
            The reduced sequence. 

        """
        if t_begin == None:
            t_begin = self.t_begin
        if t_end == None:
            t_end = self.t_end
        
        tracks = []
        for trk in self.tracks:
            if criterion == 'include':
                if trk.t_begin >= t_begin and trk.t_end <= t_end:
                    tracks.append(trk)
            elif criterion == 'overlap':
                if trk.t_begin <= t_end and trk.t_end >= t_begin:
                    tracks.append(trk)
            else:
                raise ValueError(f'Unknown criterion: {criterion}. Criterion ',
                                 'must be either "include" or "overlap".') 
        return Sequence(tracks)

    def reduce_to_time(self, time):
        tracks = []
        for trk in self.tracks:
            if time >= trk.t_begin and time <= trk.t_end:
                tracks.append(trk)
        return Sequence(tracks)

    def get_states_at_time(self, time):
        states = []

        for trk in self.tracks:
            s = trk.get_states_at_time(time)

            if s is not None:
                states.append(s)

        return np.array(s)
    
    def get_states_at_index(self, index):
        states = []
        for trk in self:
            states.append(trk.data[index,:])
        return np.array(states)


    def plot_xy(self, ax=None, colors=None, **kwargs):
        """
        Plot the sequence of tracks in the x-y plane. This only yields sensible
        results if the tracks have 'x' and 'y' features.

        Parameters
        ----------
        ax : Axes, optional
            Axes to be plotted in.
        colors : List, optional
            List of colors for the tracks. Must be None or have the same length
            as the list of tracks. The default is None, this results in all
            red tracks. 
        kwargs
            Any keyword argument of plt.plot() EXCEPT color.

        Returns
        -------
        Axes
            Axes plotted in. 

        """
        
        if ax is None:
            fig, ax = plt.subplots(1,1)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
        
        if colors is None:
            for trk in self.tracks:
                trk.plot_xy(ax, **kwargs)
        else:
            for trk, col in zip(self.tracks, colors):
                trk.plot_xy(ax, color=col, **kwargs)

        if self.sequence_id is not None:
            ax.set_title(self.sequence_id) 

        ax.set_aspect("equal")
        
        return ax
    
    def get_from_metadata(self, key):
        """Return a list of metadata values corresponding to 'key' from all tracks of this sequence
        
        Parameters
        ----------
        key : str
            A metadata key in the metadata dict of the tracks of this sequence. Must exist in all tracks.
        
        Returns
        -------
        metadata_values : list
            The list of metadata values at trk.metadata[key] for all tracks of this sequence.
        """

        metadata_values = []
        for trk in self:
            metadata_values.append(trk.metadata[key])

        return metadata_values


    def __add__(self, other):
        return Sequence(self.tracks + other.tracks)

    def __iadd__(self, other):
        if len(self.tracks) > 0 and len(other.tracks) > 0:
            self.tracks += other.tracks
            self.t_begin = min(self.t_begin, other.t_begin)
            self.t_end = max(self.t_end, other.t_end)
        elif len(self.tracks) == 0 and len(other.tracks) > 0:
            return other

        return self
        
    def __iter__(self):
        self.iter = -1
        return self
    
    def __next__(self):
        self.iter += 1
        if self.iter < len(self.tracks):
            return self.tracks[self.iter]
        else:
            raise StopIteration


class Track:
    CLASS_IDS = {
        "pedestrian": 0,
        "bicycle": 2,
        "motorcycle": 3,
        "scooter": 4,
        "stroller": 5,
        "wheelchair": 6,
        "unknown": 10,
        "car": 7,
        "truck/bus": 8,
        "interaction": 11,
    }

    def __init__(
        self,
        track_id,
        class_id,
        t,
        data,
        metadata={},
        diff_func=absolute_difference,
        yaw_feature_index=None,
        data_feature_keys=None,
    ):
        self.track_id = track_id
        self.class_id = class_id
        self.metadata = metadata
        self.diff_func = diff_func

        if data.shape[0] != len(t):
            if data.shape[1] == len(t):
                data = data.T
            else:
                raise ValueError(
                    ("data needs to have the same number of elements in the"
                     "first dimension as t!")
                )
        self.data = data

        if data_feature_keys is None:
            self.data_feature_keys = [
                f"feature#{i}" for i in range(self.data.shape[0])
            ]
        else:
            self.data_feature_keys = list(data_feature_keys)

        self.t = np.array(t)
        if not np.issubdtype(self.t.dtype, dt.datetime):
            raise TypeError(f't must be an array-like of datetime objects.')

        self.calc_time_properties()

        self.t_s = self.duration.total_seconds() / self.n

        self.yaw_feature_index = yaw_feature_index
        
    def to_dict(self, relative_time=None, features=None):
        """
        Return a dict containting all data and the timestamps of this track.

        Returns
        -------
        datadict : dict
            The dictionary with the data.
        relative_time : datetime
            If a datetime object is provided, the timestamp key 't' contains
            the relative time differences in s to relative_time. If None, the
            returned dict contains absolute time as datetime objects. If 
            the time relative to the first timestamp is desired, set 
            relative_time = self.t_begin
        features : list
            Data feature keys to be included in the dict. Each element of 
            features must be in trk.data_feature_keys. If None, all features
            are returned. The default is None. 

        """
        
        if relative_time is None:
            t = self.t
        else:
            t = self.get_relative_time(relative_time)
           
        if features is None:
            features = self.data_feature_keys
            
        datadict = {"t": t}
        
        for key in features:
            datadict[key] = self[key]

        return datadict
        
    def to_dataframe(self, relative_time=None):
        """
        Return a dataframe containing the data of this track. 

        Returns
        -------
        df : dataframe
            A dataframe containing all features and the timestamp of this run.
        relative_time : datetime, optional
            If a datetime object is provided, the timestamp column 't' contains
            the relative time differences in s to relative_time. If None, the
            returned dataframe contains absolute time as datetime objects. If 
            the time relative to the first timestamp is desired, set 
            relative_time = self.t_begin. Default is None.

        """
        
        return pd.DataFrame(data=self.to_dict(relative_time=relative_time))
    
    def write_csv(self, 
                  directory, 
                  filename, 
                  relative_time=None, 
                  write_metadata=True):
        """
        Write this track to a csv file.

        Parameters
        ----------
        directory : str
            Path of the output file.
        filename : str
            Desired filename without type. The output will be 'filename'.csv.
        relative_time : datetime
            If a datetime object is provided, the timestamp column 't' contains
            the relative time differences in s to relative_time. If None, the
            written file contains absolute timestamps. If 
            the time relative to the first timestamp is desired, set 
            relative_time = self.t_begin
        write_metadata

        Returns
        -------
        None.

        """
        assert os.path.isdir(directory), (f"Path '{directory}' does not point " 
                                          f"to an existing directory!")     
                            
        path_data = os.path.join(directory, filename+".csv")
        df = self.to_dataframe(relative_time=relative_time)
        df.to_csv(path_data, sep=';')
        
        if write_metadata:
            path_metadata = os.path.join(directory, filename+"_meta.txt")
            
            with open(path_metadata, 'w') as f:
                f.write(f"class_id: {self.class_id}\n")
                f.write(f"relative_time: {relative_time}\n")
                f.write(f"sample_time: {self.t_s}\n")
                f.write(f"n_samples: {self.n}\n")
                f.write(f"duration: {self.duration}\n")
                
                if not relative_time:
                    f.write(f"t_begin: {self.t_begin}\n")
                    f.write(f"t_end: {self.t_end}\n")
                
                for key in self.metadata.keys():
                    f.write(f"{key}: {self.metadata[key]}\n")
                
                    
                    
        
    def apply_timeshift(self, timeshift, data_feature_keys=None):
        """
        Apply a timeshift to individual features of this track

        The signal will be shifted to t+dt. Handles unevenly spaced signals
        by interpolation to the original sample times. 

        Parameters
        ----------
        timeshift : datetime.timedelta
            Timeshift to be applied.
        data_feature_keys : list, optional
            A list of features to which the timeshift should be applied. If 
            None, the shift is applied to all features. The default is None.

        Returns
        -------
        None.

        """
        
        if data_feature_keys is None:
            data_deature_keys = self.data_feature_keys
        else:
            for key in data_feature_keys:
                assert key in self.data_feature_keys, (f"The keys in data_"
                                                       "feature_keys have to "
                                                       "be any of "
                                                       "{self.data_feature_keys}"
                                                       f", not '{key}'")
            
        t_unshifted = np.array(self.t)
        t_shifted = np.array(self.t) + timeshift 
        
        t_shifted = t_shifted[(t_shifted >= self.t_begin) & 
                              (t_shifted < self.t_end)]

        if timeshift.total_seconds() < 0:
            i_crop = (0,len(t_shifted))
            i_shift = (len(self.t)-len(t_shifted), len(self.t))
        else:
            i_crop = (len(self.t)-len(t_shifted), len(self.t))
            i_shift = (0,len(t_shifted))
        t_cropped = t_unshifted[i_crop[0]:i_crop[1]]
        t_sft_rel = np.array([(ti - t_cropped[0]).total_seconds() for ti in t_shifted])
        t_crp_rel = np.array([(ti - t_cropped[0]).total_seconds() for ti in t_cropped])
        
        data_new = np.zeros((len(t_cropped), self.data.shape[1]))

        for i, key in zip(range(len(self.data_feature_keys)), 
                                self.data_feature_keys):
            if key in data_feature_keys:

                print(t_sft_rel)
                print(t_crp_rel)
                data_new[:, i] = interpolate(t_sft_rel, 
                                             self[key][i_shift[0]:i_shift[1]], 
                                             t_crp_rel)
                mkey = f"timeshift_{key}"
                if mkey in self.metadata:
                    self.metadata[mkey] += timeshift
                else:
                    self.metadata[mkey] = timeshift
            else:
                data_new[:,i] = self[key][i_crop[0]:i_crop[1]]
       
        self.data = data_new
        self.t = list(t_cropped)
        self.calc_time_properties()
                    
                    
        
    def add_features(self, data, data_feature_keys=None):
        """
        Add features to the data stored in this track

        Parameters
        ----------
        data : array-like
            Data features to be added to this track. Must be shaped (n_samples,
            n_features), where n_samples must equal the number of samples 
            that this track already has (which is Track.data.shape[0])
        data_feature_keys : list
            Keys of the added features. Must be a list of length n_features
            or None. If None, the features will be called feature#i. 
        """
        
        self.data = np.concatenate((self.data, data), axis=1)
        
        if data_feature_keys is None:
            data_feature_keys = [f"feature#{self.data.shape[0] + i}" for i in range(data.shape[0])]
        self.data_feature_keys += data_feature_keys                               

    def dissect(self, n_samples_per_snippet):
        n_snippets = int(np.floor(self.data.shape[0] / n_samples_per_snippet))
        tracks = []

        for i in range(n_snippets):
            i_begin = i * n_samples_per_snippet
            i_end = (i + 1) * n_samples_per_snippet
            tracks.append(
                Track(
                    self.track_id + f"_{i}",
                    self.class_id,
                    self.t[i_begin:i_end],
                    self.data[i_begin:i_end, :],
                    metadata=self.metadata,
                    diff_func=self.diff_func,
                    yaw_feature_index=self.yaw_feature_index,
                )
            )

        return Sequence(tracks)

    def calc_time_properties(self):
        self.t_begin = self.t[0]
        self.t_end = self.t[-1]
        self.duration = self.t_end - self.t_begin

        self.n = len(self.t)

    def _get_sampled_timeseries(self, t_begin=None, t_end=None, t_s=None):
        """Get the timeseries data sampled periodically with sampling time 
        t_s relative to in the interval [t_begin t_end].

        Parameters
        ----------
        t_s : float, optional
            Sample time in seconds. If None, the (approximate) intrinsic
            sample time of the signal is used.

        Returns
        -------
        t : numpy.ndarray
            Timestamps of each sample in sampled_data expressed in seconds
            relative to t_offset.
        sampled_data : numpy.ndarray
            Sampled track data. Array has the shape (n_samples, n_features)
            with the features x, y, z, psi, v. Corresponding to the times
            t

        """
        if t_begin is None:
            t_begin = self.t_begin
        else:
            t_begin = max(t_begin, self.t_begin)

        if t_end is None:
            t_end = self.t_end
        else:
            t_end = min(t_end, self.t_end)

        if t_s is None:
            t_s = self.t_s
        t_s = dt.timedelta(seconds=t_s)

        #i_begin, i_end = self.get_timespan_indices(t_begin, t_end)
        n_samples = int((t_end-t_begin)/t_s)+1
        t_sample = np.array([t_begin + i * t_s for i in range(n_samples)])
        #t_sample = np.arange(t_begin, t_end, t_s)

        return self._get_sampled_timeseries_at_t(t_sample)        
        
    def _get_sampled_timeseries_at_t(self, t_sample):
        """Get the timeseries data as samples sampled at times t within the 
        the interval [t_begin t_end].

        Parameters
        ----------
        t_sample : float
            Sample times in seconds. Use datetime.total_seconds() to 
            convert absolute times in seconds.

        Returns
        -------
        t : numpy.ndarray
            Timestamps of each sample in sampled_data expressed as dt.datetime.
        sampled_data : numpy.ndarray
            Sampled track data. Array has the shape (n_samples, n_features)
            with the features x, y, z, psi, v. Corresponding to the times
            t

        """
        
        assert isinstance(t_sample[0], dt.datetime), ('Sample times must be'
                                                      'provided as array of'
                                                      'datetime.datetime.')
        
        #t_begin = t_sample[0]
        #t_end = t_sample[-1]
        # get relative time and sampletime.
        t_rel = np.array(self.get_relative_time())
        t_sample_rel = np.array([(t- self.t[0]).total_seconds() for t in t_sample])

        # separate data arrays for position and rotation data
        data = self.data
        if self.yaw_feature_index is not None:
            data_yaw = data[:, self.yaw_feature_index]
            data = np.delete(data, self.yaw_feature_index, axis=1)
            sampled_data_yaw = sample_yaw(data_yaw, t_rel, t_sample_rel)

        sampled_data = np.zeros((t_sample.size, data.shape[1]))
        for i in range(data.shape[1]):
            mask_finite = np.isfinite(data[:,i])
            cs = CubicSpline(t_rel[mask_finite], data[:,i][mask_finite])
            sampled_data[:,i] = cs(t_sample_rel)

        if self.yaw_feature_index is not None:
            sampled_data = np.insert(
                sampled_data, self.yaw_feature_index, sampled_data_yaw, axis=1
            )

        #t = [None] * sampled_data.shape[0]
        #for i in range(sampled_data.shape[0]):
        #    t[i] = t_begin + dt.timedelta(seconds=t_sample[i])

        return t_sample, sampled_data

    def sample(self, t_begin=None, t_end=None, t_s=None):
        """Sample a track on the interval [t_begin, t_end] with the sample
        time t_s.

        Parameters
        ----------
        t_begin : datetime.datetime, optional
            Start time of the resampled signal. If None, the earliest
            sample of the original signal is used. Can't be
            earlier then the earliest original sample. The default is None.
        t_end : datetime.datetime, optional
            End time of the resampled signal. If None, the earliest
            latest of the original signal is used. Can't be later then the
            latest original sample. The default is None.
        t_s : TYPE, optional
            Sample time in sections. If None, the sample time of the
            original signal is used. The default is None.

        Returns
        -------
        trajdatamanager.Track
            Resampled Track.
        """

        if t_begin == None:
            t_begin = self.t_begin
        if t_end == None:
            t_end = self.t_end

        if t_begin < self.t_begin or t_end > self.t_end:
            warnings.warn(
                f"Can't extrapolate time series for track {self.track_id}"
            )
            return None

        t, sampled_data = self._get_sampled_timeseries(t_begin, t_end, t_s)
        self.t = t

        self.data = sampled_data
        self.calc_time_properties()

        self.t_s = t_s

        return self
    
    def sample_at_times(self, t):
        """Sample a track at the times t
        
        This does not extrapolate. Rather, the requested time is croped to 
        [t_begin, t_end[ of the track.
        
        Parameters
        ----------
        t : Array
            Sample times given as array of datetime.datetime
            
        Returns
        -------
        trajdatamanager.Track
            Resampled Track.
        """
        
        #crop sample times to available data
        i_begin, i_end = self.get_timespan_indices(t[0], t[-1])
        t = t[(t >= self.t[i_begin]) & (t <= self.t[i_end])]

        t, sampled_data = self._get_sampled_timeseries_at_t(t)
        
        self.t = t
        self.data = sampled_data
        self.calc_time_properties()

        return self
        

    def get_relative_time(self, t_ref=None):
        """Get the timestamp of all samples in seconds relative to a reference
        timestamp.

        Parameters
        ----------
        t_ref : datetime.datetime, optional
            Reference timestamp. If None, the timestamp of the first sample is
            used. Default is None.

        Returns
        -------
        t_rel : np.ndarray
            Array of relative sample timestamps in seconds.
        """

        if t_ref is None:
            t_ref = self.t_begin
        t_rel = np.array(self.t) - t_ref
        t_rel = [ti.total_seconds() for ti in t_rel]

        return t_rel
    
    def get_index_at_time(self, time):
        if time < self.t_begin or time > self.t_end:
            return None
        else:
            i = np.amin(np.where(np.isclose(self.t, time))[0])
            return i

    def get_states_at_time(self, time):
        i = self.get_index_at_time(time)
        if i is None:
            return None
        else:
            return (self.t[i], self.x[i], self.y[i], self.z[i], self.v[i])

    def get_timespan_indices(self, t_begin, t_end):
        if t_begin > self.t_end or t_end < self.t_begin:
            return None, None

        for i in range(0, len(self.t)):
            if self.t[i] >= t_begin:
                i_begin = i
                break

        for i in range(i_begin, len(self.t)):
            i_end = i
            if self.t[i] > t_end:
                break

        return i_begin, i_end

    def crop_to_sample(self, i_begin, i_end):
        """
        Crop a track between the samples indicated by [i_begin, i_end[

        Parameters
        ----------
        i_begin : int
            First index of the crop.
        i_end : int
            Last index of the crop (not included).

        Returns
        -------
        trajdatamanager.Track
            Cropped Track.

        """
        self.t = self.t[i_begin:i_end]
        self.data = self.data[i_begin:i_end, :]

        self.calc_time_properties()

        return self

    def crop_to_timespan(self, t_begin, t_end):
        t_begin = max(t_begin, self.t_begin)
        t_end = min(t_end, self.t_end)

        i_begin, i_end = self.get_timespan_indices(t_begin, t_end)

        if i_begin is None:
            return None

        return self.crop_to_sample(i_begin, i_end)
    

    def segment_by_indicator(self, indicator):
        """Segment a track by an indicator array.

        The indicator array should have the segment number
        at every index belonging to the a segment. Sample
        indices not belonging to any segment should have
        np.nan or np.inf. For example:

        [0,0,0,np.nan,np.nan,1,1,1,1,1,2,2,2,2,2]

        creates three segments (0,1,2) and drops 
        the data at indices 3 and 4.

        Samples with the same segment number will
        end up in the same segment, even if they are not
        in one block. 

        Parameters
        ----------
        indicator : array-like
            The indicatory array. Must be length trk.n.

        Returns
        -------
        Sequence
            A sequence of Track object each representing one 
            segment of the original track. 
        """

        segindicators = np.unique(indicator[np.isfinite(indicator)])

        segments = []

        for segid in segindicators:
            mask = indicator == segid

            # segment properties
            data_seg = self.data[mask,:]
            t_seg = self.t[mask]
            metadata_seg = copy.deepcopy(self.metadata)
            key = "segment_id"
            if key in metadata_seg:
                i=0
                key = f"segment_id{i}"
                while key in metadata_seg:
                    i += 1
                    key = f"segment_id{i}"
            metadata_seg[key] = int(segid)

            # create segment Track
            segments.append(Track(
                self.track_id + f"_{segid}",
                self.class_id,
                t_seg, 
                data_seg, 
                metadata=metadata_seg,
                diff_func=self.diff_func,
                yaw_feature_index=self.yaw_feature_index,
                data_feature_keys=self.data_feature_keys
            ))
        
        return Sequence(segments)
            
    
    def segment_by_geofencing(self, xfences):
        """
        Segment a track into multiple tracks by geofencing.
        
        Currently only supports fences parallel to the y-axis.

        Parameters
        ----------
        xfences : list
            List of x-values representing geofences parallel to the y-axis. Can
            be multiple fences

        Returns
        -------
        seqs : list
            List of Sequence objects hold thing the track segments in between
            two fences.

        """
        
        xfences_padded = np.zeros(len(xfences)+2)
        xfences_padded[1:-1]=xfences
        xfences_padded[0] = - np.inf
        xfences_padded[-1] = np.inf
        
        tracks = []
        n_splits = 0
        
        seqs = []
        
        for i in range(len(xfences_padded)-1):
            
            tracks = []
            
            inside = (xfences_padded[i] < self.data[:,0]) & (self.data[:,0] < xfences_padded[i+1])
            
            is_a_track = False
            for j in range(len(inside)):
                if inside[j]:
                    if not is_a_track:
                        is_a_track = True    
                        j_begin = j
                        n_splits += 1
                if not inside[j]:
                    if is_a_track:
                        is_a_track = False
                        j_end = j
                        
                        if (j_end-j_begin) < 2:
                            continue
                        
                        trk = Track(self.track_id+"_split"+str(n_splits),
                                    self.class_id,
                                    self.t[j_begin:j_end],
                                    self.data[j_begin:j_end,:],
                                    metadata=self.metadata,
                                    diff_func=self.diff_func,
                                    yaw_feature_index=self.yaw_feature_index,
                                    data_feature_keys=self.data_feature_keys)
                        
                        trk.metadata['geofencing_segment_nr'] = i
                        
                        tracks.append(trk)
                        
            seqs.append(Sequence(tracks))
                        
        return seqs
            
            
    def plot_xy(self, ax=None, **kwargs):
        """
        Plot the track.

        Automatically adds markers, where an o marks the start of the
        track and a filled o marks the end.

        The color of the track indicates the road user type if not specified
        otherwise:
            pedestrian : orange
            bicycle : red
            car : black
            scooter : gray
            other : blue

        Parameters
        ----------
        ax : axes, optional
            The axes to be plotted in. Creates a new axes object if None. The
            default is None.
        kwargs
            Any keyword argument of plt.plot() including color.

        -------
        ax : axes
            The axes that the plot was created in.

        """
        
        if ax is None:
            fig, ax = plt.subplots(1,1)
            ax.set_aspect("equal")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
        
        if 'color' not in kwargs:
            if self.class_id == 0:
                kwargs['color'] = "orange"
            elif self.class_id == 2:
                kwargs['color'] = "red"
            elif self.class_id == 7 or self.class_id == 8:
                kwargs['color'] = "black"
            elif self.class_id == 4:
                kwargs['color'] = "gray"
        
        if 'label' not in kwargs:
            kwargs['label'] = self.track_id

        ax.plot(
            to_finite(self.data[:, 0]), to_finite(self.data[:, 1]), **kwargs)
        

        t = Affine2D().rotate(np.atan2(self.data[1, 1]-self.data[0, 1], self.data[1, 0]-self.data[0, 0]))
        mstyle = MarkerStyle("|", fillstyle='full', transform=t)
        kwargs["marker"] = mstyle
        del(kwargs["label"])
        ax.plot(self.data[0, 0], self.data[0, 1], **kwargs)

        t = Affine2D().rotate(np.atan2(self.data[-2, 1]-self.data[-1, 1], self.data[-2, 0]-self.data[-1, 0]))
        mstyle = MarkerStyle("<", fillstyle='full', transform=t)
        kwargs["marker"] = mstyle
        ax.plot(self.data[-1, 0], self.data[-1, 1], **kwargs)

        return ax
    
    def rotate_xy(self, alpha, deg=False):
        """Rotate a track in the xy plane

        This requires the track to have the features 'x' and 'y'. Any
        orientation feature should be named 'psi' to be rotated accordingly.        

        Parameters
        ----------
        alpha : float
            Rotation angle in rad (deg=False) or deg (deg=True).
        deg : bool, optional
            If true, alpha is interpreted in degrees. The default is False.

        Returns
        -------
        None.

        """
        if deg:
            alpha = alpha/180 * np.pi
        data_new = copy.copy(self.data)
        
        ix = self.data_feature_keys.index('x')
        data_new[:,ix] = np.cos(alpha) * self['x'] + \
                        - np.sin(alpha) * self['y']
                        
        iy = self.data_feature_keys.index('y')
        data_new[:,iy] =np.sin(alpha) * self['x'] + \
                        np.cos(alpha) * self['y']
                        
        try:
            ipsi = self.data_feature_keys.index('psi')
            data_new[:,ipsi] = self['psi'] + alpha
        except:
            pass
        
        try:
            ivx = self.data_feature_keys.index('v_x')
            data_new[:,ivx] = self['v'] * np.cos(data_new[:,ipsi])
        except:
            pass
        
        try:
            ivy = self.data_feature_keys.index('v_y')
            data_new[:,ivy] = self['v'] * np.sin(data_new[:,ipsi])
        except:
            pass
               
        self.data = data_new

    def shift_xy(self, dx, dy):
        """Shift this track in the x/y-plane.

        This requires the tracks of this sequence to have the features 'x' and 
        'y'.       

        Parameters
        ----------
        dx : float
            X-shift so that x <- x + dx
        dy : float
            Y-shift so that y <- y + dy

        Returns
        -------
        None.

        """
        self['x'] += dx
        self['y'] += dy 

    def has_time_overlap(self, other, dtmin=dt.timedelta(seconds=0)):
        """Check of this track overlaps in time with another track by
        at least dtmin.

        Paramters
        ---------
        other : datamanager.Track
            Other Track to be checked against.
        dtmin : datetime.timedelta
            Minimum time for an overlap to be detected. Default is 0 s

        Return
        ------
        has_overlap : bool
            Flag indicating overlap (True) or no overlap (False)
        """
        if other.t_begin > self.t_end or other.t_end < self.t_begin:
            return False
        else:
            if self.t_begin <= other.t_begin:
                dt = self.t_end - other.t_begin
            else:
                dt = other.t_end - self.t_begin

            if dt >= dtmin:
                return True
            else:
                return False

    def plot(self, axes=None, features=None, plot_over_timestamps=False, t_plot=None, **plot_kwargs):
        """Plot all features of this track

        Parameters
        ----------
        axes : list of axes, optional
            Axes to be plotted in. Must be the same number of axes as the
            track has features. Creates a new figure per default.
        features : list, optional
            List of feature names 
        plot_over_timestamps : bool, optional
            If true, the data is plotted over timestamps. If false, the data
            is plotted over sample number. The default is False.
        t : array-like, optional
            Force plotting over a given array of t values. t must be the length of 
            the sample number. Overwrites plot_over_timestamps. Default is None.
        plot_kwargs : dict
            Keyword arguments passed to matplotlib.pyplot.plot(_,_,**plot_kwargs).

        Returns
        -------
        axes : list of axes
            The axes that the plot was created in.
        """

        if features is None:
            features = self.data_feature_keys
        for f in features:
            if f not in self.data_feature_keys:
                raise ValueError((f"'{f}' is not a feature of this track. Available features are:"
                                  f"{self.data_feature_keys}."))
        
        if axes is None:
            fig, axes = plt.subplots(len(features), 1, sharex=True)
        else:
            if len(axes) != len(features):
                raise ValueError((f"Must provide the same number of axes as the track has features."
                                  f"You provided {len(axes)} axes and {len(features)} features."))
            
        if 'label' not in plot_kwargs.keys():
            label=self.track_id

        if t_plot is not None:
            if np.array(t_plot).size != self.n:
                raise ValueError((f"The number of samples is the time array for plotting must equal "
                                  f"the number of data points n of the trajectory! n = {self.n}, "
                                  f"t_plot.size = {np.array(t_plot).size}"))
            for ax, feat  in zip(axes, features):
                ax.plot(t_plot, to_finite(self[feat]), **plot_kwargs)
                ax.set_ylabel(feat)
        elif plot_over_timestamps:
            for ax, feat  in zip(axes, features):
                if 'marker' not in plot_kwargs.keys():
                    plot_kwargs['marker'] = '.'
                finite = np.isfinite(self[feat])
                ax.plot(np.array(self.t)[finite], self[feat][finite], **plot_kwargs)
                ax.set_ylabel(feat)
        else:
            for ax, feat  in zip(axes, features):
                feat_finite, mask = to_finite(self[feat], return_mask=True)
                ax.plot(np.arange(self.n)[mask], feat_finite, **plot_kwargs)
                ax.set_ylabel(feat)

        return axes

    def __sub__(self, other):
        """Implement subtraction for tracks.

        Returns the relative trajectory between two tracks.

        """
        if not self.has_time_overlap(other):
            return Track(
                "", 11, [], [], [], [], [], [], diff_func=self.diff_func
            )

        t_begin = max(self.t_begin, other.t_begin)
        t_end = min(self.t_end, other.t_end)
        t_s = max(self.t_s, other.t_s)

        t1, data1 = self._get_sampled_timeseries(
            t_begin=t_begin, t_end=t_end, t_s=t_s
        )
        t2, data2 = other._get_sampled_timeseries(
            t_begin=t_begin, t_end=t_end, t_s=t_s
        )

        assert len(t1) == data1.shape[0]
        assert len(t2) == data2.shape[0]

        i_end = min(len(t1), len(t2))

        ddata, yaw_feature_index = self.diff_func(
            data1[0:i_end, :], data2[0:i_end, :]
        )

        track_id = f"d({self.track_id}-{other.track_id})"

        return Track(
            track_id,
            11,
            t1[0:i_end],
            ddata,
            diff_func=self.diff_func,
            yaw_feature_index=yaw_feature_index,
        )

    def keys(self):
        return self.data_feature_keys

    def __getitem__(self, key):
        """Get a data series of this track based on its name. See
        self.data_feature_keys for a list of available keys.

        Parameters
        ----------
        key : any
            Any key in self.data_feature_keys.

        Returns
        -------
        feature_data_series : numpy.ndarray.
            Time series of the requested data feature.
        """
        if key not in self.data_feature_keys:
            raise KeyError((f"Key '{key}' is not a feature of this Track. "
                            f"The features are {self.data_feature_keys}."))
        
        i = self.data_feature_keys.index(key)
        return self.data[:, i]
    
    def __setitem__(self, key, value):
        """Set a data series of this track based on its name. See
        self.data_feature_keys for a list of available keys.
        
        Parameters
        ----------
        key : any
            Any key in self.data_feature_keys.
        value : array-like
            Array-like of size self.data.shape[0].
        """
        
        if key in self.data_feature_keys:
            idx = self.data_feature_keys.index(key)
        else:
            raise KeyError((f"Key '{key}' is not a feature of this Track. "
                            f"The features are {self.data_feature_keys}."))
            
        value = np.array(value).flatten()
        
        if value.size != self.data.shape[0]:
            raise ValueError((f"Value must be array-like with size "
                              f"{self.data.shape[0]}. Instead it had size "
                              f"{value.size}."))
        
            
        self.data[:,idx] = value

    def __copy__(self):
        new = Track(self.track_id+f'_copy',
                self.class_id,
                self.t,
                self.data,
                metadata=copy.copy(self.metadata),
                diff_func = self.diff_func,
                yaw_feature_index=self.yaw_feature_index,
                data_feature_keys=self.data_feature_keys,
            )
        return new


    def get_begin_allfinite(self):
        """Get the time and index of the first index after which at which all 
        features had at least one finite value.

        Returns
        -------
        t_begin : datetime.datetime
            Timestamp of the first sample where all features have at least
            one finite value.
        i_begin : int
            Index of the first sample where all features have at least
            one finite value.
        """
        mask = np.isfinite(self.data)
        indices_first_finite = np.argmax(mask, axis=0)

        i_begin = np.max(indices_first_finite)
        t_begin = self.t[i_begin]

        return t_begin, i_begin
    

    def get_end_allfinite(self):
        """Get the time and index of the last index until which all 
        features have finite values.

        Returns
        -------
        t_end : datetime.datetime
            Timestamp of the last sample where all features still have 
            some finite value.
        i_end : int
            Index of the last sample where all features still have 
            some finite value.
        """

        mask = np.flip(np.isfinite(self.data), axis=0)
        indices_first_finite = np.argmax(mask, axis=0)

        i_end = self.data.shape[0] - np.max(indices_first_finite) - 1
        t_end = self.t[i_end]

        return t_end, i_end