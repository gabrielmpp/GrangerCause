import xarray as xr
import statsmodels as stm
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from warnings import warn
from scipy import signal


class NotEnoughContinuousData(Exception):
    pass


class NonStationaryError(Exception):
    pass


class NotEnoughData(Exception):
    pass


def _assert_xr_time_index_continuity(da, timedim='time', time_freq='1D',
                                     chunk_threshold=0):
    def numpy_test_boolean_wrapper(a, b, verbose=False):
        if verbose:
            print(a, b)
        try:
            np.testing.assert_equal(a, b)
        except:
            return False
        else:
            return True

    da = da.copy()
    da = da.sortby('time')
    # Converting the required time interval to seconds
    timedelta = int(pd.to_timedelta(time_freq).total_seconds())
    timedelta = np.timedelta64(str(timedelta), 's')
    # Converting the input array time axis to seconds
    original_time = da.time.values.copy()
    original_time_in_seconds = original_time.astype('datetime64[s]')
    # Computing time intervals from original array
    array_of_deltas = np.diff(original_time_in_seconds)
    # Creating boolean mask to highlight missing dates
    discontinuous_mask = np.array([not numpy_test_boolean_wrapper(delta, timedelta) for delta in array_of_deltas])
    if sum(discontinuous_mask) == 0:
        print('Array is continuous in time, returning original.')
        return da
    else:
        warn('Array is discontinuous, fetching longest continuous time chunk')
        positions_with_discontinuos_time = np.where(discontinuous_mask)[0]

        # Calculating size of chunks
        continuous_chunk_sizes = np.concatenate([np.array([0, ]), positions_with_discontinuos_time,
                                                np.array([discontinuous_mask.shape[0]])])
        continuous_chunk_sizes = np.diff(continuous_chunk_sizes)
        #
        if np.max(continuous_chunk_sizes) < chunk_threshold:
            raise NotEnoughContinuousData('Not enough continuous data.')

        # Finding starting a ending indexes of the largest continuous chunk
        idx_of_greater_continuous_chunk = int(np.where(continuous_chunk_sizes == np.max(continuous_chunk_sizes))[0])

        if idx_of_greater_continuous_chunk == 0: # This means that the first chunk is the largest
            end_idx = np.max(continuous_chunk_sizes)  # Ending index
            start_idx = 0
        elif idx_of_greater_continuous_chunk > 0:
            start_idx = positions_with_discontinuos_time[idx_of_greater_continuous_chunk-1] + 1 #  Plus one to skip the problematic interval
            end_idx = start_idx + np.max(continuous_chunk_sizes)
        else:
            raise ValueError('idx_of_greater_chunk_size is negative valued - this is not allowed.')

        da = da.isel({timedim: slice(start_idx, end_idx)})

        # Final verification to assert that the new array is continuous
        for delta in np.diff(da[timedim].values.astype('datetime64[s]')):
            assert timedelta == delta, "Asserting time continuous failed."

        return da


def ADF_test(da_x, featuredim, sampledim, critical_level=None, testtype='c', maxlag=None):
    """
    Assert stationary of 2D xarray dataarrays
    Parameters
    ----------
    ds_x
    featuredim:
    sampledim

    Returns
    -------

    """
    if isinstance(critical_level, type(None)):  # TODO use a complete critical level table
        critical_level = '1%'

    assert len(da_x.dims) == 2, 'Input array must be 2D.'
    da_x = da_x.copy()
    features = da_x[featuredim].values
    test_results = []
    for feature in features:
        to_test = da_x.sel({featuredim: feature}).dropna(sampledim, how='any').values
        threshold = stm.tsa.stattools.adfuller(to_test, regression=testtype)[4][critical_level]
        adf = stm.tsa.stattools.adfuller(np.diff(to_test), maxlag=maxlag, regression=testtype)[0]
        if isinstance(adf, (float, int)):
            test_results.append(adf < threshold)
        else:
            test_results.append(False)
    results = xr.DataArray(test_results, dims=[featuredim], coords={featuredim: features})
    if np.sum(~results.values) > 0:
        print('One ore more features failes the ADF test.')
    return results


class GrangerCausality:

    def __init__(self, featuredim, sampledim):

        self.featuredim = featuredim
        self.sampledim = sampledim

    def run(self, da_x, da_y, maxlag=None, detrend=False,
            granger_test='params_ftest', test_stationarity_x=True, test_stationarity_y=True,
            critical_level='5%',
            testtype='c'):
        """

        Parameters
        ----------
        da

        Returns
        -------

        """
        self.maxlag = maxlag
        assert len(da_x.dims) == 2, 'Features array must be 2D.'
        assert len(da_y.dims) == 1, 'Response array must be 1D.'

        da_x = da_x.copy()
        da_y = da_y.copy()

        da_x = da_x.dropna(self.sampledim, how='any')
        da_y = da_y.dropna(self.sampledim, how='any')
        if da_x[self.sampledim].shape[0] < 10 or da_y[self.sampledim].shape[0] < 10:
            raise NotEnoughData
        da_x = _assert_xr_time_index_continuity(da_x, timedim='time', chunk_threshold=10)
        da_y = _assert_xr_time_index_continuity(da_y, timedim='time', chunk_threshold=10)
        if detrend:
            original_dimorder = da_x.dims

            da_x = da_x.transpose(..., self.sampledim).copy(data=signal.detrend(da_x.transpose(..., self.sampledim)))
            da_x = da_x.transpose(*original_dimorder)
        da_x, da_y = xr.align(da_x, da_y)

        if test_stationarity_x:
            stationary_mask_x = ADF_test(da_x, featuredim=self.featuredim, sampledim=self.sampledim,
                                       critical_level=critical_level, testtype=testtype, maxlag=maxlag)

            da_x_dropped = da_x.where(stationary_mask_x, drop=True)
            dropped_x = set(da_x[self.featuredim].values.tolist()).difference(set(da_x_dropped[self.featuredim].values.tolist()))
            if len(dropped_x) > 0:
                warn('The following features are stationary and were dropped: ')
                print(dropped_x)
            da_x = da_x_dropped

        if test_stationarity_y:
            da_y = da_y.assign_coords(dummy_featuredim=0)
            da_y = da_y.expand_dims('dummy_featuredim')
            stationary_mask_y = ADF_test(da_y, featuredim='dummy_featuredim',
                                       sampledim=self.sampledim,
                                       critical_level=critical_level, testtype=testtype,
                                         maxlag=maxlag)

            da_y_dropped = da_y.where(stationary_mask_y, drop=True)
            dropped_y = set(da_y['dummy_featuredim'].values.tolist()).difference(set(da_y_dropped['dummy_featuredim'].values.tolist()))
            if len(dropped_y) > 0:
                raise NonStationaryError('Target variable failed the stationarity test.')
            da_y = da_y_dropped.isel(dummy_featuredim=0).drop('dummy_featuredim')

        features_x = da_x[self.featuredim].values

        if isinstance(maxlag, type(None)):
            maxlag = int((da_y.shape[0] / 3) - 2)
        Y = da_y.values
        list_of_pval_arrays = []
        for feature in features_x:
            X = da_x.sel({self.featuredim: feature}).values
            res = grangercausalitytests(np.column_stack([Y, X]), maxlag=maxlag, verbose=True)
            pvals = []
            for lag in range(1, maxlag):
                pvals.append(res[lag][0][granger_test][1])  # Fetching the p-value
            da_pvals = xr.DataArray(pvals, dims=['lags'],
                                    coords={'lags': np.arange(1, maxlag)})
            list_of_pval_arrays.append(da_pvals)
        p_array = xr.concat(list_of_pval_arrays, dim=pd.Index(features_x, name=self.featuredim))

        return p_array