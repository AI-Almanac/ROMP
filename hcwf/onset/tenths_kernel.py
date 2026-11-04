"""The numba kernel of ``hcwf.onset.ingest._to_tenths``, in a module of its own.

Numba is imported (and the kernel compiled) only when a block of rainfall is converted, not when `hcwf.config` or
`hcwf.onset.ingest` is imported. Without numba the module still imports and holds no kernel.
"""
try:
    import numba
    from numba import njit
    from numba import prange
except ImportError:  # pragma: no cover - exercised only without numba
    numba = None

import numpy as np

__all__ = []

if numba is not None:

    @njit(parallel=True, cache=True)
    def _tenths_kernel(flat, to_tenths, clip_negative, exact, keep_negative, out_tenths, out_missing, n_bad, lowest):
        """Scale, clean and round flat rain values to tenths and missing flags, in parallel over pieces.

        Compiled with numba (``njit``, ``parallel=True``). Each element goes through exactly the ``float64`` steps of
        ``_to_tenths_numpy``: multiply in ``float64``, non-finite is missing, values below the limit are missing or,
        when clipping, zero, beyond ``int32`` is missing, round half to even.

        Parameters
        ----------
        flat : numpy.ndarray, shape (n,)
            The raw rain values, any float or integer dtype.
        to_tenths : float
            The factor that takes the values to tenths of a mm.
        clip_negative : bool
            Count values below zero as bad and set them to 0, instead of making them missing.
        exact : bool
            Keep ``float64`` mm, unrounded, instead of ``int32`` tenths.
        keep_negative : bool
            Leave negative values as they are, counting none as bad.
        out_tenths : numpy.ndarray, shape (n,)
            Output: the tenths (``int32``), or ``float64`` mm when ``exact``; 0 where missing.
        out_missing : numpy.ndarray of bool, shape (n,)
            Output: True where the value is missing.
        n_bad : numpy.ndarray of int64, shape (pieces,)
            Output: the number of bad elements of each piece.
        lowest : numpy.ndarray of float64, shape (pieces,)
            Output: the lowest scaled value of the bad elements of each piece (0 if there is none).
        """
        n = flat.shape[0]
        pieces = n_bad.shape[0]
        size = (n + pieces - 1) // pieces
        limit = 1e300 if exact else 2147483647.0
        floor = -0.05 if exact else -0.5
        for c in prange(pieces):
            count = 0
            low = 0.0
            for i in range(c * size, min(n, (c + 1) * size)):
                x = np.float64(flat[i]) * to_tenths
                if not np.isfinite(x):
                    out_tenths[i] = 0
                    out_missing[i] = True
                    continue
                bad = (not keep_negative) and (x < 0.0 if clip_negative else x < floor)
                if bad:
                    count += 1
                    if x < low:
                        low = x
                    if clip_negative:
                        x = 0.0
                if (not clip_negative and bad) or abs(x) > limit:
                    out_tenths[i] = 0
                    out_missing[i] = True
                elif exact:
                    out_tenths[i] = x
                    out_missing[i] = False
                else:
                    out_tenths[i] = np.int32(np.rint(x))
                    out_missing[i] = False
            n_bad[c] = count
            lowest[c] = low
