"""The numba kernel of ingest._to_tenths, in a module of its own so that numba is imported (and the kernel
compiled) only when a block of rainfall is converted, not when hcwf.config or hcwf.onset.ingest is imported."""
try:
    import numba
    from numba import njit
    from numba import prange
except ImportError:  # pragma: no cover - exercised only without numba
    numba = None

import numpy as np

if numba is not None:

    @njit(parallel=True, cache=True)
    def _tenths_kernel(flat, to_tenths, clip_negative, exact, keep_negative, out_tenths, out_missing, n_bad, lowest):
        """Scale, clean and round the flat values to int32 tenths (float64 mm, unrounded, when exact) and missing flags
        in one pass, in parallel over pieces.

        Each element goes through exactly the float64 steps of _to_tenths_numpy (multiply in float64, non-finite is
        missing, values below the limit are missing or, when clipping, zero, beyond int32 is missing, round half to
        even); n_bad[c] and lowest[c] are the count and the lowest scaled value of the bad elements of piece c.
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
