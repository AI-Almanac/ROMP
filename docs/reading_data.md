# Reading data: the chunk rules

A big NetCDF file is stored as compressed chunks, and **a chunk is decompressed whole, whatever part of it a read
wants**. The archive files (zlib, chunks of 11 inits by 7 members by 13 steps by 33 by 34 cells, latitude stored
north to south) cost as many decodes as chunks a read touches, so what matters is that every chunk is touched by one
read, and only by the reads that need it. These are the rules the readers of `hcwf` follow, and what to do when you add one.

## When to read through `hcwf.spatial.select`

Read **through `hcwf.spatial.select`** whenever data of a lazily opened variable goes into memory and the variable can be
big (a forecast, a year of observations, a block of a detection sample):

* {py:func}`hcwf.spatial.select.load_blocks` reads a whole selection, block by block along one dim, optionally only the
  cells of a keep mask (`keep`, `compact`); {py:func}`hcwf.spatial.select.region_select` (with `read_by=`) cuts the region
  box first and applies the masks.
* {py:func}`hcwf.spatial.select.read_box` reads one box of ranges (one read per dim), which is one decode of each chunk
  inside it; {py:func}`hcwf.spatial.select.chunk_runs` gives the runs of positions that lie in one stored chunk, to cut
  your own boxes on the chunk grid (the detection sample does).
* Do not call `.load()` / `.values` on a lazy selection of a big variable, and do not write a second reader: reuse these.

Reading is fine without them where it is small metadata: a coordinate, an attribute, one value
(`hcwf.cache`, `hcwf.config.leads`, `hcwf.io.output`, the threshold attributes). A lazily opened file that is only
looked at (the warm-up, `hcwf.runs.check`) decodes nothing.

## Chunk alignment

The runs of a read are cut where the **file's rows** cross a chunk edge, not at multiples of the chunk length from the first
position: a region box usually starts inside a chunk, and a block made of whole runs then decodes each chunk once.
`select._chunk_runs` does this from the lazy selection xarray keeps (`select._lazy_key`); when the rows are not strictly
monotonic, or there is no chunk information, it falls back to cutting by position.

Blocks are about 64 MB and a whole number of chunks. A file chunked over the whole time axis, which would be decoded
again by every block along time, is blocked along the axis with the most chunks instead.

## Lists of rows and reversed axes

* **Lists**: a list of rows (the inits of a year out of a file's whole time axis, a subset of members) read as a list makes
  HDF5 decode a chunk again for every run of the list. `load_blocks` and `read_box` read the longest increasing list one chunk
  at a time as a contiguous range and pick the rows in memory; the other lists are read as the range that covers them and
  picked in memory too. The values are those of the plain read. Never `.load()` a selection that picks several inits of one
  chunk with a list.
* **Reversed axes**: a stored axis that runs backwards (latitude north to south) and a selection with a negative step are
  **read forward**, as the same rows in file order, and flipped in memory. The rows and runs above are in file order, so a flipped
  latitude and a box on it cost the same decodes as a plain one.

## NetCDF versus zarr

* **NetCDF/HDF5**: one thread decodes at a time per process, so a big read may be split over reader processes
  (`readers=`, `hcwf.parallel.reader_pool`); a worker never starts any. Several files are read one at a time (the library is
  not thread-safe).
* **zarr stores** (`data/india`, `data/ethiopia`; one init per chunk): opened with `hcwf.io.stores.open_store` and
  `chunks=None`, so nothing is decoded until a selection is read; chunks are independent, so years may be prefetched in
  threads, and a store chunked over many years is read in blocks and sliced per year (`hcwf.io.observations`).
  The same `select` functions read them: their chunk information is the store's.

## Checking it

`tests/_decode_count.py` counts the chunks each read of a NetCDF file touches (`count_decodes()`; read the totals with
`.chunks`, `.total`, `.worst()`), and `tests/test_decoded_once.py` asserts that the observation, forecast, live-forecast,
detection-sample and alignment-check readers decode every chunk at most once on small synthetic files. A new reader
gets a test there.
