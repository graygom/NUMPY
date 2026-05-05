#
# TITLE: high performance computing with python numpy
# AUTHOR: Hyunseung Yoo
# PURPOSE: 
# REVISION: 
# REFERENCE: mastering numpy for data science and machine learning (M. J. Maxell, 2025)
#


import numpy as np
import time
from typing import Optional, Tuple


#
# CH 1: Getting Started with Numpy
#

if False:
    
    # verify installation version
    print(np.__version__)

    # native python list CPU time = 6.43e-1sec @Lenovo M70q i5-10400T 6-CPUs (2020)
    nat_list = list( range(10_000_000) )
    start_time = time.time()
    sum(x*x for x in nat_list)
    end_time = time.time()
    print('Native python list CPU time = %.2e' % (end_time - start_time) )

    # numpy array CPU time = 3.81e-2sec @Lenovo M70q i5-10400T 6-CPUs (2020)
    np_array = np.arange(10_000_000)
    start_time = time.time()
    np.sum(np_array*np_array)
    end_time = time.time()
    print('Numpy array CPU time = %.2e' % (end_time - start_time) )

    # numpy array a (2 rows x 3 columns)
    a = np.array( [ [1,2,3], [4,5,6] ])
    print('Numpy array a shape:', a.shape)
    print('Numpy array a ndim:', a.ndim)
    print('Numpy array a size:', a.size)
    print('Numpy array a dtype:', a.dtype)
    print('Numpy array a itemsize:', a.itemsize, 'bytes')

    # numpy array creation
    array_1 = np.array( [1,2,3] )               # from list
    array_2 = np.array( (1,2,3), float )        # from tuple, specify dtype
    array_3 = np.zeros( (3, 4) )                # 3x4 matrix of zeros
    array_4 = np.ones( 5 )                      # vector of ones
    array_5 = np.full( (2, 3), 7 )              # filled with constant
    array_6 = np.eye(4)                         # 4 x 4 identity matrix
    array_7 = np.arange(0, 10, 2)               # start, stop, step
    array_8 = np.linspace(0, 1, 5)              # start, stop, num

    # random number API
    rng = np.random.default_rng(seed=42)
    rng.integers(0, 10, size=(2,3))             
    rng.normal(loc=0, scale=1, size=10)         # standard normal
    
    # data type
    ints16 = np.array( [1, 2, 3], dtype=np.int16)
    floats32 = np.array( [1.0, 2.0, 3.0], dtype=np.float32)
    print(ints16.dtype, floats32.dtype)
    print(ints16.itemsize, 'bytes', floats32.itemsize, 'bytes')

    # type conversion
    array = np.array([1.0, 2.0, 3.0])
    ints = array.astype(np.int32)
    print(array.dtype, array.itemsize, 'bytes')
    print(ints.dtype, ints.itemsize, 'bytes')

    # operation, upcasting automatically
    array_add = np.array([1,2,3]) + np.array([1.5])
    print(array_add, array_add.dtype, array_add.itemsize)

    # indexing and slicing
    v = np.arange(10)
    print(v[0], v[-1], v[2:7:2], v.dtype, v.itemsize)
    m = np.arange(12).reshape(3,4)
    print(m[1,2], m[0:2, 1:4], m[:,0])
    sub = m[0:2, 1:3].copy()
    sub[0,0] = 99
    print(m, sub)

    # Boolean masks
    mask = m % 2 == 0
    print(m, mask, m[mask])
    m[m<5] = -1
    print(m)

    # Fancy indexing
    rows = np.array([0, 2])
    cols = np.array([1, 3])
    print(m, m[rows, cols])
    print(m, m[rows[:,None], cols])

    # pulling it all together
    rng = np.random.default_rng(seed=123)
    data = rng.normal(loc=50, scale=15, size=(6,6)).astype(np.float32)
    mean_val = data.mean()
    high = data[data>mean_val]
    print(f'dataset mean: {mean_val:.2f}')
    print(f'values above mean: {high}')
    print(f'high-value mean: {high.mean():.2f}')



#
# CH 2: Core Array Operations
#

if False:

    # element-wise arithmetics
    a = np.array([2, 4, 6])
    b = np.array([1, 3, 5])
    print(' a = ', a, a.dtype, ' b = ', b, b.dtype)
    print(' a + b = ', a + b)
    print(' a - b = ', a - b)
    print(' a * b = ', a * b)
    print(' a / b = ', a / b)
    print(' a + 10 = ', a + 10)
    print(' a * 0.5 = ', a * 0.5)

    # broadcasting
    mat = np.arange(6).reshape(2, 3)
    vec = np.array([10, 20, 30])
    print(mat)
    print(vec)
    print(mat+vec)

    # universal functions (ufuncs)
    x = np.linspace(0, 2*np.pi, 6)
    print('x:', x)
    print('sin(x):', np.sin(x))
    print('exp(x):', np.exp(x))
    print('sqrt(x):', np.sqrt(x))

    u = np.array([1,2,3])
    v = np.array([4,5,6])
    print(np.maximum(u, v))
    print(np.power(u, v))
    
    res = np.empty_like(u)
    print(res)
    np.add(u, v, out=res)       # in-place computation
    print(u, v, res)
    
    result = np.sin(x)**2 + np.cos(x)**2
    print(result)

    def triple(x):
        return 3*x
    triple_vec = np.vectorize(triple)       # improving readability, not give the full C-level speed
    print(triple_vec([1,2,3]))

    # reductions and aggregations
    rng = np.random.default_rng(seed=42)
    data = rng.normal(size=(4,5))
    print('total sum:', data.sum())
    print('mean of all elements:', data.mean())
    print('column means:', data.mean(axis=0))
    print('row means:', data.mean(axis=1))

    print('std. dev. by cols:', data.std(axis=0))
    print('cum. sum by rows:', data.cumsum(axis=1))
    print(data)

    z_scores = ( data - data.mean(axis=0) ) / data.std(axis=0)
    print(data)
    print(data.mean(axis=0))
    print(z_scores)

    # comparisons and Boolean masks
    array = np.array([ [1,5,3],
                       [7,2,9] ])
    mask = array > 4
    print(mask)

    print('values > 4:', array[ mask ])
    array[mask] = 0
    print('after masking:', array)

    cond = (array % 2 == 0) | (array == 1)
    print('even or equal to 1:', array[cond])

    positive_mean = array[array>0].mean()
    print('mean of pos. entries:', positive_mean)

    rows = np.array([0, 1])
    cols = np.array([1, 2])
    print(rows[:,None])                     # column vector
    print(array[rows[:,None],cols])



#
# CH 3: Shape and data management
#

if False:

    # reshaping, transposing, and flattening
    a = np.arange(12)
    print(a.shape)
    b = a.reshape(3, 4)
    print(b)
    col = a.reshape(12, -1)
    row = a.reshape(-1, 12)
    print(col)
    print(row)
    print(np.shares_memory(a, b))           # check memory status
    print(np.shares_memory(a, col))         # check memory status
    print(np.shares_memory(a, row))         # check memory status

    # flattening, raveling
    f = b.flatten()
    print(np.shares_memory(b, f))           # check memory status
    r = b.ravel()
    print(np.shares_memory(b, r))           # check memory status
    print(b)
    r[0] = 99
    print(b)

    # transposing, axis moves
    t = b.T
    print(np.shares_memory(b, t))
    swapped = np.swapaxes(b, 0, 1)
    print(b)
    print(swapped)
    print(np.shares_memory(b, swapped))
    moved = np.moveaxis(b, 0, -1)
    print(b)
    print(moved)
    print(np.shares_memory(b, moved))
    print(b.strides, t.strides)
    safe = np.ascontiguousarray(t)              # C-contiguous memory
    print(b.strides, t.strides, safe.strides)

    # concatenating, splitting, and stacking
    A = np.arange(6).reshape(2,3)
    B = np.arange(6, 12).reshape(2,3)
    print(A)
    print(B)
    rows = np.concatenate([A, B], axis=0)
    print(rows)
    cols = np.concatenate([A, B], axis=1)
    print(cols)
    stacked = np.stack([A, B], axis=0)
    print(stacked)
    print(stacked.shape)

    vstack = np.vstack([A, B])
    hstack = np.hstack([A, B])
    cstack = np.column_stack([A, B])
    print(vstack)
    print(hstack)
    print(cstack)

    X = np.arange(12).reshape(3,4)
    rows = np.split(X, 3, axis=0)
    print(rows)
    cols = np.split(X, 4, axis=1)
    print(cols)

    sensor1 = np.random.rand(1000, 5)
    sensor2 = np.random.rand(1000, 3)
    X = np.hstack([sensor1, sensor2])
    print(sensor1[0], sensor2[0], X[0])
    print(np.shares_memory(sensor1, X))

    # broadcasting rules and patterns
    M = np.arange(6).reshape(2, 3)
    print(M.dtype, M.itemsize)
    v = np.array([10, 20, 30])          # row vector
    print(M + v)
    col = np.array([1,2])[:,None]       # column vector
    print(M + col)

    a = np.arange(3)[:, None]           # column vector
    b = np.arange(4)[None, :]           # row vector
    grid = a + b
    print(grid.shape)
    print(grid)

    A = np.random.rand(5, 3)
    B = np.random.rand(4, 3)
    diff = A[:, None, :] - B[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))
    print(A)
    print(B)
    print(diff)
    print(dist)

    # views vs. copies and memory considerations
    x = np.arange(9)
    s = x[::3]
    print(np.shares_memory(x, s))

    m = np.arange(6).reshape(2, 3)
    print(m.strides)                    # bytes to move along each axis

    if not m.flags['C_CONTIGUOUS']:
        m = np.ascontiguousarray(m)     # non-contiguous arrays may force hidden copies when passed to C/Fortran code

    big = np.ones((1_000_000,), dtype=np.float64)
    print(big.nbytes / 1e6, 'MB')

    array = np.ones(5)
    array *= 3.0            # in-place operations
    print(array)

    mm = np.memmap('data.at', dtype='float32', mode='w+', shape=(10000, 1000))
    mm[:] = np.random.rand(10000, 1000)
    print(mm.nbytes/1e6, 'MB')
    mm.flush()



#
# CH 4: Input and output
#

if False:
    # working with data means reading from and writing to files
    # in NumPy-level you'll encounter a variety of formats - simple text files, efficient binary blobs, memory-mapped arrays for huge datasets, and structured storage like HDF5
    # this chapter explains the practical tools you'll use everyday, why to choose one format over another, and how to avoid common pitfalls
    # each subsection contains complete, ready-to-run examples and small, practical recipes you can drop into a notebook
    
    # 4.1 reading and writing text and binary files
    # Numpy supports multiple on-disk representations
    # use text files for interoperability and human inspection; prefer binary formats for speed, exactness, and compactness

    # binary: .npy and .npz (recommended for NumPy arrays)
    # .npy stores a single array plus metadata (shape, dtype, endianness)
    # .npz is a zip archive of .npy files (multiple arrays)
    # these are portable, fast, and preserve dtype exactly
    # save and load a single array with .npy

    import numpy as np
    X = np.random.default_rng(seed=0).normal(size=(1000,1000))
    # save
    np.save('X.npy', X)                                 # write compressed metadata + raw binary
    X_loaded = np.load('X.npy', allow_pickle=False)     # allow_pickle=False for safety
    print(X_loaded.shape, X_loaded.dtype, X_loaded.nbytes/1e6, 'MB')

    # save multiple arrays with .npz
    
    y = np.arange(1000)
    np.savez('dataset.npz', X=X, y=y)                           # uncompressed .npz
    np.savez_compressed('dataset_compressed.npz', X=X, y=y)     # compressed
    # load
    data = np.load('dataset.npz')
    X2 = data['X']
    y2 = data['y']
    print(np.shares_memory(X2, data['X']), np.shares_memory(y2, data['y']))

    # why?
    # .npy/.npz: fast read/write, exact dtype, ideal for intermediate steps in a pipeline
    # use allow_pickle=False unless you intentionally saved Python objects (pickles can be unsafe from untrusted sources)
    
    # Text: savetxt, loadtxt, and genfromtxt (human readable)
    # text files (CSV, TSV) are convenient for interchage and quick inspection, but they are slower and may lose dtype information
    
    # write a 2-D array as CSV
    
    A = np.array([[1.234, 2.3456],[3.456, 4.4567]])
    # fmt controls text formatting
    # comments='' prevents '#' before header
    np.savetxt('A.csv', A, delimiter=',', header='c1,c2', comments='', fmt='%.4f')

    # read a CSV (simple numeric data)

    B = np.loadtxt('A.csv', delimiter=',', skiprows=1)      # skip header
    print(B)

    # for files with missing values, mixed types, or irregular rows, genfromtxt is more robust
    
    data = np.genfromtxt('missing_data.txt', delimiter=',', dtype=float, filling_values=np.nan)
    print(data)

    # genfromtxt handles missing values, converters, and column names (use names=True)

    # when to choose text: when interoperability or human readability matters (e.g., handoff to a collaborator who expect CSV)
    # for large arrays or repeated I/O, prefer binary formats

    # raw binary: tofile and fromfile (fast but not self-describing)
    # tofile writes raw bytes without metadata - no shape, no dtype info - so you must manage metadata seperately
    # user only in controlled settings when format is agreed
    
    array = np.arange(12, dtype=np.int32)
    array.tofile('raw.bin')             # raw bytes
    # load, specifying dtype and shape
    array2 = np.fromfile('raw.bin', dtype=np.int32).reshape(3, 4)
    print(array2)

    # warning: endianness, dtype, and shape must be managed manually
    # not recommended for general use

    # practical tips for safe writes
    # write automatically to avoid corrupted files (common when long writes or sudden process termination occur)
    # example pattern
    
    import tempfile, os
    def atomic_save_npy(array, filename):
        dirn = os.path.dirname(filename) or '.'
        with tempfile.NamedTemporaryFile(dir=dirn, delete=False) as tmp:
            np.save(tmp, array)
            tmpname = tmp.name
        os.replace(tmpname, filename)   # atomic on most OS

    # using os.replace ensures the file is replaced atomically on most platforms

    # 4.2 working with memory-mapped files
    # when arrays do not fit into RAM, np.memmap and memory-mapping options let you access data on disk as if it were an array, without loading everything at once
    
    # creating and using memmap
    # create a memmap-backed file and write to it

    # create a memmap-backed array on disk (binary format)
    shape = (10000, 1000)   # 10 million elements
    filename = 'big_array.dat'
    mm = np.memmap(filename, dtype='float32', mode='w+', shape=shape)
    # write a block (only that block touches memory)
    mm[:1000] = np.random.rand(1000, shape[1]).astype('float32')
    mm.flush()  # ensure data is written to disk
    del mm      # close view

    # re-open for read-only access (does not copy into memory)
    mm_r = np.memmap(filename, dtype='float32', mode='r', shape=shape)
    # read a small slice; only the necessary pages are loaded
    block = mm_r[500:1500, 10:20]
    print(block.shape)

    # np.load supports memory mapping of .npy files via mmap_mode
    # save as .npy first
    np.save('X.npy', np.random.rand(2000, 1000).astype('float32'))
    # load as memmap-like object
    X_mem = np.load('X.npy', mmap_mode='r')     # read-only memmap

    # patterns and performance
    # sequential access of slices works well; random, widely scattered reads incur many disk seeks and are slow
    # if you process data in blocks (streaming or batch-processing) memmap is ideal

    # a common pattern to process a large dataset chunk-by-chunk
    with np.load('X.npy', mmap_mode='r') as X_mem:
        for i in range(0, X_mem.shape[0], 100):
            chunk = X_mem[i:i+100]      # small subset fits in RAM
            # process chunk

    # concurrency
    # multiple processes can read the same memmap concurrently
    # concurrent writes require care: coordinate writes to avoid races (use file locks or separate output files)
    # flush (mm.flush()) helps ensure consistency

    # when to use memmap
    # use memmap when your data exceeds available memory and you need random access to subsets
    # for linear streaming workloads, consider chunked reads with tools like HDF5 (next section), which can offer better chunking and compression

    # personal note
    # i once processed a 120GB dataset with memmap by transforming and writing it in-place, then kept a much smaller summary array in RAM
    # memmap saved the project from needing a distributed cluster - just careful chunking and patience

    # 4.3 interfacing with CSV, JSON, and HDF5
    # NumPy can handle CSV and JSON for straightforward cases, but for complex files (mixed types, large CSVs, rich metadata),
    # using higher-level libraries such as Pandas or HDF5 tools is often more practical

    # CSV - NumPy vs. Pandas
    # for simple numeric CSVs, np.loadtxt or np.genfromtxt works, but for complex CSV files (mixed types, dates, millions of rows), rely on Pandas

    # example using NumPy for a simple CSV
    array = np.loadtxt('simple.csv', delimiter=',')
    print(array)

    # example using Pandas (recommended for robust CSV handling)
    import pandas as pd
    df = pd.read_csv('large.csv', \
                     dtype={'id':int, 'values':float}, \
                     parse_dates=['ts'], \
                     usecols={'id','values','ts'})
    X = df[['values']].to_numpy()   # convert to NumPy when you need raw arrays
    print(X)
    print(X.shape)
    # Pandas offers efficient parsing, dtype control, chunked reading (chunksize=), and robust handling of timezones, dates, and missing data

    # JSON - exchanging structured data

    # JSON is ideal for small arrays or metadata
    # NumPy types are not JSON serializable directly (e.g., no.int64), so convert to Python scalars or lists

    # write NumPy arrays to JSON
    import json
    array=np.array([1,2,3], dtype=np.int64)
    # convert to list first
    payload = {'data':array.tolist(), 'meta':{'shape':array.shape}}
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    with open('data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    array2 = np.array(data['data'])
    print(array2)

    # if you want to serialize numpy scalars automatically, implement a small encoder
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    json.dump({'array':array}, f, cls=NumpyEncoder)
    # note: JSON is not efficient for big numerical arrays - prefer binary formats for large data

    # HDF5 - large, structured, portable storage
    # HDF5 is designed for large datasets, hierarchical storage, partial reads, metadata, compression, and chunking
    # use h5py (low-level) or tables/PyTables (higher-level)
    # HDF5 is a superb choice for large scientific datasets

    # install h5py
    # pip install h5py

    # create and read datasets with h5py
    import h5py
    import numpy as np
    X = np.random.rand(10000, 100)
    y = np.random.randint(0, 2, size=(10000,))
    with h5py.File('data.h5', 'w') as f:
        # create dataset with gzip compression and chunking
        dX = f.create_dataset('X', data=X, compression='gzip', chunks=(1000, 100))
        dy = f.create_dataset('y', data=y)
        # attach metadata
        f.attrs['created_by'] = 'M. J. Maxwell'
        f.attrs['description'] = 'Feature Matrix'

    # read back a slice without loading everything
    with h5py.File('data.h5', 'r') as f:
        X_slice = f['X'][100:200]       # reads only this block
    print(X_slice)

    # HDF5 advantages: partial reads, compression, hierarchical organization, metadata
    # downsides: slightly more setup, potential probability/version issues across very different HDF5 library versions (rare in practice)

    # Pandas can read/write HDF5 via pd.to_hdf/pd.read_hdf (uses PyTables), which is convenient for table-like data
    
    # choosing a format - practical guidance
    #  use .npy/.npz for fast intermediate storage of NumPy arrays
    #  use HDF5 when you have many large arrays, hierarchical data, or need partial reads and compression
    #  use CSV for interoperability and light-weight exchange; use Pandas for robust CSV handling
    #  use JSON for small, structured metadata or lightweight data exchange; convert arrays to lists
    #  use memmap for out-of-core single-array workflows when you need random access

    # worked examples: a small pipeline
    #  read a large CSV in chunks with Pandas, convert to NumPy, process each chunk, and append to an HDF5 dataset

    import pandas as pd
    import numpy as np
    import h5py
    import time
    
    csv_file = 'large.csv'
    h5_file = 'processed.h5'

    # create HDF5 containers
    with h5py.File(h5_file, 'w') as hf:
        # create empty dataset with maxshape tp allow appending along axis=0
        dX = hf.create_dataset('X', shape=(0,100), maxshape=(None, 100), dtype='float32', compression='gzip', chunks=(1000,100))

    chunksize = 10_000
    for chunk in pd.read_csv(csv_file, chunksize=chunksize):
        X_chunk = chunk.drop(columns=['id','ts']).to_numpy(dtype=np.float32)    # preprocess
        # process X_chunk (normalization, feature engineering)
        X_chunk = (X_chunk - X_chunk.mean(axis=0)) / (X_chunk.std(axis=0) + 1e-8)
        # append to HDF5
        with h5py.File(h5_file, 'a') as hf:
            dX = hf['X']
            old_n = dX.shape[0]
            new_n = old_n + X_chunk.shape[0]
            dX.resize((new_n, dX.shape[1]))
            dX[old_n:new_n, :] = X_chunk

    # this pipeline is memory-efficient: Pandas reads CSV chunk-by-chunk, processing uses NumPy, and HDF5 stores results with compression

    # key takeaways
    #  .npy and .npz are the simplest, fastest, and most faithful formats for NumPy arrays; use allow_pickle=False when loading untrusted files
    #  text format (CSV) are readable and imteroperable but slower and less precise; for robust CSV handling prefer Pandas and chunksize for large files
    #  use np.memmap or np.load(..., mmap_mode='r') to work with arrays that exceed RAM - process in chunks for best performance
    #  HDF5 (via h5py) is ideal for large, hierarchical datasets with partial I/O and compression; it is a common standard in scientific computing
    #  tofile/fromfile are raw, fast binary options but are not self-describing - use only in controlled contexts
    #  prefer atomic writes (write-to-temp + os.replace) to avoid corrupted files during long writes
    #  always be mindful of endianness, dtype, and memory layout when exchanging files across system of libraries

    # with the I/O tools in this chapter you can build robust pipelines that scale from small experiments to datasets that exceed your laptop's memory
    # in the next part we'll apply these techniques to practical data-science workflows - linear algebra, random sampling, and feature engineering
    # using the I/O patterns you've just learned



#
# CH 5: Linear algebra essentials
#

if True:
    # linear algebra is the language of data science
    # from ordinary least squares to principal-component analysis, the building blocks are vectors and matrices and operations you perform on them
    # this chapter gives you practical, hand-on coverage of the most useful linear-algebra tools in NumPy
    # matrix & vector operations, reliable methods of solving linear systems, and spectral decompositions (eigenvalues/eigenvectors and SVD)
    # you'll get working code samples, numerical caveats and advice for real projects

    # 5.1 matrix and vector operations
    # NumPy makes matrix arithmetic concise and fast
    # the important distiction to keep in mind is element-wise as linear-algebra (matrix) operations

    # create small matrices and vectors experiment
    import numpy as np
    np.set_printoptions(precision=4, suppress=True)
    A = np.array([[1., 2., 3.],
                  [4., 5., 6.]])    # shape (2,3)
    B = np.array([[7., 8.],
                  [9., 10.],
                  [11., 12.]])      # shape (3,2)
    u = np.array([1., 2., 3.])      # shape (3,)
    v = np.array([4., 5.])          # shape (2,)

    # matrix multiplication
    # use the @ operator (or np.matmul) for true matrix multiplication
    C = A @ B       # shape (2,2)
    # equivalent to np.matmul(A, B) or np.dot(A, B) for 2-D arrays

    # element-wise multiply
    # the * operator multiplies element-wise and requires matching shapes (or broadcastable shapes)
    # element-wise multiply - shape must match
    D = A * A       # shape (2,3), element-wise square

    # matrix-vector product
    y = A @ u       # result shape (2,)
    # equivalent to np.dot(A, u)

    # outer product
    # use np.outer or broadcasting to compute and outer product
    outer1 = np.outer(u, v)         # shape (3,2)
    outer2 = u[:,None] * v[None,:]  # or with broadcasting
    print(u)
    print(v)
    print(outer1)
    print(outer2)

    # transpose and conjugate transpose
    # for real arrays, .T gives the transpose
    # for complex arrays, .conj().T for Hermitian transpose
    At = A.T    # shape (3.2)
    C = np.array([3j, 2j])
    print(C.conj().T)

    # batch (stacked) matrix multiplication
    # NumPy supports batched matrix multiplies
    # arrays with shape (batch, m, n) multiplied by (batch, n, p) produce (batch, m, p)
    X = np.random.rand( 10, 2, 3)       # 10 matrices (2, 3)
    Y = np.random.rand( 10, 3, 4)       # 10 matrices (3, 4)
    Z = X @ Y                           # shape (10, 2, 4)

    # Einstein summation
    # np.einsum expresses contractions succinctly and can sometimes be faster or clearer
    C = np.einsum('ik,kj->ij', A, B)    # same as A @ B

    # practical notes
    # perfer @/np.matmul for linear algebra expressions; * is element-wise
    # BLAS-backed dot/matmul calls are highly optimized - ensure NumPy is linked to a good BLAS (OpenBLAS, MKL) for heavy workloads
    # use np.einsum for complex index multiplications or when you want fine control over memory layout and temporary arrays

    # 5.2 solving linear systems
    # a common task: solve Ax=b
    # NumPy exposes several ways
    # choose the right one based on problem size and numerical stability

    # direct solve for square systems
    rng = np.random.default_rng(seed=0)
    A = rng.normal(size=(4,4))
    b = rng.normal(size=(4,))
    x = np.linalg.solve(A, b)   # robust direct solver (LU-based)
    # np.linalg.solve uses LAPACK routines (LU factorization with partial pivoting) and preferable to computing x = np.linalg.inv(A) @ b
    # inverting matrices directly is slower and numerically less stable

    # why not use the inverse?
    # compare solve vs inv
    x_solve = np.linalg.solve(A, b)
    x_inv   = np.linalg.inv(A) @ b
    # they are usually close, but solve is preferred
    print('||x_solve - x_inv||:', np.linalg.norm(x_solve-x_inv))
    # even when the difference is small for well-conditioned matrices,
    # solve is faster and more accurate because it reuses factorization and avoids the large intermediate the inverse

    # condition number and stability
    # the condition number tells you how sensitive solutions are to noise in A or b
    # compute with
    condA = np.linalg.cond(A)   # 2-norm condition number
    print('cond(A) =', condA)
    # large cond(A) (e.g.,>1e8) indicates potential numerical instability - solutions may be wildly inaccurate
    # for ill-conditioned or near-singular A, consider regularization (Tukhonov/ridge) or using SVD/pseudoinverse
    # SVD = singular value decomposition
    print(np.__version__)

    # least-squares and overdetermined systems
    # for A with more rows than columns (overdetermined), use np.linalg.lstsq
    # overdetermined system: more equations than unknowns
    A = rng.normal(size=(100, 3))   # 100 samples, 3 features
    b = rng.normal(size=(100,))
    coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print(coeffs, residuals, rank, s)
    # lstsq solves the normal equations with SVD internally (more stable than forming ATAA^TA directly)
    # use rcond=None with NumPy > 1.14 to get default machine-precision behavior

    
#
# CH 14: Linear Regression from Scratch
#

class LinearRegressionGD:

    def __init__(self,
                 lr: float = 1e-2,
                 n_epochs: int = 1000,
                 batch_size: Optional[int] = None,  # None -> full-batch, 1 -> SGD, -1 -> mini-batch
                 alpha: float = 0.0,                # L2 regularization strength (Ridge)
                 fit_intercept: bool = True,
                 tol: float = 1e-6,
                 shuffle: bool = True,
                 verbose: bool = False,
                 rng: Optional[np.random.Generator] = None,):

        # input parameters
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.rng = rng or np.random.default_rng(0)

        #
        self.coef_ = None   # includes intercept if fit_intercept = True
        self.loss_history = []


    def _add_bias(self,
                  X: np.ndarray) -> np.ndarray:
        #
        if not self.fit_intercept:
            return X

        #
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.concatenate([ones, X], axis=1)


    def _regularization_term(self,
                             w: np.ndarray) -> np.ndarray:
        # return vector to add to gradient for L2 penalty
        if self.alpha == 0.0:
            return 0.0
        
        if not self.fit_intercept:
            return (self.alpha / X.shape[0]) * w

        # do not regularize the intercept (first element)
        reg = (self.alpha / X.shape[0]) * w.copy()
        reg[0] = 0.0
        return reg


    def fit(self,
            X: np.ndarray,
            y: np.ndarray) -> 'LinearRegressionGD':

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n, p = X.shape
        Xb = self._add_bias(X)  # shape (n, p+1) if intercept, else (n, p)
        m = Xb.shape[1]         # init weights (small random or zeros)

        self.coef_ = np.zeros(m, dtype=float)

        # set default batch_size
        if self.batch_size is None:
            batch_size = n      # full-batch
        else:
            batch_size = int(self.batch_size)

        #
        for epoch in range(self.n_epochs):
            #
            if self.shuffle:
                perm = self.rng.permutation(n)
                Xb = Xb[perm]
                y = y[perm]

            epoch_loss = 0.0

            for i in range(0, n, batch_size):
                xb = Xb[i:i+batch_size]
                yb = y[i:i+batch_size]
                pred = xb @ self.coef_      # (batch, )
                err = pred - yb             # (batch, )
                grad = (xb.T @ err) / xb.shape[0]   # (m,)

                # add L2 regularization (do not regularize intercept)
                if self.alpha != 0.0:
                    reg = (self.alpha / n) * self.coef_.copy()
                    if self.fit_intercept:
                        reg[0] = 0.0
                    grad += reg

                # gradient step
                self.coef_ = self.coef_ - self.lr * grad

                # accumulate loss (for monitoring)
                epoch_loss += 0.5 * (err**2).sum()

            epoch_loss / n

















