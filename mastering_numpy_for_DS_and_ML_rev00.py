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

if False:
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

    # underdetermined systems and pseudoinverse
    # for fewer equations than unknowns, the Moore-Penrose pseudoinverse gives a minimum-norm solution
    A = rng.normal(size=(3,5))      # underdetermined
    b = rng.normal(size=(3,))
    x_pinv = np.linalg.pinv(A) @ b
    # np.linalg.pinv uses SVD and is numerically robust but more expensive than solve for square systems

    # practical workflow
    # if A is squre and well-conditioned -> use np.linalg.solve
    # if A is tall (overdetermined) -> use np.linalg.lstsq
    # if A is fat (underdetermined) -> use np.linalg.pinv or add constrains/regularization
    # for very large or sparse systems, use scipy.sparse.linalg iterative solvers (CG, GMRES) - they handle sparsity and scale better

    # worked example: linear regression
    # solve for coefficients in ordinary least squares with normal equations vs lstsq
    # data matrix X (n, p) and target y (n,)
    n, p = 200, 10
    X = rng.normal(size=(n,p))
    y = X @ np.arange(1, p+1) + rng.normal(scale=0.1, size=n)   # synthetic linear signal
    # using lstsq (recommend)
    coeffs_ls ,_ ,_ ,_ = np.linalg.lstsq(X, y, rcond=None)
    print(coeffs_ls)
    # using normal equations (less stabke)
    XtX = X.T @ X
    Xty = X.T @ y
    coeffs_normal = np.linalg.solve(XtX, Xty)   # okay when XtX is well-conditioned
    print(coeffs_normal)
    print('||coeffs_ls - coeffs_normal|| = ', np.linalg.norm(coeffs_ls - coeffs_normal))
    # in many practical cases lstsq is robust and involves fewer pitfalls

    # 5.3 eigenvalues, eigenvectors, and SVD
    # spectral decompositions reveal structure: modes, principal directions, and the effective rank of your matrices
    # they are fundamental to PCA (Principal Component Analysis), dimensionality reduction, and numerical analysis

    # eigenvalues, eigenvectors
    # for a square matrix A, eigenpairs (lambda, v) satisfy Av = lambda v
    M = np.array([[2., 0.],
                  [0., 3.]])
    w, v = np.linalg.eig(M)     # w: eigenvalues, v: columns are eigenvectors
    print(w, v)

    # symmetric/Hermitian matrices
    # if A is symmetric (real) or Hermitian (complex), prefer np.linalg.eigh
    # it's faster and yields real eigenvalues with orthonormal eigenvectors
    S = np.array([[4., 1.],
                  [1., 3.]])
    w_sym, v_sym = np.linalg.eigh(S)
    # verify reconstruction
    recon = (v_sym * w_sym) @ v_sym.T   # equivalent to v_sym @ np.diag(W_sym) @ v_sym.T
    print(recon)
    # eigh is what you want for covariance matrices (used in PCA)

    # SVD (Singular Value Decomposition)
    # for any m x n matrix A, SVD factors A = U SIGMA V T A where singular value in SIGMA are nonnegative and sorted descending
    A = rng.normal(size=(6,4))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)    # s is 1-D array of singular values
    SIGMA = np.diag(s)
    # reconstruction
    A_rec = U @ SIGMA @ Vt
    print(A)
    print('reconstruction error:', np.linalg.norm(A-A_rec))

    # uses of SVD
    # low-rank approximation: keep top k singular values to get the best rank-k approximation (Eckart-Young theorem)
    k = 2
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    A_k = U_k @ np.diag(s_k) @ Vt_k
    print(A_k)
    # this is useful for denoising and compression (image compression example commonly used)

    # Pseudoinverse: SVD is used to compute the pseudoinverse robustly
    A_pinv = Vt.T @ np.diag(1/s) @ U.T
    # np.linalg.pinv wraps this with tolerance for small singular values

    # Rank and numerical rank: small singular values indiate directions where the matrix is (nearly) singular
    print('singular values:', s)
    print('numerical rank (s>tol):', np.sum(s>1e-10))

    # Eigen vs. SVD
    # use eigen decomposition for square matrices where eigen-structure is meaningful (e.g., dynamic systems, spectral graph theory)
    # use SVD for general matrices for stable computations (least-squares, pseudoinverse, low-rank approximation)
    # SVD handles non-symmetric matrices naturally and is numerically robust

    # worked example: PCA (Principal Component Analysis) via SVD (Singular Value Decomposition)
    # PCA on centered data x
    # X: (n_samples, n_features)
    X = rng.normal(size=(100,5))
    X_centered = X - X.mean(axis=0)
    # SVD on centered data
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # principal components are rows of Vt
    pcs = Vt[:2]    # top 2 principal directions
    # projection of data onto top-2 PCs
    proj = X_centered @ pcs.T       # shape (100,2)
    print(proj)
    # using SVD avoids forming the covariance X^T x explicitly and numerically preferred for high-dimensional data

    # practical performance notes
    # SVD is more expensive than solve or eigen decomposition: complexity roughly O( min(m, n) )
    # for very large sparse matrices, use scipy.sparse.linalg.svds or randomized SVD (scikit-learn) which scale better
    # NumPy delegates linear-algebra work to LAPACK/BLAS
    # performance and multithreading depend on the linked implementation (MKL, OpenBLAS, etc.)
    # for heavy linear algebra, check what BLAS your NumPy uses

    # numerical example: ill-conditioned matrix and effect on solutions
    # construct near-singular matrix
    A = np.array([[1., 1.],
                  [1., 1.0000001]])
    b = np.array([2.0, 2.000000001])
    cond = np.linalg.cond(A)
    x = np.linalg.solve(A, b)
    print(cond, x)
    # small changes in b or floating errors can produce large relative changes in x
    # large condition number warns you that solutions may be unreliable
    # user regularization or SVD-based techniques in such cases

    # personal insight
    # I treat SVD as a diagnostic first: inspecting singular values tells me whether a problem is well-posed or whether I need to ass regularization
    # in practice, when a nomal equation gives surprising coefficients, I run np.linalg.svd - the singular value spectrum tells
    # the story much faster than blind debugging

    # key takeaways
    # use @ / np.matmul for marix multiplication: * is element-wise, np.einsum is powerful for custom contractions
    # solve well-conditioned, square systems with np.linalg.solve, avoid explicit matrix inversion for solving linear systems
    # for overdetermined (least-square) problems, use np.linalg.lstsq
    # for underdetermined problems, consider np.linalg.pinv or regularization
    # compute condition numbers with np.linalg.cond to judge numerical stability, large condition numbers require caution and regularization
    # use np.linalg.eig / np.linalg.eigh to inspect eigen-structure; prefer eigh for symmetruc/Hermitian matrices
    # use SVD (np.linalg.svd) for robust decomposition, pseudoinverse, rank estimation and low-rank approximation
    # for very large/sparse problems, move to scipy/scikit-learn methods (sparse solvers, randomized SVD)
    # NumPy linear algebra is fast when linked to optimized BLAS/LAPACK; for heavy workloads, verify your BLAS backend

    # with these tools you can implement regression, PCA, spectral methods, and more - reliably and efficiently
    # in the next chapter we'll look at randomness, sampling, and probability distribution,
    # so you can build simulations, bootstrap procedures, and stochastic algorithms on top of these linear-algebra primitives



#
# CH 6: Random numbers and statistics
#

if False:
    # randomness and statistics are central to data science: sampling, simulation, boostrapping, hypothesis check, Monte-Carlo methods,
    # and simple descriptive analytics all rely on generating and summarizing random data correctly and efficiently
    # NumPy provides a modern, fast random API and a compact set of statistical primitives that are the backbone for many workflows
    # this chapter explains the modern Generator API, demonstrates the most-used probability distributions, and shows how to compute
    # reliable descriptive statistics and simple inferential procedures with NumPy
    # you'll get practical, copy-and-run examples and guidance about reproducibility, parallel streams, numerical stability, and performance

    # 6.1 the numpy.random.Generator API
    # NumPy's new random API (available since NumPy 1.17) centers on the Generator class
    # it replaces the legacy RandomState with a more robust, flexible design and better seeding semantics
    # use np.random.default_rng() to create a Generator
    import numpy as np
    # recommended: create a generator with a fixed seed for reproducibility
    rng = np.random.default_rng(seed=12345)
    # draw 4 uniform [0,1) floats
    samples = rng.random(5)
    print(samples)

    # why prefer Generator over the legacy global functions?
    # clear separation between generator objects (you avoid global state)
    # better seeding and reproducibility (via SeedSequence)
    # support for modern bit generators and parallel-safe spawning
    # more consisent behavior for methos like choice, integers, and multivariate_normal

    # seeding and reproducibility
    # a reproducible run needs a seed
    # the simplest pattern is:
    rng = np.random.default_rng(seed=42)
    # if you need reproducible independent streams (for parallel or multi-state pipelines), use SeedSequence.spawn
    seed = np.random.SeedSequence(12345)
    children = seed.spawn(4)    # 4 independent child seeds
    rngs = [np.random.default_rng(s) for s in children]
    # rngs[0], rngs[1], ... produce independent streams reproducibly
    # this is safer than slicing a single RNG across processes or threads

    # Legacy RandomState (avoid for new code)
    # you may still encounter code using np.random.seed() or np.random.RandomState()
    # those APIs use an older generator with different behavior
    # for new projects, prefer default_rng
    # if you must interoperate with legacy code, convert or isolate legacy usage

    # parallel sampling patterns
    # when running parallel jobs (multiprocessing, distributed tasks), spawn independent SeedSequence children and give each worker its own Generator
    # avoid sharing a single Generator instance across processes

    # 6.2 common probability distribution
    # Generator exposes methods for the common distributions you'll need
    # two important principles
    # (1) sample vectorized (pass size), and
    # (2) prefer built-in methods over Python loops for speed

    # uniform and floats
    rng = np.random.default_rng(seed=0)
    u = rng.random(6)       # uniform on [0,1)
    u2 = rng.uniform(low=-1.0, high=1.0, size=(3,4))
    # random is equivalent to uniform(0, 1)

    # integers and permutations
    # integers replaces legacy radint
    ints = rng.integers(low=0, high=10, size=10)    # integers in [low, high)
    perm = rng.permutation(10)      # random permutation (shuffle indices)
    print(ints)
    print(perm)
    # rng.choice supports sampling with/without replacement and weights
    vals = np.arange(10)
    sample = rng.choice(vals, size=4, replace=False)    # withour replacement
    weighted = rng.choice(vals, size=5, replace=True, p=np.linspace(1,10,10)/55)
    print(sample)
    print(weighted)

    # Gaussian (normal), binomial, Poisson, etc.
    normal = rng.normal(loc=0.0, scale=1.0, size=(1000,))
    binom = rng.binomial(n=10, p=0.3, size=1000)
    poisson = rng.poisson(lam=3.5, size=1000)
    # Generator exposes many other distributions: exponential, gamma, beta, chisquare, laplace, geometric, etc.

    # multivariate normal
    mean = np.array([0.0, 1.0])
    cov = np.array([[1.0, 0.5],
                    [0.5, 2.0]])
    X = rng.multivariate_normal(mean, cov, size=500)    # shape (500, 2)
    # be careful: covariance must be positive semi-definite
    # if the covariance is ill-conditioned, consider adding a small diagonal jitter

    # vectorized sampling and memory
    # sampling thousands or millions of draws is fast, but be mindful of memory
    # example to generate a huge sample in chunks
    def large_sample(rng, total, chunk=10_000_000):
        i = 0
        while i < total:
            n = min(chunk, total - i)
            yield rng.normal(size = n)
            i += n
    # use chunked generation for very large Monte Carlo simulations to avoid consuming all RAM

    # practical recipes
    # train/test split (simple, reproducible)
    n = 1000
    rng = np.random.default_rng(seed=0)
    perm = rng.permutation(n)
    train_idx = perm[:800]
    test_idx = perm[800:]
    
    # bootstrap confidence interval for a statistic
    data = rng.normal(loc=10, scale=2, size=200)
    def bootstrap_ci(data, statfunc=np.mean, n_boot=10000, alpha=0.05):
        n = len(data)
        boots = rng.choice(data, size=(n_boot, n), replace=True)     # (n_noot, n)
        stats = statfunc(boots, axis=1)
        lower = np.percentile(stats, 100*(alpha/2))
        upper = np.percentile(stats, 100*(1-alpha/2))
        return lower, upper
    ci = bootstrap_ci(data)
    print('bootstrap 95% CI for mean:', ci)
    
    # monte carlo estimate (pi)
    def estimate_pi(rng, n=1_000_000):
        x = rng.random(n)
        y = rng.random(n)
        inside = (x*x+y*y)<=1.0
        return inside.sum() / n * 4.0
    print('pi estimate:', estimate_pi(rng, 1_000_000))

    # 6.3 statistical functions and descriptive analytics
    # NumPy provides fast vectorized functions for the essentials: means, medians, variances, percentiles, histograms, correlation, and covariances
    # for more advanced statistics (e.g., skewness, kurtosis, hypothesis tests), use scipy.stats
    # but for everyday EDA and numeric pipelines, NumPy is often sufficient

    # means, variances, and degrees of freedom
    x = rng.normal(size=(1000,))
    m = x.mean()        # mean
    s = x.std(ddof=0)   # std(ddof=0)
    s_sample = x.std(ddof=1)    # sample standard deviation (ddof=1)
    print(m, s, s_sample)
    # ddof (delta degrees of freedom) controls the denominator N-ddof
    # use ddof=1 for the unbiased sample standard deviation in many statistical contexts
    # for large N the difference is minor, but be explicit

    # nan-aware statistics
    # datasets often contain missing values
    # NumPy offers nan-aware variants
    x = np.array([1.0, 2.0, np.nan, 4.0])
    m_ = np.nanmean(x)
    md_ = np.nanmedian(x)
    s_ = np.nanstd(x)
    print(x, m_, md_, s_)
    # these functions ignore np.nan values; this is often what you want for real-world data

    # percentile and quantiles
    x = rng.normal(size=(1000,))
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    print(q25, q50, q75)
    q = np.quantile(x, [0.25, 0.50, 0.75])
    print(q)
    # be aware of interpolation options (interpolation parameter in older versions; current versions offer method choices)
    # for deterministic unit tests, document the method used

    # histrograms, bins, and densities
    # np.histogram is the basic buliding block for empirical distributions
    vals, edges = np.histogram(x, bins=20, density=False)  # probabilies per bin when density=True
    print(len(x), len(vals), len(edges))
    print(x)
    print(vals)
    print(edges)
    if False:
        import matplotlib.pyplot as plt
        plt.plot((edges[1:]+edges[:-1])/2.0, vals, 'o-')
        plt.show()
    # for integer counts and weighted histogram, np.bincount is extremely efficient
    labels = np.array([0,1,2,1,0,2,2])
    counts = np.bincount(labels)
    print(labels)
    print(counts)
    # weighted counts
    weights = np.array([1.0,0.5,1.2,1.0,0.8,1.0,0.5])
    weighted = np.bincount(labels, weights=weights)
    print(weights)
    print(weighted)
    # np.digitize maps values to bin indices (useful for bucketing features)
    bins = np.array([0, 10, 20, 50])
    indices = np.digitize([5,15,30], bins)  # returns bin indices
    print(bins)
    print(indices)

    # covariance and correlation
    # np.cov and np.corrcoef compute covariance and Pearson correlations
    # pay attention to the rowvar argument: default is rowvar=True (each row is a variable)
    X = rng.normal(size=(100, 3))       # 100 observations, 3 features
    cov = np.cov(X, rowvar=False)       # shape (3,3)
    print(cov)
    corr = np.corrcoef(X, rowvar=False)
    print(corr)
    # if you want the sample covariance matrix with unbiased estimator (division by N-1),
    # np.cov already does that by default(bias=False)

    # weighted averages
    # use up.average to compute weighted means
    vals = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.2, 0.3, 0.5])
    wmean = np.average(vals, weights=weights)
    print(wmean, vals.mean())
    # to get weighted variance, compute weighted average of squared deviations (or use specialized routines in other libraries)

    # correlation of time series (rolling windows)
    # NumPy provides building blocks (np.convolve, np.lib.stride_tricks.sliding_window_view) to build rolling statistics,
    # but for complex rolling functionality prefer pandas
    # still, a simple rolling mean using stride tricks
    from numpy.lib.stride_tricks import sliding_window_view
    data = rng.normal(size=100)
    windowed = sliding_window_view(data, window_shape=5)
    rolling_mean = windowed.mean(axis=1)    # length 96
    if False:
        import matplotlib.pyplot as plt
        plt.plot(data)
        plt.plot(rolling_mean)
        plt.show()
    # sliding_window_view returns a view (cheap), but be mindful of memory because it creates an array with an extra dimension

    # resampling and permutation tests
    # permutation tests are simple to implement with rng.permutation
    # example: test whether two samples have different means under the nul hypothesis by shuffling labels
    def permutation_test(a, b, n_perm=10000):
        rng = np.random.default_rng(seed=0)
        observed = a.mean() - b.mean()
        pooled = np.concatenate([a, b])
        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(pooled)
            a_p = perm[:len(a)]
            b_p = perm[len(a):]
            if (a_p.mean() - b_p.mean()) >= observed:
                count += 1
        pvalue = (count+1) / (n_perm+1)
        return pvalue
    pvalue = permutation_test(a=np.arange(100), b=np.arange(200))
    # for speed, vectorize the permutations in blocks (but be careful with memory)

    # practical examples & patterns
    # 1. reproducible train/test split
    rng = np.random.default_rng(seed=1)
    n = 500
    idx = rng.permutation(n)
    train_idx, val_idx = idx[:400], idx[400:450]

    # 2. stratified sampling (class proportions preserved)
    # use np.unique with return_inverse or rely on scikit-learn's train_test_split(..., stratify=...)
    # with NumPy
    labels = rng.integers(0, 3, size=200)
    unique, inv = np.unique(labels, return_inverse=True)
    train_idx = []
    for cls in unique:
        cls_idx = np.where(labels == cls)[0]
        k = int(0.8*len(cls_idx))
        chosen = rng.choice(cls_idx, size=k, replace=False)
        train_idx.append(chosen)
    train_idx = np.concatenate(train_idx)
    print(len(train_idx))
    print(labels)

    # 3. bootstrap for estimator variability
    # shown earlier; useful for non-parametric CI and small sample inference

    # 4. Monte Carlo integration
    # estimate expected value E[f(x)] by sampling
    def monte_carlo_expectation(rng, f, n=100_000):
        x = rng.random(n)
        return f(x).mean()

    # pitfalls, numerical caveats, and best practices
    # randomness & determinism: for unit tests, set explicit seeds
    # for production simulations, be aware of reproducibility requirements and document seeds or use SeedSequence hierarchies
    # legacy API differences: np.random.seed() and global functions differ from Generator
    # avoid mixing unless you understand the consequences
    # memory footprint during vectorized sampling: drawing a size=(100_000_000,) array will allocated memory accordingly
    # use chunking if necessary
    # precision and accumulation: when summing very large arrays, numerical error accumulate
    # use dtype promotion and np.sum(..., dtype=np.float64) if summing many float32s
    # for extreme accuracy consider math.fsum or compensated summation algorithm
    # interperting std and var: be default NumPy uses population formulas (ddof=0)
    # use ddof=1 for sample estimates when appropriate
    # use SciPy for inference: for t-tests, skewness, kurtosis, p-value, and advanced fitting
    # use scipy.stats rather than implementing from scratch

    # key takeaways
    # use np.random.default_rng() and the Generator API for modern, reproducible randomness
    # seed with SeedSequence and spawn child sequences for independent parallel streams
    # prefer vectorized sampling (single size cell) and chunk stream if memory is limited
    # familiarize yourself with the most used distributions: random, integers, uniform, normal, binomial, poisson, multivariate_normal
    # use NumPy's statistical functions (mean, std, median, percentile, cov, corrcoef, histrogram, bincount) for fast descriptive analytics
    # use SciPy for advanced inference
    # for real-world pipelines, combine Pandas (for robust CSV/IO and group operations) with NumPy for heavy numeric work
    # always be explicit about ddof, dtype, nan-handling, and seed choices to avoid subtle bugs

    # this chapter equips you to generate random data, build reproducible experiments, and compute key descriptive statistics directly and efficiently
    # with Numpy
    # in the next chapters we'll apply these tools to feature engineering, model preprocessing, and building a simple linear-regression model from
    # scratch



#
# CH 7: Data cleaning and feature engineering
#

if False:
    # cleaning and feature engineering turn raw data into something a model can learn from
    # good preprocessing is often what seperates a working prototype form a reliable pipelines
    # in this chapter we cover practical NumPy techniques for handling missing or invalid values, and building end-to-end NumPy pipelines
    # you can apply to real datasets
    # you'll get carefully commented, runnable examples and guidance on the trade-offs you'll face in production

    # 7.1 handling missing or invalid data
    # missing data shows up in many forms: NaNs in float arrays, sentinel values(e.g., -999), or empty strings in CSVs
    # the general workflow is (1) detect, (2) characterized (how much/where), (3) decide strategy (drop/impute/mask),
    # (4) apy consistently (fit on training data, apply to test)

    # detecting missing values
    # for true numeric NaNs use np.isnan
    # for sentinel values detect equality
    import numpy as np
    rng = np.random.default_rng(seed=0)
    # example dataset (5 rows, 3 features)
    X = rng.normal(size=(5,3))
    X[1,0] = np.nan         # insert a missing data
    X[3,2] = np.nan
    print(X)
    nan_mask = np.isnan(X)
    print(nan_mask)
    print(nan_mask.any(axis=0))     # any NaNs per column
    print(nan_mask.sum(axis=0))     # counts per column
    # if your missing values are encoded as -999 or empty strings from CSV,
    # convert them to np.nan first so you can use the same APIs

    # simple imputation: column mean/median
    # a robust, often effective choice is to replace missing feature values with the column mean or median computed from the training data
    # always compute statistics on training data and apply to validation/test sets
    def impute_mean(X, inplace=False):
        X = X if inplace else X.copy()
        col_mean = np.nanmean(X, axis=0)    # shape (n_features,)
        inds = np.where(np.isnan(X))        # tuple (rows, cols)
        X[inds] = col_mean[inds[1]]
        return X, col_mean
    X_imputed, means = impute_mean(X)
    print(X)
    print(X_imputed)
    print(means)
    # np.nanmean ignores NaN values
    # if an entire column is NaN you'll get nan back - handle that with a fallback values (e.g., 0 or median)

    # imputation bu interpolation (time series)
    # for ordered data (time series), linear interpolation often preserves temporal structure better than global mean
    def interp_fill(col):
        # col: 1D numeric array with np.nan for missing
        n = len(col)
        idx = np.arange(n)
        valid = np.where(~np.isnan(col))[0]
        if valid.size == 0:
            return np.full(n, 0.0)  # fallback
        return np.interp(idx, valid, col[valid])
    # apply per-column
    X2 = X.copy()
    for j in range(X2.shape[1]):
        X2[:,j] = interp_fill(X2[:,j])
    print(X)
    print(X2)
    # np.interp will forward/backfill endpoints with the nearest valid value
    # for more complex time-series imputation (seasonality, AR models), use domain-specific methods or pandas

    # forward-fill / backward-fill without pandas
    # you can forward-fill (carry last observation forward) with a small NumPy trick using indicies
    def forward_fill(col):
        n = len(col)
        mask = np.isnan(col)
        idx = np.where(~mask, np.arange(n), 0)
        # make index cumulative so each NaN points to last valid index
        np.maximum.accumulate(idx, out=idx)
        filled = col[idx]
        # if leading values are NaN they will map to index 0 - handle separately
        first_valid = np.where(~mask)[0]
        if first_valid.size == 0:
            return np.full(n, 0.0)
        if first_valid[0] != 0:
            filled[:first_valid[0]] = col[first_valid[0]]   # backfill leading NaNs
        #
        return filled
    X2 = X.copy()
    X2[0,1] = np.NaN
    print(X2)
    for j in range(X2.shape[1]):
        X2[:,j] = forward_fill(X2[:,j])
    print(X2)
    # this method is vectorized and fast for large 1D arrays, apply per column per column for 2D

    # masking and masked arrays
    # if you perfer to keep missing values explicit, NumPy's masked arrays(np.ma) let you perform computations while ignoring masked elements
    print(X)
    m = np.ma.masked_invalid(X)     # masks NaN and infs
    print(m)
    col_mean_masked = m.mean(axis=0).data
    # masked arrays perseve mask through many ops
    m_filled = m.filled(fill_value=-1)
    print(m_filled)
    # masked arrays are handy when you need to propagate missingness through complex calculations rather than immediately filling

    # dropping rows or columns
    # if a row has many missing values and imputation is not justified, drop it
    row_valid_counts = (~np.isnan(X)).sum(axis=1)
    keep = row_valid_counts >= 2    # require at least 2 valid features
    x_filtered = X[keep]
    print(X)
    print(row_valid_counts)
    print(keep)
    print(x_filtered)
    # similarly drop columns with too many missing values

    # putting it together: an imputer object (NumPy style)
    # create a simple object that fits on training data and can transform new data
    class SimpleImputer:
        def __init__(self, strategy='mean', fill_value=0.0):
            assert strategy in ('mean', 'median', 'constant')
            self.strategy = strategy
            self.fill_value = fill_value
            self.statistics_ = None
            print('SimpleImputer')

        def fit(self, X):
            if self.strategy == 'mean':
                self.statistics_ = np.nanmean(X, axis=0)
            elif self.strategy == 'median':
                self.statistics_ = np.nanmedian(X, axis=0)
            else:
                self.statistics_ = np.full(X.shape[1], self.fill_value)
            # fallback: replace NaN stats with fill_value
            nan_stats = np.isnan(self.statistics_)
            if nan_stats.any():
                self.statistics_[nan_stats] = self.fill_value
            #
            print(self.statistics_)
            return self

        def transform(self, X):
            X = X.copy()
            inds = np.where(np.isnan(X))
            print(inds)                     # ( row_array, col_array )
            print(inds[1])                  # col_array
            X[inds] = self.statistics_[inds[1]]
            return X
    #
    imp = SimpleImputer(strategy='mean').fit(X)
    X_imp = imp.transform(X)
    print(X)
    print(X_imp)
    # this ensures consistent behavior and easy application to test sets

    # 7.2 scaling and normalization
    # scaling makes features comparable and speeds up optimization
    # the two most common scalings are standardization (z-score) and min-max scaling; each has domain use cases

    # why scale ?
    # distance-base algorithms (k-NN, k-means), gradient-based optimization (neural nets), and regularization all benefit from features on similiar scales
    # if features are already comparable (e.g., all percentiles), scaling may not be necessary

    # standardization (z-score)
    # subtract column mean and divide by column standard deviation
    # fit on training data; apply same parameters to test
    def standard_scale(X_train, X):
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0, ddof=0)
        # avoid divide-by-zero
        std_safe = np.where(std == 0, 1.0, std)
        return (X - mean) / std_safe, mean, std_safe
    # example
    rng = np.random.default_rng(seed=0)
    X_train = rng.normal(10, 2, size=(100,3))
    X_test  = rng.normal(11, 3, size=(20,3))
    X_train_scaled, mean, std = standard_scale(X_train, X_train)
    X_test_scaled, _, _ = standard_scale(X_train, X_test)       # user X_train params
    print(X_train_scaled.min(), X_train_scaled.max(), mean, std)
    print(X_test_scaled.min(), X_test_scaled.max())
    # use ddof=0 for population std (NumPy default);
    # use ddof=1 for classical unbiased estimator when appropriate

    # Min-Max scaling
    # transforms each column to range [0, 1) (or [a, b])
    def minmax_scale(X_train, X, feature_range=(0,1)):
        X_min = X_train.min(axis=0)
        X_max = X_train.max(axis=0)
        scale = (X_max - X_min)
        scale_safe = np.where(scale == 0, 1.0, scale)
        X_std = (X - X_min) / scale_safe
        a, b = feature_range
        return X_std * (b - a) + a, X_min, X_max
    # example
    X_train_mm, Xmin, Xmax = minmax_scale(X_train, X_train)
    X_test_mm, _, _ = minmax_scale(X_train, X_test)
    print('min max scaling')
    print(Xmin, Xmax)
    print(X_train_mm)
    # if a column is constant (max == min), decide whether to leave it zero or assign some constant (e.g.,0.5)

    # Robust scaling (median and IQR)
    # robust to outliers: center by median and scale by interquartile range (IQR)
    def robust_scale(X_train, X):
        med = np.median(X_train, axis=0)
        q75 = np.percentile(X_train, 75, axis=0)
        q25 = np.percentile(X_train, 25, axis=0)
        iqr = q75 - q25
        iqr_safe = np.where(iqr == 0, 1.0, iqr)
        return (X-med) / iqr_safe, med, iqr_safe
    # example
    X_train_rs, X_train_rs_med, X_train_rs_iqr = robust_scale(X_train, X_train)
    print('robust scaling')
    print(X_train_rs_med, X_train_rs_iqr)

    # row (sample) normalization: L2/L1 norms
    # sometimes you want each row to have unit length (common in text embeddings):
    def L2_normalize_rows(X, eps=1e-12):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms_safe = np.where(norms == 0, eps, norms)
        return X / norms_safe
    X_normed = L2_normalize_rows(X_train)
    print('L2 normalization')
    print(X_normed)

    # in-place vs. out-of_place transforms
    # in-place operations save memory: X -= mean and X /= std
    # but if you need to keep originals, operate on copies
    # also convert integers to floats before scaling
    X = X.astype(np.float32, copy=False) # convert to float32 if not already
    X -= mean
    X /= std
    print(X)

    # scaling pipeline example: StandardScalar object
    class StandardScalar:
        def __init__(self, with_mean=True, with_std=True, dtype=np.float64):
            self.with_mean = with_mean
            self.with_std  = with_std
            self.dtype     = dtype

        def fit(self, X):
            X = X.astype(self.dtype, copy=False)
            self.mean_ = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1], dtype=self.dtype)
            self.scale_ = X.std(axis=0, ddof=0) if self.with_std else np.ones(X.shape[1], dtype=self.dtype)
            self.scale_[self.scale_ == 0] = 1.0
            return self

        def transform(self, X):
            X = X.astype(self.dtype, copy=True)
            if self.with_mean:
                X -= self.mean_
            if self.with_std:
                X /= self.scale_
            return X
        # practical advice
        # 1. always fit scalars on training data only
        # 2. keep scalar parameters (mean, std, min, max) with your model for reproducible transforms
        # 3. be mindful of dtype: use float32 for memory efficiency, float64 for high precision
        # 4. if features are counts or sparse, consider log transforms before scaling: np.log1p(X)

        # 7.3 encoding categorical features
        # NumPy lacks the high-level label-encoding utilities that pandas and scikit-learn provide,
        # but it can still do label encoding and one-hot encoding efficiently
        # the typical steps are: map categories to integer labels, handle unseen categories,
        # and decide between dense one-hot or a sparse representation

        # label encoding (map category -> integer)
        # use np.unique(return_inverse=True) to discover unique categories and map values to indicies
        cats = np.array(['dog', 'cat', 'cat', 'bird', 'dog'])
        unique, inverse = np.unique(cats, return_inverse=True)
        print(cats)
        print(unique)
        print(inverse)
        # inverse is the label-encoded integer array
        # save unique as the mapping (index -> category)
        # for transform on new data, use a dict lookup
        mapping = {cat:i for i, cat in enumerate(unique)}
        print(mapping)
        def transform_labels(new_cats, mapping, unknown=-1):
            return np.array([mapping.get(x, unknown) for x in new_cats], dtype=int)
        test = np.array(['cat', 'fox'])
        print(transform_labels(test, mapping))
        # mapping with a python loop is simple and fast for moderate numbers of items;
        # for very large arrays, vectorized approaches with np.searchsorted on sorted unique arrays can be faster

        # one-hot encoding (dense)
        # given integer labels inverse and k = len(unique), one-hot matrix is
        k = unique.size
        one_hot = np.eye(k, dtype=int)[inverse]     # shape (n_samples, k)
        print(k)
        print(one_hot)
        # this creates a dense matrix
        # it is fine for small cardinality
        # for high-cardinality (e.g., thousands of categories),
        # dense one-hot becomes expensive

        # on-hot with unknowns
        # if some items map to -1 for unknown categories, handle them explicitly
        def one_hot_from_labels(labels, k, unknow_index=None):
            # labels: int array with -1 for unknown
            onehot = np.zeros((labels.size, k), dtype=int)
            mask = labels >= 0
            onehot[np.arange(labels.size)[mask], labels[mask]] = 1
            if unknown_index is not None:
                # set unknown column
                onehot[~mask, unknown_index] = 1
            return onehot

        # frequency or target encoding (alternative to one-hot)
        # for very high cardinality, encode categories by statistics (frequency, mean target)
        vals, inv = np.unique(cats, return_inverse=True)
        counts = np.bincount(inv)
        freq = counts[inv] / counts.sum()   # frequency per sample
        print(cats)
        print(vals)
        print(inv)
        print(counts)
        print(freq)
        # target encoding uses average target per category (careful: risk of target leakage -
        # fit only on training data and apply smoithing)

        # hashing trick (memory-efficient, approximate)
        # hash categories to a fixed number of bins (useful in streaming or high-cardinal contexts)
        # example
        def hashing_encode(strings, n_bins=256):
            # deterministic across runs; use python's hash but take modulo
            h = np.fromiter((hash(s) % n_bins for s in strings), dtype=np.int64)
            return h    # then one-hot or use as categorical feature index
        print(hashing_encode('asdfasdfasdf'))
        # hashes are fast and memory-efficient but introduce collisions

        # practical considerations
        # preserve category-to-index mapping and include an 'unknown' bucket
        # avoid target leakage - compute encoding on train data only
        # for sparse or very high-cardinality features, consider sparse matrix (scipy.sparse) or hashing + embedding in models
        # when you can, use pandas.Categorical of sklearn.preprocessing utilities for convenience and clarity

        # 7.4 working with real datasets - an end-to-end numpy pipeline
        # below is a compact but complete example that demonstrates a realistic workflow
        # generate a synthetic mixed dataset (numerial + categorical + missing),
        # split into train/test,
        # fit imputation/encoding/scaling on train, transform test,
        # and preserve parameters for later use
        # this is intentionally implemented with pure NumPy to show the mechanics;
        # in many real projects, combining pandas for IO and bookkeeping with NumPy for heavy numeric work is pragmatic
        import numpy as np
        from dataclasses import dataclass
        rng = np.random.default_rng(seed=0)
        # 1. create synthetic dataset
        n_samples = 200
        num_features = 3    # numeric features with some NaNs
        X_num = rng.normal(loc=[10, 0, 100], scale=[2, 5, 20], size=(n_samples, num_features))
        mask_nan = rng.random(X_num.shape) < 0.1
        X_num[mask_nan] = np.nan
        # 2. categorical feature
        cats = np.array(['red', 'green', 'blue'])
        X_cat = rng.choice(cats, size=(n_samples,))     # introduce unseen category in test set later
        # 3. target (regression)
        beta = np.array([1.5, -2.0, 0.01])
        y = np.nansum(np.nan_to_num(X_num, nan=0.0) * beta, axis=1) + \
            rng.normal(scale=1.0, size=n_samples)
        # 4. train/test split (reproducible)
        perm = rng.permutation(n_samples)
        train_idx = perm[:150]
        test_idx  = perm[150:]
        X_num_train = X_num[train_idx]
        X_num_test  = X_num[test_idx]
        X_cat_train = X_cat[train_idx]
        X_cat_test  = np.copy(X_cat[test_idx])
        X_cat_test[0] = 'yellow'                    # inject unseen category in test
        y_train = y[train_idx]
        y_test  = y[test_idx]
        # 5. define preprocessing components (imputer, encoder, scalar)
        @dataclass
        class NumPyPreprocessor:
            imputer_strategy: str = 'mean'          # or 'median'
            scaler: str = 'standard'                # 'minmax', 'standard', 'robust'
            cat_unknown_token: str = '__UNK__'
            
            def fit(self, X_num, X_cat):
                # imputer
                if self.imputer_strategy == 'mean':
                    self.imputer_stats_ = np.nanmean(X_num, axis=0)
                else:
                    self.imputer_stats_ = np.nanmedian(X_num, axis=0)
                    
                # fallback for all-NaN columns
                nan_stats = np.isnan(self.imputer_stats_)
                if nan_stats.any():
                    self.imputer_stats_[nan_stats] = 0.0
                    
                # scalar params
                if self.scaler == 'standard':
                    self.scale_mean_  = np.nanmean(X_num, axis=0)
                    self.scale_scale_ = np.nanstd(X_num, axis=0, ddof=0)
                    self.scale_scale_[self.scale_scale_ == 0] = 1.0
                elif self.scaler == 'minmax':
                    self.scale_min_ = np.nanmin(X_num, axis=0)
                    self.scale_max_ = np.nanmax(X_num, axis=0)
                    diff = self.scale_max_ - self.scale_min_
                    diff[diff == 0] = 1.0
                    self.scale_min_, self.scale_max, self.scale_range_ = self.scale_min_, self.scale_max_, diff
                else:
                    # robust
                    self.scale_median_ = np.nanmedian(X_num, axis=0)
                    q75 = np.nanpercentile(X_num, 75, axis=0)
                    q25 = np.nanpercentile(X_num, 25, axis=0)
                    iqr = q75 - q25
                    iqr[iqr == 0] = 1.0
                    self.scale_median_, self.scale_iqr_ = self.scale_median, iqr
                    
                # categorical mapping
                unique, inv = np.unique(X_cat, return_inverse=True)
                self.cat_mapping_ = {cat:i for i, cat in enumerate(unique)}
                self.cat_list_ = list(unique) + [self.cat_unknown_token]
                self.n_cat_ = len(self.cat_list_)

                return self

            def transform_num(self, X_num):
                X = X_num.copy().astype(np.float64)
                # impute
                inds = np.where(np.isnan(X))
                if inds[0].size > 0:
                    X[inds] = self.imputer_stats_[inds[1]]
                # scale
                if self.scaler == 'standard':
                    X = (X - self.scale_mean_) / self.scale_scale_
                elif self.scaler == 'minmax':
                    X = (X - self.scale_min_) / self.scale_range_
                else:
                    X = (X - self.scale_median_) / self.scale_irq_

                return X

            def transform_cat(self, X_cat):
                # label-encode with unknown handling
                labels = np.full(X_cat.shape[0], -1, dtype=int)
                for i, v in enumerate(X_cat):
                    labels[i] = self.cat_mapping_.get(v, self.n_cat_-1)  # unknown -> last index
                # one-hot
                onehot = np.zeros((labels.size, self.n_cat_), dtype=np.float32)
                mask = labels >= 0
                onehot[np.arange(labels.size)[mask], labels[mask]] = 1.0

                return onehot

            def transform(self, X_num, X_cat):
                Xn = self.transform_num(X_num)
                Xc = self.transform_cat(X_cat)

                return np.hstack([Xn, Xc])

        # 6. Fit preprocessor on train, transform train/test
        pp = NumPyPreprocessor(imputer_strategy='mean', scaler='standard').fit(X_num_train, X_cat_train)
        X_train_prepared = pp.transform(X_num_train, X_cat_train)
        X_test_prepared = pp.transform(X_num_test, X_cat_test)

        print('Prepared train shape:', X_train_prepared.shape)
        print('Prepared test shape:', X_test_prepared.shape)

        # this example shows several best practices:
        # fit-on-train only: imputer, scaler, and category mappings are derived from training data
        # unknown category handling: map unseen categories to a reserved index
        # combining numeric and categorical: numeric features after scaling and dense one-hot appended
        # for high cardinality, prefer sparse encodings or dimensionality reduction
        # preserve parameters: pp stores the parameters needed for transforming future data and for deployment

        # saving preprocessing parameters
        # serialize the necessary arrays (imputer_stats_, scale_mean_, scale_scale_, cat_list) using np.savez or
        # standard file formats so the exact transformation can be reproduced in production

        np.savez('preprocessor.npz', imputer_stats=pp.imputer_stats_, scale_mean=getattr(pp, 'scale_mean_', None), \
                 scale_scale=getattr(pp, 'scale_scale_', None), cat_list=np.array(pp.cat_list_, dtype=object))

        # real-world tips
        # for tabular datasets, pandas make reading and initial cleaning (parsing dates, mixed types) easier;
        # convert to NumPy for heavy numeric transforms
        # log-transform skewed positive features (np.log1p) before scaling
        # for streaming or very large datasets, compute running statistics (online mean/variance) or fit on a representative subset
        # keep reproducibility: record RNG seeds when sampling, seeds for train/test splits, and transformation parameters

        # key takeaways
        # detect and treat missing values consistently - prefer computing imputation statistics on training data only
        # user np.nanmean, np.nanmedian, or interpolation (for time series) as appropriate
        # standardization, min-max scaling, and robust scaling each have trade-offs;
        # choose based on algorithms and data distribution
        # always apply scaling parameters computed on training data to validation/test sets
        # label encoding and one-hot encoding are straightforward with np.unique and np.eye;
        # for high-cardinality features consider frequency encoding, hashing, or sparse representation
        # build reproducible preprocessing objects (fit/transform) in NumPy and persist the transformation parameters for deployment
        # prefer vectorized, broadcasted operations for performance;
        # only use python loops when necessary (e.g.,building category maps for large string sets,
        # but even then try to vectorize using np.searchsorted or np.unique)
        # use float32 for large datasets if precision permits,
        # and pay attention to memory layout and contiguity when performance matters

        # good preprocessing saves time and improves model stability
        # the techniques in this chapter give you a solid, numpy-centric toolbox for turning real-world data reliable inputs for models and analyses
        # in the next chapter we'll explore exploratory analysis and visualization patterns that
        # help you spot data issues and validate your preprocessing choices



#
# CH 8: Exploratory Analysis and Visualization
#

if False:
    # Exploratory data analysis (EDA) is the conversation you have with your dataset before you write model
    # good EDA helps you spot distributed quirks, outliers, relationships among features and items that need cleaning or transformation
    # this chapter shows how to perform quick, reproducible EDA using numpy for the computations and matplotlib for visual checks
    # you'll get working, well-documented examples for summary statistics, distributions, outliers, pairwise relationships,
    # and visualizing covariance/correlation structure - plus practical advice on which checks to run first and why

    # 8.1 quick EDA with NumPy
    # start with a compact summary: sized, dtypes, counts of missing values, and basic statistics
    # those few numbers immediately tell you if something is wrong or surprising

    # below is a reproducible example that creates a small realistic dataset (numerical features, a categorical label, and sime missing values)
    # and runs a set of quick EDA checks
    # copy-paste int0 a notebook and run each cell to explore

    import numpy as np
    rng = np.random.default_rng(seed=0)
    # synthetic dataset: 200 samples, 4 numeric features, 1 categical label
    n = 200
    features = np.column_stack([rng.normal(loc=50, scale=10, size=n),       # feature 1
                                rng.exponential(scale=5.0, size=n),         # feature 2
                                rng.normal(loc=0, scale=1, size=n),         # feature 3
                                rng.normal(loc=1000, scale=200, size=n)])   # feature 4
    labels = rng.choice(['a', 'b', 'c'], size=n, p=[0.5, 0.3, 0.2])
    # introduce some missing values (NaNs) and an outlier
    features[rng.choice(n, size=5), rng.integers(0, 4, size=5)] = np.nan
    features[10, 3] = 10_000.0      # large outliers in feature 3
    # quick structure checks
    print('shape:', features.shape)
    print('dtype:', features.shape)
    print('n missing per column:', np.isnan(features).sum(axis=0))
    print(features[0,:])
    print(np.isnan(features)[0,:])
    # summary statistics (nan-aware)
    col_mean = np.nanmean(features, axis=0)
    col_median = np.nanmedian(features, axis=0)
    col_std = np.nanstd(features, axis=0, ddof=0)
    col_min = np.nanmin(features, axis=0)
    col_max = np.nanmax(features, axis=0)
    print('mean:', col_mean)
    print('median:', col_median)
    print('std:', col_std)
    print('min:', col_min)
    print('max:', col_max)

    # interpreting results
    # n missing per column tells you which columns need imputation or exclusion
    # large differences between mean and median indicate skew (example: exponential feature)
    # extremely large max (outlier) suggests you should inspect that row or apply a robust transform

    # outlier detection (IQR and z-sore)
    # two common, complementary approaches are IQR-based outlier detection and z-score screening
    # IQR-based (robust)
    q1 = np.nanpercentile(features, 25, axis=0)
    q3 = np.nanpercentile(features, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask_iqr = (features < lower) | (features > upper)
    print('IQR outliers per column:', np.nansum(outlier_mask_iqr, axis=0))
    # z-score based
    mean = np.nanmean(features, axis=0)
    std = np.nanstd(features, axis=0)
    zscore_mask = np.abs( (features - mean) / (std + 1e-12) ) > 3.0
    print('z-score outliers per column:', np.nansum(zscore_mask, axis=0))
    # show rows flagged by either method
    rows_flagged = np.where( np.any( outlier_mask_iqr | zscore_mask , axis=1 ) )[0]
    print('rows flagged as outliers (indices):', rows_flagged)
    # user IQR for skewed features and z-score for approximately normal features
    # both are heuristics; always inspect flagged rows before dropping them

    # distribution checks (skew, kurtosis approximations)
    # numpy doesn't have built-in skew/kurtosis;
    # you can compute sample skewness/kurtosis with formulars or use SciPy
    # Here's a quick skew estimator (Pearson moment):
    # quick sample skewness (Fisher-Pearson moment coefficient)
    def skewness(x):
        x = x[~np.isnan(x)]
        n = x.size
        m = x.mean()
        s2 = ((x-m)**2).mean()
        s = np.sqrt(s2)
        if s ==0 or n < 3:
            return np.nan
        return ((x-m)**3).mean() / (s**3)
    skews = np.array([skewness(features[:,j]) for j in range(features.shape[1])])
    print('skewness per column:', skews)
    # skew helps decide on log or box-cox transformations for features like the exponential one above

    # 8.2 integration with Matplotlib
    # numbers tell a story, but visualization reveals shape
    # Matplotlib is the standard plotting library; here we demonstrate concise, separate plots for common EDA checks -
    # histrogram, boxplots, scatter plots and PCA (Principle Component Analysis) projection
    # each plot is created in its own figure so you can view or save them individually
    import matplotlib.pyplot as plt
    # 1) histogram for a single feature (feature 1 is skewed)
    plt.figure()
    plt.hist(features[:,1][~np.isnan(features[:,1])], bins=40)
    plt.title('Histrogram - feature 1 (skewed)')
    plt.xlabel('value')
    plt.ylabel('count')
    plt.grid(ls=':')
    plt.show()
    # 2) Boxplot to inspect spread and outliers for all features
    plt.figure()
    plt.boxplot([features[:,j][~np.isnan(features[:,j])] for j in range(features.shape[1])], \
                vert=True, labels=[f'f{j}' for j in range(features.shape[1])])
    plt.title('Boxplot - all features')
    plt.ylabel('value')
    plt.grid(ls=':')
    plt.show()
    # 3) scatter plot between two features to check relationships
    plt.figure()
    x = features[:,0]
    y = features[:,3]
    mask = ~np.isnan(x) & ~np.isnan(y)
    plt.scatter(x[mask], y[mask], alpha=0.6, s=100)
    plt.title('scatter - feature 0 vs feature 3')
    plt.xlabel('feature 0')
    plt.ylabel('feature 3')
    plt.grid(ls=':')
    plt.show()
    # notes for readable visuals: include axis labels, a title, and ensure missing values are filtered before plotting (as shown)
    # use alpha to reduce overplotting ans s to control marker size

    # heatmap of correlation matrix
    # a correlation heatmap quickly reveals which features move together
    # compute correlation matrix (rowvar=False: each column is a variable)
    X = features.copy()
    # replace NaNs with column mean for visualization (do not use as imputation for modeling automatically)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = col_mean[inds[1]]
    corr = np.corrcoef(X, rowvar=False)     # (4,4) matrix
    plt.figure()
    plt.imshow(corr, vmin=-1, vmax=1)
    plt.title('correlation matrix (heatmap)')
    plt.colorbar()
    plt.xticks(np.arange(features.shape[1]), [f'f{j}' for j in range(features.shape[1])])
    plt.yticks(np.arange(features.shape[1]), [f'f{j}' for j in range(features.shape[1])])
    plt.show() 
    # imputation: near +-1 indicates strong linear association:
    # values near 0 mean weak linear association (nonlinear relationships may still exist)

    # pairwise scatter-grid (selective)
    # a full scatter matrix is many plots; often you only need a few informative pairs
    # plot each pair in its own figure to keep things simple:
    pairs = [(0,1), (0,2), (1,3)]
    for i, j in pairs:
        plt.figure()
        xi = features[:,i]
        xj = features[:,j]
        mask =  ~np.isnan(xi) & ~np.isnan(xj)
        plt.scatter(xi[mask], xj[mask], s=12, alpha=0.5)
        plt.xlabel(f'f{i}')
        plt.ylabel(f'f{j}')
        plt.title(f'scatter: f{i} vs f{j}')
        plt.grid(ls=':')
        plt.show()

    # PCA projection for a quick multivariate view
    # project on the first two principal components to see clustering or label separations
    # PCA via SVD on centered data (nan-handling: impute with col mean for visualization only)
    Xc = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1, pc2 = U[:,0] * s[0], U[:,1] * s[1]     # equivalent to Xc @ Vt.T[:, :2]
    plt.figure()
    # color by label
    unique_labels = np.unique(labels)
    colors = {lab:i for i, lab in enumerate(unique_labels)}
    for lab in unique_labels:
        mask = labels == lab
        plt.scatter(pc1[mask], pc2[mask], label=lab, s=20, alpha=0.7)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA projection (PC1 vs PC2)')
    plt.legend()
    plt.show()
    # PCA projection often helps spot class separation, clusters, or outliers that were not obvious from univariate plots
        
    # 8.3 covariance and correlation calculations
    # covariance and correlation are the quantitative backbone of many EDA decisions
    # covariance shows joint variability on units of the original features;
    # correlation standardizes covariance to a dimensionless value in [-1,1][-1,1], making cross-feature comparisions straightforward

    # computing covariance and correlation with NumPy
    # use the mean-imputed X from previous section for covariance calculation
    cov = np.cov(X, rowvar=False)   # shape (n_features, n_features)
    corr = np.corrcoef(X, rowvar=False)
    print('Covariance matrix:\n', cov)
    print('Correlation matrix:\n', corr)
    # np.cov by default uses the unbiased estimator (division by N-1)
    # if you need the population version (divided by N), pass bias=True is older NumPy versions or adjust manually

    # manual Pearson correlation (for understanding)
    # Pearson correlation between two columns x and y can be computed directly from covariance and standard deviation:
    def pearson_corr(x, y):
        mask = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[mask], y[mask]
        xm, ym = x.mean(), y.mean()
        cov = ((x - xm) * (y - ym)).mean()  # population covariance
        return cov / (x.std(ddof=0) * y.std(ddof=0))
    # example
    print('manual pearson f0 vs f3:', pearson_corr(X[:,0], X[:,3]))
    print('matrix corr f0 vs f3:', corr[0, 3])
    # manual computation emphasizes how correlation standardizes covariance by scale

    # partial correlation (intuition)
    # partial correlation measures association between two variables while controlling for others
    # NumPy does not provide a direct function, but the concept is useful when multicollinearity is a concern
    # one practical approach is to regress each variable on the control variables and compute the correlation of the residuals
    # for complex workflows, use statsmodels

    # visualizing covariance/correlation structure
    # beyond the heatmap shown earlier, plotting the correlation of each variable with the target (if you have a numeric target) helps rank features
    # suppose y is numeric target created earlier
    target_corrs = [pearson_corr(X[:,j], y) for j in range(X.shape[1])]
    for j, tx in enumerate(target_corrs):
        print(f'correlation(feature {j}, target) = {tx:.3f}')
    # sort features by absolute correlation for quick feature ranking

    # dealing with non-linear relationships
    # correlation measures linear association
    # for non-linear dependencies, scatter plots, mutual information, or rank-based correlations (Spearman) are more appropriate
    # you can compute Spearman rank correlation by ranking the arrays with np.argsort twice or use scupy.stats.spearmanr for convenience

    # practical EDA workflow (short checklist)
    # 1. inspect shape, dtypes, and missing-value counts
    # 2. look at distributions (histograms) and robust summaries (median, IQR)
    # 3. detect outliers (IQR and z-score), but always inspect flagged rows manually
    # 4. check pairwise relationships with scatter plots for suspected dependencies
    # 5. compute covariance/correlation matrices and visualize with a heatmap
    # 6. use PCA/SVD to get a multivariate overview and to see clusters or structure
    # 7. record findings and decide preprocessing steps (transformations, imputations, dropping columns)

    # personal insight
    # i keep a one-page EDA "intent memo" fir every dataset:
    # main skewed features, columns with many missing values, obvious outliers, and the top-3 features correlated with the target
    # it speeds up discussions with collaborators and prevents overfitting to transient exploratory observations

    # key takeaways
    # start EDA with concise numeric checks: shape, dtype, missing counts, mean/median/std - these reveal most immediate issues
    # use histograms and boxplots to understand distributions and outliers; scatter plots reveal pariwise structure
    # compute covariance and correlation matrices to quantify linear relationship; visualize them with a heatmap for quick interpretation
    # PCA (via SVD) is a powerful diagnostic for multivariate patterns - project onto the first two components to inspect clustering or separation
    # always filter out NaNs before plotting or compute visualizations on imputed copies only for exploration (keep imputation decisions explicit)
    # EDA is iterative: observations from plots should feed back into preprocessing decisions
    # (scaling, transforms, feature selection) and be documented

    # in the next part of the book we'll use these EDA insights to guide feature engineering, model buliding, and the creation of evaluation
    # pipelines - ensuring that model choices are grounded in the data's actual structure rather than assumptions



#
# CH 9: Vectorization and optimization
#

if False:
    # speed and clarity in numerical python come from learning one central habit:
    # think in whole-array operations, not in Python loops
    # vectorization - using Numpy's ufuncs, broadcasting, and array-oriented idioms - moves work into C/Fortran/BLAS where it executes far faster
    # but naive vectorization can also produce giant temporaries or excessive memory use
    # this chapter teaches you how to remove Python-level loops, measure where the time goes,
    # and write broadcasting patterns that are both fast and memory efficient
    # you'll get hands-on examples, profiling recipes, and practical strategies that work in real projects

    # eliminating python loops
    # why avoid python loops?
    # each iteration in python pays interpreter overhead; when you loop over millions of elements the interpreter becomes the bottleneck
    # NumPy moves that work into compiled code and makes arithmetic operations run orders of magnitude faster

    # a simple motivating example
    # suppose you want to compute the square of every elememt in a large array and sum them
    # tress approaches: python loop, list comprehension, and NumPy vectorized operations
    import numpy as np
    from time import perf_counter
    n = 5_000_000
    a = np.random.default_rng(seed=0).random(n)
    # 1) python loop
    t0 = perf_counter()
    s = 0.0
    for x in a:
        s += x * x
    t1 = perf_counter()
    print('python loop time', t1-t0)
    # 2) list comprehension + sum (faster than loop, but still python)
    t0 = perf_counter()
    s = sum([x*x for x in a])
    t1 = perf_counter()
    print('list comprehension time:', t1-t0)
    # 3) NumPy vectorized (fastest)
    t0 = perf_counter()
    s = np.sum(a*a)
    t1 = perf_counter()
    print('NumPy vectorized time:', t1-t0)
    # you'll typically see the vectorized version substantially faster
    # the exact factor depends on hardware and BLAS, but the pattern is general

    # replacing loops with vectorized idioms
    # common loop -> vector patterns
    # element-wise arithmatic: for i: b[i] = 2*a[i]  ->  b = 2*a
    # conditional assignment: for i: if a[i] > 0: b[i] = a[i] else: b[i] = 0  ->  b = np.where(a>0, a, 0)
    # aggregation: for i: total += arr[i]  ->  total = arr.sum()

    # be careful with np.vectorize
    # it wraps a python function to accept arrays, but it does not implement a true ufunc on C;
    # it's convenience only and usually slower than a real ufunc
    # prefer built-in ufuncs or np.einsum/BLAS for heavy work

    # example: pairwise Euclidean distances
    # python double loop (O(n^2d)) vs vectorized broadcasting (still O(n^2d) work, but in C):
    # two sets of points A: n x d, B: m x d -> produce n x m pairwise distances
    def pairwise_loops(A, B):
        n, d = A.shape
        m = B.shape[0]
        D = np.empty((n,m), dtype=A.dtype)
        for i in range(n):
            for j in range(m):
                D[i, j] = np.linalg.norm(A[i] - B[j])
        return D
    def pairwise_vectorized(A, B):
        # broadcasting: (n,l,d) - (l,m,d) -> (n,m,d)
        diff = A[:, None, :] - B[None, :, :]
        return np.sqrt( (diff**2).sum(axis=2))
    # the vectorized version is much faster for moderate n and m,
    # but note it creates an intermediate of shape (n,m,d)
    # for very large n and m this will blow memory

    # when loops are OK (and Numba)
    # there are cases where vectorization is awkward or causes big temperary arrays
    # two options
    # 1. write a tight loo but accelerate it with Numba (a JIT complier that produces machine code)
    # example:
    from numba import njit
    @njit
    def pairwise_loops_numba(A, B):
        n, d = A.shape
        m = B.shape[0]
        D = np.empty((n,m), A.dtype)
        for i in range(n):
            for j in range(m):
                s = 0.0
                for k in range(d):
                    tmp = A[i,k] - B[j,k]
                    s += tmp * tmp
                D[i,j] = np.sqrt(s)
        return D
    # Numba can make loop-based code as fast as vectorized BLAS-based code while keeping memory small
    # it's an excellent fallback for algorithms that don't map well to ufuncs
    # 2. write an explicit blocking algorithm (explained later) to limit peak memory while staying in NumPy
    # personal note:
    # in my work I often start with vectorized code for clarity and correctness,
    # then switch to blocked computation or Numba only when memory or speed profiling shows an issue

    # 9.2 profiling and benchmarking(%timeit, cProfile, perf_counter)
    # before optimizing, measure
    # blind micro-optimizations waste time
    # use timeit for small code snippets and cProfile for whole scripts

    # Quick timing: %timeit (Jupyter)
    # in a notebook, iPython's %timeit is the simplest way to get robust timings
    # % timeit np.sum(a * a)
    # % timeit runs the snippet multiple times, reporting mean and best time, and automatically selects
    # repetitions to reduce noise

    # script-friendly timing: time.perf_counter
    # for .py scripts, use perf_counter() to measure blocks manually:
    from time import perf_counter
    t0 = perf_counter()
    np.sum(a*a)
    t1 = perf_counter()
    print('elapsed time (s):', t1-t0)
    # wrap repeated runs to get reliable averages

    # deeper profiling: cProfile + pstats
    # cProfile profiles at the function-call level and is best for finding hot functions across a whole run
    import cProfile, pstats, io
    def run():
        # place the task you want to profile here
        np.sum(a*a)
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats(50)      # top 50 entries
    print(s.getvalue())
    # look for:
    # functions with large cumulative time (cumtime) -> targets for optimization
    # frequent Python-level calls (loops, Python functions) that could be removed into arrays or compiles

    # line-level profiling
    # for deeper line-by-line timing within a function, use line_profiler (3rd party PKG) or manual micro-timings
    # example usage (if installed):
    # using @profile decorator (requires running 'kernprof -l script.py')

    # memory profiling
    # time is only half the picture
    # for memory, inspect .nbytes or arrays:
    a.nbytes      # bytes consumed by array data
    # for runtime memory profiling use tracemalloc (standard lib) or memory_profiler (3rd party) to see allocations
    # example tracemalloc snippet:
    import tracemalloc
    tracemalloc.start()
    # code that allocates
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    for stat in top_stats[:10]:
        print(stat)
    tracemalloc.stop()
    # example flow: optimize pairwise distances
    # 1. start with a correctness-focused vectorized version
    # 2. measure with %timeit on representative data
    # 3. profile with cProfile to see hotspots
    # 4. if memory blowups occur, either block the computation or implement a numba-compiled loop

    # 9.3 efficient broadcasting strategies
    # broadcasting is powerful, but naive broadcasting can create large intermediate arrays (temporaries)
    # that blow memory and kill performance
    # learn patterns that use broadcasting safely and minimize temporaries

    # use np.newaxis purposefully
    # shape alignment often requires injecting axes with None/np.newaxis
    # that's fine - but understand what the resulting operation does
    A = np.random.rand(1000,3)      # (n, d)
    v = np.array([1.0, 2.0, 3.0])   # (d,)
    # add v to every row:
    B = A + v   # uses broadcasting (v os treated as shape (3,) -> (1, 3) -> (n, 3))
    # explicit:
    B = A + v[None, :]
    # this is memory efficient because v is not expanded into a full (n, 3) copy
    # the broadcast is virtual for simple ufuncs

    # avoid np.tile when unnecessary
    # np.tile(v, (n,1)) create a real large array
    # prefer broadcasting:
    # bad: creates big copy
    big = np.tile(v, (10000, 1))
    # good: use broadcasting, no copy
    res = A + v     # use broadcas
    print(big.nbytes, res.nbytes)

    # use out = to avoid temporaries
    # many ufuncs accept an out parameter to write results into a pre-allocated array, avoiding temporary allocation
    res = np.empty_like(A)
    print(A.nbytes, res.nbytes)
    np.add(A, v, out=res)   # write directly into res
    # or in-place (if acceptable)
    A += v
    print(A.shape, v.shape)
    # chaining operations like (A - mean) / std creates temporaries for (A - mean) then for division
    # reduce temporaries:
    # creates two temporaries
    mean, std = A.mean(), A.std()
    Z = ( A - mean ) / std
    # in-place approach
    A_minus_mean  = A - mean    # 1 temporary
    A_minus_mean /= std         # reuse same buffer for division
    # or:
    # allocate out once
    out = np.empty_like(A)
    np.subtract(A, mean, out=out)
    np.divide(out, std, out=out)

    # use np.einsum to reduce temporaries and express complex contractions
    # einsum can compute complex sums with fewer temporaries and often clearer intent:
    # compute X^T X (p x p) without building intermediate large arrays:
    # X: (n, p)
    X = np.zeros([1000, 100])
    XtX = np.einsum('ni,nj->ij', X, X)
    print(XtX.shape, XtX.nbytes)
    # this avoids forming (n, p) x (n, p) intermediate products and can be very memory-efficient

    # blocked computation to control memory
    # when an operation needs an n x m intermediate (e.g.,pairwise distances) and nxm is too large, process in blocks:
    def pairwise_dist_blocked(A, B, block=1000):
        n, d = A.shape
        m = B.shape[0]
        D = np.empty((n, m), dtype=A.dtype)
        for i in range(0, n, block):
            i_end = min(n, i + block)
            Ai = A[i:i_end]     # shape (bi, d)
            # compute distances from Ai to all B
            # using vectorized broadcasting but only (bi, m, d)
            diff = Ai[:, None, :] - B[None, :, :]
            D[i:i_end] = np.sqrt((diff**2).sum(axis=2))
            return D
    # block size controls peak memory
    # choosing it depends on available RAM and shape

    # compute pairwise distances with algebraic trick to reduce memory
    # squared Euclidean distance can be computed via dot products without create a (n, m, d) intermediate:
    # || a - b ||^2 = || a ||^2 + || b ||^2 - 2 a b
    # implement:
    def pairwise_sqdist(A, B):
        # A: (n, d), B: (m, d)
        Anorm = ( A**2 ).sum(axis=1)[:, None]   # shape (n, 1)
        Bnorm = ( B**2 ).sum(axis=1)[None, :]   # shape (1, m)
        cross = A @ B.T     # shape (n, m)
        return Anorm + Bnorm - 2*cross
    # this requires allocating (n, m) for cross plus (n, m) for result, but does avoid (n, m, d) temporary
    # if cross is still too large, compute cross in blocks

    # broadcast to avoid copies: np.broadcast_to
    # if you need a logically expanded view but don't want to allocate memory,
    # np.broadcast_to gives a readonly broadcasted view:
    v = np.array([1,2,3])
    v_big = np.broadcast_to(v, (10000, 3))  # no copy, read-only view
    # attempting to modify v_big will error; if you need writable, copy explicitly

    # beware of integer/float dtypes and promotions
    # some ufuncs promote dtypes causing extra work/copies
    # be explicit with dtypes and convert arrays beforehand if you know the precision that suffices:
    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)
    # float32 uses half the memory of float64 and often yields adequate performance for ML training

    # strides tricks - advanced and dangerous
    # np.lib.stride_tricks.as_stride can create clever views without copying,
    # but misuse produces arrays that point outside the underlying buffer and lead to crashes
    # use only when you deeply understand strides

    # putting it together - worked examples
    # goal: compute k-nearest neighbors (k-NN) distances for a dataset of size n with limited memory
    # strategy:
    # 1. precompute A_norms and store B (dataset) in a memmap if huge:
    # 2. process queries in blocks to compute cross = A_block @ B.T
    # 3. compute squared distances via Anorm + Bnorm - 2*cross
    # 4. np.argpartition to get k smallest distances per query without full sort
    # sketch:
    def knn_blocked(X, queries, k=5, block=500):
        # X: (N, d), queries: (Q, d)
        Xnorm = (X**2).sum(axis=1)
        Qnorm = (queries**2).sum(axis=1)
        N = X.shape[0]
        Q = queries.shape[0]
        knn_idx = np.empty((Q, k), dtype=int)
        knn_dist = np.empty((Q, k), dtype=float)
        for i in range(0, Q, block):
            i_end = min(Q, i+block)
            q = queries[i:i_end]    # (b, d)
            cross = q @ X.T         # (b, N)
            dists = Qnorm[i:i_end, None] + Xnorm[None,:] - 2*cross  # (b, N)
            # find k smallest per row
            idx_part = np.argpartition(dists, kth=k-1, axis=1)[:, :k]
            # for exact order, sort those k values
            rows = np.arange(dists.shape[0])[:, None]
            k_idx_sorted = idx_part[np.argsort(dists[rows, idx_part], axis=1), :]
            knn_idx[i:i_end] = k_idx_sorted
            knn_dist[i:i_end] = dists[rows, k_idx_sorted]
        return knn_idx, knn_dist
    # this approach avoids building huge (Q, N, d) intermediates
    # and uses blocking and argpartition to keep memory and work manageable

    # key takeaways
    # think in arrays, not loops
    # replace python iteration with ufuncs and reductions wherever possible
    # this moves work into optimized C/BLAS
    # measure first
    # use %timeit in notebooks for microbenchmarks and
    # cProfile (with pstats) for end-to-end profiling;
    # inspect memory with .nbytes and tracemalloc
    # avoid naive broadcasting that create huge temporaries
    # use out= parameters, np.einsum, and blocked algorithms to reduce peak memory
    # use np.newaxis and np.broadcast_to to express shape changes without copies
    # avoid np.tile unless you really new a writable copy
    # numba is a pragmatic fallback
    # if vectorization forces impossible memory usage,
    # a numba-compiled loop can be both memory- and time-efficient
    # use algebric identities
    # (e.g., norms and inner products) to reduce temporary shapes (pairwise distance trick)
    # be mindful of dtype and contiguity
    # float32 halves memory;
    # contiguous arrays and an optimized BLAS backend improve compute throughput
    # block large computations
    # processing data in blocks is often the simplest, most robust strategy
    # to make large O(n2) tasks tractable in limited RAM

    # by mastering these vectorization and optimization patterns
    # you'll write code that's not only fast but also maintainable and predictable
    # in the next chapter we'll dig into memory layout and advanced indexing,
    # giving you the tools needed to squeeze every bit of performance out of NumPy arrays



#
# CH 10: memory layout and advanced indexing
#

if False:
    # when you reach for peak performance in NumPy,
    # the way your arrays are laid out in memory and
    # indexing patterns you uses often matter more than micro-optimizing arithmetic
    # this chapter explains why stride and contiguity affect speed and safety,
    # demonstrates advanced indexing patterns that let you express complex element selection concisely,
    # and shows memory-efficient slicing strategies for very large arrays
    # you'll get practrical code you can copy into a notebook,
    # clear rules-of-thumb, and a few guarded tricks for advanced use

    # 10.1 strides and contiguous arrays
    # NumPy stores the array data as a single block of memory and uses shape + strides to map
    # n-dimensional indices to byte offsets inside that block
    # arr.strides gives the number of bytes to step to move one element along each axis
    # understanding strides explains why arr.T can be instantaneous (it simply changes strides)
    # and why some views are non-contiguous

    # create a small examples to inspect shape, dtype, strides, and the formula that computes a memory offset:
    import numpy as np
    a = np.arange(12, dtype=np.int64)   # 1-D array: 12 elements
    print('a:', a)
    print('a.shape:', a.shape)
    print('a.strides:', a.strides)      # (8,) because int64 uses 8 bytes
    m = a.reshape(3, 4)     # shape (3, 4)
    print('\nm\n:', m)
    print('m.shape:', m.shape)
    print('m.strides:', m.strides)  # (32, 8): to move one row (axis 0) move 32 bytes; one column 8 bytes
    # a general formula to compute the byte offset of element with index (i, j, k, ...) is:
    # offset = i * strides[0] + j * strides[1] + k * strides[2] + ...
    # so m[1,2] maps to base + 1*32 + 2*8 = base + 48 bytes
    # equivalent to the 6th element in row-major order
    # C-contiguous vs F-contiguous (row-major vs column-major)
    # matters for algorithms that iterate along a particular axis
    # NumPy defaults to C order: rows are contiguous in memory
    # use .flags to inspect contiguity:
    print('m.flags:\n', m.flags)    # include 'C_CONTIGUOUS' and 'F_CONTIGUOUS' flags
    # m.T (transpose) usually returns a view with changed strides but not a copy:
    mt = m.T
    print('mt.strides:', mt.strides)    # often (8, 32) - reversed
    print('mt.flags:\n', mt.flags)
    # because mt is non-contiguous in C order,
    # passing it to some C/Fortran libraries or BLAS routine may cause NumPy to make an internal contiguous copy,
    # which can hurt performance
    # when calling external code, perfer to supply a contiguous buffer:
    mt_c = np.ascontiguousarray(mt)     # explicit C-contiguous copy if needed
    # ascontiguousarray returns the original array
    # if it's already contiguous, otherwise it makes a copy

    # practical examples and consequences
    # 1. if you iterate over rows of a C-contiguous (n, p) array,
    # that traversal will be cache-friendly and fast
    # iterating over columns repeatedly is slower because elements are not contiguous in memory
    # 2. matrix-multiply routine (BLAS) are fastest when inputs are in the expected memory order (often C-contiguous for NumPy)
    # if you observe poor performance in A @ B, check .flags['C-CONTIGUOUS']
    # and make ascontiguousarray calls where needed
    # 3. slicing with a step (e.g.,a[::2]) produces a view with a stride larger than element size
    # that's fine and cheap, but memory access will jump, which can be slower in tight loops

    # negative strides and reversed arrays
    # slicing with a negative step produces a view with a negative stride:
    r = a[::-1]
    print(a)
    print(r)
    print('r.strides:', r.strides)      # negative strides
    # this is a view and no copy is made,
    # but some libraries don't expect negative strides and may copy

    # computing offsets manually - sanity check
    # if you ever suspect mis-indexing, you can compute the flat index and then bytes offset:
    i, j = 1, 2
    flat_index = i * (m.shape[1]) + j   # only valid for C-contiguous
    offset = i * m.strides[0] + j * m.strides[1]
    print(i, j)
    print(m.shape)
    print(m.strides)
    print(flat_index, offset)
    # but prefer to reason with shape/strides rather than manual offset arithmetic in production code

    # best practice synopsis for contiguity
    # if your code performs heavy numeric operations or passes arrays to external libraries:
    # check .flags['C_CONTIGUOUS'] and .flags['F_CONTIGUOUS'] when debugging performance
    # make an explicit copy with np.ascontiguousarray(arr) or
    # np.asfortranarray(arr) once and reuse it rather than letting internal functions copy repeatedly
    # convert dtypes explicitly (astype(np.float32, copy=False)) to avoid unexpected copies

    # 10.2 advanced indexing tricks
    # indexing in NumPy is rich
    # there are two fundamentally different indexing styles:
    # basic slicing (slices, integers, :) which returns view when possible,
    # and advanced indexing (integer arrays, boolean masks) which generally returns copies
    # knowing whch returns a view vs a copy is crucial for correctness and memory use

    # slicing (views)
    M = np.arange(12).reshape(3,4)
    s = M[0:2, 1:3]         # view - shares memory with M
    s[0, 0] = 999           # modifies M

    # fancy indexing (copies)
    idx = np.array([0, 2])
    rows = M[idx]           # not a view - this is copy
    rows[0,0]=-1            # does not change M

    # because fancy indexing returns a copy, modifications to the result don't affect the source
    # if you want to assign into the original array using fancy indicies, assign into the original directly:
    M[idx, 1] = 100         # write into M at rows 0 and 2, column 1

    # per-row selection (very common pattern)
    # selecting one element per row using integer indices is efficient and idiomatic
    arr = np.array([[10, 11, 12],
                    [20, 21, 22],
                    [30, 31, 32]])
    col_indices = np.array([2, 0, 1])   # one column index per row
    selected = arr[np.arange(arr.shape[0]), col_indices]    # shape (3, )
    print(arr)
    print(arr.shape)
    print(selected)
    # this pattern is useful for selecting the predicted class per row, picking maxima, etc

    # np.takg_along_axis and np.put_along_axis
    # when you have per-row indicies and want to get or set values along a particular axis,
    # take_along_axis and put_along_axis are modern, efficient helpers:
    # get top-2 values per row
    scores = np.random.rand(5, 10)
    topk_idx = np.argpartition(scores, -2, axis=1)[:,-2:]
    top2 = np.take_along_axis(scores, topk_idx, axis=1)     # shape (5, 2)
    # set per-row chosen
    np.put_along_axis(scores, topk_idx, 0.0, axis=1)
    # these functions preserve the alignment of shapes and are preferable to clumsy broadcasting hacks

    # np.ix_ for outer-product indexing
    # if you want the cross-product of selected rows and columns
    rows = np.array([0, 2])
    cols = np.array([1, 3])
    sub = M[np.ix_(rows, cols)]     # shape (2, 2) picks rows x cols in outer-product fashion
    # np.ix_ makes intent explicit: choose these rows and these columns

    # boolean masks (filtering and in-place modification)
    # boolean masks are common and usually memory efficient:
    mask = M % 2 == 0   # boolean array same shape as M
    even =  M[mask]     # retuns 1-D array (copy)
    M[mask] = -1        # in-place modify only selected entries
    # nasty pitfall: mask.nonzero() returns tuple if arrays; usin them in fancy indexing returns a copy:
    ri, ci = np.nonzero(mask)
    # M[ri, ci] is a copy - modifying it does not affect M unless you assign back
    M[ri, ci] = -2  # this *does* write back because assignment syntax targets M

    # selecting top-k per row without creating full sorts
    # argpartition is a neat performance trick:
    # get k smaller or largest indicies without a full sort
    k = 3
    vals = np.random.rand(1000, 50)
    k_small_idx = np.argpartition(vals, kth=k-1, axis=1)[:, :k]
    k_small_vals = np.take_along_axis(vals, k_small_idx, axis=1)
    print(k_small_vals.shape)
    # you can then sort the small block if you need them ordered

    # np.where with axis-aware broadcasting
    # np.where(cond, x, y) is powerful for elementwise selection
    # combined with broadcasting,
    # you can replace elements conditionally without temporaries:
    A = np.arange(12).reshape(3, 4)
    result = np.where(A % 2 ==0, A, -A)     # elementwise choose positive for evens, negative for odds

    # np.ndindex and np.ndenumerate - controlled, readable iteration
    # when you must iterate but want readable index loops, np.ndindex yields multi-dimensional indices:
    for idx in np.ndindex(A.shape):
        print(idx, A[idx])
    for idx, val in np.ndenumerate(A):
        print(idx, val)
    # np.ndenumerate yields (index, value) pairs

    # as_strided - powerful and dangerous
    # np.lib.stride_tricks.as_strided can create views with arbitrary shape/strides
    # this can implement rolling windows without copies,
    # but if misused you can create views that point outside the underlying array (undefined behavier)
    # Perfer sliding_window_view if availiable
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.arange(10)
    windows = sliding_window_view(x, window_shape=3)    # shape (8,3) view
    # window[i] is x[i:i+3] and is a view (no copy)
    print(windows)
    print(x[windows[0]].mean())
    # sliding_window_view is safe and readable
    # only use as_strided if you are certain about the layout and bounds

    # 10.3 memory-efficient slicing for large data
    # when data size approaches or exceeed RAM, two strategies dominate:
    # work in views and blocks, and avoid unnecessary copies
    # combine memmaped arrays and careful block-processing or streaming
    # here are robust patterns

    # chunked processing pattern (row by blocks)
    # this pattern applies a function f to each block of rows and
    # aggregate results without loading the whole matrix:
    def process_in_blocks(X, block_rows, func, *args, **kwargs):
        n = X.shape[0]
        results = []
        for i in range(0, n, block_rows):
            block = X[i:i+block_rows]       # view (cheap)
            res = func(block, *args, **kwargs)
            resuts.append(res)
        return np.concatenate(results, axis=0)  # depending on func, adjust aggregation
    # if X is a memmap or HDF5 dataset,
    # block = X[i:i+block_rows] reads only that chunk into memory

    # compute column means by chunking
    # a common building block:
    # compute column means without loading full array:
    def col_mean_chunked(filename, shape, dtype=np.float32, chunk=1000):
        mm = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
        total = np.zeros(shape[1], dtype=np.float64)
        count = 0
        for i in range(0, shape[0], chunk):
            block = mm[i:i+chunk]
            total += np.nansum(block, axis=0)   # handle NaN if necessary
            count += (~np.isnan(block)).sum(axis=0)
        return total / count
    # this uses memmap to avoid loading everything

    # choosing block size
    # block size should fit comfortably in RAM and exploit CPU cache
    # test with a few sizes and profile memory and time;
    # often multiples of a few megabytes work well

    # avoiding copies with selection patterns
    # if you need to compute a function that reads rows but must access columns inside the loop,
    # prefer row-major iteration for C-contiguous arrays
    # because per-row slices are contiguous views
    # if you must repeatedly access columns,
    # it can be faster to transpose once and iterate over rows of the transposed array
    # (i.e.,columns of original), making those accesses contiguous
    # slow: loop over columns in a C-contiguous array
    for j in range(A.shape[1]):
        col = A[:,j]    # this is a view but non-contiguous step-by-step access can be slow
    # fast: transpose to make columns contiguous
    A_T = np.ascontiguousarray(A.T)
    for j in range(A_T.shape[0]):   # now rows of A_T are contiguous
        row = A_T[j]

    # using memmap + sliding windows (time series)
    # for long time series stored on disk,
    # sliding windows for feature extraction can be done with sliding_window_view
    # applied on small blocks:
    # mm = np.memmap('long_series.dat', dtype=np.float32, mode='r', shape=(10_000_000,))
    # window = 128
    # step = 64
    # features= []
    # for i in range(0, mm.shape[0] - window + 1, step):
    #     w = mm[i:i+window]  # small view
    #     feat = extract_features(w)  # small computation in RAM
    #     features.append(feat)
    # features = np.vstack(features)
    # this avoids creating an enormous (n_windows, window) intermediate

    # be caerful about views that look huge
    # operation like X[:,None.:] - Y[None,:,:] create a view shapeed(n, m, d)
    # that is virtual but will be materialized when used in arithmetic, causing memory blow-up
    # use algebric tricks (norms + dot products) or
    # blocked computation ot avoid this

    # when you must copy - do it once
    # if an operation requires a contiguous copy for performance,
    # make an explicit copy once and reuse it
    # repeated implicit copies inside tight loops are the silent performance killer

    # practical debugging checklist for memory & indexing bugs
    # 1. if performance is poor, put .strides, .flags, and .nbytes to inspect layout and memory
    # 2. use np.share_memory(a, b) to check whethera view actually shares memory
    # 3. when you expect a view but observe unexpected behavior,
    # check the .base attribute (non-None if this array is a view)
    # 4. if indexing returns unexpected copies,
    # remember that advanced indexing (integer arrays, boolean masks) returns copies
    # use np.where assignments or put_along_axis to write back into the original array

    # key takwaways
    # understanding memory layout and indexing choices is the fastest path from 'it works' to 'it runs well'
    # ndarray uses shape + strides to map indices to offsets;
    # transpose and reshaps are often views that change strides not data
    # advanced indexing (integer arrays, boolean masks) typically returns copies -
    # know when to expect a view vs a copy
    # for very large arrays, prcess in blocks, prefer memmap or chunked I/O,
    # and broadcasting patterns that temporarily materialize enormous arrays
    # use np.ascontiguousarray where external libraries expect C order,
    # and favor sliding_window_view over manual as_strided for rolling-window operations
    # finally, when performance surprises you,
    # inspect .strides, .flags, .nbytes, and use micro-benchmarks to guide optimizations rather than guessing

    # when these techniques you'll avoid many subtle bugs, reduce memory pressure,
    # and write NumPy code that hehaves predictably at scale
    # in the next chapter we'll look at ways to extend NumPy with compiled code and accelerators
    # - numba, cython, and quick tour of GPU options



#
# CH 11: extending numpy
#

if False:
    # numpy is extremely powerful, but sometimes you need more:
    # very tight loops that NumPy can't express efficiently,
    # GPU acceleration, or interoperability with legacy C/Fortran code and libraries
    # this chapter covers pragmatic, production-ready paths to extend NumPy:

    # 1. JIT and compiled Python (Numba, Cython) for high performance loops
    # 2. GPU acceleration with CuPy (overview and practical notes)
    # 3. interfacing with C and Fortran (ctypes, f2py and best practices)

    # for each tool you'll see clear, working examples, reasons to pick that tool,
    # and practical caveats that matter when you move from prototype to production

    # 11.1 using Numba and Cython for speed
    # when vectorization becomes awkward or creates huge temporaries,
    # you can either restructure your algirithm or compile the hot loops
    # numba and cython are the two practical choices for most users

    # Numba - JIT (Just-In Time) compilation with minimal friction
    # Numba compiles annotated python functions to machine code using LLVM (Low Level Virtual Machine)
    # typical pattern: import numba and decorate a function with @njit (a shorthand for @numba.njit),
    # optionally parallel=True for multi-threading

    # benefits:
    # very fast for numeric loops (close to C speeds)
    # minimal code changes for many algorithms
    # support @njit(parallel=True) with numba.prange for parallel loops

    # limitations:
    # supports a subset of python and numpy features
    # first call has compilation overhead
    # debugging compiled code can be harder

    # example:
    # pairwise squared Euclidean distances with Numba vs python loop
    import numpy as np
    from time import perf_counter
    from numba import njit, prange
    rng = np.random.default_rng(seed=1)
    A = rng.random((1000, 64))
    B = rng.random((1000, 64))
    # pure python double loop (for comparison)
    def pairwise_py(A, B):
        n = A.shape[0]
        m = B.shape[0]
        D = np.empty((n, m), dtype=A.dtype)
        for i in range(n):
            for j in range(m):
                s = 0.0
                for k in range(A.shape[1]):
                    tmp = A[i, k] - B[j, k]
                    s += tmp * tmp
                D[i, j] = s
        return D
    # numba-compiled version (single-thread)
    @njit
    def pairwise_numba(A, B):
        n, m, d = A.shape[0], B.shape[0], A.shape[1]
        D = np.empty((n, m), dtype=A.dtype)
        for i in range(n):
            for j in range(m):
                s = 0.0
                for k in range(d):
                    tmp = A[i, k] - B[j, k]
                    s += tmp * tmp
                D[i, j] = s
        return D
    # numba parallel version
    @njit(parallel=True, fastmath=True)
    def pairwise_numba_parallel(A, B):
        n, m, d = A.shape[0], B.shape[0], A.shape[1]
        D = np.empty((n, m), dtype=A.dtype)
        for i in range(n):
            for j in range(m):
                s = 0.0
                for k in range(d):
                    tmp = A[i, k] - B[j, k]
                    s += tmp * tmp
                D[i, j] = s
        return D
    # timing: python
    t0 = perf_counter()
    D_py = pairwise_py(A, B)
    t1 = perf_counter()
    print('pairwise_py(A, B) > time = %.6f sec' % (t1 - t0))
    # timing: compilation happens on first call
    t0 = perf_counter()
    D_nb = pairwise_numba(A, B)
    t1 = perf_counter()
    print('pairwise_numba(A, B) > 1st time = %.6f sec' % (t1 - t0))
    # timing: cached compiled code
    t0 = perf_counter()
    D_nb = pairwise_numba(A, B)
    t1 = perf_counter()
    print('pairwise_numba(A, B) > 2nd time = %.6f sec' % (t1 - t0))
    t0 = perf_counter()
    D_nb = pairwise_numba(A, B)
    t1 = perf_counter()
    print('pairwise_numba(A, B) > 3rd time = %.6f sec' % (t1 - t0))
    # timing: compilation happens on first call (may be faster on multi-core)
    t0 = perf_counter()
    D_nb_p = pairwise_numba_parallel(A, B)
    t1 = perf_counter()
    print('pairwise_numba_parallel(A, B) > 1st time = %.6f sec' % (t1 - t0))
    # timing: cached compiled code
    t0 = perf_counter()
    D_nb_p = pairwise_numba_parallel(A, B)
    t1 = perf_counter()
    print('pairwise_numba_parallel(A, B) > 2nd time = %.6f sec' % (t1 - t0))
    t0 = perf_counter()
    D_nb_p = pairwise_numba_parallel(A, B)
    t1 = perf_counter()
    print('pairwise_numba_parallel(A, B) > 3rd time = %.6f sec' % (t1 - t0))
    # practical notes:
    # use fastmath=True only when small floating-point reorderings are acceptable
    # for parallel=True use prange instead of range
    # keep input arrays contiguous (np.ascontiguousarray) and typed (float32/64) for best performance
    # numba supports a CUDA target for writing GPU kernels (numba.cuda),
    # but that requires a CUDA toolchain and is a different API

    # Cython - static typing for compiled extensions
    # cython compiles .pyx files to C extension
    # it gives fine control and works well when you want to write code in a Python-like syntax
    # but with static type annotations

    # benefits:
    # very efficient when you add types
    # best interop with C libraries and OpenMP
    # excellent control over the C ABI(Application Binary Interface) and memory layout

    # limitations:
    # requires a build step (compilation), packaging considerations
    # more boilerplate (setup files), but modern pyproject workflows help

    # examples:
    # cython module to compute a row-wise squared norms
    # save this as fastops.pyx:
    # fastops.pyx
    #cimport cython
    #import numpy as np
    #cimport numpy as np
    #from cython.parallel import prange
    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    #def row_sqnorms(np.narray[np.float64_t, ndim=2] A):
    #    cdef int n = A.shape[0]
    #    cdef int d = A.shape[1]
    #    cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(n, dtype=np.float64)
    #    cdef int i, k
    #    for i in prange(n, nogil=True):
    #        cdef double s = 0.0
    #        for k in range(d):
    #            s += A[i,k] * A[i,k]
    #        out[i] = s
    #    return out
    # a minimal setup.py to build (traditional approach):
    # setup.py
    #from setuptools import setup
    #from cython.build import cythonize
    #import numpy as np
    #setup(
    #   ext_modules = cythonize('fastops.pyx', annotate=True),
    #   include_dirs=[np.get_include()],
    #)
    # build in shell:
    #python setup.py build_ext --inplace
    # then in python:
    #import numpy as np
    #import fastops
    #A = np.random.rand(5000,128).astype(np.float64)
    #out = fastops.row_sqnorms(A)

    # tips
    # use typed memoryviews(double[:,:]) for modern Cython style
    # turn off bounds checking and wraparound for speed
    # use nogil=True with prange to parallelize loops;
    # you'll need an OpenMP-capable compiler and to pass -
    # fopenmp and link flags (Cython docs provide details)
    # Cython is ideal when you need C-level control,
    # call external C APIs, or packages a compiled wheel for distribution

    # when to pick Numba vs Cython
    # choose Numba if you want minimal code changes, fast iteration, and JIT compilation
    # choose Cython when you need maximum control, interoperability with C libraries,
    # or when building a distributable compiled extension with fine-grained optimizations

    # 11.2 GPU acceleration with CuPy overview
    # GPUs can massively accelerate array computations,
    # especially dense linear algebra and element-wise operations with high arithmetic intensity
    # CuPy is the most practical path for many NumPy users:
    # it provides a NumPy-compatible API that runs on NVIDIA GPUs

    # CuPy's model:
    # replace import numpy as np with import cupy as cp
    # allocate device arrays with cp.array, cp.zeros, or cp.asarray
    # use the same syntax (@, ufuncs, reductions)
    # transfer data between host and device with cp.asnumpy() and cp.asarray()

    # example:
    # simple workflow using CuPy
    # requires a CUDA-enabled GPU and CuPy installed (matching your CUDA)
    #import cupy as cp
    #import numpy as np
    # Host -> device
    #x_cpu = np.random.rand(1_000_000).astype(np.float32)
    #x_gpu = cp.asarray(x_cpu)  # copy to device
    # perform heavy computation on GPU
    #y_gpu = cp.sin(x_gpu) * cp.exp(x_gpu)  # fully on GPU
    # synchronize and copy back
    #y_cpu = cp.asnumpy(y_gpu)  # copy results to host

    # practical considerations:
    # transfer costs matter
    # copying data to/from GPU is relatively expensive
    # for small array, CPU may be faster
    # put as much work as possible on the device before bringing data back
    # installation
    # CuPy provides binary wheels like cupy-cudallx matching different CUDA versions
    # (e.g.,cupy-cuda11x), or pip install cupy for a fallback that builds from source (slow)
    # conda packages can make installation easier for complex CUDA setups
    # you must have a compatible NVIDIA driver + CUDA toolkit or runtime installed
    # memory
    # device memory is limited
    # watch cp.cuda.runtime.memGetInfo() and
    # use cp.cuda.memory pools for managing allocations
    # streams and async
    # GPU kernels are asynchronous
    # use cp.cuda.Stream for overlapping compute and transfer;
    # remember to synchronize(stream.synchronize() or cp.cuda.Stream.null.synchronize())
    # when you need deterministic host-side timing
    # 3rd-party ecosystem
    # many linbraries(CuPy, PAPIDS, cuML, cuDF, PyTorch, TensorFlow) offer GPU-enabled tooling
    # CuPy integrates well with CUDA libraries for linear algebra and FFTs
    # compatibility caveat
    # not all numpy functions are implmented in CuPy; but many common ones are
    # for missing functionality, you can write custom CUDA kernels with CuPy's RawKernel or use Numba's CUDA
    # minimal matrix multiply example:
    # large matrix multiply on GPU
    #a = cp.random.rand(5000, 5000, dtype=cp.float32)
    #b = cp.random.rand(5000, 5000, dtype=cp.float32)
    # time GPU operation (remember to synchronize)
    #import time
    #t0 = time.time()
    #c = a @ b
    #cp.cuda.Stream.null,synchronize()  # wait for completion
    #t1 = time.time()
    #print('GPU matmaul time:', t1-t0)

    # guidance
    # use GPU when you have large, compute-bound arrays and can amortize transfer overhead
    # profile both data transfer and kernel time;
    # CuPy includes cupy.cuda.profiler hooks and integrates with nvprof/N sight
    # for production systems, consider mixed CPU/GPU pipelines,
    # memory pooling, and error handling (out-of-memory conditions)

    # 11.3 interfacing with C and Fortran libraries
    # many high-performance numerical routines exist in C or Fortran
    # calling them directly avoids reimplementing optimized algorithms and leverages battle-tested code

    # passing numpy arrays to C with ctypes (simple, no build system)
    # write a tiny C function that modifies a double array in place:
    #/* double_inplace.c */
    ##include <stddef.h>
    #void double_inplace(double *arr, int n)
    #{
    #   for (int i = 0; i < n; i++) {
    #       arr[i] *= 2.0l
    #   }
    #}
    # compile it to a shared library:
    #gcc -O3 -fPIC -shared double_inplace.c -o libdouble.so
    # call it from python using ctypes
    # import ctypes
    # import numpy as np
    # lib = ctypes.CDLL('./libdouble.so')
    # lib.double_inplace.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.c_int)
    # lib.double_inplace.restype = None
    # a = np.arange(10, dtype=np.float64)
    # ensure contiguous
    # a_ct = np.ascontiguousarray(a, dtype=np.float64)
    # ptr = a_ct.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # lib.double_inplace(ptr, a_ct.size)
    # print(a_ct)

    # notes:
    # always ensure the array has the expected dtype and memory layout
    # use np.ascontiguousarray or np.asfortranarray as required
    # numpy.ctypeslib provides helpers for convenience
    # for multi-dimensional array,
    # pass a pointer and dimensions, and compute offsets in C using strides (or flatten before calling)

    # Fortran vi f2py - easy way to wrap fortran code
    # fortran remains popular for numerical code
    # f2py complies fortran subroutines and generate python module
    # fortran subroutine example scale_array.f90:
    #! scale_array.f90
    #subroutine scale_array(n, a)
    #  implicit none
    #  integer, intent(in) :: n
    #  real(8), intent(inout) :: a(n)
    #  integer :: i
    #  do i = 1, n
    #    a(i) = a(i) * 2.0d0
    #  end do
    #end subroutine scale_array
    # compile with f2py
    #f2py -c -m f2mod scale_array.f90
    # then in python:
    #import numpy as np 
    #import f2mod   # generated module
    #a = np.arange(10, dtype=np.float64)
    #f2mod.scale_array(a)   # modifies array in place
    # f2py handles many interop details (fortran ordering, dtype matching) for you
    # it's an excellent choice when you need to reuse fortran libraries

    # Pybind11 and modern C++ bindings
    # if your codebase is in C++, pybind11 is a modern, ergonomic way to expose functions and classes as Python modules
    # it supports zero-copy conversion between NumPy array and Eigen/raw buffers
    # example is beyond this chapter's length,
    # but note: pybind11 produce clean, modern bindings and integrates with CMake and setuptools

    # best practices when calling native code
    # contiguity & dtype
    # always ensure arrays are contiguous and of the exact dtype expected by native code
    # (use np.ascontiguousarray and astype(copy=False))
    # ownership & lifetime
    # be careful with temporary arrays;
    # avoid passing pointers to ephemeral arrays that will be freed bt python
    # use .ctypes or create a persistent array in python or C
    # threading
    # if the C/Fortran code uses threads (OpenMP, BLAS), coordinate with python-level threads/processes
    # avoid nesting multi-threading layers that oversubscribe cores
    # use environment variables (OMP_NUM_THREADS) or library-specific controls to limit thread count
    # error handling
    # native code must not crash python (segfault)
    # add bounds checks and validate inputs before passing them
    # consider writing thin wrapper layers that validate arguments
    # build & packaging
    # for portability, build wheels on target platforms or use conda packages for compiled extensions
    # requiring users to compile native code locally creates friction

    # choosing the right extension path
    # if the bottleneck is a tight numerical loop you can't express in NumPy:
    # try Numba first (fast turnaround),
    # then Cython if you need distribution-grade extensions or deeper C interop
    # if you have massive parallelism needs that can run on the GPU and you can tolerate extra complexity:
    # try CuPy (or framework like PyTorch) and miminize host/device transfers
    # if you need to call existing C/Fortran libraries (BLAS, LAPACK, custom Fortran routines):
    # use ctypes/f2py/pybind11 or Cython bindings
    # for packaging:
    # compiled Cython/pybind11 extensions can be distributed as wheels;
    # rely on CI and manylinux wheels for portability

    # key takeaways
    # numba
    # fastest way to accelerate tight numeric loops with minimal code changes
    # great for prototype -> production transitions when you want JIT speed without a compile step
    # Cython
    # provide static typing, tight control, and robust C interop
    # prefer when you need a compiled extension disbtributed as a wheel or
    # when you want to call C APIs directly
    # CuPy
    # a NumPy-like GPU array library - excellent for large, compute-bound tasks
    # if you can amortize host/device transfer costs and have a compatible CUDA environment
    # C/Fortran
    # ctypes and f2py are practical and efficient for calling native libraries;
    # ensure array contiguity, dtyoe correctness, and careful lifetime management
    # always profile first (CPU vs, GPU, memory vs compute)
    # use np.ascontiguousarray and check .dtype to prevent hidden copiesl
    # use blocking, memory pools, and device streams when working on GPU
    # for production, invest in packaging, and reproducible builds (compiled wheels, conda packages)
    # so users do not have to compile native code locally

    # extending NumPy unlocks real speed and scalability, but it also brings complexity:
    # toolchains, memory management, and platform variability
    # start with the simplest effective tool (often numba),
    # measure carefully, and escalate to custom C/Fortran or GPU implementations
    # when the performance gains justify the added build and maintenance costs



#
# CH 12: working with the wider ecosystem
#

if False:
    # NumPy is the foundation - but most real-world data work happens in an ecosystem of libraries built on top of or around NumPy
    # in this chapter we'll walk through practical, hands-on patterns for working with
    # Pandas, SciPy, and the major machine-learning framework (scikit-learn, TensorFlow, PyTorch)
    # you'll see how data flows between these libraries, common pitfalls (dtype, memory, missing values),
    # and simple, reproducible code examples you can paste into a notebook
    # this emphasis is pragmatic: how to move arrays around reliably,
    # preserve performance and memory where possible, and make each layer do what it does best

    # 12.1 how pandas builds on numpy
    # pandas provides high-level, table-oriented data structures (Series, DataFrame) that wrap NumPy arrays (and other array backends)
    # the tight integration means:
    # most numeric data in a DataFrame is backed by NumPy ndarray memory (unless an extension dtype is used)
    # converting between Pandas and NumPy is cheap and idiomatic:
    # df.to_numpy() or df.values -> NumPy:
    # pd.DataFrame(arr) -> Pandas
    # pandas handles labels, alignment, missing-value semantics, and many convenience operations
    # that are inconvenient in raw numpy

    # below is a compact end-to-end example showing typical interop patterns and import caveats
    import numpy as np
    import pandas as pd
    # create a dataframe from numpy array
    rng = np.random.default_rng(seed=0)
    arr = rng.normal(size=(6, 3))
    df  = pd.DataFrame(arr, columns=['height', 'weight', 'age'])
    df.loc[2, 'weight'] = np.nan    # injecting missin value
    # 1) convert dataframe -> numpy safely (for numeric-only array)
    X = df.to_numpy()       # shape (6,3), dtype=float64, NaNs preserved
    print('X shape, dtype:', X.shape, X.dtype)
    # 2) convert only a subset of columns (common pattern)
    X_num = df[ ['height', 'weight'] ].to_numpy(dtype=np.float32)   # explicit dtype
    print('X_num shape, dtype:', X_num.shape, X_num.dtype)
    # 3) convert categorical column to codes
    df['color'] = pd.Categorical(['red','blue','red','green','blue','red'])
    codes = df['color'].cat.codes.to_numpy()    # integer codes (Numpy)
    print('category mapping:', dict(enumerate(df['color'].cat.categories)))
    # important subtitles and best practices
    # df.values vs df.to_numpy():
    # both return an ndarray, but to_numpy() is the recommended explicit API
    # if columns use mixed dtypes or pandas extension dtype (nullable integers, pd.NA),
    # the result may be an object-dtype array.
    # use df.select_dtypes() to request numeric columns only
    # pandas may use special extension dtypes (e.g., int64, boolean) that are not plain NumPy dtype
    # converting those columns to NumPy may give an object array or require astype/fillna first
    # missing values:
    # pandas' NaN handling is compatible with NumPy floats
    # for integer columns with missing values, prefer pandas nullable integer types (Int64)
    # but convert to floats before sending to NumPy-based ML routines (which expect np.nan for missingness) or impute first
    # label alignment:
    # Pandas aligns by index/column names for arithmetic - a powerful feature
    # when you convert to NumPy and back, you lose labels; keep track of column order explicitly

    # when to use Pandas vs. Numpy
    # use Pandas for
    # reading/parsing files, exploratory data cleaning, grouping/aggregation with labels, time, series, join/merge operations
    # use NumPy for
    # heavy numeric kernels, vectorized transforms, linear algebra, high-performance loops,
    # often the workflow is:
    # ingestion/cleaning in Pandas -> numeric transforms in NumPy (or scikit-learn) -> back to Pandas for reporting

    # 12.2 interoperability with SciPy
    # SciPy builds on NumPy and provides higher-level scientific functionality:
    # advanced linear algebra, optimization, signal processing, sparse matrices, statistics beyond the basics, and more
    # interoperability is straightforward:
    # Scipy functions accept Numpy arrays and often return Numpy arrays (or Scipy objects that wrap arrays)

    # examples and patterns

    # Dense Linear Algebra
    import numpy as np
    import scipy as sc
    A = np.random.default_rng(seed=0).normal(size=(5,5))
    # Scipy wraps LAPACK and offers extra routines:
    lu, piv = sc.linalg.lu_factor(A)    # LU factorization
    x = sc.linalg.lu_solve((lu, piv), np.random.rand(5))
    print(A)
    print(lu)
    print(piv)
    print(x)
    # scipy.linalg sometimes offers more functionality and tunable control than np.linalg
    # (e.g., specialized solvers, condition estimators)

    # sparse matrices (memory & speed for sparse data)
    # Scipy's sparse module is the standard for sparse linear algebra
    # it provide CSR(Compressed Sparse Row)/CSC(Compressed Sparse Column)/COO(Coordinate List) format that interoperate with Numpy:
    import scipy as sc
    # build a CSR(Compressed Sparse Row) sparse matrix form NumPy arrays
    rows = np.array([0,1,2,2])
    cols = np.array([1,2,0,2])
    data = np.array([10.0, 20.0, 30.0, 40.0])
    S = sc.sparse.csr_matrix((data, (rows,cols)), shape=(4,4))
    # convert back to dense Numpy if you need to
    dense = S.toarray()
    print(S)
    print(dense)
    # use sparse matrices whenever the matrix is mostly zeros and you need memory or computational savings
    # many scipy linear solvers accept sparse matrices directly
    # and scikit-learn works seamlessly with scipy sparse matrices in many estimators
    # (e.g., LogisticRegression with sparse input)

    # signal processing, optimization, statistics
    # scipy exports lots of functionality:
    # scipy.optimize(minimizers), scipy.signal(filters), scipy.stats(tests, distributions)
    # they accept numpy arrays and return numpy arrays or scalars

    # example: T-test using scipy (a common complement to numpy's descriptive stats:
    import scipy as sc
    x = np.random.normal(0.0, 1.0, size=100)
    y = np.random.normal(0.2, 1.0, size=100)
    tstat, pval = sc.stats.ttest_ind(x, y, equal_var=False)
    print(tstat)
    print(pval)

    # practical advice
    # prefer scipy for algorithms that numpy doesn't provide (sparse, optimization, special functions)
    # conversions between dense numpy amd scipy sparse formats are explicit:
    # sparse.csr_matrix(X) and .toarray() as needed - be mindful of memory
    # many scipy routines return tuples of numpy arrays (e.g., decomposition outputs)
    # keep track of shapes and dtypes

    # 12.3: bridging to machine learning frameworks (scikit-learn, tensorflow, pytorch)
    # machine learning libraries accept numpy arrays as primary inputs
    # they typically expect numeric, contiguous arrays with no NaNs (unless specifically supported)
    # this section shows common, robust patterns to move data
    # between numpy/pandas and ML frameworks, handle dtypes, and preserve performance and memory where possible

    # scikit-learn (classical ML)
    # scikit-learn's API was designed around numpy arrays
    # most estimators accept X as a numpy array or a pandas DataFrame and y as a 1-D array/series
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    # sample dataset (Numpy)
    X = np.random.default_rng(seed=0).normal(size=(200,10))
    y = (X[:,0] + 0.5*X[:,1] > 0).astype(int)
    # train/test split (works with numpy arrays)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # pipeline: scaling + PCA + logistic regression
    pipe = make_pipeline(StandardScaler(), PCA(n_components=5), LogisticRegression())
    pipe.fit(X_train, y_train)
    print('test score:', pipe.score(X_test, y_test))

    # key interoperability notes
    # fit/transform often return numpy arrays
    # newer scikit-learn versions provide set_output(transform='pandas') to request DataFrame outputs
    # from transforms when a DataFrame was given, but earlier code expects Numpy arrays
    # (if you rely on column names downstreams, either keep track of them or use transforms that preserve them)
    # OneHotEncoder can produce a sparse output (sparse=True),
    # which is memory efficient for high-cardinality categoricals
    # you can pass sparse matrices directly to many classifiers which accept scipy sparse input
    # scikit-learn works natually with scipy sparse matrices
    # e.g., text data transformed by CountVectorizer -> sparse matrix
    # SVD (Singular Value Decomposition)
    # PCA (Principal Component Analysis)
    
    # example: OneHotEncoder -> sparse -> classifier
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import CountVectorizer
    import scipy as sc
    cats = np.array(['red','green','blue','red'])
    enc = OneHotEncoder(handle_unknown='ignore')
    S = enc.fit_transform(cats.reshape(-1,1))       # sparse matrix
    print(type(S), S.shape)
    print(S)
    print(cats)

    # practical tips
    # always fit scalers/encoders on the training set and apply them to validation/test sets
    # for pipelines deployed in production,
    # persist both the trained model and the preprocessing steps (joblib.dump)
    # and ensure deterministic versions of libraries
    # keep an eye on memory:
    # prefer sparse outputs where appropriate and prefer float32 for large datasets
    # (but check estimator compatibility)

    # tensorflow (deep learning)
    # tensorflow accepts NumPy arrays and provides tf.data for performant input pipelines
    # converting is straightforward:
    import numpy as np
    import tensorflow as tf
    X = np.random.rand(1000, 32).astype(np.float32)
    y = np.random.randint(0, 2, size=(1000,))
    # convert Numpy -> tensorflow tensor (copy by default)
    tX = tf.convert_to_tensor(X)    # usually copies into TF-managed buffer
    # build a tf.data.Dataset from numpy arrays
    ds = tf.data.Dataset.from_tensor_slices((X,y))
    ds = ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    # simple model
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(32,)),
                                 tf.keras.layers.Dense(64, activation='relu'),
                                 tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(ds, epochs=3)

    # notes on memory and copies
    # tf.convert_to_tensor or from_tensor_slices will typically copy data into TensorFlow's buffers;
    # there is no shared memory with NumPy arrays
    # for large datasets, prefer tf.data pipelines that read from disk (TFRecord, HDF5) or use generators to avoid huge memory peaks
    # TensorFlow operations run on CPU or GPU depending on placement
    # transfer of large NumPy arrays to GPU happens implicitly when data reaches GPU-backed tensors
    # be mindful of transfer cost and batch appropriately

    # PyTorch (deep learning) - close NumPy interop
    # PyTorch has a tight, zero-copy interop with NumPy for CPU tensors
    import numpy as np
    import torch
    X = np.random.rand(1000, 32).astype(np.float32)
    t = torch.from_numpy(X)     # shares memory with X (no copy) if dtype is compatible
    t[:,0] = 0.0                # this modifies X as well
    print('t[0,0], X[0,0]', t[0,0], X[0,0])
    # to move tensor to GPU (requires CUDA)
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        t_gpu = t.to('cuda')        # copies to device
        # compute on GPU...
        t_back = t_gpu.to('cpu')    # copy back
        arr_back = t_back.numpy()

    # caveats and best practices
    # torch.from_numpy shares memory with the numpy array
    # if you need independent data, call .clone() or np.copy() first
    # pytorch expects C-contiguous arrays;
    # if x is fortran-ordered or has negative strides,
    # torch.from_numpy may copy - ensure X = np.ascontiguousarray(X) beforehand
    # moving tensors to GPU incurs a copy;
    # keep data on the device where computation happens and minimize host <-> device transfers

    # putting it together - a small cross-framework example
    # a common pattern is: ingest with pandas -> preprocess with numpy/scikit-learn -> train with pytorch/tensorflow
    # example sketch(PyTorch)
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    # load with pandas
    #df = pd.read_csv('some.csv')   # assume numeric columns present
    #X = df[['f1','f2','f3']].to_numpy(dtype=np.float32)
    #y = df['label'].to_numpy(dtype=np.int64)
    # split and scale with scikit-learn
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    #scaler = StandardScaler().fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_val = scaler.transform(X_val)
    # create pytorch dataloaders (shares memory if using torch.from_numpy on contiguous arrays)
    #train_ds = TensorDataset(torch.from_numpy(X_train), torch_from_numpy(y_train))
    #train_loader = DataLoader(trin_ds, batch_size=64, suffle=True)
    # model and training code here...
    # this pattern keeps each library in its sweet spot and minimizes unnecessary copies

    # pitfalls, practical advice, and my experience
    # 1. Dtype discipline
    # decide early whether to use float32 or float64
    # float32 is often fine for ML and halves memory
    # float64 may be needed for high-precision scientific work
    # convert explicitly (astype) and avoid implicit upcasts when combining arrays of different dtypes
    # 2. missing values
    # fill or encode missing values before passing to frameworks that don't accept NaN
    # (many scikit-learn estimators throw errors)
    # pandas makes this easy, but remenber to persist the imputation parameters from training to inference
    # 3. memory sharing surprises:
    # pytorch from_numpy shares memory - modifying the tensor affects the numpy array and vice versa
    # tensorflow typically copies
    # always be explicit about where data is copies or shared
    # 4. sparse data
    # if your features are sparse (text, one-hot heavy), use scipy sparse matrices and choose
    # estimators that support sparse input
    # converting sparse -> dense can blow RAM instantly
    # 5. pipelines and reproducibility
    # keep preprocessing and model steps packaged as Pipeline objects (scikit-learn)
    # or saved artifacts(joblib, pickle) so you don't forget to apply the same transforms at inference time
    # 6. batching and device placement
    # for GPU training, batch sizes and how you transfer data matter
    # move large arrays once per batch (not per element) and reuse devices buffers where possible

    # personal note:
    # in produnction systems I almost always persist two things:
    # (a) fitted preprocessing parameters (scalers, encoders) and
    # (b) a canonical list of feature names and order
    # losing the exact feature order is the most common source of mysterious model regressions

    # key takeaways
    # pandas is the user-friendly front end for ingestion and wrangling;
    # numpy remains the workhorse for raw numeric computaion
    # convert deliberately with to_numpy() and manage dtype/NaN semantics
    # Scipy complements Numpy with advanced linear algebra, sparse matrices, optimization, and statistics
    # pass Numpy array directly to Scipy and mind dense <-> sparse conversions
    # scikit-learn expects and returns Numpy arrays (or pandas DataFrames);
    # prefer Pipeline objects and fit transformations on training data only
    # tensorflow typically copies Numpy arrays into tensors;
    # pytorch from_numpy can share memory - know when you shared vs copied buffers
    # for GPUs, minimize host <-> device transfers and batch data
    # use sparse representations where appropriate, persist preprocessing artifacts,
    # and enforce consistent dtypes/contiguity before crossing library boundaries
    # in practice, keep each library doing the job it does best:
    # pandas for labels and complex IO,
    # numpy/scipy for numerics,
    # scikit-learn for conventional ML,
    # and tensorflow/pytorch for GPU-accelerated deep learning

    # working fluently across this ecosystem lets you prototype quickly and scale reliably
    # the next part of the book turns toward model evaluation, deployment considerations, and production robustness



#
# CH 13: machine learning preprocessing
#

if False:
    # good preprocessing is the scaffolding of successful machine learning
    # a careful, reproducible preprocessing pipeline prevents data leakage, stabilizes training, and makes models portable into production
    # in this chapter we cover three pragmatic topics you'll apply immediately:

    # 13.1 train/test splits and shuffling - reproducible, stratified, and fold-based splitting using Numpy
    # 13.2 feature-scaling pipelines - robust, test-safe fit/transform objects you can persist
    # 13.3 dimensionality reduction with Numpy - PCA (SVD-based), explained variance, and memory-efficient PCA for large data
    # you'll get clear, runnable code for each task, step-by-step explainations, and practical tips from real projects

    # 13.1 train/test splits and shuffling
    # splitting data correctly is deceptively important
    # common mistakes include:
    # (a) shuffling after splitting,
    # (b) leaking information from validation into training when computing statistics, and
    # (c) non-reproducible splits
    # here are reproducible, numpy-only patterns for simple random splits, stratified splits, and K-fold cross-validation

    # deterministic random splits
    # always use a Generator for reproducible randomness:
    import numpy as np
    def train_test_split_np(n_samples, test_size=0.2, rng=None, shuffle=True):
        # return train_idx, test_idx (integer array)
        # n_samples: number of rows
        # test_size: fraction of integer count
        # rng: np.random.Generator or None -> default_rng(0)
        if rng is None:
            rng = np.random.default_rng(seed=0)
        idx = np.arange(n_samples)
        if shuffle:
            rng.shuffle(idx)
        if 0 < test_size < 1:
            n_test = int(n_samples * test_size)
        else:
            n_test = int(test_size)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        return train_idx, test_idx
    # example
    rng = np.random.default_rng(seed=42)
    train_idx, test_idx = train_test_split_np(1000, test_size=0.25, rng=rng)
    print(len(train_idx), len(test_idx))
    # important patterns
    # fit any preprocessing (imputer, scaler) on X[train_idx] and apply transform on both train and test - never fit on full data
    # keep rng seed-controlled (default_rng(seed)) for reproducibility across runs and machines

    # stratified splits with numpy
    # for classification tasks you often want to preserve class proportions
    # use np.unique with grouping and sample with each class:
    def stratified_split(labels, test_size=0.2, rng=None):
        # labels: 1D array-like of class labels
        # returns train_idx, test_idx preserving label proportions
        labels = np.asarray(labels)
        if rng is None:
            rng = np.random.default_rng(seed=0)
        unique, inv = np.unique(labels, return_inverse=True)
        train_mask = np.zeros(labels.shape[0], dtype=bool)
        test_idx_list = []
        train_idx_list = []
        for cls in range(unique.size):
            cls_idx = np.where(inv == cls)[0]
            rng.shuffle(cls_idx)
            if 0 < test_size < 1:
                n_test = int(len(cls_idx) * test_size)
            else:
                n_test = int(test_size)
            test_idx_list.append(cls_idx[:n_test])
            train_idx_list.append(cls_idx[n_test:])
        test_idx = np.concatenate(test_idx_list)
        train_idx = np.concatenate(train_idx_list)
        # shuffle the final indices
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        return train_idx, test_idx
    # example
    y = np.repeat([0, 1, 2], [50, 30, 20])
    # imbalanced toy levels
    rng = np.random.default_rng(seed=1)
    train_idx, test_idx = stratified_split(y, test_size=0.2, rng=rng)
    print('train counts:', np.bincount(y[train_idx]))
    print('test counts:', np.bincount(y[test_idx]))
    # this pattern is simple and avoid external dependencies
    # for multi-label or complex stratification, you may prefer scikit-learn utilities,
    # but the numpy pattern above covers most tabular cases

    # K-fold and stratified K-fold generation
    # lightweight K-fold generator in NumPy:
    def kfold_indices(n_samples, n_splits=5, shuffle=True, rng=None):
        if rng is None:
            rng = np.random.default_rng(seed=0)
        idx = np.arange(n_samples)
        if shuffle:
            rng.shuffle(idx)
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        print(fold_sizes)
        fold_sizes[: n_samples % n_splits] += 1
        print(fold_sizes)
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = idx[start:stop]
            train_idx = np.concatenate([idx[:start], idx[stop:]])
            yield train_idx, test_idx
            current = stop
    # example
    for fold, (tr, te) in enumerate(kfold_indices(23, n_splits=5, rng=rng)):
        print(f'fold {fold}: train {len(tr)}, test {len(te)}')
    # for stratified k-fold, group indices by label then distiribute class numbers across folds evenly
    # implementation is similar to stratified split but with round-robin assignment into fold buckets

    # cross-validation tips
    # use the same random seed for splitting across experiments for comparability
    # avoid shuffling time-series data unless you use time aware cross-validation
    # (e.g., rolling window) - next section: time-series considerations
    # when using stratified k-fold, ensure minimum class counts exceed n_splits to avoid empty test folds

    # 13.2 feature scaling pipelines
    # scaling is about reproducibility and safty
    # implment simple fit / transform / save / load classes that operate on NumPy arrays only
    # below we build a small, well-documented StandardScaler, MinMaxScaler, and a lightweight Pipeline
    # that chains steps and persists parameters with np.savez

    # a robust StandardScaler (Numpy implementation)
    import numpy as np
    import matplotlib.pyplot as plt
    class StandardScalerNP:
        def __init__(self, with_mean=True, with_std=True, dtype=np.float64):
            self.with_mean = with_mean
            self.with_std  = with_std
            self.dtype     = dtype
            self.mean_     = None
            self.scale_    = None

        def fit(self, X):
            X = np.asarray(X, dtype=self.dtype)
            if self.with_mean:
                self.mean_ = np.nanmean(X, axis=0)
            else:
                self.mean_ = np.zeros(X.shape[1], dtype=self.dtype)
            if self.with_std:
                self.scale_ = np.nanstd(X, axis=0, ddof=0)
                # avoid division by zero
                self.scale_[self.scale_ == 0.0] = 1.0
            else:
                self.scale_ = np.ones(X.shape[1], dtype=self.dtype)
            return self

        def transform(self, X):
            if self.mean_ is None or self.scale_ is None:
                raise ValueError('Scaler has not been fitted.')
            X = np.asarray(X, dtype=self.dtype, copy=True)
            # imprecision: if X has NaNs, substraction preserves NaNs (desired)
            X -= self.mean_
            X /= self.scale_
            return X

        def fit_transform(self, X):
            return self.fit(X),transform(X)

        def save(self, path):
            np.savez(path, mean=self.mean_, scale=self.scale_)

        def load(self, path):
            d = np.load(path)
            self.mean_ = d['mean']
            self.scale_ = d['scale']
            return self
    # usage
    rng = np.random.default_rng(seed=0)
    X_train = rng.normal(loc=5, scale=2, size=(100, 3))
    X_test  = rng.normal(loc=5.1, scale=2.2, size=(20, 3))
    scaler = StandardScalerNP().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    plt.plot(X_train, 'bo')
    plt.plot(X_test, 'bx')
    plt.plot(X_train_s, 'ro')
    plt.plot(X_test_s, 'rx')
    plt.grid(ls=':')
    plt.close()

    # min-max and robust scalers
    class MinMaxScalerNP:
        def __init__(self, feature_range=(0.0, 1.0), dtype=np.float64):
            self.feature_range = feature_range
            self.dtype = dtype
            self.data_mim_ = None
            self.data_max_ = None
            self.data_range_ = None

        def fit(self, X):
            X = np.asarray(X, dtype=self.dtype)
            self.data_min_ = np.nanmin(X, axis=0)
            self.data_max_ = np.nanmax(X, axis=0)
            self.data_range_ = self.data_max_ - self.data_min_
            self.data_range_[self.data_range_ == 0] = 1.0
            return self

        def transform(self, X):
            if self.data_min_ is None:
                raise ValueError('not fitted')
            X = np.asarray(X, dtype=self.dtype, copy=True)
            X -= self.data_min_
            X /= self.data_range_
            a, b = self.feature_range
            return X * (b - a) + a

        # robust (median + IQR)
        class RobustScalerNP:
            def __init__(self, dtype=np.float64):
                self.dtype = dtype
                self.center_ = None
                self.scale_ = None

            def fit(self, X):
                X = np.asarray(X, dtype=self.dtype)
                self.center_ = np.nanmedian(X, axis=0)
                q75 = np.nanpercentile(X, 75, axis=0)
                q25 = np.nanpercentile(X, 25, axis=0)
                iqr = q75 - q25
                iqr[iqr==0] = 1.0
                self.scale_ = iqr
                return self

            def transform(self, X):
                X = np.asarray(X, dtype=self.dtype, copy=True)
                X -= self.center_
                X /= self.scale_
                return X

        # A minimal Pipeline utility
        class NumpyPipeline:
            def __init__(self, steps):
                # steps: list of (name, transformer) where transformer implements fit/transform
                self.steps = steps

            def fit(self, X):
                Xt = X
                for name, transformer in self.steps:
                    transformer.fit(Xt)
                    Xt = transformer.transform(Xt)
                return self

            def transform(self, X):
                Xt = X
                for name, transformer in self.steps:
                    Xt = transformer.transform(Xt)
                return Xt

            def fit_transform(self, X):
                self.fit(X)
                return self.transform(X)

            def save(self, basepath):
                params = {}
                for name, transformer in self.steps:
                    # save each transform to a separate file: '${basepath}__{name}.npz'
                    transformer.save(f'{basepath}__{name}.npz')
        
        # important practical patterns
        # always fit on training data only
        # use copy=True semantics when you want to preserve original arrays
        # persist scaler parameters (npz or joblib) along with meta-data (column order, dtype)
        # document dtypes - many production systems expect float32

        # why this matters in production
        # during deployment you may receive single-row inference requests
        # your transformation must be deterministic and use the exact same mean, scale and category maps used in training
        # save and version those artifacts

        # 13.3 dimensionality reduction with numpy
        # PCA vis SVD is the canonical dimensionality-reduction technique
        # we'll cover:
        # centering and SVD-based PCA (direct)
        # interpreting explained variance and choosing components
        # memory-efficient PCA for large n using chunked covariance accumulation
        # (when feature dimension p is manageable)

        # PCA by SVD (standard, robust)
        # given data matrix x with shape (n_samples, n_features):
        # 1. center x by column means: Xc = X - X.mean(axis=0)
        # 2. compute SVD: U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        # 3. pricipal components (directions) are rows of Vt
        #    project data: Z = Xc @ Vt.T or Z = U * s (scaled scores)
        # 4. explained variance: var_explained = s**2 / (n_samples - 1)
        #    proportion = var_explained / var_explained.sum()

        # working code:
        import numpy as np
        def pca_svd(X, n_components=None, center=True):
            X = np.asarray(X, dtype=float)
            n, p = X.shape                                      # shape (n, p)
            if center:
                mean = np.mean(X, axis=0)
                Xc = X - mean
            else:
                mean = np.zeros(p)
                Xc = X
            U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            #plt.plot( np.cumsum(s**2/(n-1)/np.sum(s**2/(n-1))), 'o')
            #plt.grid(ls=':')
            #plt.show()
            if n_components is None:
                n_components = Vt.shape[0]
            components = Vt[:n_components]                      # shape (k, p)
            scores = U[:, :n_components] * s[:n_components]     # shape (n, k)
            # explained variance
            explained_variance = (s**2) / (n-1)
            explained_variance_ratio = explained_variance / explained_variance.sum()
            #
            return {'components': components,
                    'scores': scores,
                    'explained_variance': explained_variance[:n_components],
                    'explained_variance_ratio': explained_variance_ratio[:n_components],
                    'mean': mean,
                    'singular_values': s[:n_components]}
        # example:
        rng = np.random.default_rng(seed=0)
        X = rng.normal(size=(200,50))
        res = pca_svd(X, n_components=5)
        print('explained ratios:', res['explained_variance_ratio'])

        # projecting new data: to project a new matrix x_new:
        def project_new(X_new, components, mean):
            Xn = np.asarray(X_new, dtype=float) - mean
            return Xn @ components.T
        # use components (k, p) and mean saved from training

        # choosing number of components
        # choose k by
        # cumulative explained variance threshold (e.g.,95%):
        # pick smallest k with sum(ratio[:k]) >= 0.95
        # scree plot (plot singular values or explained variance) to find elbow
        # downstream model performance via cross-validation

        cumvar = np.cumsum(res['explained_variance_ratio'])
        k = np.searchsorted(cumvar, 0.95) + 1
        print('components for 95%:', k)

        # memory-efficient PCA for large n (many samples, moderate features)
        # if n is huge but p (number of features) is moderate (e.g.,p=100-10k),
        # compute the sample covariance matrix in chunks and eigendecompose the p x p covariance:
        # 1. compute column means by chunking (chapter 4 patterns)
        # 2. compute covariance: accumulate Xc.T @ Xc in chunks
        # 3. divide by (n-1) to get covariance C (shape p x p)
        # 4. compute eigendecomposition w, v = np.linang.eigh(C) (symmetric) and sort decending

        # code sketch:
        def incremental_mean_cov(source, n_rows, n_features, chunk_rows=10000, dtype=np.float64):
            # source: function(start, stop) -> returns array (stop-start, n_features)
            #         e.g., a memmap slice or reader function
            # 1) compute mean
            total = np.zeros(n_features, dtype=dtype)
            count = 0
            for i in range(0, n_rows, chunk_rows):
                block = source(i, min(n_rows, i+chunk_rows)).astype(dtype)
                total += np.nansum(block, axis=0)
                count += (~np.isnan(block)).sum(axis=0)
            mean = total / count    # element-wise
            # 2) compute covariance accumulator
            cov_acc = np.zeros((n_features, n_features), dtype=dtype)
            for i in range(0, n_rows, chunk_rows):
                block = source(i, min(n_rows, i+chunk_rows)).astype(dtype)
                # subtract column means, taking care of NaNs
                inds = np.where(np.isnan(block))
                block[inds] = 0.0
                block -= mean
                cov_acc += block.T @ block  # (p, k) @ (k, p) -> (p, p)
            # divide by (n-1) or actual effective count per pair - approximate when NaNs exist
            cov = cov_acc / (n_rows - 1)
            return mean, cov
        # eigendecompose
        # w, V = np.linalg.eigh(cov)    # ascending order
        # sort descending:
        # idx = np.argsort(w)[::-1]
        # w = w[idx]; V = V[:, idx]
        
        # this approach avoids forming n x p matrices in memory at once
        # and computes PCA via covariance eigendecomposition
        # it's efficient if p is small enough to fit p x p in memory
        # handling NaNs and varying counts per pair requires more care
        # (use pairwise counts and divide per element),
        # but for many datasets full rows are present and the simple approach is fine

        # randomized PCA (sketch)
        # for very large p and n, randomized SVD algorithms (Halko et al.) are efficient
        # implementing a robust randomized SVD is non-trivial; instead, you can:
        # use scikit-learn's randomized_svd or IncrementalPCA which wrap efficient C code
        # if you must stick to numpy only, implement a simple power-iteration style
        # randomized SVD (beyond this chapter's scope), or use block processing with
        # np.linalg.svd on smaller projected data

        # practical PCA checklist
        # center data before SVD - do not compute PCA on raw data unless you understand the consequences
        # save mean and components for inference
        # for numerical stability prefer SVD over eigendecomposition of X.T @ X when n and p are comparable
        # for p << n, the covariance-based approach (eigendecomposition of p x p) is cheaper
        # for data with NaNs, either impute first or use algorithms that can handle missingness
        # impute using train-only statistics

        # personal insight:
        # i often do PCA as a diagnostic first - singular-value spectrum quickly tells me
        # if the data is effectively low-rank (a handful of large singular values) or high-dimensional noise
        # if the spectrum drops quickly,
        # a low-dimensional projection both speeds training and often improves generalization

        # key takeaways
        # always split data reproducibly:
        # use np.random.default_rng(seed) and fit preprocessing on training data only
        # for classification use stratified splits to preserve class proportions
        # build small, explicit fit/transform objects for scalars (standard. minmax, robust);
        # persist their parameters and apply them consistently at inference time
        # PCA vis SVD is robust and gives both components and explained variance;
        # save mean and components to tranform new data
        # choose k by cumulative explained-variance or downstream validation
        # for very large n but moderate p,
        # compute covariance in chunks and eigendecompose the p x p matrox (memory-efficient PCA)
        # for extremen scale or high p, use specialized randomized or incremental PCA implementations (e.g., scikit-learn)
        # document and presist metadata (column order, dtype, scaler parameters, category mapping)
        # this is the single most practical habit for maintaining reproducible pipelines
        


#
# CH 14: Linear Regression from Scratch
#

if True:
    # linear regression is the canonical first supervised learning algorithm:
    # simple to state, but rich enough to teach best practices for optimization, numerical stability, and model evaluation
    # in this chapter we build linear regression from first priciples
    # closed-form solution and gradient-based optimization
    # then show how to evaluate and diagnose models so you can trust them in practice
    # we'll proceed step-by-step, with clear vectorized numpy code you can paste into a notebook and run
    # at the end you'll understand trade-offs (when to use the normal equation vs gradient descent),
    # how to add regularization, and how to check whether a model actually learned something useful

    # 14.1 implementing gradient descent
    # problem setup and notation (very short)
    # given an input matrix X (real, n x p) and target y (real, n), linear regression models
    # y = X w + b
    # we'll implement the bias b by augmenting X with a constant column so we only solve for a single parameter vector w)
    # the squared-error loss (mean squared error) we minimize is
    # J_w = 1/2n || X w - y ||^2 = 1/2n (X w - y).T @ (X w - y)
    # we include the 1/2 to simplify gradients
    # its gradient is
    # del J_w = 1/n X.T @ (X w - y)
    # gradient descent update w <- w - eta del J_w with step size (learning rate) eta

    # closed-form (normal equation) - quick reference
    # if you want the exact solution (and p is not too large),
    # the normal equation is
    # w* = (X.T @ X)^-1 @ (X.T @ y)
    # with L2 (Ridge) regularization lambda, the closed form becomes
    # w* = (X.T @ X + lambda I)^-1 @ (X.T @ y)
    # where the bias is usually excluded from regularization
    # you can handle that by augmenting II appropriately
    # drawbacks
    # computing and inverting X.T @ X costs O(p^3) and can be numerically unstable
    # if X.T @ X is ill-conditioned or p is very large

    # vectorized batch gradient descent (implementation)
    # below is a single, self-constrained. well-documented numpy implementation that includes options for:
    # fit intercept (bias)
    # L2 regularization (Ridge)
    # batch / mini-batch / stochastic updates
    # learning-rate schedule
    # early stopping by tolerance
    # returning loss history for diagnostics

    import numpy as np
    from typing import Optional, Tuple
    
    class LinearRegressionGD:
        # lienar regression using gradient descent (supports mini-batches and L2 regularization)
        # minimize (1/(2n)) * || X w - y ||^2 + (alpha/(2n)) * || w_reg ||^2
        # where w_reg excluded the bias term (if fit_intercept=True)
        # function annotation (kind of comments)
        #  (a)  :  metadata of input parameters
        #  (b) ->  metadata of return value
        def __init__(self,
                     lr: float = 1e-2,
                     n_epochs: int = 1000,
                     batch_size: Optional[int] = None,  # None -> full-batch, 1 -> SGD, >1 -> mini-batch
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
            #
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float).reshape(-1)  # 1-D array flattening
            #
            n, p = X.shape
            Xb = self._add_bias(X)  # shape (n, p+1) if intercept, else (n, p)
            m = Xb.shape[1]         # init weights (small random or zeros)
            #
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
                #
                epoch_loss = 0.0
                #
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





