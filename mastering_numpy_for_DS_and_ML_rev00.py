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

if True:
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




































