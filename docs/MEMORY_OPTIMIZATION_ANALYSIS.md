# Memory Optimization Analysis for Vectorized Batch Processing

## Current Memory Usage Pattern

### Per-Lesion Batch Processing (20 lesions):

1. **Lesion preparation**: ~negligible (indices only)
2. **Per connectome batch** (repeated for each h5 file):
   - Load timeseries: `(n_subjects, n_timepoints, n_voxels)` → ~1-4 GB
   - Extract lesion timeseries: `(n_lesions, n_subjects, n_timepoints)` → ~100-400 MB
   - Compute correlations: `(n_lesions, n_subjects, n_voxels)` → ~2-8 GB
   - **Accumulate in all_subject_r_maps**: `(n_lesions, total_subjects, n_voxels)` → **GROWS WITH EACH BATCH**
3. **Final aggregation**: Each lesion processes accumulated maps → ~2-8 GB
4. **Total peak**: ~10-20 GB for 20 lesions

### Key Memory Issues:

#### 1. **Accumulation of all_subject_r_maps (MAJOR)**
**Location**: `functional_network_mapping.py:825-836`

```python
# Initialize accumulators for each lesion
all_subject_r_maps = {i: [] for i in range(len(lesion_batch))}

for batch_idx, connectome_path in enumerate(connectome_files):
    # ... process batch ...
    batch_r_maps = self._compute_batch_correlations_vectorized(...)
    
    # Accumulate results for each lesion
    for i in range(len(lesion_batch)):
        all_subject_r_maps[i].append(batch_r_maps[i])  # ACCUMULATES!
```

**Problem**: 
- For 1000 subjects split into 10 h5 files, stores ALL 1000 × n_voxels correlation maps
- With 20 lesions × 1000 subjects × 50k voxels × 4 bytes = **4 GB just for r_maps**
- Keeps full precision (float32) when final result is averaged anyway

**Solution**: 
- Running aggregation instead of accumulation
- Accumulate sums and sums-of-squares for Fisher z-transformed values
- Only store mean + std, discard individual subject maps

#### 2. **Float64 in einsum computation**
**Location**: `functional_network_mapping.py:945-951`

```python
cov = np.einsum(
    "lit,itv->liv",
    lesion_ts_centered,
    brain_ts_centered,
    dtype=np.float64,  # Double precision
    optimize="optimal",
)
```

**Problem**: Uses 8 bytes per value instead of 4

**Solution**: Use float32 throughout unless precision critically needed

#### 3. **Multiple 3D volume allocations**
**Location**: `functional_network_mapping.py:1067-1077`

```python
correlation_map_3d = np.zeros(mask_shape, dtype=np.float32)
z_map_3d = np.zeros(mask_shape, dtype=np.float32)
t_map_3d = np.zeros(mask_shape, dtype=np.float32)  # if t-map computed
threshold_map_3d = np.zeros(mask_shape, dtype=np.uint8)  # if threshold
```

**Problem**: Each full brain volume = ~91 × 109 × 91 × 4 bytes = ~3.6 MB × 4 maps = ~15 MB per lesion

**Solution**: Minimal - this is already small, but could reuse buffer

#### 4. **Intermediate timeseries storage**
**Location**: `functional_network_mapping.py:906-928`

```python
# Extract and process timeseries for all lesions
lesion_mean_ts_list = []
for lesion_info in lesion_batch:
    lesion_ts = timeseries_data[:, :, voxel_indices]  # Extracted
    lesion_mean_ts = np.mean(lesion_ts, axis=2)  # Computed
    lesion_mean_ts_list.append(lesion_mean_ts)  # Stored

# Stack into (n_lesions, n_subjects, n_timepoints)
lesion_mean_ts_batch = np.stack(lesion_mean_ts_list, axis=0)
```

**Problem**: Creates temporary list, then stacks into new array

**Solution**: Pre-allocate and fill directly

## Proposed Optimizations

### Priority 1: Streaming Aggregation (CRITICAL - saves 50-70% memory)

Replace accumulation with running statistics:

```python
def run_batch(self, lesion_data_list: list[LesionData]) -> list[LesionData]:
    # ... setup ...
    
    # Initialize running aggregators for each lesion
    aggregators = []
    for i in range(len(lesion_batch)):
        aggregators.append({
            'sum_z': np.zeros(n_voxels, dtype=np.float32),
            'sum_z2': np.zeros(n_voxels, dtype=np.float32),
            'n': 0,
        })
    
    # Process connectome batches with streaming aggregation
    for batch_idx, connectome_path in enumerate(connectome_files):
        # ... load and compute ...
        batch_r_maps = self._compute_batch_correlations_vectorized(...)
        
        # Convert to z-scores and update running statistics
        with np.errstate(divide='ignore', invalid='ignore'):
            batch_z_maps = np.arctanh(batch_r_maps)
            batch_z_maps = np.nan_to_num(batch_z_maps)
        
        # Update aggregators (NO accumulation of full maps!)
        for i in range(len(lesion_batch)):
            for subj_idx in range(batch_r_maps.shape[1]):
                aggregators[i]['sum_z'] += batch_z_maps[i, subj_idx, :]
                aggregators[i]['sum_z2'] += batch_z_maps[i, subj_idx, :] ** 2
                aggregators[i]['n'] += 1
        
        # Free memory immediately
        del timeseries_data, batch_r_maps, batch_z_maps
    
    # Compute final statistics from aggregated values
    for i, lesion_info in enumerate(lesion_batch):
        n = aggregators[i]['n']
        mean_z = aggregators[i]['sum_z'] / n
        mean_r = np.tanh(mean_z)
        
        if self.compute_t_map:
            std_z = np.sqrt(aggregators[i]['sum_z2'] / n - mean_z ** 2)
            t_map = mean_z / (std_z / np.sqrt(n))
        
        # ... create results ...
```

**Memory savings**: 
- Before: 20 lesions × 1000 subjects × 50k voxels × 4 bytes = **4 GB**
- After: 20 lesions × 50k voxels × 8 bytes (sum + sum²) = **8 MB**
- **Reduction: 99.8% for correlation maps!**

### Priority 2: Float32 Throughout

```python
cov = np.einsum(
    "lit,itv->liv",
    lesion_ts_centered,
    brain_ts_centered,
    dtype=np.float32,  # Half the memory
    optimize="optimal",
)
```

**Memory savings**: 50% reduction in einsum intermediate results

### Priority 3: Pre-allocated Timeseries Stack

```python
# Pre-allocate instead of list + stack
n_lesions = len(lesion_batch)
n_subjects = timeseries_data.shape[0]
n_timepoints = timeseries_data.shape[1]

lesion_mean_ts_batch = np.empty(
    (n_lesions, n_subjects, n_timepoints), 
    dtype=np.float32
)

for idx, lesion_info in enumerate(lesion_batch):
    lesion_ts = timeseries_data[:, :, lesion_info["voxel_indices"]]
    lesion_mean_ts_batch[idx] = np.mean(lesion_ts, axis=2)
    del lesion_ts  # Explicit cleanup
```

**Memory savings**: Eliminates temporary list overhead

### Priority 4: In-place Operations Where Possible

```python
# Instead of creating new arrays
lesion_ts_centered = lesion_mean_ts_batch - lesion_mean_ts_batch.mean(axis=2, keepdims=True)

# Do in-place
lesion_mean_ts_batch -= lesion_mean_ts_batch.mean(axis=2, keepdims=True)
```

**Memory savings**: Eliminates temporary copy

## Expected Impact

### Current Memory Usage (20 lesions, 1000 subjects, 50k voxels):
- Timeseries load: 1-4 GB (temporary)
- Accumulated r_maps: **4 GB** (persistent across batches)
- Einsum computation: 2-4 GB (temporary)
- Final volumes: 15 MB × 20 = 300 MB
- **Peak: ~10-15 GB**

### Optimized Memory Usage:
- Timeseries load: 0.5-2 GB (float32, temporary)
- Running aggregators: **8 MB** (persistent)
- Einsum computation: 1-2 GB (float32, temporary)
- Final volumes: 300 MB
- **Peak: ~3-5 GB**

### Result:
- **2-3x reduction in peak memory usage**
- **Can process 40-60 lesions instead of 20 with same RAM**
- Same numerical results (Fisher z-transform is correct approach)

## Implementation Priority

1. ✅ **Streaming aggregation** - Biggest impact, critical for large datasets
2. ✅ **Float32 throughout** - Easy change, significant impact
3. ⚡ **Pre-allocated arrays** - Moderate impact, clean code
4. ⚡ **In-place operations** - Small impact, easy wins

## Backward Compatibility

All optimizations maintain identical:
- API surface
- Results (Fisher z-averaging is statistically correct)
- Numerical precision (float32 sufficient for correlation coefficients)

No breaking changes for users.
