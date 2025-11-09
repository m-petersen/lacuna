# Batch Processing Time Monitoring

## Overview

When processing large datasets with many connectome batches, the toolkit now provides real-time timing information to help you monitor progress and estimate completion time.

## Example Output

```
======================================================================
PROCESSING CONNECTOME BATCHES
======================================================================

Batch 1/100: 10 subjects - completed in 0.33s

Batch 2/100: 10 subjects - completed in 0.95s

Batch 3/100: 10 subjects - completed in 2.00s (est. 194.0s remaining)

Batch 4/100: 10 subjects - completed in 2.42s (est. 163.2s remaining)

Batch 5/100: 10 subjects - completed in 1.73s (est. 142.3s remaining)

...

Batch 100/100: 10 subjects - completed in 0.53s (est. 0.0s remaining)

✓ Processed 1000 total subjects across 100 batches
✓ Total processing time: 156.20s (avg: 1.56s per batch)
```

## Features

### 1. **Per-Batch Timing**
Each batch shows completion time:
```
Batch 1/100: 10 subjects - completed in 0.33s
```

### 2. **Estimated Time Remaining** 
After the 3rd batch, shows estimated time to completion:
```
Batch 3/100: 10 subjects - completed in 2.00s (est. 194.0s remaining)
```

The estimate updates with each batch based on the average time per batch so far.

### 3. **Summary Statistics**
After all batches complete:
```
✓ Processed 1000 total subjects across 100 batches
✓ Total processing time: 156.20s (avg: 1.56s per batch)
```

## How It Works

1. **First 2 batches:** Shows only completion time (establishing baseline)
2. **Batch 3+:** Calculates running average and estimates remaining time
3. **Final summary:** Total time and average per batch

### Calculation

```python
# After each batch
avg_time_per_batch = sum(all_batch_times) / num_completed_batches
remaining_batches = total_batches - num_completed_batches
estimated_remaining = avg_time_per_batch * remaining_batches
```

## Enabling/Disabling

Timing information is shown when `verbose=True`:

```python
from ldk.analysis import FunctionalNetworkMapping

# Show timing (default)
analysis = FunctionalNetworkMapping(
    connectome_path="...",
    verbose=True  # Timing shown
)

# Hide timing
analysis = FunctionalNetworkMapping(
    connectome_path="...",
    verbose=False  # No timing output
)
```

## Use Cases

### 1. Planning Your Workflow
```
Batch 3/100: completed in 2.00s (est. 194.0s remaining)
```
→ "~3 minutes remaining, I can grab coffee" ☕

### 2. Performance Monitoring
```
✓ Total processing time: 156.20s (avg: 1.56s per batch)
```
→ Track performance across runs, identify slow batches

### 3. Capacity Planning
```
Average: 1.56s per batch × 100 batches = 156s total
```
→ Estimate resources needed for larger datasets

## Performance Expectations

Typical times per batch (with 10 subjects, 100 timepoints, 50k voxels):

| Lesion Batch Size | Time per Connectome Batch |
|-------------------|---------------------------|
| 1 lesion | 0.5-1.0s |
| 10 lesions | 0.8-1.5s |
| 20 lesions | 1.0-2.0s |
| 50 lesions | 1.5-3.0s |

**Note:** Times vary based on:
- CPU speed
- Number of voxels in connectome
- Number of timepoints
- Lesion size (number of voxels in lesion mask)

## Interpreting Results

### Fast Batches (< 1s)
✅ Good! System is performing well
- Consider increasing `lesion_batch_size` for faster total time

### Slow Batches (> 5s)
⚠️ May indicate:
- Very large connectome (high voxel count)
- Slow disk I/O (check if reading from network drive)
- Insufficient RAM (swapping to disk)

### Inconsistent Times
⚠️ Large variation between batches:
- First batch often slower (JIT compilation, cache warming)
- Background system activity
- Disk cache effects

## Example: Real Dataset

For the example dataset (96 lesions, 1000 subjects in 100 batches):

```
Expected output:
================
Batch 1/100: 10 subjects - completed in 1.23s
Batch 2/100: 10 subjects - completed in 1.45s
Batch 3/100: 10 subjects - completed in 1.67s (est. 155.0s remaining)
...
Batch 100/100: 10 subjects - completed in 1.52s (est. 0.0s remaining)

✓ Processed 1000 total subjects across 100 batches
✓ Total processing time: 152.30s (avg: 1.52s per batch)
```

**Interpretation:**
- ~2.5 minutes total processing time
- ~1.5 seconds per batch
- Consistent performance

## Tips

### 1. **Monitor First Few Batches**
The estimate stabilizes after ~5-10 batches. Early estimates may be inaccurate.

### 2. **Plan for Aggregation Time**
Batch processing time + aggregation time:
```
Batch processing: 152s (shown in timing)
Aggregation: ~5-10s (not included in estimate)
Total: ~157-162s
```

### 3. **Use for Optimization**
If seeing slow times:
1. Check CPU usage (`htop`)
2. Monitor memory (`free -h`)
3. Check disk I/O (`iotop`)
4. Consider reducing `lesion_batch_size`

## Code Location

Implementation in `src/ldk/analysis/functional_network_mapping.py`:

```python
def run_batch(self, lesion_data_list):
    # ...
    for batch_idx, connectome_path in enumerate(connectome_files):
        batch_start_time = time.time()
        
        # Process batch
        batch_r_maps = self._compute_batch_correlations_vectorized(...)
        
        # Update statistics
        # ...
        
        # Show timing
        batch_elapsed = time.time() - batch_start_time
        if self.verbose:
            if len(batch_times) > 2:
                avg_time = sum(batch_times) / len(batch_times)
                remaining = avg_time * (total_batches - completed_batches)
                print(f"completed in {batch_elapsed:.2f}s (est. {remaining:.1f}s remaining)")
```

## Summary

✅ **Real-time progress monitoring**  
✅ **Estimated completion time**  
✅ **Performance tracking**  
✅ **No performance overhead** (just prints to console)  
✅ **Always enabled when verbose=True**  

The timing information helps you:
- Know when processing will complete
- Monitor system performance
- Plan your workflow
- Identify performance issues
