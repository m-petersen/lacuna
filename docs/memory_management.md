# Memory Management in Batch Processing

## Overview

The Lacuna provides sophisticated memory management capabilities for processing large numbers of lesions efficiently. This guide explains how to optimize memory usage while maintaining high performance.

## Key Concepts

### 1. Lesion Batch Size

The `lesion_batch_size` parameter controls how many lesions are processed together in each batch:

```python
from lacuna.batch import batch_process

results = batch_process(
    lesions,
    analysis,
    strategy='vectorized',
    lesion_batch_size=20,  # Process 20 lesions at a time
)
```

**Benefits:**
- **Memory Control**: Limits peak memory usage to batch_size × memory_per_lesion
- **Progress Tracking**: Enables monitoring progress through large datasets
- **Fault Isolation**: A failure in one batch doesn't affect others

**Guidelines:**
- For 50-100 lesions: Use `batch_size=20-30`
- For 100-500 lesions: Use `batch_size=10-20`
- For 500+ lesions: Use `batch_size=5-10`
- RAM consideration: ~2-4 GB per batch of 20 lesions (depends on connectome size)

### 2. Batch Result Callback

The `batch_result_callback` enables incremental result saving, freeing memory immediately after each batch:

```python
def save_batch_results(batch_results):
    """Save results immediately after batch processing."""
    for result in batch_results:
        subject_id = result.metadata['subject_id']
        # Save correlation map
        rmap_path = output_dir / f"{subject_id}_rmap.nii.gz"
        flnm_results = result.results['FunctionalNetworkMapping']
        nib.save(flnm_results['correlation_map'], rmap_path)

results = batch_process(
    lesions,
    analysis,
    strategy='vectorized',
    lesion_batch_size=20,
    batch_result_callback=save_batch_results,  # Save after each batch
)
```

**Benefits:**
- **Memory Efficiency**: Results saved to disk immediately, memory freed
- **Crash Recovery**: Partial results saved even if processing fails later
- **Real-time Output**: Files available during processing
- **Constant Memory**: Memory usage doesn't grow with total lesion count

### 3. Combined Strategy

For maximum efficiency with large datasets, combine both features:

```python
from pathlib import Path
import nibabel as nib
from lacuna.batch import batch_process

output_dir = Path("/path/to/output")
output_dir.mkdir(exist_ok=True)

def save_batch_results(batch_results):
    """Incrementally save results after each batch."""
    for result in batch_results:
        subject_id = result.metadata['subject_id']
        flnm_results = result.results.get('FunctionalNetworkMapping', {})
        
        # Save correlation map (r-map)
        if 'correlation_map' in flnm_results:
            rmap_path = output_dir / f"{subject_id}_rmap.nii.gz"
            nib.save(flnm_results['correlation_map'], rmap_path)
        
        # Save t-statistic map if computed
        if 't_map' in flnm_results:
            tmap_path = output_dir / f"{subject_id}_tmap.nii.gz"
            nib.save(flnm_results['t_map'], tmap_path)
        
        # Save binary threshold map if computed
        if 'binary_map' in flnm_results:
            binmap_path = output_dir / f"{subject_id}_binary.nii.gz"
            nib.save(flnm_results['binary_map'], binmap_path)
        
        print(f"  ✓ Saved {subject_id}")

# Process with memory-controlled batching and incremental saving
results = batch_process(
    lesions,
    analysis,
    strategy='vectorized',
    lesion_batch_size=20,  # Control memory
    batch_result_callback=save_batch_results,  # Save incrementally
    show_progress=True,
)
```

## Memory Usage Examples

### Example 1: Small Dataset (10-50 lesions)

```python
# Simple approach - process all together
results = batch_process(lesions, analysis, strategy='vectorized')
```

**Memory usage:** Peak = all_lesions × memory_per_lesion (~10-20 GB for 50 lesions)

### Example 2: Medium Dataset (50-200 lesions)

```python
# Balanced approach - batch with saving
results = batch_process(
    lesions,
    analysis,
    strategy='vectorized',
    lesion_batch_size=20,
    batch_result_callback=save_batch_results,
)
```

**Memory usage:** Peak = 20 × memory_per_lesion (~8-16 GB constant)

### Example 3: Large Dataset (500+ lesions)

```python
# Memory-efficient approach - small batches
results = batch_process(
    lesions,
    analysis,
    strategy='vectorized',
    lesion_batch_size=10,
    batch_result_callback=save_batch_results,
)
```

**Memory usage:** Peak = 10 × memory_per_lesion (~4-8 GB constant)

## Performance Trade-offs

| Batch Size | Memory Usage | Processing Speed | Best For |
|------------|--------------|------------------|----------|
| None (all) | Very High    | Fastest          | < 50 lesions |
| 50-100     | High         | Fast             | 50-100 lesions |
| 20-30      | Medium       | Good             | 100-500 lesions |
| 10-20      | Low          | Acceptable       | 500+ lesions |
| 5-10       | Very Low     | Slower           | Limited RAM |

## Best Practices

1. **Always use callback for large datasets:**
   - For 100+ lesions, always provide `batch_result_callback`
   - Without callback, all results accumulate in memory

2. **Monitor memory during first run:**
   - Start with conservative batch_size (10-20)
   - Monitor with `htop` or similar tool
   - Increase batch_size if memory allows

3. **Save all result types:**
   - Correlation maps (r-maps) are always computed
   - T-statistic maps if `compute_t_map=True`
   - Binary threshold maps if `t_threshold` is set

4. **Error handling in callbacks:**
```python
def save_batch_results(batch_results):
    for result in batch_results:
        try:
            # Save logic here
            pass
        except Exception as e:
            print(f"Warning: Failed to save {result.metadata['subject_id']}: {e}")
            # Continue processing other results
```

5. **Progress tracking:**
```python
batch_count = 0

def save_batch_results(batch_results):
    global batch_count
    batch_count += 1
    print(f"Completed batch {batch_count} ({len(batch_results)} lesions)")
    # Save logic here
```

## Troubleshooting

### Out of Memory Errors

**Problem:** Process killed due to memory exhaustion

**Solution:**
1. Reduce `lesion_batch_size` (try 10, 5, or even 2)
2. Ensure `batch_result_callback` is provided
3. Check connectome size - larger connectomes need smaller batches

### Slow Processing

**Problem:** Processing takes too long

**Solution:**
1. Increase `lesion_batch_size` if memory allows
2. Ensure you're using `strategy='vectorized'`
3. Check disk I/O isn't bottleneck (callback writing to slow disk)

### Empty Results Directory

**Problem:** No files saved despite callback

**Solution:**
1. Check callback is actually saving files
2. Verify output directory exists and is writable
3. Check for exceptions in callback (add try-except)
4. Ensure results contain expected keys ('correlation_map', etc.)

## Implementation Example

See `test_batch_flnm.py` for a complete working example that demonstrates:
- Loading lesions from directory
- Creating analysis with appropriate settings
- Using lesion_batch_size and batch_result_callback
- Conditional logic for batched vs. non-batched processing
- Comprehensive result saving

```python
# From test_batch_flnm.py
LESION_BATCH_SIZE = 20  # Configure batch size

def save_batch_results(batch_results: list[LesionData]) -> None:
    """Save results after each batch to free memory."""
    for lesion_data in batch_results:
        # Save all result types
        # ...

# Process with memory management
if USE_VECTORIZED and LESION_BATCH_SIZE:
    print(f"🔄 Processing with batch_size={LESION_BATCH_SIZE} (incremental saving)")
    results = batch_process(
        lesions,
        analysis,
        strategy="vectorized",
        lesion_batch_size=LESION_BATCH_SIZE,
        batch_result_callback=save_batch_results,
    )
```

## Technical Details

### How It Works

1. **Batch Formation:**
   - Input lesions split into chunks of size `lesion_batch_size`
   - Each chunk processed independently

2. **Vectorized Processing:**
   - Within each batch, Einstein summation for correlation computation
   - Optimized linear algebra (BLAS/LAPACK) for speed
   - Memory allocated once per batch, reused for all lesions

3. **Callback Execution:**
   - Called immediately after each batch completes
   - Receives list of LesionData objects with results attached
   - Callback can save, analyze, or transform results
   - Return value ignored (side effects only)

4. **Memory Lifecycle:**
   ```
   Loop: for each batch
     ├─ Load batch of lesions
     ├─ Compute correlations (vectorized)
     ├─ Call callback(batch_results)  ← Results saved here
     ├─ Release batch memory
     └─ Continue to next batch
   ```

### Callback Signature

```python
from typing import List
from lacuna.core.lesion_data import LesionData

def batch_result_callback(batch_results: List[LesionData]) -> None:
    """
    Process results after each batch completes.
    
    Parameters
    ----------
    batch_results : List[LesionData]
        List of LesionData objects with analysis results attached.
        Each object has .results dict with analysis outputs.
    
    Returns
    -------
    None
        Callback is invoked for side effects (e.g., saving files).
    """
    pass
```

## See Also

- [Batch Processing Guide](batch_processing.md)
- [Vectorized Strategy Documentation](vectorized_strategy.md)
- [Performance Optimization](performance.md)
