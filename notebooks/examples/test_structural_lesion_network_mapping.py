from ldk import batch_process
from ldk.analysis import StructuralNetworkMapping
from ldk.io import load_bids_dataset

# Load all subjects
dataset = load_bids_dataset("/path/to/bids")
lesions = list(dataset.values())

# Configure analysis
analysis = StructuralNetworkMapping(
    tractogram_path="data/connectomes/dTOR_full_tractogram.tck",
    whole_brain_tdi="data/connectomes/dTOR_tdi_2mm.nii.gz",
    template="data/templates/MNI152_T1_2mm_Brain.nii.gz",
    n_jobs=10,
)

# Batch process all subjects
results = batch_process(lesions, analysis, n_jobs=4, show_progress=True)

# Export to CSV for statistics
from ldk.io import batch_export_to_csv

batch_export_to_csv(results, "sLNM_results.csv")
