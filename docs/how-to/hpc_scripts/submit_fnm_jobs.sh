#!/bin/bash

# --- Configuration ---
BATCH_SIZE=200
BIDS_DIR="/path/to/bids"

# --- Count subjects ---
num_subjects=$(find "$BIDS_DIR" -maxdepth 1 -name "sub-*" -type d | wc -l)

if [ "$num_subjects" -eq 0 ]; then
    echo "Error: No subjects found in $BIDS_DIR"
    exit 1
fi

# --- Calculate array limit ---
num_batches=$(( (num_subjects + BATCH_SIZE - 1) / BATCH_SIZE ))
array_limit=$(( num_batches - 1 ))

# --- Submit ---
echo "Found $num_subjects subjects."
echo "Batch size: $BATCH_SIZE"
echo "Submitting $num_batches jobs (array indices 0-$array_limit)."

sbatch --array=0-$array_limit --export=BATCH_SIZE=$BATCH_SIZE,ALL lacuna_fnm.batch
