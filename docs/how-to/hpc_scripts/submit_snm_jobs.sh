#!/bin/bash

# --- Configuration ---
BATCH_SIZE=20
BIDS_DIR="/path/to/bids"

# --- Determine subjects ---
# Pass subject names as arguments to process a subset, e.g.:
#   bash submit_snm_jobs.sh sub-001 sub-002 sub-003
# If no arguments are given, all subjects in BIDS_DIR are used.
if [ $# -gt 0 ]; then
    num_subjects=$#
    SUBJECT_LIST="$*"
else
    num_subjects=$(find "$BIDS_DIR" -maxdepth 1 -name "sub-*" -type d | wc -l)
    SUBJECT_LIST=""
fi

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

sbatch --array=0-$array_limit --export=BATCH_SIZE=$BATCH_SIZE,SUBJECT_LIST="$SUBJECT_LIST",ALL lacuna_snm.batch
