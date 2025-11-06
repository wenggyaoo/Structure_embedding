#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
import time
import os
import gc
import pickle
import sys
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm


sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Import real structure embedding methods
from real_space2_method import create_real_space2_embeddings
from real_protein_holography import create_real_protein_holography_embeddings
from real_antiberta2_cssp_method import create_antiberta2_cssp_structure_embeddings
from unified_structure_embedding import create_unified_structure_embeddings


# Set up logging - simple version that works with SLURM
def setup_logging(output_dir="logs"):
    """Setup logging that works properly with SLURM and local environments"""
    # Check if running in SLURM environment
    slurm_job_id = os.environ.get('SLURM_JOB_ID')

    if slurm_job_id:
        # Running in SLURM - only use console output (goes to SLURM log files)
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',  # Simplified format for SLURM
            handlers=[logging.StreamHandler()],
            force=True
        )
        logger = logging.getLogger(__name__)
        print(f"SLURM job {slurm_job_id} started")
        print(f"Node: {os.environ.get('SLURM_NODELIST', 'Unknown')}")
        return logger, f"SLURM_job_{slurm_job_id}"
    else:
        # Running locally - create log files
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"structure_embedding_{timestamp}.log"
        log_path = os.path.join(output_dir, log_filename)

        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',  # Simplified format for better readability
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True
        )

        logger = logging.getLogger(__name__)
        print(f"Local run - Log file: {log_path}")
        return logger, log_path

# Initialize logger (will be properly set up in main())
logger = None


class PerformanceTracker:
    """Track and compare performance between different structure backends"""

    def __init__(self):
        self.timing_data = {}
        self.method_stats = {}

    def start_timing(self, method_name: str, backend: str):
        """Start timing for a method with specific backend"""
        key = f"{method_name}_{backend}"
        start_time = time.time()
        self.timing_data[key] = {
            'start_time': start_time,
            'method': method_name,
            'backend': backend,
            'sequences_processed': 0
        }

        # Log start time
        start_timestamp = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"PERFORMANCE_TIMING: {method_name.upper()}_{backend.upper()}_START at {start_timestamp}")

    def end_timing(self, method_name: str, backend: str, sequences_count: int, success: bool = True):
        """End timing and record results"""
        key = f"{method_name}_{backend}"
        if key in self.timing_data:
            end_time = time.time()
            total_time = end_time - self.timing_data[key]['start_time']
            time_per_sequence = total_time / sequences_count if sequences_count > 0 else 0

            self.timing_data[key].update({
                'end_time': end_time,
                'total_time': total_time,
                'sequences_processed': sequences_count,
                'time_per_sequence': time_per_sequence,
                'success': success
            })

            # Log detailed timing results
            end_timestamp = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
            print(f"PERFORMANCE_TIMING: {method_name.upper()}_{backend.upper()}_END at {end_timestamp}")
            print(f"PERFORMANCE_RESULT: {method_name.upper()}_{backend.upper()} - "
                       f"Total: {total_time:.2f}s, "
                       f"Per_sequence: {time_per_sequence:.2f}s, "
                       f"Sequences: {sequences_count}, "
                       f"Success: {success}")

            # Store in method stats for comparison
            if method_name not in self.method_stats:
                self.method_stats[method_name] = {}

            self.method_stats[method_name][backend] = {
                'total_time': total_time,
                'sequences_processed': sequences_count,
                'time_per_sequence': time_per_sequence,
                'success': success
            }

    def get_simple_report(self) -> str:
        """Generate a simple performance report for current run"""
        if not self.method_stats:
            return "No performance data available."

        report = []
        report.append("PERFORMANCE SUMMARY")
        report.append("="*60)

        for method_name, backends in self.method_stats.items():
            for backend_name, stats in backends.items():
                success_status = "SUCCESS" if stats.get('success', True) else "FAILED"
                report.append(f"{method_name.upper()} ({backend_name.upper()}):")
                report.append(f"  Total time: {stats['total_time']:.2f}s")
                report.append(f"  Time per sequence: {stats['time_per_sequence']:.3f}s")
                report.append(f"  Sequences processed: {stats['sequences_processed']}")
                report.append(f"  Status: {success_status}")
                report.append("")

        return "\n".join(report)

    def save_results(self, output_dir: str, log_path: str = None):
        """Save basic timing results"""
        # Create a simple performance log entry
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Append to performance log file
        perf_log_file = os.path.join(output_dir, 'performance_log.txt')
        with open(perf_log_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Performance Log Entry - {timestamp}\n")
            f.write(f"{'='*60}\n")
            f.write(self.get_simple_report())
            f.write(f"Log file: {log_path}\n")
            f.write(f"{'='*60}\n")

        print(f"Performance data appended to: {perf_log_file}")


# Global performance tracker
performance_tracker = PerformanceTracker()


def save_checkpoint(data, checkpoint_path):
    """Save checkpoint data"""
    try:
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Checkpoint saved: {checkpoint_path}")
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")

def load_checkpoint(checkpoint_path):
    """Load checkpoint data"""
    try:
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Checkpoint loaded: {checkpoint_path}")
            return data
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
    return None

def load_bcr_dataset_sequences():
    """
    Load BCR sequences from the unified dataset - FIXED VERSION
    """
    print("="*80)
    print("LOADING BCR AIRR DATASET")
    print("="*80)

    # Load the dataset - fix path for current working directory
    dataset_path = 'dataset/bcr_airr_dataset.tsv'
    if not os.path.exists(dataset_path):
        dataset_path = '../dataset/bcr_airr_dataset.tsv'

    df = pd.read_csv(dataset_path, sep='\t')
    print(f"Loaded {len(df)} total sequences from dataset")

    # Filter for productive sequences only
    df = df[df['productive'] == True].copy()
    print(f"After filtering productive: {len(df)} sequences")

    # Remove sequences with missing specificity
    df = df[df['specificity'].notna()].copy()
    print(f"After removing missing specificity: {len(df)} sequences")

    # Add sequence quality filtering
    def is_valid_sequence(seq):
        if pd.isna(seq) or len(seq) < 50:  # Too short
            return False
        if len(seq) > 500:  # Too long
            return False
        # Check for non-standard amino acids (keep only standard 20)
        standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
        if not all(aa in standard_aa for aa in seq.upper()):
            return False
        return True

    # Filter sequences by quality
    initial_count = len(df)
    df = df[df['sequence_vdj_aa'].apply(is_valid_sequence)].copy()
    print(f"After sequence quality filtering: {len(df)} sequences (removed {initial_count - len(df)} low-quality sequences)")

    # Extract heavy and light chain sequences
    heavy_chains = df[df['chain_type'] == 'H'].copy()
    light_chains = df[df['chain_type'] == 'L'].copy()
    print(f"Heavy chain sequences: {len(heavy_chains)}")
    print(f"Light chain sequences: {len(light_chains)}")

    # Verify we have the expected number of unique cells
    unique_cells = df['cell_id'].nunique()
    print(f"Unique cells in dataset: {unique_cells}")

    # Check specificity distribution
    cell_specificity = heavy_chains.groupby('cell_id')['specificity'].first()
    spec_counts = cell_specificity.value_counts()
    print(f"")
    print(f"DATASET SUMMARY:")
    print(f"  Original dataset: {unique_cells} cells")
    s_plus = spec_counts.get('S+', 0)
    s_minus = spec_counts.get('S-', 0)
    print(f"  S+ (positive): {s_plus} cells")
    print(f"  S- (negative): {s_minus} cells")
    print(f"  Total: {s_plus + s_minus} cells ({s_plus + s_minus} antibody pairs)")
    print(f"")

    # Create paired sequence mapping with metadata (Heavy + Light as complete antibodies)
    # This matches the format of your successful embedding file: 858683_heavy_4424_light
    sequence_data = []

    # Process ALL cells - should be 15,538 paired antibodies
    print("Creating Heavy+Light paired antibodies...")
    for cell_id in heavy_chains['cell_id'].unique():
        heavy_row = heavy_chains[heavy_chains['cell_id'] == cell_id]
        light_row = light_chains[light_chains['cell_id'] == cell_id]

        # Handle different pairing scenarios
        if len(heavy_row) == 1 and len(light_row) == 1:
            # Perfect pairing (expected for this dataset)
            heavy_data = heavy_row.iloc[0]
            light_data = light_row.iloc[0]

            sequence_data.append({
                'sequence_id': f"{heavy_data['sequence_id']}_{light_data['sequence_id']}",
                'heavy_chain': heavy_data['sequence_vdj_aa'],
                'light_chain': light_data['sequence_vdj_aa'],
                'specificity': heavy_data['specificity'],
                'subject': heavy_data['subject'],
                'cell_id': cell_id,
                'heavy_id': heavy_data['sequence_id'],
                'light_id': light_data['sequence_id']
            })
        elif len(heavy_row) == 1 and len(light_row) == 0:
            # Heavy chain only
            heavy_data = heavy_row.iloc[0]
            sequence_data.append({
                'sequence_id': f"{heavy_data['sequence_id']}_only",
                'heavy_chain': heavy_data['sequence_vdj_aa'],
                'light_chain': '',  # No light chain
                'specificity': heavy_data['specificity'],
                'subject': heavy_data['subject'],
                'cell_id': cell_id,
                'heavy_id': heavy_data['sequence_id'],
                'light_id': None
            })
        elif len(heavy_row) > 1 or len(light_row) > 1:
            # Multiple chains per cell - take the first one of each type
            print(f"Cell {cell_id} has {len(heavy_row)} heavy and {len(light_row)} light chains. Taking first of each.")

            if len(heavy_row) > 0:
                heavy_data = heavy_row.iloc[0]

                if len(light_row) > 0:
                    light_data = light_row.iloc[0]
                    sequence_data.append({
                        'sequence_id': f"{heavy_data['sequence_id']}_{light_data['sequence_id']}",
                        'heavy_chain': heavy_data['sequence_vdj_aa'],
                        'light_chain': light_data['sequence_vdj_aa'],
                        'specificity': heavy_data['specificity'],
                        'subject': heavy_data['subject'],
                        'cell_id': cell_id,
                        'heavy_id': heavy_data['sequence_id'],
                        'light_id': light_data['sequence_id']
                    })
                else:
                    sequence_data.append({
                        'sequence_id': f"{heavy_data['sequence_id']}_only",
                        'heavy_chain': heavy_data['sequence_vdj_aa'],
                        'light_chain': '',
                        'specificity': heavy_data['specificity'],
                        'subject': heavy_data['subject'],
                        'cell_id': cell_id,
                        'heavy_id': heavy_data['sequence_id'],
                        'light_id': None
                    })
        else:
            # No heavy chain for this cell (shouldn't happen)
            print(f"Cell {cell_id} has no heavy chain. Skipping.")

    print(f"Final paired dataset: {len(sequence_data)} antibody pairs")
    paired_count = sum(1 for seq in sequence_data if seq['light_chain'])
    heavy_only_count = len(sequence_data) - paired_count
    print(f"  - Paired (H+L): {paired_count}")
    print(f"  - Heavy only: {heavy_only_count}")

    # Verify we got all expected cells
    final_spec_counts = {}
    for seq in sequence_data:
        spec = seq['specificity']
        final_spec_counts[spec] = final_spec_counts.get(spec, 0) + 1

    print(f"Final specificity distribution:")
    for spec, count in final_spec_counts.items():
        print(f"  {spec}: {count}")

    # Ensure we have the expected 15,538 paired antibodies
    expected_total = 15538
    if len(sequence_data) != expected_total:
        print(f"Expected {expected_total} paired antibodies but got {len(sequence_data)}!")
        print("This indicates a problem with the data loading logic.")
    else:
        print(f"Successfully loaded all {expected_total} expected paired antibodies")

    return sequence_data

def generate_and_save_real_space2_embeddings(sequence_data, output_dir, batch_size=10, use_imgt_numbering=True, structure_backend="igfold"):
    """
    Generate real SPACE2 structure embeddings and save to TSV with checkpoint support

    Args:
        sequence_data: List of sequence dictionaries
        output_dir: Output directory for embeddings
        batch_size: Batch size for processing
        use_imgt_numbering: Whether to use IMGT numbering for optimal SPACE2 performance
        structure_backend: Structure prediction backend ("igfold" or "abodybuilder2")
    """
    print("\n" + "="*60)
    print(f"GENERATING REAL SPACE2 STRUCTURE EMBEDDINGS ({structure_backend.upper()} BACKEND)")
    print("="*60)

    # Start performance tracking
    performance_tracker.start_timing("space2", structure_backend)

    print(f"Processing all {len(sequence_data)} sequences for SPACE2")

    # Setup checkpoint
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'space2_{structure_backend}_checkpoint.pkl')

    # Load existing checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path)
    if checkpoint_data:
        processed_batches = checkpoint_data.get('processed_batches', 0)
        all_embeddings = checkpoint_data.get('embeddings', [])
        processed_sequences = processed_batches * batch_size
        print(f"RESUMING FROM CHECKPOINT:")
        print(f"  Processed batches: {processed_batches}")
        print(f"  Processed sequences: {processed_sequences}")
        print(f"  Remaining sequences: {len(sequence_data) - processed_sequences}")
    else:
        processed_batches = 0
        all_embeddings = []
        print(f"STARTING FROM BEGINNING")

    # Pass the full sequence data (with heavy and light chains) to SPACE2
    sequences = sequence_data
    total_batches = (len(sequences) - 1) // batch_size + 1

    print(f"")
    print(f"PROCESSING PLAN:")
    print(f"  Total sequences: {len(sequences)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total batches: {total_batches}")
    print(f"  Remaining batches: {total_batches - processed_batches}")

    # Estimate processing time based on previous runs
    if processed_batches > 0:
        # We have checkpoint data, can estimate based on previous performance
        avg_time_per_batch = 25  # seconds (estimated from log analysis)
        remaining_batches = total_batches - processed_batches
        estimated_hours = (remaining_batches * avg_time_per_batch) / 3600
        print(f"  Estimated remaining time: {estimated_hours:.1f} hours")
        print(f"  (Based on average {avg_time_per_batch} seconds/batch)")
    else:
        # First run, provide general estimate
        avg_time_per_batch = 25  # seconds
        estimated_hours = (total_batches * avg_time_per_batch) / 3600
        print(f"  Estimated total time: {estimated_hours:.1f} hours")
        print(f"  (Based on estimated {avg_time_per_batch} seconds/batch)")

    print(f"")

    # Process remaining batches with progress bar
    remaining_batches = total_batches - processed_batches
    print(f"Starting batch processing with progress bar...")

    # Create progress bar
    pbar = tqdm(
        range(processed_batches, total_batches),
        desc="SPACE2 Embedding",
        initial=processed_batches,
        total=total_batches,
        unit="batch",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    for batch_idx in pbar:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]

        # Calculate progress
        current_sequence = batch_idx * batch_size + 1
        end_sequence = min(current_sequence + len(batch_sequences) - 1, len(sequences))
        progress_percent = ((batch_idx + 1) / total_batches) * 100

        # Update progress bar description
        pbar.set_description(f"SPACE2 Batch {batch_idx + 1}/{total_batches} ({progress_percent:.1f}%)")

        # Log detailed info every 50 batches or at start
        if batch_idx % 50 == 0 or batch_idx == processed_batches:
            print(f"")
            print(f"Processing SPACE2 batch {batch_idx + 1}/{total_batches}")
            print(f"  Sequence range: {current_sequence}-{end_sequence}")
            print(f"  Batch size: {len(batch_sequences)}")
            print(f"  Overall progress: {progress_percent:.1f}%")

        try:
            start_time = time.time()

            # Clear memory before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # batch_sequences already contains paired antibody data in the correct format
            # Each item has 'heavy_chain' and 'light_chain' fields
            batch_embeddings = create_real_space2_embeddings(batch_sequences, device='cpu', use_imgt_numbering=use_imgt_numbering, structure_backend=structure_backend)
            generation_time = time.time() - start_time

            if batch_embeddings is not None:
                all_embeddings.append(batch_embeddings)

                # Calculate processing speed
                sequences_per_second = len(batch_sequences) / generation_time
                total_processed = (batch_idx + 1) * batch_size
                remaining_sequences = len(sequences) - total_processed
                estimated_remaining_time = remaining_sequences / sequences_per_second / 3600  # hours

                print(f"SPACE2 batch {batch_idx + 1} completed!")
                print(f"  Processing time: {generation_time:.2f} seconds")
                print(f"  Processing speed: {sequences_per_second:.2f} sequences/second")
                print(f"  Completed: {total_processed}/{len(sequences)} sequences")
                print(f"  Estimated remaining time: {estimated_remaining_time:.1f} hours")

                # Save checkpoint
                checkpoint_data = {
                    'processed_batches': batch_idx + 1,
                    'embeddings': all_embeddings,
                    'total_sequences_processed': end_idx
                }
                save_checkpoint(checkpoint_data, checkpoint_path)

                # Save intermediate results
                if all_embeddings:
                    combined_embeddings = np.vstack(all_embeddings)
                    processed_data = sequence_data[:len(combined_embeddings)]

                    embedding_df = pd.DataFrame(combined_embeddings)
                    embedding_df.index = [item['sequence_id'] for item in processed_data]

                    temp_output_path = os.path.join(output_dir, f'real_space2_{structure_backend}_structure_embedding_temp.tsv')
                    embedding_df.to_csv(temp_output_path, sep='\t')
                    print(f"SPACE2 intermediate results saved: {len(combined_embeddings)} embeddings")
                    print(f"  Temporary file: {temp_output_path}")
            else:
                print(f"SPACE2 batch {batch_idx + 1} failed - skipping")

        except Exception as e:
            print(f"Error processing SPACE2 batch {batch_idx + 1}: {e}")
            import traceback
            print(f"  Detailed error: {traceback.format_exc()}")
            continue

        # Every 100 batches, log a summary
        if (batch_idx + 1) % 100 == 0:
            total_processed = (batch_idx + 1) * batch_size
            print(f"")
            print(f"MILESTONE: Completed {batch_idx + 1} batches")
            print(f"  Processed sequences: {total_processed}/{len(sequences)}")
            print(f"  Completion percentage: {(total_processed/len(sequences)*100):.1f}%")
            print(f"")

    # Close progress bar
    pbar.close()
    print(f"Batch processing completed!")

    # Final processing
    if all_embeddings:
        embeddings = np.vstack(all_embeddings)
        print(f"Generated real SPACE2 embeddings: {embeddings.shape}")

        # Create DataFrame with sequence IDs and embeddings
        print(f"DEBUG: About to save embeddings with shape: {embeddings.shape}")
        print(f"DEBUG: Embeddings stats - Min: {np.min(embeddings):.6f}, Max: {np.max(embeddings):.6f}, Mean: {np.mean(embeddings):.6f}")
        print(f"DEBUG: Non-zero count: {np.count_nonzero(embeddings)}/{embeddings.size}")

        embedding_df = pd.DataFrame(embeddings)
        embedding_df.index = [item['sequence_id'] for item in sequence_data[:len(embeddings)]]

        # Save final results with backend-specific naming
        output_path = os.path.join(output_dir, f'real_space2_{structure_backend}_structure_embedding.tsv')
        embedding_df.to_csv(output_path, sep='\t')
        print(f"Real SPACE2 embeddings saved to: {output_path}")

        # DEBUG: Verify saved file
        print(f"DEBUG: Verifying saved file...")
        saved_df = pd.read_csv(output_path, sep='\t', index_col=0)
        print(f"DEBUG: Saved file shape: {saved_df.shape}")
        print(f"DEBUG: Saved file stats - Min: {saved_df.values.min():.6f}, Max: {saved_df.values.max():.6f}, Mean: {saved_df.values.mean():.6f}")

        # Clean up
        temp_path = os.path.join(output_dir, f'real_space2_{structure_backend}_structure_embedding_temp.tsv')
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        # End performance tracking
        performance_tracker.end_timing("space2", structure_backend, len(embeddings), success=True)

        return True
    else:
        print("Failed to generate real SPACE2 embeddings")
        # End performance tracking with failure
        performance_tracker.end_timing("space2", structure_backend, 0, success=False)
        return False

def generate_and_save_real_holography_embeddings(sequence_data, output_dir, batch_size=10, use_imgt_numbering=True, structure_backend="igfold"):
    """
    Generate real Protein Holography structure embeddings and save to TSV with checkpoint support

    Args:
        sequence_data: List of sequence dictionaries
        output_dir: Output directory for embeddings
        batch_size: Batch size for processing
        use_imgt_numbering: Whether to use IMGT numbering
        structure_backend: Structure prediction backend ("igfold" or "abodybuilder2")
    """
    print("\n" + "="*60)
    print(f"GENERATING REAL PROTEIN HOLOGRAPHY STRUCTURE EMBEDDINGS ({structure_backend.upper()} BACKEND)")
    print("="*60)

    # Start performance tracking
    performance_tracker.start_timing("holography", structure_backend)

    print(f"Processing all {len(sequence_data)} sequences for Protein Holography")

    # Setup checkpoint
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'holography_{structure_backend}_checkpoint.pkl')

    # Load existing checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path)
    if checkpoint_data:
        processed_batches = checkpoint_data.get('processed_batches', 0)
        all_embeddings = checkpoint_data.get('embeddings', [])
        print(f"Resuming Holography from batch {processed_batches}")
    else:
        processed_batches = 0
        all_embeddings = []

    # Pass the full sequence data (with heavy and light chains) to Holography
    sequences = sequence_data
    total_batches = (len(sequences) - 1) // batch_size + 1

    # Process remaining batches
    for batch_idx in range(processed_batches, total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]

        print(f"Processing Holography batch {batch_idx + 1}/{total_batches} ({len(batch_sequences)} sequences)")

        try:
            start_time = time.time()

            # Clear memory before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            batch_embeddings = create_real_protein_holography_embeddings(batch_sequences, device='cpu', use_imgt_numbering=use_imgt_numbering, structure_backend=structure_backend)
            generation_time = time.time() - start_time

            if batch_embeddings is not None:
                all_embeddings.append(batch_embeddings)
                print(f"Holography batch {batch_idx + 1} completed in {generation_time:.2f} seconds")

                # Save checkpoint
                checkpoint_data = {
                    'processed_batches': batch_idx + 1,
                    'embeddings': all_embeddings,
                    'total_sequences_processed': end_idx
                }
                save_checkpoint(checkpoint_data, checkpoint_path)

                # Save intermediate results
                if all_embeddings:
                    combined_embeddings = np.vstack(all_embeddings)
                    processed_data = sequence_data[:len(combined_embeddings)]

                    embedding_df = pd.DataFrame(combined_embeddings)
                    embedding_df.index = [item['sequence_id'] for item in processed_data]

                    temp_output_path = os.path.join(output_dir, f'real_holography_{structure_backend}_structure_embedding_temp.tsv')
                    embedding_df.to_csv(temp_output_path, sep='\t')
                    print(f"Holography intermediate results: {len(combined_embeddings)} embeddings")
            else:
                print(f"Holography batch {batch_idx + 1} failed - skipping")

        except Exception as e:
            print(f"Error processing Holography batch {batch_idx + 1}: {e}")
            continue

    # Final processing
    if all_embeddings:
        embeddings = np.vstack(all_embeddings)
        print(f"Generated real Protein Holography embeddings: {embeddings.shape}")

        # Create DataFrame with sequence IDs and embeddings
        embedding_df = pd.DataFrame(embeddings)
        embedding_df.index = [item['sequence_id'] for item in sequence_data[:len(embeddings)]]

        # Save final results with backend-specific naming
        output_path = os.path.join(output_dir, f'real_holography_{structure_backend}_structure_embedding.tsv')
        embedding_df.to_csv(output_path, sep='\t')
        print(f"Real Protein Holography embeddings saved to: {output_path}")

        # Clean up
        temp_path = os.path.join(output_dir, f'real_holography_{structure_backend}_structure_embedding_temp.tsv')
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        # End performance tracking
        performance_tracker.end_timing("holography", structure_backend, len(embeddings), success=True)

        return True
    else:
        print("Failed to generate real Protein Holography embeddings")
        # End performance tracking with failure
        performance_tracker.end_timing("holography", structure_backend, 0, success=False)
        return False

def generate_and_save_antiberta2_embeddings(sequence_data, output_dir, batch_size=10, max_sequences=None, use_imgt_numbering=True, structure_backend="igfold"):
    """
    Generate AntiBERTa2-CSSP structure-aware embeddings following official methodology and save to TSV with batch processing

    Official AntiBERTa2-CSSP approach:
    - Uses IMGT numbering for antibody alignment
    - Incorporates CDRH3 loop structure similarity (RMSD between backbone atoms)
    - Applies dynamic time warp algorithm for length-independent alignment
    - Uses [CLS] token embeddings with latent structural information

    Args:
        sequence_data: List of sequence dictionaries
        output_dir: Output directory for embeddings
        batch_size: Batch size for processing
        max_sequences: Maximum number of sequences to process
        use_imgt_numbering: Whether to use IMGT numbering (essential for AntiBERTa2-CSSP)
        structure_backend: Structure prediction backend ("igfold" or "abodybuilder2")
    """
    print("\n" + "="*60)
    print(f"GENERATING ANTIBERTA2-CSSP STRUCTURE-AWARE EMBEDDINGS ({structure_backend.upper()} BACKEND)")
    print("="*60)
    print("Following official methodology:")
    print(f"{structure_backend.upper()} structure generation with IMGT renumbering")
    if structure_backend == "igfold":
        print("PyRosetta refinement for structural accuracy")
    print("[CLS] token embeddings with latent structural information")

    # Start performance tracking
    performance_tracker.start_timing("antiberta2-cssp", structure_backend)
    print("CDRH3 loop structure similarity training")

    # Limit sequences if specified
    if max_sequences and len(sequence_data) > max_sequences:
        print(f"Limiting to first {max_sequences} sequences for AntiBERTa2-CSSP")
        sequence_data = sequence_data[:max_sequences]
    else:
        print(f"Processing all {len(sequence_data)} sequences for AntiBERTa2-CSSP")

    # Setup checkpoint
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'antiberta2_cssp_{structure_backend}_checkpoint.pkl')

    # Load existing checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path)
    if checkpoint_data:
        processed_batches = checkpoint_data.get('processed_batches', 0)
        all_embeddings = checkpoint_data.get('embeddings', [])
        print(f"Resuming AntiBERTa2-CSSP from batch {processed_batches}")
    else:
        processed_batches = 0
        all_embeddings = []
        print("Starting AntiBERTa2-CSSP from beginning")

    # Process in batches
    total_batches = (len(sequence_data) + batch_size - 1) // batch_size
    failed_count = 0

    for batch_idx in range(processed_batches, total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sequence_data))
        batch_sequences = sequence_data[start_idx:end_idx]

        print(f"Processing AntiBERTa2-CSSP batch {batch_idx + 1}/{total_batches} (sequences {start_idx + 1}-{end_idx})")

        try:
            start_time = time.time()
            # Use the new structure-aware version with selected backend
            batch_embeddings = create_antiberta2_cssp_structure_embeddings(
                batch_sequences,
                device='cpu',
                structure_backend=structure_backend,  # Use selected backend
                do_refine=True,       # Enable PyRosetta refinement (IgFold only)
                do_renum=True         # Enable IMGT renumbering (critical for AntiBERTa2-CSSP)
            )
            batch_time = time.time() - start_time

            if batch_embeddings is not None and len(batch_embeddings) > 0:
                # Convert to list if it's a numpy array
                if isinstance(batch_embeddings, np.ndarray):
                    batch_embeddings_list = batch_embeddings.tolist()
                else:
                    batch_embeddings_list = batch_embeddings

                all_embeddings.extend(batch_embeddings_list)
                print(f"AntiBERTa2-CSSP batch {batch_idx + 1} completed in {batch_time:.2f}s: {len(batch_embeddings)} embeddings")

                # Save checkpoint (fix parameter order)
                checkpoint_data = {
                    'processed_batches': batch_idx + 1,
                    'embeddings': all_embeddings,
                    'total_sequences': len(sequence_data)
                }
                save_checkpoint(checkpoint_data, checkpoint_path)

                # Save intermediate results
                if all_embeddings:
                    combined_embeddings = np.vstack(all_embeddings)
                    processed_data = sequence_data[:len(combined_embeddings)]

                    embedding_df = pd.DataFrame(combined_embeddings)
                    embedding_df.index = [item['sequence_id'] for item in processed_data]

                    temp_output_path = os.path.join(output_dir, f'real_antiberta2_cssp_{structure_backend}_structure_embedding_temp.tsv')
                    embedding_df.to_csv(temp_output_path, sep='\t')
                    print(f"AntiBERTa2-CSSP intermediate results: {len(combined_embeddings)} embeddings")
            else:
                print(f"AntiBERTa2-CSSP batch {batch_idx + 1} failed - skipping")

        except Exception as e:
            print(f"AntiBERTa2-CSSP batch {batch_idx + 1} failed with error: {e}")
            failed_count += 1
            continue

    # Final processing
    if all_embeddings:
        # Convert all embeddings to numpy array
        try:
            # all_embeddings is a list of lists, convert to numpy array
            embeddings = np.array(all_embeddings)
            print(f"Generated real AntiBERTa2-CSSP embeddings: {embeddings.shape}")
        except Exception as e:
            print(f"Failed to convert embeddings to numpy array: {e}")
            print(f"all_embeddings type: {type(all_embeddings)}")
            if all_embeddings:
                print(f"First embedding type: {type(all_embeddings[0])}")
                print(f"First embedding shape: {np.array(all_embeddings[0]).shape if hasattr(all_embeddings[0], '__len__') else 'scalar'}")
            return False

        # Create DataFrame with sequence IDs and embeddings
        embedding_df = pd.DataFrame(embeddings)
        embedding_df.index = [item['sequence_id'] for item in sequence_data[:len(embeddings)]]

        # Save final results with backend-specific naming
        output_path = os.path.join(output_dir, f'real_antiberta2_cssp_{structure_backend}_structure_embedding.tsv')
        embedding_df.to_csv(output_path, sep='\t')
        print(f"Real AntiBERTa2-CSSP embeddings saved to: {output_path}")

        # Clean up
        temp_path = os.path.join(output_dir, f'real_antiberta2_cssp_{structure_backend}_structure_embedding_temp.tsv')
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        # End performance tracking
        performance_tracker.end_timing("antiberta2-cssp", structure_backend, len(embeddings), success=True)
        print(f"AntiBERTa2-CSSP completed successfully. Failed batches: {failed_count}")
        return True
    else:
        print("Failed to generate real AntiBERTa2-CSSP embeddings")
        # End performance tracking with failure
        performance_tracker.end_timing("antiberta2-cssp", structure_backend, 0, success=False)
        return False

def create_embedding_summary(sequence_data, output_dir):
    """
    Create a summary file mapping sequence IDs to metadata
    """
    print("\nCreating embedding summary...")
    
    summary_data = []
    for item in sequence_data:
        summary_data.append({
            'sequence_id': item['sequence_id'],
            'specificity': item['specificity'],
            'subject': item['subject'],
            'cell_id': item['cell_id'],
            'heavy_chain_length': len(item['heavy_chain']),
            'light_chain_length': len(item['light_chain']) if item['light_chain'] else 0,
            'has_light_chain': bool(item['light_chain'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'real_structure_sequence_metadata.tsv')
    summary_df.to_csv(summary_path, sep='\t', index=False)
    print(f"Sequence metadata saved to: {summary_path}")

def generate_and_save_unified_structure_embeddings(sequence_data, output_dir, structure_backend="abodybuilder2",
                                                 batch_size=10, use_imgt_numbering=True):
    """
    Generate unified structure embeddings using specified backend and save to TSV with checkpoint support
    Args:
        sequence_data: List of sequence dictionaries
        output_dir: Output directory for embeddings
        structure_backend: "abodybuilder2" or "igfold"
        batch_size: Batch size for processing (default: 10)
        use_imgt_numbering: Whether to use IMGT numbering for optimal performance

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*60)
    print(f"GENERATING UNIFIED STRUCTURE EMBEDDINGS ({structure_backend.upper()} BACKEND)")
    print("="*60)

    checkpoint_path = os.path.join(output_dir, f'{structure_backend}_checkpoint.pkl')

    # Check for existing checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path)
    if checkpoint_data:
        print(f"Resuming from checkpoint: processed {checkpoint_data['processed_count']} sequences")
        processed_embeddings = checkpoint_data['embeddings']
        processed_ids = checkpoint_data['sequence_ids']
        start_idx = checkpoint_data['processed_count']
    else:
        processed_embeddings = []
        processed_ids = []
        start_idx = 0

    # Limit sequences if specified
    total_sequences = len(sequence_data)
    print(f"Processing sequences {start_idx} to {total_sequences} for {structure_backend}")

    # Process in batches
    batch_count = 0
    failed_count = 0

    for i in range(start_idx, total_sequences, batch_size):
        batch_end = min(i + batch_size, total_sequences)
        batch_data = sequence_data[i:batch_end]
        batch_count += 1

        print(f"Processing batch {batch_count}: sequences {i+1}-{batch_end}")

        try:
            # Extract heavy chain sequences for this batch (unified structure method uses heavy chain only)
            sequences = [item['heavy_chain'] for item in batch_data]

            start_time = time.time()
            batch_embeddings = create_unified_structure_embeddings(
                sequences,
                structure_backend=structure_backend,
                device='cpu',
                use_imgt_numbering=use_imgt_numbering
            )
            generation_time = time.time() - start_time

            if batch_embeddings is not None:
                print(f"Generated {structure_backend} embeddings for batch {batch_count} in {generation_time:.2f} seconds")
                print(f"Batch embedding shape: {batch_embeddings.shape}")

                # Add to processed embeddings
                if len(processed_embeddings) == 0:
                    processed_embeddings = batch_embeddings
                else:
                    processed_embeddings = np.vstack([processed_embeddings, batch_embeddings])

                # Add sequence IDs
                batch_ids = [item['sequence_id'] for item in batch_data]
                processed_ids.extend(batch_ids)

                # Save checkpoint
                checkpoint_data = {
                    'embeddings': processed_embeddings,
                    'sequence_ids': processed_ids,
                    'processed_count': len(processed_ids)
                }
                save_checkpoint(checkpoint_data, checkpoint_path)

                print(f"Checkpoint saved: {len(processed_ids)} sequences processed")

            else:
                print(f"Failed to generate {structure_backend} embeddings for batch {batch_count}")
                failed_count += 1

                # Add zero embeddings for failed batch
                zero_embeddings = np.zeros((len(batch_data), 128))  # Default embedding size
                if len(processed_embeddings) == 0:
                    processed_embeddings = zero_embeddings
                else:
                    processed_embeddings = np.vstack([processed_embeddings, zero_embeddings])

                batch_ids = [item['sequence_id'] for item in batch_data]
                processed_ids.extend(batch_ids)

        except Exception as e:
            print(f"Error processing batch {batch_count}: {e}")
            failed_count += 1

            # Add zero embeddings for failed batch
            zero_embeddings = np.zeros((len(batch_data), 128))
            if len(processed_embeddings) == 0:
                processed_embeddings = zero_embeddings
            else:
                processed_embeddings = np.vstack([processed_embeddings, zero_embeddings])

            batch_ids = [item['sequence_id'] for item in batch_data]
            processed_ids.extend(batch_ids)

        # Memory cleanup
        gc.collect()

    if len(processed_embeddings) > 0:
        print(f"Generated {structure_backend} embeddings for {len(processed_ids)} sequences")
        print(f"Final embedding shape: {processed_embeddings.shape}")
        print(f"Failed batches: {failed_count}")

        # Create DataFrame with sequence IDs and embeddings
        embedding_df = pd.DataFrame(processed_embeddings)
        embedding_df.index = processed_ids

        # Save to TSV
        output_path = os.path.join(output_dir, f'real_{structure_backend}_structure_embedding.tsv')
        embedding_df.to_csv(output_path, sep='\t')
        print(f"Real {structure_backend} embeddings saved to: {output_path}")

        # Clean up checkpoint
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("Checkpoint file cleaned up")

        return True
    else:
        print(f"Failed to generate real {structure_backend} embeddings")
        return False


# Backward compatibility function
def generate_and_save_real_abodybuilder2_embeddings(sequence_data, output_dir, batch_size=10, use_imgt_numbering=True):
    """Backward compatibility wrapper for ABodyBuilder2 embeddings"""
    return generate_and_save_unified_structure_embeddings(
        sequence_data, output_dir, "abodybuilder2", batch_size, use_imgt_numbering
    )

def check_dependencies():
    """
    Check if required dependencies are installed
    """
    print("Checking dependencies...")
    
    dependencies = {
        'IgFold': False,
        'SPACE2': False,
        'protein_holography': False,
        'PyRosetta': False,
        'transformers': False,
        'ABodyBuilder2': False
    }
    
    # Check IgFold
    try:
        from igfold import IgFoldRunner
        dependencies['IgFold'] = True
        print("IgFold available")
    except ImportError:
        print("IgFold not available")
    
    # Check SPACE2
    try:
        import SPACE2
        dependencies['SPACE2'] = True
        print("SPACE2 available")
    except ImportError:
        print("SPACE2 not available")
    
    # Check protein_holography
    try:
        import protein_holography
        dependencies['protein_holography'] = True
        print("protein_holography available")
    except ImportError:
        print("protein_holography not available")
    
    # Check PyRosetta
    try:
        import pyrosetta
        dependencies['PyRosetta'] = True
        print("PyRosetta available")
    except ImportError:
        print("PyRosetta not available")
    
    # Check transformers
    try:
        from transformers import AutoTokenizer, AutoModel
        dependencies['transformers'] = True
        print("transformers available")
    except ImportError:
        print("transformers not available")

    # Check ABodyBuilder2 (for paired heavy+light chain sequences)
    try:
        from ImmuneBuilder import ABodyBuilder2
        dependencies['ABodyBuilder2'] = True
        print("ABodyBuilder2 available")
    except ImportError:
        print("ABodyBuilder2 not available")

    return dependencies

def main():
    """
    Main function to generate all real structure embeddings with optimization
    """
    parser = argparse.ArgumentParser(description="Generate real structure embeddings for BCR dataset (optimized)")
    parser.add_argument("--output_dir", default="embeddings",
                       help="Output directory for embeddings")
    parser.add_argument("--log_dir", default="logs",
                       help="Directory for log files. Default: logs")
    parser.add_argument("--no_extra_log", action="store_true",
                       help="Don't create extra log files (useful when using SLURM logging)")
    parser.add_argument("--methods", nargs='+', default=['space2', 'holography', 'antiberta2-cssp', 'structure'],
                       choices=['space2', 'holography', 'antiberta2-cssp', 'structure'],
                       help="Which methods to generate. Use 'structure' for unified structure embeddings")
    parser.add_argument("--structure_backend", default="abodybuilder2",
                       choices=['abodybuilder2', 'igfold'],
                       help="Structure prediction backend for SPACE2 and Holography methods. Default: abodybuilder2")
    parser.add_argument("--max_sequences_antiberta2", type=int, default=None,
                       help="Maximum number of sequences for AntiBERTa2-CSSP. Default: None (all sequences)")
    parser.add_argument("--batch_size", type=int, default=5,
                       help="Batch size for processing sequences. Default: 5 for structure methods")

    args = parser.parse_args()

    # Setup logging with timestamped files
    global logger
    logger, log_path = setup_logging(args.log_dir)

    # Test logging immediately
    print("Testing logging system...")
    print("Logger test - this should appear in both console and log file")
    print("If you see this print but not the logger message above, there's a logging issue.")

    # IMGT numbering is enabled by default and automatically disabled if AbNumber is not available
    args.use_imgt_numbering = True

    # Check if AbNumber is available for IMGT numbering
    try:
        from abnumber import Chain
        abnumber_available = True
        print("AbNumber available - IMGT numbering enabled for optimal performance")
    except ImportError:
        abnumber_available = False
        args.use_imgt_numbering = False
        print("WARNING: AbNumber not available - IMGT numbering disabled")
        print("  Install AbNumber for optimal SPACE2 and AntiBERTa2-CSSP performance:")
        print("  conda install -c bioconda abnumber")

    print("="*80)
    print("REAL STRUCTURE EMBEDDING GENERATION FOR BCR DATASET")
    print("="*80)
    print(f"Log file: {log_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max sequences (AntiBERTa2): {args.max_sequences_antiberta2 or 'All sequences'}")
    print(f"IMGT numbering: {'Enabled' if args.use_imgt_numbering else 'Disabled (AbNumber not available)'}")
    print("Structure methods (SPACE2, Holography) will process all sequences")
    
    # Check dependencies
    dependencies = check_dependencies()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    sequence_data = load_bcr_dataset_sequences()

    # Create metadata summary
    create_embedding_summary(sequence_data, args.output_dir)
    
    # Generate embeddings
    success_count = 0
    total_methods = len(args.methods)
    
    if 'space2' in args.methods:
        print(f"\nGenerating SPACE2 embeddings using {args.structure_backend} backend...")
        backend_available = dependencies.get('ABodyBuilder2', False) if args.structure_backend == 'abodybuilder2' else dependencies.get('IgFold', False)

        if backend_available:
            if generate_and_save_real_space2_embeddings(sequence_data, args.output_dir, args.batch_size, args.use_imgt_numbering, args.structure_backend):
                success_count += 1
        else:
            backend_name = "ABodyBuilder2" if args.structure_backend == 'abodybuilder2' else "IgFold"
            print(f"Skipping SPACE2: {backend_name} not available")

    if 'holography' in args.methods:
        print(f"\nGenerating Protein Holography embeddings using {args.structure_backend} backend...")
        backend_available = dependencies.get('ABodyBuilder2', False) if args.structure_backend == 'abodybuilder2' else dependencies.get('IgFold', False)

        if backend_available:
            if generate_and_save_real_holography_embeddings(sequence_data, args.output_dir, args.batch_size, args.use_imgt_numbering, args.structure_backend):
                success_count += 1
        else:
            backend_name = "ABodyBuilder2" if args.structure_backend == 'abodybuilder2' else "IgFold"
            print(f"Skipping Protein Holography: {backend_name} not available")

    if 'antiberta2-cssp' in args.methods:
        print(f"\nGenerating AntiBERTa2-CSSP embeddings using {args.structure_backend} backend...")
        backend_available = dependencies.get('ABodyBuilder2', False) if args.structure_backend == 'abodybuilder2' else dependencies.get('IgFold', False)

        if dependencies['transformers'] and backend_available:
            if generate_and_save_antiberta2_embeddings(sequence_data, args.output_dir, args.batch_size, args.max_sequences_antiberta2, args.use_imgt_numbering, args.structure_backend):
                success_count += 1
        else:
            if not dependencies['transformers']:
                print("Skipping AntiBERTa2-CSSP: transformers not available")
            if not backend_available:
                backend_name = "ABodyBuilder2" if args.structure_backend == 'abodybuilder2' else "IgFold"
                print(f"Skipping AntiBERTa2-CSSP: {backend_name} not available")

    if 'structure' in args.methods:
        print(f"\nGenerating structure embeddings using {args.structure_backend} backend...")
        backend_available = dependencies.get('ABodyBuilder2', False) if args.structure_backend == 'abodybuilder2' else dependencies.get('IgFold', False)

        if backend_available:
            if generate_and_save_unified_structure_embeddings(
                sequence_data, args.output_dir, args.structure_backend, args.batch_size, args.use_imgt_numbering
            ):
                success_count += 1
        else:
            backend_name = "ABodyBuilder2" if args.structure_backend == 'abodybuilder2' else "IgFold"
            print(f"Skipping structure embeddings: {backend_name} not available")
            if args.structure_backend == 'abodybuilder2':
                print("  Install ImmuneBuilder with: pip install ImmuneBuilder")
            else:
                print("  Install IgFold following the official instructions")

    print("\n" + "="*80)
    print("OPTIMIZED REAL STRUCTURE EMBEDDING GENERATION COMPLETED")
    print("="*80)
    print(f"Successfully generated {success_count}/{total_methods} embedding methods")
    print(f"Embeddings saved to: {args.output_dir}")

    if success_count > 0:
        print("\nGenerated files:")
        for filename in os.listdir(args.output_dir):
            if filename.startswith('real_') and filename.endswith('_embedding.tsv'):
                filepath = os.path.join(args.output_dir, filename)
                file_size = os.path.getsize(filepath) / (1024*1024)  # MB
                print(f"  - {filename} ({file_size:.1f} MB)")

    # Log simple performance summary
    if len(performance_tracker.method_stats) > 0:
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)

        for method_name, backends in performance_tracker.method_stats.items():
            for backend, stats in backends.items():
                print(f"{method_name.upper()} ({backend.upper()}):")
                print(f"  Total time: {stats['total_time']:.2f}s")
                print(f"  Time per sequence: {stats['time_per_sequence']:.3f}s")
                print(f"  Sequences processed: {stats['sequences_processed']}")
                print(f"  Status: {'SUCCESS' if stats.get('success', True) else 'FAILED'}")

        # Save performance data
        performance_tracker.save_results(args.output_dir, log_path)
        print(f"Performance data saved to: {args.output_dir}/performance_log.txt")
        
    # Final log summary
    print("\n" + "="*80)
    print("LOGGING SUMMARY")
    print("="*80)
    print(f"Complete log saved to: {log_path}")
    print(f"Latest log symlink: {os.path.join(args.log_dir, 'structure_embedding_latest.log')}")
    print(f"Performance data: {os.path.join(args.output_dir, 'performance_comparison.json')}")
    print("="*80)

if __name__ == "__main__":
    main()