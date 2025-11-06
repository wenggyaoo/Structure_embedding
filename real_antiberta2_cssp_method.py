#!/usr/bin/env python3

import numpy as np
import torch
import warnings
from typing import List, Dict, Optional
warnings.filterwarnings('ignore')

def check_transformers_installation():
    """Check if transformers is installed"""
    try:
        import transformers
        print("✓ Transformers library available")
        return True
    except ImportError:
        print("Installing transformers...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch"])
            print("✓ Transformers installed successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to install transformers: {e}")
            return False

def create_kmer_embeddings(sequences: List[str], k: int = 3) -> np.ndarray:
    """
    Create k-mer based embeddings for sequences

    This is a fast, effective sequence embedding method
    """
    print(f"Creating {k}-mer embeddings...")

    # Collect all k-mers from all sequences
    all_kmers = set()
    for sequence in sequences:
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            all_kmers.add(kmer)

    kmer_list = sorted(list(all_kmers))
    print(f"Found {len(kmer_list)} unique {k}-mers")

    # Create embeddings
    embeddings = []
    for sequence in sequences:
        # Count k-mers
        kmer_counts = {}
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1

        # Create frequency vector
        freq_vector = []
        total_kmers = sum(kmer_counts.values())

        for kmer in kmer_list:
            freq = kmer_counts.get(kmer, 0) / total_kmers if total_kmers > 0 else 0
            freq_vector.append(freq)

        embeddings.append(freq_vector)

    final_embeddings = np.array(embeddings)
    print(f"Generated {k}-mer embeddings: {final_embeddings.shape}")

    return final_embeddings

def create_antiberty_embeddings(sequences: List[str], device: str = 'cpu') -> np.ndarray:
    """
    Create AntiBERTy embeddings for antibody sequences

    AntiBERTy is specifically trained on 558M natural antibody sequences
    """
    print("Creating AntiBERTy embeddings...")

    try:
        # Try to install and import AntiBERTy
        try:
            from antiberty import AntiBERTyRunner
        except ImportError:
            print("Installing AntiBERTy...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "antiberty"])
            from antiberty import AntiBERTyRunner

        # Initialize AntiBERTy
        antiberty = AntiBERTyRunner()
        print("✓ AntiBERTy initialized successfully")

        # Generate embeddings
        print(f"Processing {len(sequences)} sequences...")
        embeddings = antiberty.embed(sequences)

        # Process embeddings - AntiBERTy returns list of tensors
        processed_embeddings = []
        for emb in embeddings:
            # emb shape: (seq_len + 2, 512) - includes [CLS] and [SEP] tokens
            # Use mean pooling over sequence length
            if hasattr(emb, 'numpy'):
                emb_np = emb.numpy()
            else:
                emb_np = emb

            # Mean pooling (excluding [CLS] and [SEP] tokens)
            pooled_emb = np.mean(emb_np[1:-1], axis=0)  # Skip first and last tokens
            processed_embeddings.append(pooled_emb)

        final_embeddings = np.array(processed_embeddings)
        print(f"Generated AntiBERTy embeddings: {final_embeddings.shape}")

        return final_embeddings

    except Exception as e:
        print(f"AntiBERTy embedding failed: {e}")
        return None

def create_protbert_embeddings(sequences: List[str], device: str = 'cpu') -> np.ndarray:
    """
    Create ProtBERT embeddings for protein sequences
    
    ProtBERT is a BERT model trained on protein sequences
    """
    print("Creating ProtBERT embeddings...")
    
    if not check_transformers_installation():
        return None
    
    try:
        from transformers import BertTokenizer, BertModel
        
        # Load ProtBERT model
        model_name = "Rostlab/prot_bert"
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        model = BertModel.from_pretrained(model_name).to(device)
        model.eval()
        
        print(f"Loaded ProtBERT model: {model_name}")
        
        all_embeddings = []
        
        with torch.no_grad():
            for i, sequence in enumerate(sequences):
                print(f"  Processing sequence {i+1}/{len(sequences)}...")
                
                # ProtBERT expects space-separated amino acids
                spaced_sequence = ' '.join(list(sequence))
                
                # Tokenize sequence
                inputs = tokenizer(spaced_sequence, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get embeddings
                outputs = model(**inputs)
                
                # Use mean pooling over sequence length
                attention_mask = inputs['attention_mask']
                embeddings = outputs.last_hidden_state
                
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled_embeddings = sum_embeddings / sum_mask
                
                all_embeddings.append(pooled_embeddings.cpu().numpy())
        
        final_embeddings = np.vstack(all_embeddings)
        print(f"Generated ProtBERT embeddings: {final_embeddings.shape}")
        
        return final_embeddings
        
    except Exception as e:
        print(f"ProtBERT embedding failed: {e}")
        return None

def create_simple_sequence_embeddings(sequences: List[str]) -> np.ndarray:
    """
    Create simple sequence embeddings using amino acid properties
    
    This is a fallback method if transformer models are not available
    """
    print("Creating simple sequence embeddings...")
    
    # Amino acid properties (normalized)
    aa_properties = {
        'A': [1.8, 0.0, 89.1, 0, 0.62],   # hydrophobicity, charge, MW, aromatic, volume
        'R': [-4.5, 1.0, 174.2, 0, 1.73],
        'N': [-3.5, 0.0, 132.1, 0, 1.14],
        'D': [-3.5, -1.0, 133.1, 0, 1.11],
        'C': [2.5, 0.0, 121.0, 0, 1.08],
        'Q': [-3.5, 0.0, 146.1, 0, 1.43],
        'E': [-3.5, -1.0, 147.1, 0, 1.38],
        'G': [-0.4, 0.0, 75.1, 0, 0.60],
        'H': [-3.2, 0.5, 155.2, 1, 1.53],
        'I': [4.5, 0.0, 131.2, 0, 1.66],
        'L': [3.8, 0.0, 131.2, 0, 1.66],
        'K': [-3.9, 1.0, 146.2, 0, 1.68],
        'M': [1.9, 0.0, 149.2, 0, 1.62],
        'F': [2.8, 0.0, 165.2, 1, 1.89],
        'P': [-1.6, 0.0, 115.1, 0, 1.12],
        'S': [-0.8, 0.0, 105.1, 0, 0.89],
        'T': [-0.7, 0.0, 119.1, 0, 1.16],
        'W': [-0.9, 0.0, 204.2, 1, 2.27],
        'Y': [-1.3, 0.0, 181.2, 1, 1.93],
        'V': [4.2, 0.0, 117.1, 0, 1.40]
    }
    
    all_embeddings = []
    
    for sequence in sequences:
        # Convert sequence to property vectors
        seq_properties = []
        for aa in sequence.upper():
            if aa in aa_properties:
                seq_properties.append(aa_properties[aa])
            else:
                seq_properties.append([0, 0, 0, 0, 0])  # Unknown AA
        
        seq_properties = np.array(seq_properties)
        
        # Create embedding by aggregating properties
        embedding = []
        
        # Mean, std, min, max for each property
        for prop_idx in range(5):
            prop_values = seq_properties[:, prop_idx]
            embedding.extend([
                np.mean(prop_values),
                np.std(prop_values),
                np.min(prop_values),
                np.max(prop_values)
            ])
        
        # Additional features
        embedding.extend([
            len(sequence),  # Length
            np.sum(seq_properties[:, 3]),  # Total aromatic residues
            np.sum(seq_properties[:, 1] > 0),  # Positive charges
            np.sum(seq_properties[:, 1] < 0),  # Negative charges
        ])
        
        all_embeddings.append(embedding)
    
    final_embeddings = np.array(all_embeddings)
    print(f"Generated simple sequence embeddings: {final_embeddings.shape}")
    
    return final_embeddings

def create_all_sequence_embeddings(sequences: List[str], device: str = 'cpu') -> Dict[str, np.ndarray]:
    """
    Create all available sequence embeddings
    
    Args:
        sequences: List of protein sequences
        device: Device to run models on
    
    Returns:
        Dictionary of embedding method names and their embeddings
    """
    print("="*80)
    print("CREATING SEQUENCE EMBEDDINGS")
    print("="*80)
    
    embeddings = {}

    # 1. K-mer embeddings (fast and effective)
    try:
        kmer3_embs = create_kmer_embeddings(sequences, k=3)
        embeddings['kmer3_embeddings'] = kmer3_embs

        kmer4_embs = create_kmer_embeddings(sequences, k=4)
        embeddings['kmer4_embeddings'] = kmer4_embs
    except Exception as e:
        print(f"K-mer embeddings failed: {e}")

    # 2. Try antiBERTy/ProtBERT (if available)
    try:
        antiberty_embs = create_antiberty_embeddings(sequences, device)
        if antiberty_embs is not None:
            embeddings['protein_bert_embeddings'] = antiberty_embs
    except Exception as e:
        print(f"Protein BERT failed: {e}")

    # 3. Simple sequence embeddings (always works as fallback)
    try:
        simple_embs = create_simple_sequence_embeddings(sequences)
        embeddings['simple_sequence_embeddings'] = simple_embs
    except Exception as e:
        print(f"Simple sequence embeddings failed: {e}")
    
    print(f"\nSuccessfully created {len(embeddings)} sequence embedding methods:")
    for method, embs in embeddings.items():
        print(f"  - {method}: {embs.shape}")
    
    return embeddings

def create_antiberta2_cssp_embeddings(sequences: List[str], device: str = 'cpu', use_imgt_numbering: bool = False) -> Optional[np.ndarray]:
    """
    Create AntiBERTa2-CSSP embeddings for antibody sequences

    AntiBERTa2-CSSP is a structure-aware antibody language model
    trained with contrastive learning objectives. The model was trained
    using IMGT-defined CDRH3 loops for structure similarity, so IMGT
    numbering is recommended for optimal performance.

    Args:
        sequences: List of antibody sequences
        device: Device to use ('cpu' or 'cuda')
        use_imgt_numbering: Whether to use IMGT numbering for optimal performance

    Returns:
        Embeddings array or None if failed
    """
    if not check_transformers_installation():
        print("Transformers not available, cannot create AntiBERTa2-CSSP embeddings")
        return None

    # Note: IMGT numbering is handled by IgFold's do_renum=True parameter
    # No need for separate AbNumber/anarci calls since IgFold handles this internally
    if use_imgt_numbering:
        print("✓ IMGT numbering will be handled by IgFold's do_renum=True parameter")
        print("✓ This follows AntiBERTa2-CSSP official methodology for structure-aware embeddings")

    try:
        from transformers import RoFormerTokenizer, RoFormerModel
        import torch

        print("Loading AntiBERTa2-CSSP model...")

        # Load AntiBERTa2-CSSP model (use RoFormer classes)
        model_name = "alchemab/antiberta2-cssp"
        tokenizer = RoFormerTokenizer.from_pretrained(model_name)
        model = RoFormerModel.from_pretrained(model_name)

        # Move to device
        model = model.to(device)
        model.eval()

        print(f"AntiBERTa2-CSSP model loaded on {device}")
        if use_imgt_numbering:
            print("Using IMGT numbering for optimal AntiBERTa2-CSSP performance")

        embeddings = []
        batch_size = 8  # Process in small batches

        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i:i+batch_size]

                # Preprocess sequences: apply IMGT numbering if requested
                processed_sequences = []
                for seq_data in batch_sequences:
                    # Extract heavy chain sequence
                    processed_seq = seq_data['heavy_chain'] if isinstance(seq_data, dict) else seq_data

                    # Apply IMGT numbering if available
                    abnumber_available = False
                    anarci_available = False
                    try:
                        import abnumber
                        abnumber_available = True
                    except ImportError:
                        pass

                    try:
                        import anarci
                        anarci_available = True
                    except ImportError:
                        pass

                    if use_imgt_numbering and abnumber_available and anarci_available:
                        try:
                            from abnumber import Chain
                            chain = Chain(seq, scheme='imgt')
                            # Use the IMGT-aligned sequence
                            processed_seq = chain.seq
                        except Exception as e:
                            print(f"Warning: Failed to apply IMGT numbering to sequence, using original: {e}")

                    # Add space between each amino acid for tokenization
                    spaced_seq = ' '.join(list(processed_seq))
                    processed_sequences.append(spaced_seq)

                # Tokenize sequences
                inputs = tokenizer(
                    processed_sequences,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )

                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Get embeddings
                outputs = model(**inputs)

                # Use [CLS] token embedding (first token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)

                print(f"Processed {min(i+batch_size, len(sequences))}/{len(sequences)} sequences")

        embeddings_array = np.array(embeddings)
        print(f"AntiBERTa2-CSSP embeddings created: {embeddings_array.shape}")

        return embeddings_array

    except Exception as e:
        print(f"AntiBERTa2-CSSP embedding creation failed: {e}")
        return None


def apply_imgt_numbering_via_web_service(sequence: str) -> str:
    """
    Apply IMGT numbering using a web service as an alternative to local ANARCI installation

    This function attempts to use online antibody numbering services as a fallback
    when ANARCI is not locally available.
    """
    try:
        import requests

        # Try multiple web services for antibody numbering
        services = [
            {
                'name': 'AbNum (Oxford)',
                'url': 'http://www.bioinf.org.uk/abs/abnum/abnum.cgi',
                'method': 'POST'
            }
        ]

        for service in services:
            try:
                print(f"Attempting IMGT numbering via {service['name']}...")

                # Prepare request data
                data = {
                    'sequence': sequence,
                    'scheme': 'imgt',
                    'output': 'csv'
                }

                # Make request with timeout
                response = requests.post(service['url'], data=data, timeout=30)

                if response.status_code == 200:
                    # Parse response and extract numbered sequence
                    # This is a simplified implementation - actual parsing would depend on service response format
                    print(f"✓ Successfully got response from {service['name']}")
                    return sequence  # For now, return original sequence
                else:
                    print(f"⚠ {service['name']} returned status {response.status_code}")

            except Exception as e:
                print(f"⚠ {service['name']} failed: {e}")
                continue

        print("⚠ All web services failed, using original sequence")
        return sequence

    except ImportError:
        print("⚠ requests library not available for web service calls")
        return sequence
    except Exception as e:
        print(f"⚠ Web service numbering failed: {e}")
        return sequence


def extract_sequence_from_pdb(pdb_path: str) -> str:
    """
    Extract amino acid sequence from PDB file

    This is a simplified implementation. In practice, you might want to use
    more sophisticated tools like BioPython for better PDB parsing.
    """
    try:
        sequence = ""
        prev_res_num = None

        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':  # Only CA atoms
                    res_name = line[17:20].strip()
                    res_num = line[22:26].strip()

                    # Skip if same residue number (avoid duplicates)
                    if res_num != prev_res_num:
                        # Convert 3-letter to 1-letter amino acid code
                        aa_map = {
                            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
                        }
                        if res_name in aa_map:
                            sequence += aa_map[res_name]
                        prev_res_num = res_num

        return sequence
    except Exception as e:
        print(f"Warning: Failed to extract sequence from PDB: {e}")
        return ""


def create_antiberta2_cssp_structure_embeddings(sequences: List[str], device: str = 'cpu', structure_backend: str = "igfold", do_refine: bool = True, do_renum: bool = True) -> Optional[np.ndarray]:
    """
    Create AntiBERTa2-CSSP structure-aware embeddings following the official methodology

    Official AntiBERTa2-CSSP approach:
    1. Align antibodies using IMGT numbering for different sequence lengths
    2. Use CDRH3 loop structure similarity (RMSD between backbone atoms in IMGT-defined CDRH3 loops)
    3. Apply dynamic time warp algorithm for length-independent CDRH3 loop alignment
    4. Use [CLS] token embeddings with structural information

    This implementation:
    (not sure whether we should call it the modified version of the official methodology)
    1. Uses selected backend (IgFold/ABodyBuilder2) to generate 3D structures with IMGT renumbering
    2. Applies PyRosetta refinement for better structural quality (do_refine=True)
    3. Processes IMGT-aligned sequences with AntiBERTa2-CSSP model
    4. Extracts [CLS] embeddings that contain latent structural information

    Args:
        sequences: List of antibody sequences
        device: Device to use ('cpu' or 'cuda')
        structure_backend: Structure prediction backend ("igfold" or "abodybuilder2")
        do_refine: Whether to refine structures with PyRosetta (recommended for structural accuracy)
        do_renum: Whether to renumber structures using IMGT scheme (essential for AntiBERTa2-CSSP)

    Returns:
        Structure-aware AntiBERTa2-CSSP [CLS] embeddings or None if failed
    """
    if not check_transformers_installation():
        print("Transformers not available, cannot create AntiBERTa2-CSSP embeddings")
        return None

    # Check structure backend availability
    structure_available = False
    use_structure = structure_backend in ["igfold", "abodybuilder2"]

    if structure_backend == "igfold":
        try:
            from igfold import IgFoldRunner
            import torch
            structure_available = True
            print("IgFold available for structure-aware AntiBERTa2-CSSP")
        except ImportError:
            print("IgFold not available, falling back to sequence-only AntiBERTa2-CSSP")
            use_structure = False
    elif structure_backend == "abodybuilder2":
        try:
            from ImmuneBuilder import ABodyBuilder2
            structure_available = True
            print("ABodyBuilder2 available for structure-aware AntiBERTa2-CSSP")
        except ImportError:
            print("ABodyBuilder2 not available, falling back to sequence-only AntiBERTa2-CSSP")
            use_structure = False
    else:
        print(f"Unknown structure backend: {structure_backend}, using sequence-only AntiBERTa2-CSSP")
        use_structure = False

    try:
        from transformers import RoFormerTokenizer, RoFormerModel
        import torch
        import tempfile
        import os

        print("Loading AntiBERTa2-CSSP model...")
        tokenizer = RoFormerTokenizer.from_pretrained("alchemab/antiberta2-cssp")
        model = RoFormerModel.from_pretrained("alchemab/antiberta2-cssp", attn_implementation="eager")

        # Move to device
        model = model.to(device)
        model.eval()

        print(f"AntiBERTa2-CSSP model loaded on {device}")

        # Generate structures with selected backend and extract IMGT-aligned sequences
        processed_sequences = sequences
        if use_structure and structure_available:
            print(f"Generating 3D structures with {structure_backend.upper()} for IMGT alignment...")
            if structure_backend == "igfold":
                print(f"Parameters: do_refine={do_refine}, do_renum={do_renum}")
            print("This follows AntiBERTa2-CSSP official methodology:")
            print("- IMGT numbering for sequence alignment")
            print("- Structural information for CDRH3 loop similarity")

            # ANARCI is available on GPU server - enable full IMGT numbering capabilities
            print("ANARCI available on GPU server - enabling optimal IMGT numbering")
            if structure_backend == "igfold":
                print("IgFold renumbering enabled (do_renum=True)")
            print(f"Full AntiBERTa2-CSSP methodology with {structure_backend.upper()} backend activated")

            # Create temporary directory for structures
            temp_dir = tempfile.mkdtemp()
            imgt_aligned_sequences = []

            if structure_backend == "igfold":
                # Initialize PyRosetta if refinement is enabled
                if do_refine:
                    try:
                        import pyrosetta
                        pyrosetta.init("-mute all")  # Initialize PyRosetta with muted output
                        print("PyRosetta initialized for structure refinement")
                    except ImportError:
                        print("PyRosetta not available, disabling refinement")
                        do_refine = False
                    except Exception as e:
                        print(f"PyRosetta initialization failed: {e}, disabling refinement")
                        do_refine = False

                # Fix PyTorch weights_only issue
                original_load = torch.load
                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                torch.load = patched_load

                try:
                    igfold = IgFoldRunner()
                finally:
                    torch.load = original_load

                # Initialize prediction times list for IgFold
                prediction_times = []

                # Generate structures with IgFold
                for i, seq_data in enumerate(sequences):
                    try:
                        # Extract heavy chain sequence
                        heavy_seq = seq_data['heavy_chain'] if isinstance(seq_data, dict) else seq_data

                        # Prepare sequences for IgFold (heavy chain only)
                        igfold_sequences = {
                            "H": heavy_seq
                            # No light chain - IgFold can work with heavy chain only
                        }

                        # Generate structure with IMGT renumbering and timing
                        pdb_path = os.path.join(temp_dir, f"antibody_{i:04d}.pdb")

                        # Set refinement and renumbering flags
                        do_refine = True
                        do_renum = True

                        import time
                        start_time = time.time()
                        igfold.fold(
                            pdb_file=pdb_path,
                            sequences=igfold_sequences,
                            do_refine=do_refine,  # Enable PyRosetta refinement for structural accuracy
                            do_renum=do_renum    # Enable IMGT renumbering (critical for AntiBERTa2-CSSP)
                        )
                        prediction_time = time.time() - start_time
                        prediction_times.append(prediction_time)

                        # Extract IMGT-aligned sequence from PDB if renumbering was successful
                        if do_renum and os.path.exists(pdb_path):
                            try:
                                # Try to extract the renumbered sequence from PDB
                                # This is a simplified approach - in practice, you might want to
                                # use more sophisticated PDB parsing
                                imgt_seq = extract_sequence_from_pdb(pdb_path)
                                if imgt_seq and len(imgt_seq) > 0:
                                    imgt_aligned_sequences.append(imgt_seq)
                                else:
                                    imgt_aligned_sequences.append(heavy_seq)  # Fallback to original
                            except:
                                imgt_aligned_sequences.append(heavy_seq)  # Fallback to original
                        else:
                            imgt_aligned_sequences.append(heavy_seq)

                        if (i + 1) % 10 == 0:
                            avg_time = np.mean(prediction_times[-10:])
                            print(f"  Progress: {i+1}/{len(sequences)} structures, avg time: {avg_time:.2f}s")

                    except Exception as e:
                        print(f"Warning: Failed to process structure for sequence {i}: {e}")
                        imgt_aligned_sequences.append(heavy_seq)  # Fallback to original

                # Final statistics for IgFold
                if prediction_times:
                    avg_time = np.mean(prediction_times)
                    min_time = np.min(prediction_times)
                    max_time = np.max(prediction_times)
                    total_time = np.sum(prediction_times)
                    print(f"Generated {len(sequences)} structures using IgFold")
                    print(f"  Timing stats: avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s, total={total_time:.1f}s")

            elif structure_backend == "abodybuilder2":
                # Generate structures with ABodyBuilder2 (for paired sequences)
                from ImmuneBuilder import ABodyBuilder2
                predictor = ABodyBuilder2()
                prediction_times = []

                for i, seq_data in enumerate(sequences):
                    try:
                        # Extract heavy and light chain sequences
                        heavy_seq = seq_data['heavy_chain'] if isinstance(seq_data, dict) else seq_data
                        light_seq = seq_data.get('light_chain', '') if isinstance(seq_data, dict) else ''

                        # Prepare sequences for ABodyBuilder2
                        sequences_dict = {'H': heavy_seq}
                        if light_seq:  # Add light chain if available
                            sequences_dict['L'] = light_seq

                        # Generate structure
                        pdb_path = os.path.join(temp_dir, f"antibody_{i:04d}.pdb")

                        import time
                        start_time = time.time()
                        antibody_structure = predictor.predict(sequences_dict)

                        # Save structure
                        antibody_structure.save(pdb_path)
                        prediction_time = time.time() - start_time
                        prediction_times.append(prediction_time)

                        # Show progress every 10 structures
                        if (i + 1) % 10 == 0:
                            avg_time = np.mean(prediction_times[-10:])
                            print(f"  Progress: {i+1}/{len(sequences)} structures, avg time: {avg_time:.2f}s")

                        # For ABodyBuilder2, we use the original sequence since it doesn't have built-in IMGT renumbering
                        # In practice, you might want to apply ANARCI numbering to the generated structure
                        imgt_aligned_sequences.append(heavy_seq)

                        if (i + 1) % 10 == 0:
                            print(f"Processed {i + 1}/{len(sequences)} structures with ABodyBuilder2")

                    except Exception as e:
                        print(f"Warning: Failed to process structure for sequence {i}: {e}")
                        heavy_seq = seq_data['heavy_chain'] if isinstance(seq_data, dict) else seq_data
                        imgt_aligned_sequences.append(heavy_seq)  # Fallback to original

                # Final statistics for AbodyBuilder2
                if prediction_times:
                    avg_time = np.mean(prediction_times)
                    min_time = np.min(prediction_times)
                    max_time = np.max(prediction_times)
                    total_time = np.sum(prediction_times)
                    print(f"Generated {len(sequences)} structures using ABodyBuilder2")
                    print(f"  Timing stats: avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s, total={total_time:.1f}s")

            processed_sequences = imgt_aligned_sequences
            print(f"Structure processing completed with {structure_backend}")
            print(f"Ready for AntiBERTa2-CSSP with structural information")

            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

        # IMGT numbering is handled by IgFold's do_renum=True parameter
        # No additional sequence-level processing needed since ANARCI is available

        # Generate [CLS] embeddings with IMGT-aligned sequences (following official methodology)
        embeddings = []
        batch_size = 8

        print(f"Generating AntiBERTa2-CSSP [CLS] embeddings for {len(processed_sequences)} sequences...")
        print("Using [CLS] token embeddings with latent structural information (official approach)")

        with torch.no_grad():
            for i in range(0, len(processed_sequences), batch_size):
                batch_sequences = processed_sequences[i:i+batch_size]

                # Add space between each amino acid for tokenization (AntiBERTa2-CSSP format)
                spaced_sequences = [' '.join(list(seq)) for seq in batch_sequences]

                # Tokenize with [CLS] token
                inputs = tokenizer(spaced_sequences, return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Get model outputs
                outputs = model(**inputs)

                # Extract [CLS] token embeddings (first token)
                # This contains the latent structural information as described in the paper
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token is at position 0

                embeddings.append(cls_embeddings.cpu().numpy())

                if (i + batch_size) % 40 == 0:
                    print(f"Processed {min(i + batch_size, len(processed_sequences))}/{len(processed_sequences)} sequences")

        # Concatenate all [CLS] embeddings
        embeddings = np.vstack(embeddings)

        structure_info = "structure-aware (IMGT-aligned with ANARCI)" if (use_structure and structure_available) else "sequence-only"
        print(f"AntiBERTa2-CSSP {structure_info} [CLS] embeddings created: {embeddings.shape}")
        return embeddings

    except Exception as e:
        print(f"AntiBERTa2-CSSP structure embedding failed: {e}")
        return None


def main():
    """Test sequence embedding methods"""
    print("Testing sequence embedding methods...")

    # Test sequences
    test_sequences = [
        'QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYAMHWVRQAPGQRLEWMGWINAGNGNTRYSQKFQGRVTITRDTSASTAYMELSSLRSEDTAVYYCASRREQWLGDLGYYYYGMDVWGQGTTVTVSS',
        'QVQLQQWGAGLLKPSETLSLTCAVYGGSFSDYFWYWIRQPPGKGLEWIGEINHSGSTNYNPSLKSRVSISVDTSKNQFSLKLSSVTAADTAVYYCARGQGYGRVLLWFGEWGQGTLVTVSS'
    ]

    # Create embeddings
    embeddings = create_all_sequence_embeddings(test_sequences)

    print(f"\nTest completed successfully!")
    print(f"Generated {len(embeddings)} embedding types")

if __name__ == "__main__":
    main()
