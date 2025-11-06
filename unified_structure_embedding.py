#!/usr/bin/env python3
"""

This module implements a unified approach to structure embedding generation that can use
different structure prediction backends (ABodyBuilder2, IgFold) and apply the same
embedding extraction methods to all of them.

1. Structure prediction backend (ABodyBuilder2 vs IgFold)
2. Embedding extraction methods (coordinate, distance, chemical features)

"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import ABodyBuilder2
try:
    from ImmuneBuilder import ABodyBuilder2
    ABODYBUILDER2_AVAILABLE = True
    print("ABodyBuilder2 successfully imported")
except ImportError as e:
    ABODYBUILDER2_AVAILABLE = False
    print(f"ABodyBuilder2 not available: {e}")

# Try to import IgFold
try:
    from igfold import IgFoldRunner
    IGFOLD_AVAILABLE = True
    print("IgFold successfully imported")
except ImportError as e:
    IGFOLD_AVAILABLE = False
    print(f"IgFold not available: {e}")

# Try to import structure analysis libraries
try:
    import Bio
    from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue
    from Bio.PDB.vectors import calc_dihedral, calc_angle
    from Bio.PDB.Polypeptide import PPBuilder
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("BioPython not available for structure analysis")


class StructurePredictionBackend:
    """Base class for structure prediction backends"""
    
    def __init__(self, output_dir: str = "structures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_structures(self, sequences: List[Dict]) -> List[str]:
        """Generate 3D structures and return PDB file paths"""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if this backend is available"""
        raise NotImplementedError


class ABodyBuilder2Backend(StructurePredictionBackend):
    """ABodyBuilder2 structure prediction backend"""
    
    def __init__(self, output_dir: str = "abodybuilder2_structures"):
        super().__init__(output_dir)
        self.predictor = None
        self.available = ABODYBUILDER2_AVAILABLE
        
        if self.available:
            try:
                print("Initializing ABodyBuilder2 predictor...")
                self.predictor = ABodyBuilder2()
                print("ABodyBuilder2 predictor initialized successfully")
            except Exception as e:
                print(f"Failed to initialize ABodyBuilder2: {e}")
                self.available = False
    
    def is_available(self) -> bool:
        return self.available
    
    def generate_structures(self, sequences: List[Dict]) -> List[str]:
        """Generate structures using ABodyBuilder2"""
        if not self.available:
            print("ABodyBuilder2 not available")
            return []
        
        print(f"Generating structures for {len(sequences)} antibodies using ABodyBuilder2...")
        
        pdb_files = []
        successful = 0
        failed = 0
        
        for i, seq_data in enumerate(sequences):
            antibody_id = seq_data.get('antibody_id', f'antibody_{i:04d}')
            heavy_seq = seq_data.get('heavy_chain_aa', '')
            light_seq = seq_data.get('light_chain_aa', '')
            
            if not heavy_seq:
                print(f"  ✗ {antibody_id}: No heavy chain sequence")
                failed += 1
                continue
            
            try:
                start_time = time.time()
                
                # Prepare sequences for ABodyBuilder2
                sequences_dict = {'H': heavy_seq}
                if light_seq:
                    sequences_dict['L'] = light_seq
                
                # Generate structure
                pdb_file = self.output_dir / f"{antibody_id}.pdb"
                
                print(f"  Predicting structure for {antibody_id}...")
                antibody_structure = self.predictor.predict(sequences_dict)
                
                # Save structure
                antibody_structure.save(str(pdb_file))
                
                processing_time = time.time() - start_time
                
                if pdb_file.exists():
                    pdb_files.append(str(pdb_file))
                    successful += 1
                    print(f"  ✓ {antibody_id}: Generated in {processing_time:.2f}s")
                else:
                    print(f"  ✗ {antibody_id}: Structure file not created")
                    failed += 1
                
            except Exception as e:
                failed += 1
                print(f"  ✗ {antibody_id}: {str(e)}")
        
        print(f"ABodyBuilder2 structure generation: {successful} successful, {failed} failed")
        return pdb_files


class IgFoldBackend(StructurePredictionBackend):
    """IgFold structure prediction backend"""

    def __init__(self, output_dir: str = "igfold_structures"):
        super().__init__(output_dir)
        self.igfold = None
        self.available = IGFOLD_AVAILABLE

        if self.available:
            try:
                self.igfold = IgFoldRunner()
                print("IgFold backend initialized successfully")
            except Exception as e:
                print(f"Failed to initialize IgFold: {e}")
                self.available = False
    
    def is_available(self) -> bool:
        return self.available
    
    def generate_structures(self, sequences: List[Dict]) -> List[str]:
        """Generate structures using IgFold"""
        if not self.available:
            print("IgFold not available")
            return []
        
        print(f"Generating structures for {len(sequences)} antibodies using IgFold...")
        
        pdb_files = []
        successful = 0
        failed = 0
        
        for i, seq_data in enumerate(sequences):
            antibody_id = seq_data.get('antibody_id', f'antibody_{i:04d}')
            heavy_seq = seq_data.get('heavy_chain_aa', '')
            light_seq = seq_data.get('light_chain_aa', '')
            
            if not heavy_seq:
                print(f"  ✗ {antibody_id}: No heavy chain sequence")
                failed += 1
                continue
            
            try:
                start_time = time.time()
                
                # Prepare sequences for IgFold
                sequences_dict = {"H": heavy_seq}
                if light_seq:
                    sequences_dict["L"] = light_seq
                
                pdb_file = self.output_dir / f"{antibody_id}.pdb"
                
                print(f"  Predicting structure for {antibody_id}...")
                
                # Generate structure using IgFold
                self.igfold.fold(
                    str(pdb_file),
                    sequences=sequences_dict,
                    do_refine=False,
                    do_renum=False
                )
                
                processing_time = time.time() - start_time
                
                if pdb_file.exists():
                    pdb_files.append(str(pdb_file))
                    successful += 1
                    print(f"  ✓ {antibody_id}: Generated in {processing_time:.2f}s")
                else:
                    print(f"  ✗ {antibody_id}: Structure file not created")
                    failed += 1
                
            except Exception as e:
                failed += 1
                print(f"  ✗ {antibody_id}: {str(e)}")
        
        print(f"IgFold structure generation: {successful} successful, {failed} failed")
        return pdb_files


class UnifiedStructureEmbeddingMethod:
    """
    Unified structure embedding method that can use different structure prediction backends
    and apply the same embedding extraction methods to all of them
    """
    
    def __init__(self,
                 structure_backend: str = "abodybuilder2",
                 embedding_dim: int = 128,
                 output_dir: str = "unified_structures",
                 do_refine: bool = True,
                 do_renum: bool = True):
        """
        Initialize unified structure embedding method

        Args:
            structure_backend: "abodybuilder2" or "igfold"
            embedding_dim: Dimension of output embeddings
            output_dir: Directory for structure files
            do_refine: Whether to refine structures (IgFold only)
            do_renum: Whether to renumber structures (IgFold only)
        """
        self.embedding_dim = embedding_dim
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize structure prediction backend
        if structure_backend.lower() == "abodybuilder2":
            self.backend = ABodyBuilder2Backend(
                output_dir=str(self.output_dir / "abodybuilder2")
            )
        elif structure_backend.lower() == "igfold":
            self.backend = IgFoldBackend(
                output_dir=str(self.output_dir / "igfold"),
                do_refine=do_refine,  # Use high-quality settings like other methods
                do_renum=do_renum
            )
        else:
            raise ValueError(f"Unknown structure backend: {structure_backend}")
        
        self.structure_backend = structure_backend
        
        # Structure analysis capabilities
        self.biopython_available = BIOPYTHON_AVAILABLE
        if self.biopython_available:
            self.pdb_parser = PDBParser(QUIET=True)
        
        print(f"Unified Structure Embedding Method initialized:")
        print(f"  - Structure backend: {structure_backend}")
        print(f"  - Backend available: {self.backend.is_available()}")
        print(f"  - Embedding dimension: {embedding_dim}")
        print(f"  - Output directory: {self.output_dir}")
    
    def is_available(self) -> bool:
        """Check if the method is available"""
        return self.backend.is_available()

    def extract_coordinate_features(self, pdb_file: str) -> np.ndarray:
        """Extract coordinate-based geometric features from PDB structure"""
        try:
            if not self.biopython_available:
                return self._fallback_coordinate_features(pdb_file)

            structure = self.pdb_parser.get_structure('antibody', pdb_file)

            # Extract coordinates and calculate geometric features
            coordinates = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.has_id('CA'):  # Alpha carbon
                            ca_atom = residue['CA']
                            coordinates.append(ca_atom.get_coord())

            if not coordinates:
                return np.zeros(self.embedding_dim // 3)

            coordinates = np.array(coordinates)

            # Calculate geometric features
            features = []

            # Center of mass
            center_of_mass = np.mean(coordinates, axis=0)
            features.extend(center_of_mass)

            # Radius of gyration
            distances_from_center = np.linalg.norm(coordinates - center_of_mass, axis=1)
            radius_of_gyration = np.sqrt(np.mean(distances_from_center**2))
            features.append(radius_of_gyration)

            # Principal components
            centered_coords = coordinates - center_of_mass
            cov_matrix = np.cov(centered_coords.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            features.extend(eigenvalues)
            features.extend(eigenvectors.flatten()[:6])  # First 6 components

            # Distance statistics
            pairwise_distances = []
            for i in range(len(coordinates)):
                for j in range(i+1, min(i+10, len(coordinates))):  # Local distances
                    dist = np.linalg.norm(coordinates[i] - coordinates[j])
                    pairwise_distances.append(dist)

            if pairwise_distances:
                features.extend([
                    np.mean(pairwise_distances),
                    np.std(pairwise_distances),
                    np.min(pairwise_distances),
                    np.max(pairwise_distances)
                ])
            else:
                features.extend([0, 0, 0, 0])

            # Pad or truncate to desired size
            target_size = self.embedding_dim // 3
            if len(features) < target_size:
                features.extend([0] * (target_size - len(features)))

            return np.array(features[:target_size])

        except Exception as e:
            print(f"Error extracting coordinate features from {pdb_file}: {e}")
            return np.zeros(self.embedding_dim // 3)

    def _fallback_coordinate_features(self, pdb_file: str) -> np.ndarray:
        """Fallback coordinate feature extraction without BioPython"""
        try:
            # Simple PDB parsing
            coordinates = []
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') and ' CA ' in line:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coordinates.append([x, y, z])

            if not coordinates:
                return np.zeros(self.embedding_dim // 3)

            coordinates = np.array(coordinates)

            # Basic geometric features
            features = []
            center = np.mean(coordinates, axis=0)
            features.extend(center)

            # Simple statistics
            features.extend([
                np.std(coordinates[:, 0]),
                np.std(coordinates[:, 1]),
                np.std(coordinates[:, 2]),
                np.mean(np.linalg.norm(coordinates - center, axis=1))
            ])

            # Pad to desired size
            target_size = self.embedding_dim // 3
            while len(features) < target_size:
                features.append(0.0)

            return np.array(features[:target_size])

        except Exception as e:
            print(f"Fallback coordinate extraction failed for {pdb_file}: {e}")
            return np.zeros(self.embedding_dim // 3)

    def extract_distance_features(self, pdb_file: str) -> np.ndarray:
        """Extract distance-based features from PDB structure"""
        try:
            if not self.biopython_available:
                return self._fallback_distance_features(pdb_file)

            structure = self.pdb_parser.get_structure('antibody', pdb_file)

            # Extract CA coordinates
            ca_coords = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.has_id('CA'):
                            ca_coords.append(residue['CA'].get_coord())

            if len(ca_coords) < 2:
                return np.zeros(self.embedding_dim // 3)

            ca_coords = np.array(ca_coords)

            # Calculate distance matrix
            n_residues = len(ca_coords)
            distance_matrix = np.zeros((n_residues, n_residues))

            for i in range(n_residues):
                for j in range(i+1, n_residues):
                    dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist

            # Extract distance-based features
            features = []

            # Distance statistics
            upper_triangle = distance_matrix[np.triu_indices(n_residues, k=1)]
            if len(upper_triangle) > 0:
                features.extend([
                    np.mean(upper_triangle),
                    np.std(upper_triangle),
                    np.min(upper_triangle),
                    np.max(upper_triangle),
                    np.median(upper_triangle)
                ])
            else:
                features.extend([0, 0, 0, 0, 0])

            # Local distance patterns (sequential neighbors)
            local_distances = []
            for i in range(n_residues - 1):
                local_distances.append(distance_matrix[i, i+1])

            if local_distances:
                features.extend([
                    np.mean(local_distances),
                    np.std(local_distances)
                ])
            else:
                features.extend([0, 0])

            # Contact map features (distances < 8 Å)
            contact_map = distance_matrix < 8.0
            contact_density = np.sum(contact_map) / (n_residues * n_residues)
            features.append(contact_density)

            # Compactness measure
            compactness = np.sum(upper_triangle < 10.0) / len(upper_triangle) if len(upper_triangle) > 0 else 0
            features.append(compactness)

            # Pad or truncate to desired size
            target_size = self.embedding_dim // 3
            if len(features) < target_size:
                features.extend([0] * (target_size - len(features)))

            return np.array(features[:target_size])

        except Exception as e:
            print(f"Error extracting distance features from {pdb_file}: {e}")
            return np.zeros(self.embedding_dim // 3)

    def _fallback_distance_features(self, pdb_file: str) -> np.ndarray:
        """Fallback distance feature extraction without BioPython"""
        try:
            # Simple PDB parsing for CA atoms
            coordinates = []
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') and ' CA ' in line:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coordinates.append([x, y, z])

            if len(coordinates) < 2:
                return np.zeros(self.embedding_dim // 3)

            coordinates = np.array(coordinates)

            # Calculate simple distance features
            features = []

            # Pairwise distances for first 20 residues (to keep it manageable)
            n_coords = min(20, len(coordinates))
            distances = []

            for i in range(n_coords):
                for j in range(i+1, n_coords):
                    dist = np.linalg.norm(coordinates[i] - coordinates[j])
                    distances.append(dist)

            if distances:
                features.extend([
                    np.mean(distances),
                    np.std(distances),
                    np.min(distances),
                    np.max(distances)
                ])
            else:
                features.extend([0, 0, 0, 0])

            # Sequential distances
            seq_distances = []
            for i in range(len(coordinates) - 1):
                dist = np.linalg.norm(coordinates[i] - coordinates[i+1])
                seq_distances.append(dist)

            if seq_distances:
                features.extend([
                    np.mean(seq_distances),
                    np.std(seq_distances)
                ])
            else:
                features.extend([0, 0])

            # Pad to desired size
            target_size = self.embedding_dim // 3
            while len(features) < target_size:
                features.append(0.0)

            return np.array(features[:target_size])

        except Exception as e:
            print(f"Fallback distance extraction failed for {pdb_file}: {e}")
            return np.zeros(self.embedding_dim // 3)

    def extract_chemical_features(self, pdb_file: str) -> np.ndarray:
        """Extract chemical property-based features from PDB structure"""
        try:
            # Amino acid properties
            aa_properties = {
                'ALA': [1.8, 0.0, 0.0, 0.0],   # hydrophobicity, charge, polar, aromatic
                'ARG': [-4.5, 1.0, 1.0, 0.0],
                'ASN': [-3.5, 0.0, 1.0, 0.0],
                'ASP': [-3.5, -1.0, 1.0, 0.0],
                'CYS': [2.5, 0.0, 0.0, 0.0],
                'GLN': [-3.5, 0.0, 1.0, 0.0],
                'GLU': [-3.5, -1.0, 1.0, 0.0],
                'GLY': [-0.4, 0.0, 0.0, 0.0],
                'HIS': [-3.2, 0.5, 1.0, 1.0],
                'ILE': [4.5, 0.0, 0.0, 0.0],
                'LEU': [3.8, 0.0, 0.0, 0.0],
                'LYS': [-3.9, 1.0, 1.0, 0.0],
                'MET': [1.9, 0.0, 0.0, 0.0],
                'PHE': [2.8, 0.0, 0.0, 1.0],
                'PRO': [-1.6, 0.0, 0.0, 0.0],
                'SER': [-0.8, 0.0, 1.0, 0.0],
                'THR': [-0.7, 0.0, 1.0, 0.0],
                'TRP': [-0.9, 0.0, 0.0, 1.0],
                'TYR': [-1.3, 0.0, 1.0, 1.0],
                'VAL': [4.2, 0.0, 0.0, 0.0]
            }

            if not self.biopython_available:
                return self._fallback_chemical_features(pdb_file, aa_properties)

            structure = self.pdb_parser.get_structure('antibody', pdb_file)

            # Extract residue information
            residue_properties = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        res_name = residue.get_resname()
                        if res_name in aa_properties and residue.has_id('CA'):
                            residue_properties.append(aa_properties[res_name])

            if not residue_properties:
                return np.zeros(self.embedding_dim // 3)

            residue_properties = np.array(residue_properties)

            # Calculate chemical features
            features = []

            # Overall composition
            hydrophobicity = residue_properties[:, 0]
            charge = residue_properties[:, 1]
            polarity = residue_properties[:, 2]
            aromaticity = residue_properties[:, 3]

            features.extend([
                np.mean(hydrophobicity),
                np.std(hydrophobicity),
                np.mean(charge),
                np.std(charge),
                np.mean(polarity),
                np.std(polarity),
                np.mean(aromaticity),
                np.std(aromaticity)
            ])

            # Charge distribution
            positive_residues = np.sum(charge > 0)
            negative_residues = np.sum(charge < 0)
            neutral_residues = np.sum(charge == 0)
            total_residues = len(charge)

            features.extend([
                positive_residues / total_residues,
                negative_residues / total_residues,
                neutral_residues / total_residues
            ])

            # Hydrophobic patches (simplified)
            hydrophobic_residues = hydrophobicity > 2.0
            hydrophobic_fraction = np.sum(hydrophobic_residues) / total_residues
            features.append(hydrophobic_fraction)

            # Polar surface area approximation
            polar_residues = polarity > 0
            polar_fraction = np.sum(polar_residues) / total_residues
            features.append(polar_fraction)

            # Aromatic content
            aromatic_residues = aromaticity > 0
            aromatic_fraction = np.sum(aromatic_residues) / total_residues
            features.append(aromatic_fraction)

            # Pad or truncate to desired size
            target_size = self.embedding_dim - 2 * (self.embedding_dim // 3)  # Remaining space
            if len(features) < target_size:
                features.extend([0] * (target_size - len(features)))

            return np.array(features[:target_size])

        except Exception as e:
            print(f"Error extracting chemical features from {pdb_file}: {e}")
            return np.zeros(self.embedding_dim // 3)

    def _fallback_chemical_features(self, pdb_file: str, aa_properties: Dict) -> np.ndarray:
        """Fallback chemical feature extraction without BioPython"""
        try:
            # Simple PDB parsing for residue names
            residue_names = []
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') and ' CA ' in line:
                        res_name = line[17:20].strip()
                        residue_names.append(res_name)

            if not residue_names:
                return np.zeros(self.embedding_dim // 3)

            # Calculate basic chemical properties
            features = []
            properties = []

            for res_name in residue_names:
                if res_name in aa_properties:
                    properties.append(aa_properties[res_name])

            if not properties:
                return np.zeros(self.embedding_dim // 3)

            properties = np.array(properties)

            # Basic statistics for each property
            for i in range(4):  # hydrophobicity, charge, polar, aromatic
                prop_values = properties[:, i]
                features.extend([
                    np.mean(prop_values),
                    np.std(prop_values)
                ])

            # Composition
            total = len(properties)
            features.extend([
                np.sum(properties[:, 1] > 0) / total,  # positive charge fraction
                np.sum(properties[:, 1] < 0) / total,  # negative charge fraction
                np.sum(properties[:, 2] > 0) / total,  # polar fraction
                np.sum(properties[:, 3] > 0) / total   # aromatic fraction
            ])

            # Pad to desired size
            target_size = self.embedding_dim // 3
            while len(features) < target_size:
                features.append(0.0)

            return np.array(features[:target_size])

        except Exception as e:
            print(f"Fallback chemical extraction failed for {pdb_file}: {e}")
            return np.zeros(self.embedding_dim // 3)

    def create_embeddings(self, sequences: List[Dict]) -> np.ndarray:
        """
        Create structure embeddings using the selected backend

        Args:
            sequences: List of sequence dictionaries

        Returns:
            Structural embeddings array
        """
        print(f"Creating structure embeddings using {self.structure_backend} backend...")

        # Step 1: Generate 3D structures using selected backend
        pdb_files = self.backend.generate_structures(sequences)

        if not pdb_files:
            print("No structures generated, returning zero embeddings")
            return np.zeros((len(sequences), self.embedding_dim))

        # Step 2: Extract three types of features from each structure
        embeddings = []

        for pdb_file in pdb_files:
            try:
                # Extract coordinate features
                coord_features = self.extract_coordinate_features(pdb_file)

                # Extract distance features
                distance_features = self.extract_distance_features(pdb_file)

                # Extract chemical features
                chemical_features = self.extract_chemical_features(pdb_file)

                # Combine all features
                combined_features = np.concatenate([
                    coord_features,
                    distance_features,
                    chemical_features
                ])

                # Ensure correct dimensionality
                if len(combined_features) < self.embedding_dim:
                    padding = np.zeros(self.embedding_dim - len(combined_features))
                    combined_features = np.concatenate([combined_features, padding])
                elif len(combined_features) > self.embedding_dim:
                    combined_features = combined_features[:self.embedding_dim]

                embeddings.append(combined_features)

            except Exception as e:
                print(f"Error processing structure {pdb_file}: {e}")
                # Add zero embedding for failed structure
                embeddings.append(np.zeros(self.embedding_dim))

        # Ensure we have embeddings for all input sequences
        while len(embeddings) < len(sequences):
            embeddings.append(np.zeros(self.embedding_dim))

        embeddings_array = np.array(embeddings)
        print(f"Created {self.structure_backend} embeddings: {embeddings_array.shape}")
        return embeddings_array


# Convenience functions for creating embeddings with different backends
def create_unified_structure_embeddings(sequences,
                                      structure_backend: str = "abodybuilder2",
                                      device: str = 'cpu',
                                      use_imgt_numbering: bool = True,
                                      embedding_dim: int = 128) -> Optional[np.ndarray]:
    """
    Create structure embeddings using specified backend (ABodyBuilder2 or IgFold)

    Args:
        sequences: List of antibody sequences (strings) or sequence dictionaries (with heavy/light chains)
        structure_backend: "abodybuilder2" or "igfold"
        device: Device (kept for compatibility)
        use_imgt_numbering: Whether to use IMGT numbering
        embedding_dim: Dimension of output embeddings

    Returns:
        Structure embeddings or None if failed
    """
    try:
        # Convert sequences to the format expected by the method
        sequence_dicts = []
        for i, seq_data in enumerate(sequences):
            if isinstance(seq_data, str):
                # Old format - just heavy chain sequence
                sequence_dicts.append({
                    'antibody_id': f'seq_{i:04d}',
                    'heavy_chain_aa': seq_data,
                    'light_chain_aa': ''  # No light chain for heavy-only sequences
                })
            else:
                # New format - paired data with heavy and light chains
                sequence_dicts.append({
                    'antibody_id': seq_data.get('sequence_id', f'seq_{i:04d}'),
                    'heavy_chain_aa': seq_data['heavy_chain'],
                    'light_chain_aa': seq_data.get('light_chain', '')
                })

        # Create unified method instance
        method = UnifiedStructureEmbeddingMethod(
            structure_backend=structure_backend,
            embedding_dim=embedding_dim,
            output_dir=f"temp_{structure_backend}_structures"
        )

        if not method.is_available():
            print(f"{structure_backend} backend not available")
            return None

        # Generate embeddings
        embeddings = method.create_embeddings(sequence_dicts)

        return embeddings

    except Exception as e:
        print(f"Unified structure embedding creation failed: {e}")
        return None


# Backward compatibility functions
def create_real_abodybuilder2_embeddings(sequences: List[str],
                                       device: str = 'cpu',
                                       use_imgt_numbering: bool = True) -> Optional[np.ndarray]:
    """Create ABodyBuilder2 embeddings (backward compatibility)"""
    return create_unified_structure_embeddings(
        sequences=sequences,
        structure_backend="abodybuilder2",
        device=device,
        use_imgt_numbering=use_imgt_numbering
    )


def create_real_igfold_embeddings(sequences: List[str],
                                device: str = 'cpu',
                                use_imgt_numbering: bool = True) -> Optional[np.ndarray]:
    """Create IgFold embeddings using the same embedding extraction methods"""
    return create_unified_structure_embeddings(
        sequences=sequences,
        structure_backend="igfold",
        device=device,
        use_imgt_numbering=use_imgt_numbering
    )


# Legacy class for backward compatibility
class RealABodyBuilder2Method(UnifiedStructureEmbeddingMethod):
    """Legacy ABodyBuilder2 method for backward compatibility"""

    def __init__(self,
                 output_dir: str = "abodybuilder2_structures",
                 use_imgt_numbering: bool = True,
                 embedding_dim: int = 128):
        """Legacy constructor for backward compatibility"""
        super().__init__(
            structure_backend="abodybuilder2",
            embedding_dim=embedding_dim,
            output_dir=output_dir
        )
        self.use_imgt_numbering = use_imgt_numbering


if __name__ == "__main__":
    # Test the unified structure embedding method
    print("Testing Unified Structure Embedding Method")

    # Test sequences
    test_sequences = [
        {
            'antibody_id': 'test_001',
            'heavy_chain_aa': 'EVQLVESGGGVVQPGGSLRLSCAASGFTFNSYGMHWVRQAPGKGLEWVAFIRYDGGNKYYADSVKGRFTISRDNSKNTLYLQMKSLRAEDTAVYYCANLKDSRYSGSYYDYWGQGTLVTVS',
            'light_chain_aa': 'VIWMTQSPSSLSASVGDRVTITCQASQDIRFYLNWYQQKPGKAPKLLISDASNMETGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCQQYDNLPFTFGPGTKVDFK'
        }
    ]

    # Test ABodyBuilder2 backend
    print("\n" + "="*60)
    print("TESTING ABODYBUILDER2 BACKEND")
    print("="*60)

    ab2_method = UnifiedStructureEmbeddingMethod(structure_backend="abodybuilder2")
    if ab2_method.is_available():
        ab2_embeddings = ab2_method.create_embeddings(test_sequences)
        print(f"ABodyBuilder2 embeddings shape: {ab2_embeddings.shape}")
        print(f"Sample embedding (first 10 values): {ab2_embeddings[0][:10]}")
    else:
        print("ABodyBuilder2 backend not available")

    # Test IgFold backend
    print("\n" + "="*60)
    print("TESTING IGFOLD BACKEND")
    print("="*60)

    igfold_method = UnifiedStructureEmbeddingMethod(structure_backend="igfold")
    if igfold_method.is_available():
        igfold_embeddings = igfold_method.create_embeddings(test_sequences)
        print(f"IgFold embeddings shape: {igfold_embeddings.shape}")
        print(f"Sample embedding (first 10 values): {igfold_embeddings[0][:10]}")
    else:
        print("IgFold backend not available")

    # Test standalone functions
    print("\n" + "="*60)
    print("TESTING STANDALONE FUNCTIONS")
    print("="*60)

    sequences_list = [seq['heavy_chain_aa'] for seq in test_sequences]

    # Test ABodyBuilder2 standalone
    ab2_standalone = create_real_abodybuilder2_embeddings(sequences_list)
    if ab2_standalone is not None:
        print(f"ABodyBuilder2 standalone embeddings shape: {ab2_standalone.shape}")
    else:
        print("ABodyBuilder2 standalone failed")

    # Test IgFold standalone
    igfold_standalone = create_real_igfold_embeddings(sequences_list)
    if igfold_standalone is not None:
        print(f"IgFold standalone embeddings shape: {igfold_standalone.shape}")
    else:
        print("IgFold standalone failed")

    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60)
    print("\nUsage examples:")
    print("1. Use ABodyBuilder2 backend:")
    print("   method = UnifiedStructureEmbeddingMethod(structure_backend='abodybuilder2')")
    print("   embeddings = method.create_embeddings(sequences)")
    print("\n2. Use IgFold backend:")
    print("   method = UnifiedStructureEmbeddingMethod(structure_backend='igfold')")
    print("   embeddings = method.create_embeddings(sequences)")
    print("\n3. Switch backends easily:")
    print("   for backend in ['abodybuilder2', 'igfold']:")
    print("       method = UnifiedStructureEmbeddingMethod(structure_backend=backend)")
    print("       if method.is_available():")
    print("           embeddings = method.create_embeddings(sequences)")
    print("           print(f'{backend} embeddings: {embeddings.shape}')")
