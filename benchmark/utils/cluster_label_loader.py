"""
Cluster Label Loader - Simplified Version
Looks up cluster labels based on filenames only
"""

import json
from pathlib import Path
from typing import Optional, Dict
from loguru import logger


class ClusterLabelLoader:
    """
    Loads cluster label mappings for SAM3 text prompts
    Quickly looks up corresponding clothing category labels by filename
    """

    def __init__(
        self,
        label_mapping_path: str,
        visual_buckets_path: str
    ):
        """
        Initialize cluster label loader

        Args:
            label_mapping_path: Path to cluster_label_mapping.json
                                Maps cluster_id → label (e.g., "0" → "Tank Top")
            visual_buckets_path: Path to visual_buckets.json
                                Maps cluster_id → list of source image paths
        """
        logger.info(f"Loading cluster label mapping from {label_mapping_path}")
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            self.cluster_to_label: Dict[str, str] = json.load(f)

        logger.info(f"Loading visual buckets from {visual_buckets_path}")
        with open(visual_buckets_path, 'r', encoding='utf-8') as f:
            cluster_to_paths: Dict[str, list] = json.load(f)

        # Build filename → cluster_id mapping
        logger.info("Building filename → cluster_id index...")
        self.filename_to_cluster: Dict[str, str] = {}

        for cluster_id, paths in cluster_to_paths.items():
            for path in paths:
                filename = Path(path).name
                if filename in self.filename_to_cluster:
                    # Duplicate filename, keep the first occurrence
                    logger.warning(
                        f"Duplicate filename {filename} in clusters "
                        f"{self.filename_to_cluster[filename]} and {cluster_id}. "
                        f"Keeping first occurrence."
                    )
                else:
                    self.filename_to_cluster[filename] = cluster_id

        self.num_clusters = len(self.cluster_to_label)
        self.num_filenames = len(self.filename_to_cluster)

        logger.success(
            f"Cluster label loader initialized:\n"
            f"  - {self.num_clusters} clusters\n"
            f"  - {self.num_filenames} filenames indexed"
        )

    def get_label_for_filename(self, filename: str) -> Optional[str]:
        """
        Get cluster label by filename

        Args:
            filename: Image filename (e.g., "000003071394.png")

        Returns:
            Cluster label (e.g., "Crew Neck T-shirt") or None
        """
        # Extract filename (supports passing full path)
        filename = Path(filename).name

        cluster_id = self.filename_to_cluster.get(filename)
        if cluster_id is None:
            return None

        return self.cluster_to_label.get(cluster_id)

    def get_stats(self) -> Dict:
        """Get statistics"""
        return {
            'num_clusters': self.num_clusters,
            'num_filenames': self.num_filenames,
            'sample_labels': list(self.cluster_to_label.values())[:5]
        }
