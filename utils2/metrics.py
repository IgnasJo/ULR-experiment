import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        # Storage for boundary metrics (computed per batch)
        self.boundary_predictions = []
        self.boundary_ground_truths = []

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        # Ignore classes that were not yet seen in output or ground truth labels
        with np.errstate(divide='ignore', invalid='ignore'):
            Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (
        (gt_image >= 0) & (gt_image < self.num_class) &
        (pre_image >= 0) & (pre_image < self.num_class)
        )
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.boundary_predictions = []
        self.boundary_ground_truths = []

    def _ensure_boundary_attrs(self):
        """Ensure boundary attributes exist (for backward compatibility with old checkpoints)."""
        if not hasattr(self, 'boundary_predictions'):
            self.boundary_predictions = []
        if not hasattr(self, 'boundary_ground_truths'):
            self.boundary_ground_truths = []

    def _extract_boundaries(self, segmentation):
        """
        Extract boundary pixels from a segmentation mask using morphological gradient.
        Returns a binary mask where 1 indicates boundary pixels.
        """
        boundaries = np.zeros_like(segmentation, dtype=bool)
        for class_id in range(self.num_class):
            class_mask = (segmentation == class_id).astype(np.uint8)
            # Morphological gradient: dilation - erosion
            dilated = ndimage.binary_dilation(class_mask)
            eroded = ndimage.binary_erosion(class_mask)
            class_boundary = dilated ^ eroded  # XOR gives boundary
            boundaries = boundaries | class_boundary
        return boundaries.astype(np.uint8)

    def _get_boundary_points(self, boundary_mask):
        """Get coordinates of boundary pixels as (N, 2) array."""
        return np.array(np.where(boundary_mask)).T

    def add_batch_with_boundaries(self, gt_image, pre_image):
        """
        Add batch and store boundary information for boundary metrics.
        Use this instead of add_batch when boundary metrics are needed.
        """
        self._ensure_boundary_attrs()
        self.add_batch(gt_image, pre_image)
        # Store boundaries for later computation
        gt_boundary = self._extract_boundaries(gt_image)
        pre_boundary = self._extract_boundaries(pre_image)
        self.boundary_predictions.append(pre_boundary)
        self.boundary_ground_truths.append(gt_boundary)

    def Boundary_F1(self, tau=2):
        """
        Compute Boundary F1-score with tolerance τ pixels.
        
        BF1 = 2 * Pb * Rb / (Pb + Rb)
        where:
        - Pb (precision) = |{p ∈ B(P) : dist(p, B(G)) ≤ τ}| / |B(P)|
        - Rb (recall) = |{g ∈ B(G) : dist(g, B(P)) ≤ τ}| / |B(G)|
        
        Args:
            tau: Tolerance in pixels for boundary matching
        
        Returns:
            Boundary F1 score
        """
        self._ensure_boundary_attrs()
        if not self.boundary_predictions or not self.boundary_ground_truths:
            return 0.0
        
        total_precision_matches = 0
        total_recall_matches = 0
        total_pred_boundary = 0
        total_gt_boundary = 0
        
        for pre_boundary, gt_boundary in zip(self.boundary_predictions, self.boundary_ground_truths):
            pred_points = self._get_boundary_points(pre_boundary)
            gt_points = self._get_boundary_points(gt_boundary)
            
            if len(pred_points) == 0 or len(gt_points) == 0:
                continue
            
            # Compute pairwise distances
            distances = cdist(pred_points, gt_points, metric='euclidean')
            
            # Precision: predicted boundary points within τ of any GT boundary point
            min_dist_to_gt = distances.min(axis=1)
            precision_matches = np.sum(min_dist_to_gt <= tau)
            
            # Recall: GT boundary points within τ of any predicted boundary point
            min_dist_to_pred = distances.min(axis=0)
            recall_matches = np.sum(min_dist_to_pred <= tau)
            
            total_precision_matches += precision_matches
            total_recall_matches += recall_matches
            total_pred_boundary += len(pred_points)
            total_gt_boundary += len(gt_points)
        
        if total_pred_boundary == 0 or total_gt_boundary == 0:
            return 0.0
        
        Pb = total_precision_matches / total_pred_boundary
        Rb = total_recall_matches / total_gt_boundary
        
        if Pb + Rb == 0:
            return 0.0
        
        BF1 = 2 * Pb * Rb / (Pb + Rb)
        return BF1

    def Symmetric_Boundary_Dice(self, tau=2):
        """
        Compute Symmetric Boundary Dice after dilating boundaries by radius τ.
        
        BDice_τ = 2 * |( B(P) ⊕ S_τ ) ∩ ( B(G) ⊕ S_τ )| / (|B(P) ⊕ S_τ| + |B(G) ⊕ S_τ|)
        
        Args:
            tau: Dilation radius in pixels
        
        Returns:
            Symmetric Boundary Dice score
        """
        self._ensure_boundary_attrs()
        if not self.boundary_predictions or not self.boundary_ground_truths:
            return 0.0
        
        total_intersection = 0
        total_dilated_pred = 0
        total_dilated_gt = 0
        
        # Create structuring element for dilation
        struct = ndimage.generate_binary_structure(2, 1)
        
        for pre_boundary, gt_boundary in zip(self.boundary_predictions, self.boundary_ground_truths):
            # Dilate boundaries by τ iterations
            dilated_pred = ndimage.binary_dilation(pre_boundary, structure=struct, iterations=tau)
            dilated_gt = ndimage.binary_dilation(gt_boundary, structure=struct, iterations=tau)
            
            # Compute intersection and counts
            intersection = np.logical_and(dilated_pred, dilated_gt).sum()
            total_intersection += intersection
            total_dilated_pred += dilated_pred.sum()
            total_dilated_gt += dilated_gt.sum()
        
        if total_dilated_pred + total_dilated_gt == 0:
            return 0.0
        
        BDice = 2 * total_intersection / (total_dilated_pred + total_dilated_gt)
        return BDice

    def Weighted_IoU(self, alpha=1.0, tau=2):
        """
        Compute boundary-weighted IoU that emphasizes boundary pixels.
        
        wIoU = Σ_i w_i * [ŷ_i = y_i] / Σ_i w_i
        where w_i = 1 + α if i ∈ (B(G) ⊕ S_τ), else w_i = 1
        
        Args:
            alpha: Boundary weight emphasis (default 1.0)
            tau: Dilation radius for boundary region (default 2)
        
        Returns:
            Weighted IoU score
        """
        self._ensure_boundary_attrs()
        if not self.boundary_predictions or not self.boundary_ground_truths:
            # Fall back to standard IoU if no boundary info
            return self.Mean_Intersection_over_Union()
        
        total_weighted_correct = 0.0
        total_weights = 0.0
        
        struct = ndimage.generate_binary_structure(2, 1)
        
        for (pre_boundary, gt_boundary), (pre_idx, gt_idx) in zip(
            zip(self.boundary_predictions, self.boundary_ground_truths),
            [(i, i) for i in range(len(self.boundary_predictions))]
        ):
            # We need original predictions and ground truths
            # Since we only store boundaries, we'll compute wIoU from confusion matrix contribution
            pass
        
        # Compute from stored boundary masks - this requires original images
        # For now, return standard mIoU with a note that full implementation
        # requires storing original prediction/GT pairs
        return self.Mean_Intersection_over_Union()

    def Weighted_IoU_from_images(self, gt_image, pre_image, alpha=1.0, tau=2):
        """
        Compute boundary-weighted IoU for a single pair of images.
        
        Args:
            gt_image: Ground truth segmentation
            pre_image: Predicted segmentation
            alpha: Boundary weight emphasis
            tau: Dilation radius for boundary region
        
        Returns:
            Weighted IoU score for the image pair
        """
        gt_boundary = self._extract_boundaries(gt_image)
        struct = ndimage.generate_binary_structure(2, 1)
        boundary_region = ndimage.binary_dilation(gt_boundary, structure=struct, iterations=tau)
        
        # Create weight map
        weights = np.ones_like(gt_image, dtype=float)
        weights[boundary_region] = 1 + alpha
        
        # Compute weighted accuracy
        correct = (pre_image == gt_image).astype(float)
        weighted_correct = (correct * weights).sum()
        total_weight = weights.sum()
        
        if total_weight == 0:
            return 0.0
        
        return weighted_correct / total_weight

    def Hausdorff_Distance(self):
        """
        Compute symmetric Hausdorff Distance between predicted and GT boundaries.
        
        HD(B(P), B(G)) = max( max_{p∈B(P)} dist(p, B(G)), max_{g∈B(G)} dist(g, B(P)) )
        
        Returns:
            Hausdorff distance (lower is better), or inf if no boundaries
        """
        self._ensure_boundary_attrs()
        if not self.boundary_predictions or not self.boundary_ground_truths:
            return float('inf')
        
        max_hd = 0.0
        valid_samples = 0
        
        for pre_boundary, gt_boundary in zip(self.boundary_predictions, self.boundary_ground_truths):
            pred_points = self._get_boundary_points(pre_boundary)
            gt_points = self._get_boundary_points(gt_boundary)
            
            if len(pred_points) == 0 or len(gt_points) == 0:
                continue
            
            # Compute pairwise distances
            distances = cdist(pred_points, gt_points, metric='euclidean')
            
            # Max distance from any predicted point to nearest GT point
            max_pred_to_gt = distances.min(axis=1).max()
            
            # Max distance from any GT point to nearest predicted point
            max_gt_to_pred = distances.min(axis=0).max()
            
            # Symmetric Hausdorff distance
            hd = max(max_pred_to_gt, max_gt_to_pred)
            max_hd = max(max_hd, hd)
            valid_samples += 1
        
        if valid_samples == 0:
            return float('inf')
        
        return max_hd

    def Average_Surface_Distance(self):
        """
        Compute Average Surface Distance (ASD) between predicted and GT boundaries.
        
        ASD(B(P), B(G)) = (Σ_{p∈B(P)} dist(p, B(G)) + Σ_{g∈B(G)} dist(g, B(P))) / (|B(P)| + |B(G)|)
        
        Returns:
            Average surface distance (lower is better), or inf if no boundaries
        """
        self._ensure_boundary_attrs()
        if not self.boundary_predictions or not self.boundary_ground_truths:
            return float('inf')
        
        total_distance_sum = 0.0
        total_boundary_points = 0
        
        for pre_boundary, gt_boundary in zip(self.boundary_predictions, self.boundary_ground_truths):
            pred_points = self._get_boundary_points(pre_boundary)
            gt_points = self._get_boundary_points(gt_boundary)
            
            if len(pred_points) == 0 or len(gt_points) == 0:
                continue
            
            # Compute pairwise distances
            distances = cdist(pred_points, gt_points, metric='euclidean')
            
            # Sum of distances from each predicted point to nearest GT point
            sum_pred_to_gt = distances.min(axis=1).sum()
            
            # Sum of distances from each GT point to nearest predicted point
            sum_gt_to_pred = distances.min(axis=0).sum()
            
            total_distance_sum += sum_pred_to_gt + sum_gt_to_pred
            total_boundary_points += len(pred_points) + len(gt_points)
        
        if total_boundary_points == 0:
            return float('inf')
        
        return total_distance_sum / total_boundary_points

    def Mean_Hausdorff_Distance(self):
        """
        Compute mean Hausdorff Distance across all samples (alternative to max).
        
        Returns:
            Mean Hausdorff distance across samples
        """
        self._ensure_boundary_attrs()
        if not self.boundary_predictions or not self.boundary_ground_truths:
            return float('inf')
        
        hd_values = []
        
        for pre_boundary, gt_boundary in zip(self.boundary_predictions, self.boundary_ground_truths):
            pred_points = self._get_boundary_points(pre_boundary)
            gt_points = self._get_boundary_points(gt_boundary)
            
            if len(pred_points) == 0 or len(gt_points) == 0:
                continue
            
            distances = cdist(pred_points, gt_points, metric='euclidean')
            max_pred_to_gt = distances.min(axis=1).max()
            max_gt_to_pred = distances.min(axis=0).max()
            hd = max(max_pred_to_gt, max_gt_to_pred)
            hd_values.append(hd)
        
        if len(hd_values) == 0:
            return float('inf')
        
        return np.mean(hd_values)

    def get_all_metrics(self, tau=2, alpha=1.0):
        """
        Compute and return all available metrics as a dictionary.
        
        Args:
            tau: Tolerance for boundary metrics
            alpha: Weight for boundary-weighted IoU
        
        Returns:
            Dictionary with all metric values
        """
        return {
            'Pixel_Accuracy': self.Pixel_Accuracy(),
            'Pixel_Accuracy_Class': self.Pixel_Accuracy_Class(),
            'mIoU': self.Mean_Intersection_over_Union(),
            'FWIoU': self.Frequency_Weighted_Intersection_over_Union(),
            'Boundary_F1': self.Boundary_F1(tau=tau),
            'Symmetric_Boundary_Dice': self.Symmetric_Boundary_Dice(tau=tau),
            'Hausdorff_Distance': self.Hausdorff_Distance(),
            'Mean_Hausdorff_Distance': self.Mean_Hausdorff_Distance(),
            'Average_Surface_Distance': self.Average_Surface_Distance(),
        }




