import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries

class BleedingEdgeEvaluator:
    def __init__(self, max_dist=20, bf_thresholds=range(1, 11)):
        self.max_dist = max_dist
        self.bf_thresholds = list(bf_thresholds)
        
        # 1. For Distance-to-Edge Plot
        self.error_counts = np.zeros(max_dist + 1)
        self.total_counts = np.zeros(max_dist + 1)
        
        # 2. For BF Score Curve
        # Store [precision_sum, recall_sum, count] for each theta
        self.bf_data = {theta: {'p_num': 0, 'p_den': 0, 'r_num': 0, 'r_den': 0} 
                        for theta in self.bf_thresholds}

    def add_batch(self, gt, pred):
        # 1. Boundary Detection (all class transitions)
        gt_boundary = find_boundaries(gt, mode='thick')
        pred_boundary = find_boundaries(pred, mode='thick')
        
        # 2. Distance Transforms
        # dist_to_gt: distance from every pixel to the nearest GT boundary
        dist_to_gt = distance_transform_edt(~gt_boundary)
        # dist_to_pred: distance from every pixel to the nearest Pred boundary
        dist_to_pred = distance_transform_edt(~pred_boundary)
        
        # --- TASK A: Distance-to-Edge Error Calculation ---
        errors = (gt != pred)
        dist_map_clipped = np.clip(dist_to_gt.astype(int), 0, self.max_dist)
        for d in range(self.max_dist + 1):
            mask = (dist_map_clipped == d)
            self.error_counts[d] += np.sum(errors[mask])
            self.total_counts[d] += np.sum(mask)

        # --- TASK B: BF Score Multi-Threshold Calculation ---
        for theta in self.bf_thresholds:
            # Precision: Predicted points within theta of GT
            precision_hits = np.logical_and(pred_boundary, dist_to_gt <= theta)
            self.bf_data[theta]['p_num'] += np.sum(precision_hits)
            self.bf_data[theta]['p_den'] += np.sum(pred_boundary)
            
            # Recall: GT points within theta of Predicted
            recall_hits = np.logical_and(gt_boundary, dist_to_pred <= theta)
            self.bf_data[theta]['r_num'] += np.sum(recall_hits)
            self.bf_data[theta]['r_den'] += np.sum(gt_boundary)

    def plot_bleeding_edge(self, save_path=None):
        safe_total = np.where(self.total_counts == 0, 1, self.total_counts)
        error_prob = self.error_counts / safe_total
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(self.max_dist + 1), error_prob, 'r-o', linewidth=2)
        plt.title("Distance-to-Edge Error Plot (All Transitions)")
        plt.xlabel("Distance from Nearest Boundary (pixels)")
        plt.ylabel("Error Probability")
        plt.grid(True, alpha=0.3)
        if save_path: plt.savefig(save_path)
        plt.show()

    def plot_bf_curve(self, save_path=None):
        thetas = self.bf_thresholds
        f1_scores = []
        
        for t in thetas:
            d = self.bf_data[t]
            p = d['p_num'] / d['p_den'] if d['p_den'] > 0 else 0
            r = d['r_num'] / d['r_den'] if d['r_den'] > 0 else 0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
            f1_scores.append(f1)
            
        plt.figure(figsize=(8, 5))
        plt.plot(thetas, f1_scores, 'b-s', linewidth=2)
        plt.title("Boundary F1 (BF) Score vs. Distance Tolerance (θ)")
        plt.xlabel("Tolerance θ (pixels)")
        plt.ylabel("BF Score")
        plt.xticks(thetas)
        plt.grid(True, alpha=0.3)
        if save_path: plt.savefig(save_path)
        plt.show()