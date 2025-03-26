import torch
import numpy as np
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
from dataset import RAITEDataset
from argparse import ArgumentParser
from arguments import EvalParams

import sys

import os
import json

class DetectionEvaluator:
    """
    DetectionEvaluator provides flexibility to evaluate object detection models
    with either a trained model or precomputed predictions from a JSON file.
    It supports custom IoU and confidence thresholds, and configurable area
    ranges for small, medium, and large object categories.

    Args:
    - model (torch.nn.Module, optional): The object detection model, e.g., Faster R-CNN.
    - dataset (Dataset, optional): The dataset object (e.g., RAITEDataset).
    - predictions_json (str or dict, optional): Path to or dictionary of precomputed predictions.
    - iou_thresholds (list of float, optional): IoU thresholds to evaluate AP at.
    - confidence_thresholds (list of float, optional): List of confidence thresholds.
    - area_ranges (dict, optional): Dictionary defining area ranges for small, medium, and large objects.
    - display_predictions (bool, optional): Whether to display predictions during evaluation.

    Raises:
    - ValueError: If neither a model nor predictions JSON is provided.
    """
    def __init__(self, model=None, dataset=None, predictions_json=None,
                 iou_thresholds=[0.5, 0.75],
                 confidence_thresholds=np.arange(0,1,.1).tolist(),  #[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0],
                 area_ranges={'S': [0, 32**2], 'M': [32**2, 96**2], 'L': [96**2, np.inf]},
                 display_predictions=False):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.display_predictions = display_predictions
        self.iou_thresholds = iou_thresholds
        self.confidence_thresholds = confidence_thresholds
        self.area_ranges = area_ranges

        # Validate inputs and set attributes
        if model is not None:
            self._initialize_with_model(model, dataset)
        elif predictions_json is not None:
            self._initialize_with_json(predictions_json)
        else:
            raise ValueError("Either a model or predictions_json must be provided for evaluation.")


    def _initialize_with_model(self, model, dataset):
        """Initialize evaluator with a model and dataset."""
        self.model = model.to(self.device)
        self.dataset = dataset
        self.predictions = None


    def _initialize_with_json(self, predictions_json):
        """Initialize evaluator with precomputed predictions from JSON."""
        self.model = None
        self.dataset = None
        self.predictions = self._load_predictions_from_json(predictions_json)


    def _load_predictions_from_json(self, predictions_json):
        """Load predictions from a JSON file or dictionary.

        Args:
        - predictions_json (str or dict): File path or dictionary with predictions.

        Returns:
        - dict: Loaded predictions data.
        """
        if isinstance(predictions_json, str):
            with open(predictions_json, 'r') as file:
                return json.load(file)
        elif isinstance(predictions_json, dict):
            return predictions_json
        else:
            raise TypeError("predictions_json must be a file path (str) or a dictionary.")


    def evaluate(self):
        """
        Evaluates the model on the dataset.
        Args:
        - confidence_threshold_step: Step size for confidence thresholding.
        """
        if self.predictions is not None:
            # Use precomputed predictions from JSON
            predictions = self.predictions
            res_dict = self.compute_tp_fp_fn_for_all_params()
            res_dict = self._calculate_ap(res_dict)
            area_based_results = None # TODO: implement
        else:
            self.model.eval()
            
            res_dict = self.compute_tp_fp_fn_for_all_params()

            # Compute AP at different IoU thresholds
            res_dict = self._calculate_ap(res_dict)
            
            # Compute area-based AP (AS, AM, AL)
            area_based_results = self._calculate_area_based_ap()

        return res_dict, area_based_results
    

    def compute_tp_fp_fn_for_all_params(self):
        '''
        Takes in the iou thresholds and interval of confidence score to evaluate across, and computes dicitionary of
        TP, FP, FN.

        Returns:
            - Dictionary for each IOU threshold and each score threshold.
        '''
        res_dict = {iou_thresh: {confidence_thresh: {'TP': 0, 'FP': 0, 'FN': 0} 
                              for confidence_thresh in self.confidence_thresholds}
                 for iou_thresh in self.iou_thresholds}
        
        if self.predictions is not None:

            for iou_thresh in self.iou_thresholds:
                for confidence_thresh in self.confidence_thresholds:
                    for frame_id, data in self.predictions.items():

                         # Define the device, e.g., 'cuda:0' if available
                        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                        
                        # Convert prediction data to tensors on the correct device
                        boxes_pred = torch.tensor(data['boxes'], device=device, dtype=torch.float32)
                        scores_pred = torch.tensor(data['scores'], device=device, dtype=torch.float32)
                        labels_pred = torch.tensor(data['labels'], device=device, dtype=torch.int64)
                        
                        predictions = [{
                            'boxes': boxes_pred,
                            'scores': scores_pred,
                            'labels': labels_pred
                        }]
                        
                        # Convert target data to tensors on the correct device
                        boxes_true = torch.tensor(data['target_boxes'], device=device, dtype=torch.float32)
                        labels_true = torch.tensor(data['target_labels'], device=device, dtype=torch.int64)
                        
                        target = {
                            'boxes': boxes_true,
                            'labels': labels_true
                        }

                        tp, fp, fn = self._compute_tp_fp_fn(predictions, target, iou_thresh, confidence_thresh) # add predictions in right format

                        res_dict[iou_thresh][confidence_thresh]['TP'] += tp
                        res_dict[iou_thresh][confidence_thresh]['FP'] += fp
                        res_dict[iou_thresh][confidence_thresh]['FN'] += fn

        else:

            for iou_thresh in self.iou_thresholds:
                for confidence_thresh in self.confidence_thresholds:
                    for image, target, _, _ in self.dataset:

                        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() # Convert from (H, W, C) to (C, H, W)

                        with torch.no_grad():
                            predictions = self.model([image_tensor.to(self.device)])
                            tp, fp, fn = self._compute_tp_fp_fn(predictions, target, iou_thresh, confidence_thresh)

                            res_dict[iou_thresh][confidence_thresh]['TP'] += tp
                            res_dict[iou_thresh][confidence_thresh]['FP'] += fp
                            res_dict[iou_thresh][confidence_thresh]['FN'] += fn

                            if self.display_predictions:
                                self._display_predictions(image, predictions[0], target)
                    
        return res_dict
    

    def _compute_tp_fp_fn(self, predictions, target, iou_thresh, confidence_thresh):
        """
        Computes true positives, false positives, and false negatives.
        Args:
        - predictions: Model predictions.
        - target: Ground truth annotations.

        Returns:
        - tp, fp, fn: True positives, false positives, and false negatives.
        """
        boxes_pred = predictions[0]['boxes'].to(self.device)  # Move predictions to the same device
        labels_pred = predictions[0]['labels'].to(self.device)  # Ensure labels are also on the same device
        scores_pred = predictions[0]['scores'].to(self.device)  # Ensure scores are on the same device

        boxes_true = target['boxes'].to(self.device)  # Move ground truth boxes to the same device
        labels_true = target['labels'].to(self.device)

        tp_count = 0
        fp_count = 0
        fn_count = 0
        
       #  this function operates on each image in the test set.

        if len(boxes_pred) == 0:  # No predictions
            fn_count += len(boxes_true)  # All true boxes are false negatives
            return tp_count, fp_count, fn_count  # Return counts
        
        if len(boxes_true) == 0:
            fp_count += len(boxes_pred)
            return tp_count, fp_count, fn_count

        ious = box_iou(boxes_pred, boxes_true) # TODO: verify the logic here
        max_iou, max_idx = ious.max(dim=1)

        # Initialize a boolean array to keep track of matched ground truth boxes
        matched_gt = torch.zeros(boxes_true.size(0), dtype=torch.bool, device=self.device)

        for i, (score, iou, label_pred) in enumerate(zip(scores_pred, max_iou, labels_pred)):
            if score > confidence_thresh:  # Use the provided confidence threshold
                gt_idx = max_idx[i]
                label_true = labels_true[gt_idx]
               
                if iou >= iou_thresh and label_pred == label_true:
                    tp_count += 1
                    matched_gt[gt_idx] = True  # Mark this ground truth box as matched
                elif iou < iou_thresh and label_pred == label_true:  # Correct class but IoU less than threshold:
                    fp_count += 1
                else:
                    fp_count += 1
        
        fn_count = (len(boxes_true) - matched_gt.sum().item())

        return tp_count, fp_count, fn_count


    def _calculate_ap(self, res_dict):
        """
        Calculates average precision at different IoU thresholds.
        """
        ap_results = {}

        for i,iou_thresh in enumerate(self.iou_thresholds):
            ap_results[iou_thresh] = {}  # Initialize a sub-dictionary for this IoU threshold
        
            # Iterate through each confidence threshold
            ap_terms = []
            for i, confidence_thresh in enumerate(self.confidence_thresholds):

                if i == (len(self.confidence_thresholds)-1): break
                # Retrieve TP, FP, FN counts for the current IoU and confidence threshold
                tp_sum = res_dict[iou_thresh][confidence_thresh]['TP']
                fp_sum = res_dict[iou_thresh][confidence_thresh]['FP']
                fn_sum = res_dict[iou_thresh][confidence_thresh]['FN']

                # Calculate precision and recall
                precision = tp_sum / (tp_sum + fp_sum + 1e-10)
                recall = tp_sum / (tp_sum + fn_sum + 1e-10)

                 # Retrieve TP, FP, FN counts for the current IoU and confidence threshold
                tp_sum = res_dict[iou_thresh][self.confidence_thresholds[i+1]]['TP']
                fn_sum = res_dict[iou_thresh][self.confidence_thresholds[i+1]]['FN']

                next_recall = tp_sum / (tp_sum + fn_sum)

                # Calculate AP using the formula
                ap_terms.append((recall - next_recall)*precision)

                # Store the AP result in the ap_results dictionary
                ap_results[iou_thresh][confidence_thresh] = [precision, recall] # previously stored the result at each confidence

            ap_results[iou_thresh]['final_ap'] = np.sum(ap_terms)

        # Add the AP results to the original res_dict
        res_dict['ap_results'] = ap_results

        return res_dict
        

    def _calculate_area_based_ap(self):
        """
        Calculate area-based AP (small, medium, large objects).
        """
        area_results = {'AS': 0, 'AM': 0, 'AL': 0}
        # Calculate AP for each area range
        for area, (min_area, max_area) in self.area_ranges.items():
            # Area-specific logic goes here, similar to how AP is computed
            pass

        return area_results
    
    def find_highest_f1_score(self, ap_results_dict):
        """
        Finds the highest F1 score across all IoU and confidence thresholds.

        Args:
            ap_results_dict (dict): Dictionary containing precision and recall results at different IoU and confidence thresholds.

        Returns:
            dict: A dictionary containing the highest F1 score and its corresponding precision, recall, IoU, and confidence threshold.
        """
        max_f1 = 0
        best_result = {
            'f1_score': 0,
            'precision': 0,
            'recall': 0,
            'iou_threshold': None,
            'confidence_threshold': None
        }

        for iou_thresh in self.iou_thresholds:
            for conf_thresh in self.confidence_thresholds:
                # Get precision and recall from the results dictionary
                res = ap_results_dict["ap_results"].get(iou_thresh, {}).get(conf_thresh)
                if res is None:
                    continue  # Skip if no result for this threshold

                precision = res[0]
                recall = res[1]
                
                # Calculate F1 score
                f1 = self.f1_score(precision=precision, recall=recall)
                
                # Update if this is the highest F1 score found so far
                if f1 > max_f1:
                    max_f1 = f1
                    best_result.update({
                        'f1_score': f1,
                        'precision': precision,
                        'recall': recall,
                        'iou_threshold': iou_thresh,
                        'confidence_threshold': conf_thresh
                    })

        return best_result
    
    def f1_score(self, precision, recall):
        if precision + recall == 0:
            return 0  # Avoid division by zero
        return 2 * (precision * recall) / (precision + recall)
    

    def plot_precision_recall_curve(self, ap_results_dict, savePath, highlight_point=None):
        """
        Plots separate Precision-Recall (PR) curves for each IoU threshold and highlights the point
        with the highest F1 score if provided.

        Args:
            ap_results_dict (dict): Dictionary containing precision and recall results at different IoU and confidence thresholds.
            savePath (str): Directory path to save the plot.
            highlight_point (dict, optional): Dictionary with keys 'f1_score', 'precision', 'recall',
                                            'iou_threshold', and 'confidence_threshold' to mark
                                            the best F1 score on the plot.
        """
        plt.style.use("ggplot")
        # Iterate over IoU thresholds
        for iou_thresh in self.iou_thresholds:
            plt.figure(figsize=(12, 8))  # Create a new figure for each IoU threshold

            precisions = []
            recalls = []

            # Iterate over confidence thresholds
            for i, conf_thresh in enumerate(self.confidence_thresholds):
                if i == (len(self.confidence_thresholds) - 1): 
                    break  # Stop at the last threshold

                res = ap_results_dict["ap_results"][iou_thresh][conf_thresh]
                precisions.append(res[0])
                recalls.append(res[1])

            # Plot the PR curve for the current IoU threshold
            plt.plot(recalls, precisions, label=f'Confidence thresholds : Step size 0.001', marker='o', markersize=6, linewidth=2,  zorder=1)

            # Highlight the point with the highest F1 score, if provided and matches the current IoU
            if highlight_point and highlight_point['iou_threshold'] == iou_thresh:
                plt.scatter(
                    highlight_point['recall'],
                    highlight_point['precision'],
                    color='orange',
                    s=200,              # Increase size for visibility
                    marker='*',
                    edgecolor='black',  # Add black edge for contrast
                    linewidth=1,
                    zorder=3,
                    label=f'Best F1: {highlight_point["f1_score"]:.2f}'
                )

            # Add labels, title, and legend
            plt.xlabel('Recall', fontsize=15)
            plt.ylabel('Precision', fontsize=15)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title(f'Precision-Recall Curve for IoU {iou_thresh}', fontsize=15, weight='bold')

            # Customize legend with larger font and shadow
            plt.legend(loc='lower left', fontsize=15, shadow=True, fancybox=True, framealpha=0.8)

            # Customize the grid lines to make them dashed and light gray
            plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

            # Set axis limits to fit the full PR curve range
            plt.xlim(0, 1)
            plt.ylim(0, 1)

            # Save the plot with a unique filename for each IoU threshold
            plt.savefig(f'{savePath}/pr_curve_iou_{iou_thresh}.png')

            # Close the figure to free up memory
            plt.close()




###########################

def evaluate_all_test_sets(model_path='models/ugvs/fasterrcnn_resnet50_fpn_ugv_v7.pth', dir="data/archive/test_sets/ugv"):
    '''
    
    '''
    model_name = os.path.basename(model_path)
    model_name = os.path.splitext(model_name)[0]
    model = torch.load(model_path)

    average_precisions_50 = []
    average_precisions_75 = []
    folder_paths = [os.path.join(dir, name) for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]
    for path in folder_paths:

        dataset = RAITEDataset(f'{path}/images', f'{path}/labels',400 , 400, 2)
        evaluator = DetectionEvaluator(model, dataset)
        ap_results, area_results = evaluator.evaluate()

        # for each test set, create a folder f in: test_sets/results/model_type
        # or results and print results to it.
        folder_name = os.path.basename(path)

        dir_name = os.path.dirname(dir)
        base_name = os.path.basename(dir)
        results_dir = f'{dir_name}/results/{base_name}/{model_name}'
        os.makedirs(results_dir, exist_ok=True)

        fp = os.path.join(results_dir, folder_name) # makes a folder for each test for the model
        os.makedirs(fp, exist_ok=True)
        result_file = os.path.join(fp, "results.txt") # results file for that test

        # ALL PLOTS HERE:
        # Add all plots to the results folder for each test
        evaluator.plot_precision_recall_curve(ap_results, savePath=fp)

        # DICTIONARY DATA HERE:
        # Dump dictionary into the text file.
        with open(result_file, 'w') as file:
            json.dump(ap_results, file, indent=4)

        # SAVE AVERAGE PRECISION FOR TEST TO BE USED FOR mAP

        average_precisions_50.append(ap_results['ap_results'][0.5]['final_ap'])
        average_precisions_75.append(ap_results['ap_results'][0.75]['final_ap'])

    # Once complete, find mAP across all.
    file_path = os.path.join(results_dir, "overall_results.txt") # creates results.txt 
    mAP_50 = sum(average_precisions_50) / len(average_precisions_50)  
    mAP_75 = sum(average_precisions_75) / len(average_precisions_75)

    with open(file_path, 'a') as f:  # Use 'a' to append to the file
        f.write(f'Mean Average Precision at 50% IoU : {mAP_50:.4f}\n')
        f.write(f'Mean Average Precision at 75% IoU : {mAP_75:.4f}\n')

    
if __name__ == '__main__':
    # dataset = RAITEDataset('data/archive/test_sets/ugv/t2_autonomyPark150/images', 
    #                        'data/archive/test_sets/ugv/t2_autonomyPark150/labels', 
    #                        400, 400, 2, {0:0, 4:4, 6:6})
    # model = torch.load('models/ugvs/fasterrcnn_resnet50_fpn_ugv_v3.pth')
    # evaluator = DetectionEvaluator(model=model, dataset=dataset, predictions_json=None)
    # Set up command line argument parser
    
    parser = ArgumentParser(description="evaluation script parameters")
    eval_params = EvalParams(parser)
    args = parser.parse_args(sys.argv[1:])
    eval_args = eval_params.extract(args)

    # If only JSON is provided:
    evaluator = DetectionEvaluator(model=None, dataset=None, predictions_json=eval_args.json)
    ap_results, area_results = evaluator.evaluate()
    f1 = evaluator.find_highest_f1_score(ap_results)
    print(f1)
    evaluator.plot_precision_recall_curve(ap_results, savePath="/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify",highlight_point=f1)

    #print(ap_results)
    # print(area_results)
    #evaluate_all_test_sets(model_path='models/drones/fasterrcnn_resnet50_fpn_drone_v3.pth', dir="data/archive/test_sets/drone")

    # evaluate_all_test_sets(model_path='models/drones/fasterrcnn_resnet50_fpn_drone_comp_v1.pth',
    #  dir="data/archive/test_sets/drone")
