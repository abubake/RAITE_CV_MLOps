import torch
import numpy as np
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
from dataset import RAITEDataset

import os
import json

class DetectionEvaluator:
    def __init__(self, model, dataset, iou_thresholds=[0.5, 0.75],
                confidence_thresholds=[.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0],
                  area_ranges={'S': [0, 32**2], 'M': [32**2, 96**2], 'L': [96**2, np.inf]},
                  display_predictions=False):
        # TODO: implement display predictions
        """
        Initializes the evaluator.

        Args:
        - model: The object detection model (e.g., Faster R-CNN).
        - dataset: The dataset object (e.g., RAITEDataset).
        - iou_thresholds: List of IoU thresholds to evaluate AP at.
        - area_ranges: Dictionary defining the area ranges for small, medium, and large objects.
        """
        self.model = model
        self.dataset = dataset
        self.iou_thresholds = iou_thresholds
        self.confidence_thresholds = confidence_thresholds
        self.area_ranges = area_ranges
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.display_predictions = display_predictions

    
    def evaluate(self):
        """
        Evaluates the model on the dataset.
        Args:
        - confidence_threshold_step: Step size for confidence thresholding.
        """
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

        ious = box_iou(boxes_pred, boxes_true) # TODO: verify the logic here
        max_iou, max_idx = ious.max(dim=1)

        # Initialize a boolean array to keep track of matched ground truth boxes
        matched_gt = torch.zeros(boxes_true.size(0), dtype=torch.bool, device=self.device)

        for i, (score, iou) in enumerate(zip(scores_pred, max_iou)):
            if score > confidence_thresh:  # Use the provided confidence threshold
                if iou >= iou_thresh and labels_pred[i] == 1:
                    tp_count += 1
                    matched_gt[max_idx[i]] = True  # Mark this ground truth box as matched
                elif iou < iou_thresh and labels_pred[i] == 1:  # Correct class but IoU less than threshold:
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
                precision = tp_sum / (tp_sum + fp_sum)
                recall = tp_sum / (tp_sum + fn_sum)

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
    
    def plot_precision_recall_curve(self, ap_results_dict, savePath):
        """
        Plots separate Precision-Recall (PR) curves for each IoU threshold.

        Args:
            ap_results_dict (dict): Dictionary containing precision and recall results at different IoU and confidence thresholds.
        """
        # Iterate over IoU thresholds
        for iou_thresh in self.iou_thresholds:
            plt.figure(figsize=(10, 6))  # Create a new figure for each IoU threshold

            precisions = []
            recalls = []

            # Iterate over confidence thresholds
            for i, conf_thresh in enumerate(self.confidence_thresholds):
                if i == (len(self.confidence_thresholds)-1): break  # Stop at the last threshold

                res = ap_results_dict["ap_results"][iou_thresh][conf_thresh]
                precisions.append(res[0])
                recalls.append(res[1])

            plt.plot(recalls, precisions, label=f'IoU: {iou_thresh}', marker='o', markersize=6, linewidth=2)

        # Adding labels and title
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves for IoU 0.5 and 0.75')
        plt.legend(loc='lower left')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Set axis limits dynamically if precision-recall trade-offs are small
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Adjust aspect ratio to give a better visual representation
        plt.gca().set_aspect('equal', adjustable='box')

        # Save the plot for the selected IoU thresholds
        plt.savefig(f'{savePath}/pr_curve_iou_0.5_and_0.75.png')

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
    #                        400, 400, 2)
    # model = torch.load('models/ugvs/fasterrcnn_resnet50_fpn_ugv_v3.pth')

    # evaluator = DetectionEvaluator(model, dataset)
    # ap_results, area_results = evaluator.evaluate()
    # evaluator.plot_precision_recall_curve(ap_results)

    # print(ap_results)
    # print(area_results)
    #evaluate_all_test_sets(model_path='models/drones/fasterrcnn_resnet50_fpn_drone_v3.pth', dir="data/archive/test_sets/drone")
    evaluate_all_test_sets(model_path='models/drones/fasterrcnn_resnet50_fpn_drone_comp_v1.pth', dir="data/archive/test_sets/drone")
