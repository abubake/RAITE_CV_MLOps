def _compute_tp_fp_fn(self, predictions, target, iou_thresh, confidence_thresh):
    """
    Computes true positives, false positives, and false negatives.
    Args:
    - predictions: Model predictions.
    - target: Ground truth annotations.
    - iou_thresh: IoU threshold for determining TP and FP.
    - confidence_thresh: Confidence threshold for considering predictions.

    Returns:
    - tp_count, fp_count, fn_count: Counts of true positives, false positives, and false negatives.
    """
    # Move predictions and target annotations to the same device
    boxes_pred = predictions[0]['boxes'].to(self.device)
    labels_pred = predictions[0]['labels'].to(self.device)
    scores_pred = predictions[0]['scores'].to(self.device)

    boxes_true = target['boxes'].to(self.device)
    labels_true = target['labels'].to(self.device)

    tp_count = 0
    fp_count = 0
    fn_count = 0

    # Handle the case where there are no predictions
    if len(boxes_pred) == 0:
        fn_count += len(boxes_true)  # All true boxes are false negatives
        return tp_count, fp_count, fn_count

    # Compute IoUs between predicted and ground truth boxes
    ious = box_iou(boxes_pred, boxes_true)
    max_iou, max_idx = ious.max(dim=1)

    # Track matched ground truth boxes
    matched_gt = torch.zeros(boxes_true.size(0), dtype=torch.bool, device=self.device)

    for i, (score, iou, label_pred) in enumerate(zip(scores_pred, max_iou, labels_pred)):
        if score > confidence_thresh:  # Filter by confidence threshold
            gt_idx = max_idx[i]
            label_true = labels_true[gt_idx]
            
            # Check if predicted label matches the ground truth label and IoU threshold
            if iou >= iou_thresh and label_pred == label_true:
                tp_count += 1
                matched_gt[gt_idx] = True  # Mark the ground truth box as matched
            else:
                fp_count += 1  # False positive if IoU or label match fails

    # Count unmatched ground truth boxes as false negatives
    fn_count = (len(boxes_true) - matched_gt.sum().item())

    return tp_count, fp_count, fn_count


# OLD
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
