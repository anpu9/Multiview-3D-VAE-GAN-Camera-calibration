import torch
def calculate_iou(pred_voxels, gt_voxels, threshold=0.5):
    """
    Calculate the Intersection over Union (IoU) between predicted and ground truth voxel grids.

    Args:
        pred_voxels: Predicted voxel grid (float tensor with values between 0 and 1)
        gt_voxels: Ground truth voxel grid (binary tensor)
        threshold: Threshold for binarizing predictions (default: 0.5)

    Returns:
        iou: Scalar IoU value
    """
    # Ensure inputs are on the same device
    if pred_voxels.device != gt_voxels.device:
        gt_voxels = gt_voxels.to(pred_voxels.device)

    # Binarize predictions
    pred_voxels_binary = (pred_voxels > threshold).float()

    # Ensure gt_voxels is binary
    gt_voxels_binary = gt_voxels.float()

    # Calculate intersection and union
    intersection = torch.sum(pred_voxels_binary * gt_voxels_binary)
    union = torch.sum(pred_voxels_binary) + torch.sum(gt_voxels_binary) - intersection

    # Avoid division by zero
    if union < 1e-6:
        return 0.0

    iou = intersection / union

    return iou.item()


# Example usage in a test function
def test_model(model, test_dataloader, threshold=0.5):
    model.eval()
    total_iou = 0.0
    num_samples = 0

    with torch.no_grad():
        for data in test_dataloader:
            # Handle different data formats based on model type
            if isinstance(data, tuple) and len(data) == 2:
                image, gt_voxels = data
                pred_voxels = model(image)
            elif isinstance(data, tuple) and len(data) >= 3:
            # Handle multi-view or pose cases as needed
            # ...

            batch_iou = calculate_iou(pred_voxels, gt_voxels, threshold)
            total_iou += batch_iou
            num_samples += 1

    average_iou = total_iou / num_samples
    print(f"Average IoU: {average_iou:.4f}")
    return average_iou