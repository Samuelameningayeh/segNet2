import torch

# ==========================================
# 3. UTILS: CLASS BALANCING
# ==========================================
def calculate_class_weights(loader, num_classes=11):
    print("Calculating class weights (Median Frequency Balancing)...")
    class_counts = torch.zeros(num_classes)
    total_pixels = 0
    
    for _, masks in loader:
        flattened = masks.flatten()
        counts = torch.bincount(flattened, minlength=num_classes)
        class_counts += counts
        total_pixels += flattened.numel()
    
    frequencies = class_counts / total_pixels
    median_freq = torch.median(frequencies[frequencies > 0])
    weights = median_freq / frequencies
    weights[torch.isinf(weights)] = 0
    weights[torch.isnan(weights)] = 0
    
    print(f"Calculated Weights: {weights}")
    return weights
