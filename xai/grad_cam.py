import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # Ensure input requires grad to force backprop even if model is frozen
        if not x.requires_grad:
            x.requires_grad = True
            
        # Forward pass
        logits = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)
        
        # Zero grads
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients.data.cpu().numpy()[0] # [C, H, W]
        activations = self.activations.data.cpu().numpy()[0] # [C, H, W]
        
        weights = np.mean(gradients, axis=(1, 2)) # Global Average Pooling on gradients
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0) # ReLU
        cam = cv2.resize(cam, (x.shape[3], x.shape[2])) # Resize to input image size
        cam -= np.min(cam)
        cam /= np.max(cam) # Normalize
        
        return cam, class_idx

def overlay_cam(img_tensor, cam, alpha=0.5):
    """
    Overlays CAM on the original image.
    img_tensor: [C, H, W] tensor, normalized
    cam: [H, W] numpy array
    """
    # Denormalize image for visualization
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img_tensor.detach().cpu().numpy() * std + mean
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0)) # [H, W, C]
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1] # BGR to RGB
    
    overlay = heatmap * alpha + img * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)
    
    return overlay
