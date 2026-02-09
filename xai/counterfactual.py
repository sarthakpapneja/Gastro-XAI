import torch
import torch.nn.functional as F

class CounterfactualGenerator:
    def __init__(self, model):
        self.model = model

    def generate(self, input_tensor, target_class=None, steps=50, epsilon=0.01):
        """
        Simple FGSM-like iterative attack to find a counterfactual.
        If target_class is None, tries to flip to *any* other class.
        """
        image = input_tensor.clone().detach()
        image.requires_grad = True
        
        optimizer = torch.optim.SGD([image], lr=epsilon)
        
        # Initial prediction
        logits = self.model(image)
        initial_pred = logits.argmax(dim=1).item()
        
        print(f"Initial prediction: {initial_pred}")
        
        for i in range(steps):
            logits = self.model(image)
            current_pred = logits.argmax(dim=1).item()
            
            if target_class is not None:
                if current_pred == target_class:
                    print(f"Counterfactual found at step {i}")
                    return image.detach()
            else:
                 if current_pred != initial_pred:
                    print(f"Counterfactual found at step {i} (flipped to {current_pred})")
                    return image.detach()
            
            # Loss to maximize likelihood of target (or minimize likelihood of initial)
            if target_class is not None:
                loss = -F.cross_entropy(logits, torch.tensor([target_class]).to(image.device))
            else:
                # Untargeted: maximize Loss(initial_pred)
                loss = F.cross_entropy(logits, torch.tensor([initial_pred]).to(image.device)) # We want to Increase this loss
            
            self.model.zero_grad()
            loss.backward()
            
            # Update image
            # For untargeted, we want to move in direction of gradient to increase loss
            # For targeted (implemented as minimizing negative loss), we execute standard step
            image.data = image.data + epsilon * image.grad.sign()
            image.grad.zero_()
            
            # Clip to valid range (assuming normalized, this is tricky, but we'll clip roughly)
            # image.data = torch.clamp(image.data, -2.5, 2.5) # Approximate range for ImageNet normalization
            
        print("Counterfactual generation failed to flip class.")
        return image.detach()
