import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

from models.classifier import GastroClassifier
from xai.grad_cam import GradCAM, overlay_cam
from xai.tcav_module import TCAVWrapper
from xai.similarity import SimilaritySearch
from xai.counterfactual import CounterfactualGenerator
from data.loader import KvasirDataset

def load_model(model_path=None, num_classes=8):
    model = GastroClassifier(num_classes=num_classes)
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print("Warning: Loading untrained/ImageNet model (no custom weights found).")
    model.eval()
    return model

def generate_report(image_path, model_path, output_dir="reports"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Load Model & Image
    model = load_model(model_path)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    orig_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(orig_img).unsqueeze(0) # [1, C, H, W]
    
    # Prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_prob, top_class = torch.topk(probs, 1)
        prediction_idx = top_class.item()
        confidence = top_prob.item()

    print(f"Prediction: Class {prediction_idx} with confidence {confidence:.2f}")

    # 2. Grad-CAM
    print("Running Grad-CAM...")
    grad_cam = GradCAM(model, model.get_last_conv_layer())
    cam, _ = grad_cam(input_tensor, prediction_idx)
    cam_overlay = overlay_cam(input_tensor[0], cam)
    
    cam_path = os.path.join(output_dir, "grad_cam.png")
    cv2.imwrite(cam_path, (cam_overlay * 255).astype(np.uint8))

    # 3. TCAV (Mock/Check)
    print("Running TCAV check...")
    tcav_score = "N/A (Concept data missing)"
    # Example logic:
    # tcav = TCAVWrapper(model, ['layer4'], 'data/concepts')
    # if concepts_exist:
    #    res = tcav.interpret(input_tensor, prediction_idx, ...)
    #    tcav_score = res...

    # 4. Example-based (KNN)
    print("Running Similarity Search...")
    # Requires training data loader. Mocking for inference script if data missing.
    similar_images_paths = []
    # if os.path.exists('data/kvasir-dataset-v2'):
    #     train_loader, _, _ = get_data_loaders('data/kvasir-dataset-v2')
    #     sim_search = SimilaritySearch(model, train_loader)
    #     indices, dists = sim_search.find_nearest(input_tensor)
    #     # Retrieve paths from indices...
    
    # 5. Counterfactuals
    print("Generating Counterfactual...")
    cf_gen = CounterfactualGenerator(model)
    cf_tensor = cf_gen.generate(input_tensor)
    cf_img = overlay_cam(cf_tensor[0], np.zeros((224, 224)), alpha=0) # Just denormalize
    cf_path = os.path.join(output_dir, "counterfactual.png")
    cv2.imwrite(cf_path, (cf_img * 255).astype(np.uint8))

    # 6. Generate HTML Report
    html_content = f"""
    <html>
    <head><title>XAI Report</title>
    <style>body {{ font-family: sans-serif; }} .container {{ display: flex; }} .box {{ margin: 10px; }} img {{ max-width: 300px; }}</style>
    </head>
    <body>
        <h1>Gastrointestinal Disease Classification Report</h1>
        <h2>Prediction: Class {prediction_idx} ({confidence:.2%})</h2>
        
        <div class='container'>
            <div class='box'>
                <h3>Original Image</h3>
                <img src='{os.path.abspath(image_path)}'>
            </div>
            <div class='box'>
                <h3>Visual Attention (Grad-CAM)</h3>
                <img src='grad_cam.png'>
                <p>Highlights regions contributing to the decision.</p>
            </div>
        </div>

        <div class='container'>
            <div class='box'>
                <h3>Counterfactual Analysis</h3>
                <img src='counterfactual.png'>
                <p>Minimal perturbation required to change the prediction.</p>
            </div>
        </div>
        
        <h3>TCAV Score</h3>
        <p>{tcav_score}</p>
        
        <h3>Similar Cases</h3>
        <p><i>(Requires full dataset indexing - Placeholder)</i></p>
    </body>
    </html>
    """
    
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(html_content)
    
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <path_to_image>")
        # Create a dummy image for testing if none provided
        dummy_path = "dummy_test.jpg"
        Image.new('RGB', (224, 224), color='red').save(dummy_path)
        print(f"Created dummy image {dummy_path} for testing.")
        generate_report(dummy_path, "models/best_model.pth")
    else:
        generate_report(sys.argv[1], "models/best_model.pth")
