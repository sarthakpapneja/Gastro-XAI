from captum.concept import TCAV
from captum.concept import Concept
from captum.concept import ConceptInterpreter
import torch
from torch.utils.data import DataLoader, Dataset
from glob import glob
from PIL import Image
from torchvision import transforms

# Simple dataset for concepts
class ConceptDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_paths = glob(f"{root_dir}/*.jpg") + glob(f"{root_dir}/*.png")
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

class TCAVWrapper:
    def __init__(self, model, layers, concepts_dir):
        """
        model: PyTorch model
        layers: List of layer names (strings) or modules to probe
        concepts_dir: Directory containing subfolders for each concept (e.g., 'striped', 'random')
        """
        self.model = model
        self.layers = layers
        self.concepts_dir = concepts_dir
        self.tcav = TCAV(model=model, layers=layers)
        
    def load_concepts(self, concept_names, transform):
        """
        concept_names: List of strings matching subfolders in concepts_dir
        """
        concepts = []
        for name in concept_names:
            path = f"{self.concepts_dir}/{name}"
            dataset = ConceptDataset(path, transform=transform)
            loader = DataLoader(dataset, batch_size=16)
            concepts.append(Concept(id=name, name=name, data_iter=loader))
        return concepts

    def interpret(self, inputs, target_class, experimental_sets):
        """
        inputs: Input tensor [B, C, H, W]
        target_class: Int
        experimental_sets: List of lists of Concepts. E.g. [[concept_A, random_1], [concept_A, random_2]]
        """
        results = self.tcav.interpret(
            inputs,
            experimental_sets=experimental_sets,
            target=target_class
        )
        return results

    # Helper to calculate TCAV score
    def get_tcav_score(self, results, concept_name):
        # Captum's return structure is complex, this parses it simplistically
        # result is a dict of layer -> {concept_name -> score}
        scores = {}
        for layer in self.layers:
            # This logic depends on Captum version return type, usually valid
            if layer in results:
                layer_res = results[layer]
                if concept_name in layer_res:
                     scores[layer] = layer_res[concept_name]
        return scores
