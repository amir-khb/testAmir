import os
import random
import numpy as np
import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Class definitions
new_classes = {
    0: "Background",
    1: "Grass healthy",
    2: "Grass stressed",
    3: "Trees",
    4: "Water",
    5: "Residential buildings",
    6: "Non-residential buildings",
    7: "Road",
    8: "Unknown"
}

new_classesDioni = {
    0: "Background",
    1: "Dense Urban Fabric",
    2: "Mineral Extraction Sites",
    3: "Non Irrigated Arable Land",
    4: "Fruit Trees",
    5: "Olive Groves",
    6: "Coniferous Forest",
    7: "Sparse Sclerophyllous Vegetation",
    8: "Rocks and Sand",
    9: "Coastal Water",
    10: "Unknown"
}

new_classesPavia = {
    0: 'Background',
    1: 'Trees',
    2: 'Asphalt',
    3: 'Bricks',
    4: 'Bitumen',
    5: 'Meadows',
    6: 'Shadow',
    7: 'Bare Soil',
    8: 'Unknown'
}

# Colors for visualization
colors = [
    "#000000",  # Background
    "#FF0000",  # Grass healthy
    "#00FF00",  # Grass stressed
    "#0000FF",  # Trees
    "#FFFF00",  # Water
    "#00FFFF",  # Residential buildings
    "#FF00FF",  # Non-residential buildings
    "#8B4513",  # Road
    "#FFFFFF"  # Unknown
]

colors_dioni = [
    "#000000",  # 0: Background - Black
    "#8B4513",  # 1: Dense Urban Fabric - Saddle Brown
    "#A0522D",  # 2: Mineral Extraction Sites - Sienna
    "#D2691E",  # 3: Non Irrigated Arable Land - Chocolate
    "#FF6347",  # 4: Fruit Trees - Tomato
    "#9ACD32",  # 5: Olive Groves - Yellow Green
    "#228B22",  # 6: Coniferous Forest - Forest Green
    "#ADFF2F",  # 7: Sparse Sclerophyllous Vegetation - Green Yellow
    "#B22222",  # 8: Rocks and Sand - Fire Brick
    "#4169E1",  # 9: Coastal Water - Royal Blue
    "#FFFFFF"  # 10: Unknown - White
]

# Set environment variables for reproducibility
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def set_full_determinism(seed=42):
    """Comprehensive seed setting for maximum reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Full determinism enabled with seed {seed}")

    return lambda worker_id: random.seed(seed + worker_id)


# Initialize determinism
worker_init_fn = set_full_determinism(109)
g = torch.Generator()
g.manual_seed(109)