import torch
from model_pointnet import PointNetCls
from torchinfo import summary
from torchview import draw_graph
import os
import sys

# Force output to UTF-8 for Windows terminals to avoid 'charmap' errors
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Auto-detection of Graphviz path on Windows
    common_graphviz_paths = [
        r"C:\Program Files\Graphviz\bin",
        r"C:\Program Files (x86)\Graphviz\bin",
        os.path.join(os.getenv("LOCALAPPDATA", ""), "bin", "Graphviz", "bin"),
    ]
    for path in common_graphviz_paths:
        if os.path.exists(path):
            os.environ["PATH"] += os.pathsep + path
            break

def inspect():


    # 1. Initialize Model
    # We use k=2 for the classification (Safe/Danger)
    model = PointNetCls(k=2, feature_transform=True)
    
    print("="*30)
    print("RESUME DU MODELE POINTNET")
    print("="*30)
    
    # 2. Console Summary (Textual)
    # PointNet expects input of shape (Batch, 3, N_points)
    batch_size = 16
    n_points = 1024
    stats = summary(model, input_size=(batch_size, 3, n_points), 
                    col_names=["input_size", "output_size", "num_params", "kernel_size"],
                    depth=3)
    print(stats)

    # 3. Graphical Visualization (Saved as image or shown if possible)
    print("\nGeneration du graphique du modele...")
    if not os.path.exists('images'):
        os.makedirs('images')
        
    # 4. EXPORT ONNX (For Netron.app)
    print("\nExportation du modele au format ONNX...")
    try:
        model.cpu() # Ensure model is on CPU for export
        dummy_input = torch.randn(batch_size, 3, n_points).cpu()
        onnx_path = "models/pointnet_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['lidar_points'],
            output_names=['classification', 'transform'],
            dynamic_axes={'lidar_points': {0: 'batch_size'}, 'classification': {0: 'batch_size'}}
        )
        print(f"Modele exporte avec succès : {onnx_path}")
        print("CONSEIL : Glisse ce fichier sur https://netron.app/ pour voir l'arbre interactif !")
    except Exception as e:
        print(f"Erreur lors de l'export ONNX : {e}")


    # 5. Graphical Visualization (Traditional PNG)
    print("\nGeneration du graphique du modele (PNG)...")




if __name__ == "__main__":
    inspect()
