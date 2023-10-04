import torch
import utils
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.color import rgb2lab, deltaE_cie76
from collections import defaultdict

PATH =""

def load_model(filename):
    return torch.load(filename)

def visualize(args, model, dataset):
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Determine the total number of samples in the test dataset
    total_samples = len(test_loader.dataset)

    # Set a random seed for reproducibility (optional)
    torch.manual_seed(27)

    # Randomly select 1000 indices from the test dataset
    selected_indices = torch.randperm(total_samples)[:1000]

    # Create a Subset of the test dataset containing the selected samples
    from torch.utils.data import Subset
    subset = Subset(test_loader.dataset, selected_indices)

    # Create a DataLoader for the selected samples
    selected_test_loader = DataLoader(subset, batch_size=test_loader.batch_size, shuffle=True)

    model = load_model(PATH)
    finetuned_model = model.resnet
    features = extract_features(finetuned_model, args.device, selected_test_loader)

def extract_features(model, device, test_loader):
    # ! Do not modify the code in this function

    """
    Evaluate the model with the given dataset
    Args:
         model (keras.Model): model to be evaluated
         dataset (tf.data.Dataset): evaluation dataset
    Returns:
         AP (list): Average Precision for all classes
         MAP (float): mean average precision
    """
    
    
    with torch.no_grad():
        test_features = []
        gt_classes = []
        for data, target, wgt in test_loader:
            data = data.to(device)
            output = model(data)
            test_features.append(output.cpu().numpy())
            gt_classes.append(target.cpu().numpy())

    
    # Ensure that the features are standardized (mean=0, std=1)
    scaler = StandardScaler()
    test_features = scaler.fit_transform(test_features)

    # Perform PCA for dimensionality reduction (optional, but can help)
    # pca = PCA(n_components=50)  # You can adjust the number of components
    # test_features_pca = pca.fit_transform(test_features)

    # Perform t-SNE on the features
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(test_features_pca)

    # Prepare colors for class labels
    unique_classes = np.unique(gt_classes)
    class_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))

    # Create a dictionary to store the mean color for each class
    class_mean_colors = defaultdict(list)

    # Create a scatter plot with color-coded points
    plt.figure(figsize=(10, 8))
    for i, class_label in enumerate(unique_classes):
        mask = (gt_classes == class_label)
        class_tsne_results = tsne_results[mask]

        # Calculate the mean color for this class
        mean_color = np.mean(class_colors[i:i+1], axis=0)
        class_mean_colors[class_label] = mean_color

        plt.scatter(class_tsne_results[:, 0], class_tsne_results[:, 1], label=f'Class {class_label}', c=[mean_color])

    # Create a legend with class labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Class {label}', 
                        markersize=10, markerfacecolor=color) for label, color in class_mean_colors.items()]
    plt.legend(handles=handles, title='Object Classes')

    # Add labels and title
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Projection of ImageNet Features Color-Coded by GT Class')

    # Show the plot
    plt.show()PCA tries to reduce dimensionality by maximizing variance in the data while t-SNE tries to do the same by keeping similar data points together (and dissimilar data points apart) in both higher and lower dimensions. Because of these reasons, t-SNE can easily outperform PCA in dimensionality reduction.

    
    file_path = "t-SNE plot.png"  

    # Save the plot to the specified file
    plt.savefig(file_path, dpi=300)

    # Close the plot (optional)
    plt.close()

    # Print a message indicating that the plot has been saved
    print(f"Plot saved as {file_path}")