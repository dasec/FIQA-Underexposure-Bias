# Imports
import os
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from PIL import Image
from tqdm import tqdm
import argparse
import csv


# Data-Preprocessing
def prepare_data(data_dir, batch_size=32):
    # Input images MUST have been aligned and cropped beforehand
    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),  # Resize images to fit the input size of the model
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, dataset


# Function to test the model
def test_model(model, dataloader, dataset, device):
    correct = 0
    total = 0
    predictions = []

    model.to(device)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(inputs.size(0)):
                image_path = dataset.imgs[total - len(inputs) + i][0]
                image_name = os.path.basename(image_path)
                prob_class_0 = probabilities[i][0].item()
                prob_class_1 = probabilities[i][1].item()
                is_correct = predicted[i].item() == labels[i].item()

                predicted_class = dataset.classes[predicted[i].item()]

                result = {
                    "image": image_name,
                    "label": dataset.classes[labels[i].item()],
                    "predicted": predicted_class,
                    "prob_class_0": prob_class_0,
                    "prob_class_1": prob_class_1,
                    "correct": is_correct,
                }
                predictions.append(result)

    accuracy = 100 * correct / total
    return accuracy, predictions


def write_results_to_csv(results, filename="results.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Image", "Label", "Predicted", "Prob_Class_0", "Prob_Class_1", "Correct"]
        )

        for result in results:
            writer.writerow(
                [
                    result["image"],
                    result["label"],
                    result["predicted"],
                    result["prob_class_0"],
                    result["prob_class_1"],
                    result["correct"],
                ]
            )


if __name__ == "__main__":

    # ArgumentParser Setup
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="../models/deit_base_underexposure_checkpoint5.pth",
        help="Path to the model checkpoint .pth file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../demo_data",
        help="Path to the directory of the image. The folder structure must be as shown in the demo",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demo_predictions.csv",
        help="Path to the output CSV file for storing the predictions",
    )

    args = parser.parse_args()

    # Check if the model path exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    # Check if the data directory exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the Vision Transformer model architecture
    model = timm.create_model("deit_base_patch16_224", pretrained=False)

    # Modify the last layer to output num_classes
    num_classes = 2  # Update with the correct number of classes
    model.head = nn.Linear(model.head.in_features, num_classes)

    checkpoint = torch.load(args.model, map_location=device)

    # Load the state_dict from the checkpoint
    model.load_state_dict(checkpoint)

    # Set the model to evaluation mode
    model.eval()

    # Prepare data
    dataloader, dataset = prepare_data(args.input, batch_size=32)

    # Test model
    accuracy, results = test_model(model, dataloader, dataset, device)
    print(accuracy)
    # Write results to CSV
    write_results_to_csv(results, args.output)
    print(f"Predictions written to {args.output}")
