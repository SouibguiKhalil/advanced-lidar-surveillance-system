import torch

# 1. Import your model architecture and your NEW test_dataloader
from model import PointNetClassifier
from datasetloader import test_dataloader 

def evaluate_model():
    # ==========================================
    # SETUP & LOAD WEIGHTS
    # ==========================================
    # We use map_location in torch.load later just in case you trained on Colab GPU but are testing on PC CPU
    device = torch.device("cuda" )
    print(f"Evaluating on device: {device}")

    # Instantiate the exact same model architecture
    model = PointNetClassifier(num_classes=4).to(device)

    # Load the trained weights you downloaded from Colab
    weight_path = "pointnet_custom_weights.pth"
    try:
        # map_location=device ensures it loads safely whether you have a GPU locally or not
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("Successfully loaded trained weights!")
    except FileNotFoundError:
        print(f"Error: Could not find '{weight_path}'. Make sure it is in the exact same folder as this script.")
        return

    # ==========================================
    # EVALUATION LOOP
    # ==========================================
    # GOLDEN RULE 1: Put the model in evaluation mode! 
    # (Locks BatchNorm and turns off Dropout)
    model.eval()

    correct_predictions = 0
    total_predictions = 0

    print("Starting evaluation on unseen test data...")

    # GOLDEN RULE 2: Turn off gradient calculation!
    # (Saves memory and speeds up your PC)
    with torch.no_grad():
        # Notice we are looping over TEST_dataloader now!
        for inputs, labels in test_dataloader:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get the raw logit predictions from the model
            logits = model(inputs)
            
            # The logits output shape is [Batch_Size, 4]. 
            # We want the index of the highest score (the predicted class: 0, 1, 2, or 3)
            # torch.max returns (max_values, max_indices). We want the indices [1].
            _, predicted_classes = torch.max(logits, dim=1)
            
            # Count how many predictions matched the actual labels in this batch
            total_predictions += labels.size(0)
            correct_predictions += (predicted_classes == labels).sum().item()

    # ==========================================
    # FINAL RESULTS
    # ==========================================
    # Calculate final accuracy percentage
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\n=== Real-World Evaluation Complete ===")
        print(f"Total Unseen Objects Tested: {total_predictions}")
        print(f"Correct Predictions:         {correct_predictions}")
        print(f"True Model Accuracy:         {accuracy:.2f}%")
    else:
        print("Error: No data found in the test_dataloader.")

if __name__ == "__main__":
    evaluate_model()