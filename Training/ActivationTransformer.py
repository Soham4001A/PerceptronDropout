import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from BaselineTransformerTraining import TransformerBlock

# Load the JSON file
with open("/Users/sohamsane/Documents/Coding Projects/PerceptronDropout/Data/GroupedPerceptronDataWithInputs.json", "r") as f:
    activation_data = json.load(f)

# Prepare training data
inputs = []
targets = []

for entry in activation_data:
    input_activations = [data["output"] for data in entry["perceptrons"].values()]
    target_nullify = [1 if perceptron_id in ["Dense_Embed_0", "Dense_Embed_1"] else 0  # Example logic for nullify target
                      for perceptron_id in entry["perceptrons"].keys()]
    
    inputs.append(input_activations)
    targets.append(target_nullify)

inputs = np.array(inputs)
targets = np.array(targets)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.2, random_state=42)

def build_secondary_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))  # Input layer with the correct shape
    x = Dense(128, activation="relu")(inputs)  # Connect Dense layer to inputs
    x = Dropout(0.2)(x)  # Connect Dropout to the previous layer
    x = Dense(64, activation="relu")(x)  # Connect Dense layer to the previous layer
    x = Dense(output_dim, activation="sigmoid")(x)  # Connect the final Dense layer

    model = Model(inputs=inputs, outputs=x)  # Use the tensor `x` as the output
    return model

secondary_model = build_secondary_model(input_dim=inputs.shape[1], output_dim=targets.shape[1])
secondary_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
secondary_model.summary()

# Train the secondary model
history = secondary_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32
)

def nullify_activations(original_activations, nullify_mask):
    """
    Nullify specific perceptrons based on the mask.
    Set the activation to a very small value (e.g., 0.00001) for nullified perceptrons.
    """
    modified_activations = original_activations.copy()  # Modify only the activations
    for i, should_nullify in enumerate(nullify_mask):
        if should_nullify:
            modified_activations[i] = 0.00001
    return modified_activations

def calculate_loss(baseline_model, X_data, y_data, secondary_model):
    total_loss = 0
    num_samples = len(X_data)

    for i in range(num_samples):
        # Get the original input for this sample
        original_input = X_data[i]  # Shape: (6,)
        true_output = y_data[i]     # True label

        # Predict nullify mask using the secondary model
        activations = np.zeros(70)  # Example placeholder for perceptron activations (mocked for simplicity)
        nullify_mask = secondary_model.predict(np.expand_dims(activations, axis=0))[0]
        nullify_mask = (nullify_mask > 0.5).astype(int)  # Threshold to determine nullification

        # Nullify perceptrons (this modifies only internal activations)
        modified_activations = nullify_activations(activations, nullify_mask)

        # Use the original input for baseline model prediction
        modified_prediction = baseline_model(tf.expand_dims(original_input, axis=0))  # Shape: (1, 6)

        # Calculate loss (e.g., mean squared error)
        loss = tf.reduce_mean(tf.square(modified_prediction - true_output))
        total_loss += loss.numpy()

    return total_loss / num_samples

for epoch in range(10):  # Number of feedback iterations
    print(f"Epoch {epoch + 1}")
    
    # Train the secondary model on activations
    secondary_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=32)
    
    #Load the Baseline Model
    baseline_model = tf.keras.models.load_model('/Users/sohamsane/Documents/Coding Projects/PerceptronDropout/baseline_model.keras', custom_objects={'TransformerBlock': TransformerBlock})

    # Evaluate the secondary model using the loss calculated via the baseline model
    feedback_loss = calculate_loss(baseline_model, X_val, y_val, secondary_model)
    print(f"Feedback Loss: {feedback_loss}")