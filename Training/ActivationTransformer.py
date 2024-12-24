"""
ActivationTransformer.py — PerceptronDropout

**Skeleton Structure:**

1. **Baseline Model Initialization:**
   - Load and freeze the baseline model to prevent its weights from updating.

2. **Data Preparation:**
   - Load and preprocess the data.
   - Split the data into training and validation sets.

3. **Perceptron Activation Extraction:**
   - Define a `PerceptronExtractor` class to capture activations from Dense layers.

4. **Activation Transformer Model:**
   - Build a transformer-based secondary model that predicts which perceptrons to nullify.

5. **Nullification Function:**
   - Define a function to apply the predicted masks to the perceptron activations.

6. **Custom Training Loop:**
   - Implement `train_step` and `train_activation_transformer` functions for training the Activation Transformer.

7. **Evaluation:**
   - Define evaluation functions to assess the model's performance on validation and unseen data.

8. **Unseen Data Predictions:**
   - Generate both unmasked and masked predictions on unseen data.
   - Calculate and compare percent differences.

9. **Visualization:**
   - Plot the distribution of mask values to understand the masking behavior.

"""

# ------------------------------------------------
# 1) Imports and Setup
# ------------------------------------------------
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Import custom TransformerBlock from BaselineTransformerTraining
from BaselineTransformerTraining import TransformerBlock

# ------------------------------------------------
# 2) Load and Freeze Baseline Model
# ------------------------------------------------
baseline_model_path = "/Users/sohamsane/Documents/Coding Projects/PerceptronDropout/baseline_model.keras"

baseline_model = tf.keras.models.load_model(
    baseline_model_path,
    custom_objects={"TransformerBlock": TransformerBlock}
)

# Freeze the baseline model to prevent its weights from updating during training
for layer in baseline_model.layers:
    layer.trainable = False

# ------------------------------------------------
# 3) Data Preparation
# ------------------------------------------------
file_path = "/Users/sohamsane/Documents/Coding Projects/PerceptronDropout/Data/Model1_Omni/Asus_laptops.csv"
data = pd.read_csv(file_path).dropna(subset=["Price"])

# Define input features and target column
input_columns = ["Brand", "Processor_Speed", "RAM_Size", "Storage_Capacity", "Screen_Size", "Weight"]
target_col = "Price"

X = data[input_columns]
y = data[target_col].astype(np.float32).values.reshape(-1, 1)  # Ensure target is float32 for regression

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=["object"]).columns
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns

# Define preprocessing pipelines for numerical and categorical data
num_transformer = Pipeline([
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_features),
        ("cat", cat_transformer, categorical_features),
    ]
)

# Fit and transform the input data
X_preprocessed = preprocessor.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42
)

# ------------------------------------------------
# 4) Perceptron Activation Extraction
# ------------------------------------------------
class PerceptronExtractor(tf.keras.Model):
    """
    A wrapper around the baseline model that outputs both the final prediction
    and a dictionary of {perceptron_id: activation} for each Dense layer.
    """
    def __init__(self, baseline_model):
        super().__init__()
        self.baseline_model = baseline_model
        self.perceptron_ids = [layer.name for layer in baseline_model.layers if isinstance(layer, tf.keras.layers.Dense)]

    def call(self, inputs, training=False):
        """
        Forward pass that returns the final output and a dictionary of perceptron activations.
        
        Args:
            inputs (tf.Tensor): Input tensor.
            training (bool): Whether the model is in training mode.
        
        Returns:
            tuple: (final_output, perceptron_activations)
        """
        x = inputs
        perceptron_dict = {}
        for layer in self.baseline_model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            if isinstance(layer, tf.keras.layers.Dense):
                x = layer(x)
                perceptron_dict[layer.name] = x
            else:
                x = layer(x, training=training)
        return x, perceptron_dict

# Instantiate the PerceptronExtractor
extractor = PerceptronExtractor(baseline_model)

# ------------------------------------------------
# 5) Activation Transformer Model
# ------------------------------------------------
def build_activation_transformer(num_input_features, total_perceptrons, num_heads=3, ff_dim=128, num_transformer_blocks=2, dropout_rate=0.1):
    """
    Builds a transformer-based Activation Transformer model.
    
    Args:
        num_input_features (int): Number of original input features.
        total_perceptrons (int): Total number of perceptrons across all Dense layers.
        num_heads (int): Number of attention heads in Transformer blocks.
        ff_dim (int): Feed-forward network dimension within Transformer blocks.
        num_transformer_blocks (int): Number of Transformer blocks to stack.
        dropout_rate (float): Dropout rate for Transformer blocks.
    
    Returns:
        tf.keras.Model: Compiled Activation Transformer model.
    """
    inputs_ = Input(shape=(num_input_features + total_perceptrons,), name="ActivationTransformer_Input")
    
    # Initial Dense layer for embedding inputs
    x = Dense(64, activation="relu", name="ActivationTransformer_Dense_1")(inputs_)
    x = Reshape((1, 64), name="ActivationTransformer_Reshape")(x)  # Shape: (batch_size, seq_len=1, embed_dim=64)
    
    # Add multiple Transformer blocks
    for i in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim=64, num_heads=num_heads, ff_dim=ff_dim, rate=dropout_rate, name=f"TransformerBlock_{i}")(x)
    
    x = Flatten(name="ActivationTransformer_Flatten")(x)  # Shape: (batch_size, features)
    x = Dense(ff_dim, activation="relu", name="ActivationTransformer_Dense_2")(x)
    x = Dense(ff_dim, activation="relu", name="ActivationTransformer_Dense_3")(x)
    x = Dense(64, activation="relu", name="ActivationTransformer_Dense_4")(x)
    mask_output = Dense(total_perceptrons, activation="sigmoid", name="ActivationTransformer_MaskOutput")(x)  # Shape: (batch_size, total_perceptrons)
    
    model = Model(inputs=inputs_, outputs=mask_output, name="ActivationTransformer")
    return model

# Calculate total number of perceptrons across all Dense layers in the baseline model
total_perceptrons = sum([layer.units for layer in baseline_model.layers if isinstance(layer, tf.keras.layers.Dense)])

# Determine the number of preprocessed input features
num_input_features = X_train.shape[1]

# Instantiate the Activation Transformer model
secondary_model = build_activation_transformer(
    num_input_features=num_input_features,
    total_perceptrons=total_perceptrons,
    num_heads=3,
    ff_dim=128,
    num_transformer_blocks=2,
    dropout_rate=0.1
)

# Display the Activation Transformer model summary
secondary_model.summary()

# ------------------------------------------------
# 6) Nullification Function
# ------------------------------------------------
def nullify_activations(perceptron_dict, mask, layer_units, null_value=0.001):
    """
    Applies the predicted mask to the perceptron activations to nullify specific perceptrons.
    
    Each perceptron's activation is set to `null_value` if its corresponding mask value > 0.5.
    Otherwise, the original activation is retained.
    
    Args:
        perceptron_dict (dict): Dictionary of {layer_name: activation_tensor}.
        mask (tf.Tensor): Tensor of shape (batch_size, total_perceptrons) with mask values between 0 and 1.
        layer_units (list): List containing the number of units in each Dense layer.
        null_value (float): The value to set for nullified perceptrons.
    
    Returns:
        dict: Dictionary with updated activations after nullification.
    """
    new_dict = {}
    start_idx = 0

    for (layer_name, activation), units in zip(perceptron_dict.items(), layer_units):
        end_idx = start_idx + units
        layer_mask = mask[:, start_idx:end_idx]  # Extract mask for the current layer
        start_idx = end_idx

        # Create a binary mask: 1 where mask > 0.5, else 0
        binary_mask = tf.cast(layer_mask > 0.5, dtype=activation.dtype)

        # Apply nullification: set to null_value where binary_mask is 1, else keep original activation
        new_activation = activation * (1.0 - binary_mask) + null_value * binary_mask

        new_dict[layer_name] = new_activation

    return new_dict

# ------------------------------------------------
# 7) Custom Training Loop
# ------------------------------------------------
mse_loss_fn = MeanSquaredError()
optimizer = Adam(learning_rate=0.001)  # Set a reasonable learning rate

@tf.function
def train_step(x_batch, y_batch):
    """
    Performs a single training step for the Activation Transformer.
    
    Args:
        x_batch (tf.Tensor): Batch of input features.
        y_batch (tf.Tensor): Batch of target values.
    
    Returns:
        tf.Tensor: Calculated loss for the batch.
    """
    with tf.GradientTape() as tape:
        # Forward pass through the extractor to get baseline predictions and perceptron activations
        _, perceptron_dict = extractor(x_batch, training=False)

        # Gather the number of units in each Dense layer
        layer_units = [layer.units for layer in baseline_model.layers if isinstance(layer, tf.keras.layers.Dense)]

        # Concatenate all perceptron activations into a single vector per sample
        all_activations = list(perceptron_dict.values())
        cat_activations = tf.concat(all_activations, axis=1)  # Shape: (batch_size, total_perceptrons)

        # Combine original input features with concatenated perceptron activations
        combined_input = tf.concat([x_batch, cat_activations], axis=1)

        # Predict the mask using the Activation Transformer
        mask_pred = secondary_model(combined_input, training=True)  # Shape: (batch_size, total_perceptrons)

        # Apply the predicted masks to nullify perceptron activations
        masked_perceptron_dict = nullify_activations(perceptron_dict, mask_pred, layer_units, null_value=0.001)

        # Re-run the baseline model with masked activations
        x = x_batch
        for layer in baseline_model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            elif isinstance(layer, tf.keras.layers.Dense):
                layer_name = layer.name
                x = masked_perceptron_dict[layer_name]  # Use masked activations
            else:
                x = layer(x, training=False)
        baseline_out_masked = x  # Final output after masking

        # Calculate the Mean Squared Error loss
        loss_value = mse_loss_fn(y_batch, baseline_out_masked)

        # Add L1 regularization to encourage sparsity in the mask
        l1_loss = tf.reduce_sum(tf.abs(mask_pred))
        loss_value += 0.001 * l1_loss

    # Compute gradients with respect to the Activation Transformer's trainable variables
    grads = tape.gradient(loss_value, secondary_model.trainable_variables)

    # Apply gradients to update the Activation Transformer's weights
    optimizer.apply_gradients(zip(grads, secondary_model.trainable_variables))

    return loss_value

def train_activation_transformer(X_train_np, y_train_np, epochs, batch_size=32):
    """
    Trains the Activation Transformer using a custom training loop.
    
    Args:
        X_train_np (np.ndarray): Training input features.
        y_train_np (np.ndarray): Training target values.
        epochs (int): Number of training epochs.
        batch_size (int, optional): Size of each training batch. Defaults to 32.
    """
    dataset_size = X_train_np.shape[0]
    for epoch in range(epochs):
        # Shuffle the training data at the beginning of each epoch
        idx = np.arange(dataset_size)
        np.random.shuffle(idx)
        X_train_shuffled = X_train_np[idx]
        y_train_shuffled = y_train_np[idx]

        batch_losses = []
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            x_batch_np = X_train_shuffled[start:end]
            y_batch_np = y_train_shuffled[start:end]

            # Convert NumPy arrays to TensorFlow tensors
            x_batch = tf.convert_to_tensor(x_batch_np, dtype=tf.float32)
            y_batch = tf.convert_to_tensor(y_batch_np, dtype=tf.float32)

            # Perform a training step and record the loss
            batch_loss = train_step(x_batch, y_batch)
            batch_losses.append(batch_loss.numpy())

        # Calculate and display the average loss for the epoch
        epoch_loss = np.mean(batch_losses)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# ------------------------------------------------
# 8) Evaluation Functions
# ------------------------------------------------
@tf.function
def evaluate_step(x_batch, y_batch):
    """
    Evaluates the Activation Transformer on a batch of data.
    
    Args:
        x_batch (tf.Tensor): Batch of input features.
        y_batch (tf.Tensor): Batch of target values.
    
    Returns:
        tf.Tensor: Calculated loss for the batch.
    """
    # Forward pass through the extractor to get baseline predictions and perceptron activations
    _, perceptron_dict = extractor(x_batch, training=False)

    # Gather the number of units in each Dense layer
    layer_units = [layer.units for layer in baseline_model.layers if isinstance(layer, tf.keras.layers.Dense)]

    # Concatenate all perceptron activations into a single vector per sample
    all_activations = list(perceptron_dict.values())
    cat_activations = tf.concat(all_activations, axis=1)  # Shape: (batch_size, total_perceptrons)

    # Combine original input features with concatenated perceptron activations
    combined_input = tf.concat([x_batch, cat_activations], axis=1)

    # Predict the mask using the Activation Transformer
    mask_pred = secondary_model(combined_input, training=False)  # Shape: (batch_size, total_perceptrons)

    # Apply the predicted masks to nullify perceptron activations
    masked_perceptron_dict = nullify_activations(perceptron_dict, mask_pred, layer_units, null_value=0.001)

    # Re-run the baseline model with masked activations
    x = x_batch
    for layer in baseline_model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        elif isinstance(layer, tf.keras.layers.Dense):
            layer_name = layer.name
            x = masked_perceptron_dict[layer_name]  # Use masked activations
        else:
            x = layer(x, training=False)
    baseline_out_masked = x  # Final output after masking

    # Calculate the Mean Squared Error loss
    loss_val = mse_loss_fn(y_batch, baseline_out_masked)
    return loss_val

def evaluate_activation_transformer(X_val_np, y_val_np, batch_size=32):
    """
    Evaluates the Activation Transformer on the validation dataset.
    
    Args:
        X_val_np (np.ndarray): Validation input features.
        y_val_np (np.ndarray): Validation target values.
        batch_size (int, optional): Size of each evaluation batch. Defaults to 32.
    
    Returns:
        float: Average loss over the validation set.
    """
    dataset_size = X_val_np.shape[0]
    all_losses = []
    for start in range(0, dataset_size, batch_size):
        end = start + batch_size
        x_batch_np = X_val_np[start:end]
        y_batch_np = y_val_np[start:end]
        x_batch = tf.convert_to_tensor(x_batch_np, dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_batch_np, dtype=tf.float32)

        # Perform an evaluation step and record the loss
        val_loss = evaluate_step(x_batch, y_batch)
        all_losses.append(val_loss.numpy())
    return np.mean(all_losses)

# ------------------------------------------------
# 9) Training the Activation Transformer
# ------------------------------------------------
# Train the Activation Transformer for 125 epochs with a batch size of 32
train_activation_transformer(X_train, y_train, epochs=125, batch_size=32)

# ------------------------------------------------
# 10) Evaluate on Validation Set
# ------------------------------------------------
val_loss = evaluate_activation_transformer(X_val, y_val)
print(f"Validation Loss with Masking: {val_loss:.4f}")

# ------------------------------------------------
# 11) Unseen Data Predictions
# ------------------------------------------------

# ------------------------------------------------
# A) Regular Baseline Predictions on Unseen Data
# ------------------------------------------------
# Define the path to the unseen data
file_path_unseen = '/Users/sohamsane/Documents/Coding Projects/PerceptronDropout/Data/Model2_Split/Acer_Lenovo_laptops.csv'
unseen_data = pd.read_csv(file_path_unseen)
unseen_data.fillna(0, inplace=True)  # Handle missing values

# Preprocess the unseen data using the same preprocessor
unseen_preprocessed = preprocessor.transform(unseen_data[input_columns])

# Get the baseline model’s predictions (unmasked)
unmasked_preds = baseline_model.predict(unseen_preprocessed)

# Initialize and fit the OneHotEncoder for decoding predictions
trim_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
trim_encoder.fit(y.reshape(-1, 1))  # Fit on training targets or appropriate data

# Decode the unmasked predictions
unmasked_preds_decoded = trim_encoder.inverse_transform(unmasked_preds)

# Extract the true prices from the last column of the unseen data
true_prices = unseen_data.iloc[:, -1].values.reshape(-1, 1)

print("\n=== UNSEEN DATA SET: REGULAR BASELINE PREDICTIONS ===")
print("True Price:", true_prices[:10].flatten())
print("Predicted Price:", unmasked_preds_decoded[:10].flatten())

# Compute and display percent differences for the first 10 test cases
unmasked_diffs = []
for i in range(min(10, len(true_prices))):
    true_val = float(true_prices[i][0])
    pred_val = float(unmasked_preds_decoded[i][0])
    diff = 100 * abs(true_val - pred_val) / (true_val if true_val != 0 else 1)
    print(f"Percent Difference for Test Case {i+1}: {diff:.2f}%")
    unmasked_diffs.append(diff)

print(f"Average Percent Difference (Unmasked Baseline): {np.mean(unmasked_diffs):.2f}%")

# ------------------------------------------------
# B) Masked Predictions on Unseen Data Using Activation Transformer
# ------------------------------------------------
print("\n=== UNSEEN DATA SET: MASKED PREDICTIONS WITH ACTIVATION TRANSFORMER ===")

# Convert the unseen preprocessed data to TensorFlow tensors
unseen_tensor = tf.convert_to_tensor(unseen_preprocessed, dtype=tf.float32)

# Forward pass through the extractor to get perceptron activations
_, perceptron_dict = extractor(unseen_tensor, training=False)

# Gather the number of units in each Dense layer
layer_units = [layer.units for layer in baseline_model.layers if isinstance(layer, tf.keras.layers.Dense)]

# Concatenate all perceptron activations into a single vector per sample
all_activations = list(perceptron_dict.values())
cat_activations = tf.concat(all_activations, axis=1)  # Shape: (batch_size, total_perceptrons)

# Combine original input features with concatenated perceptron activations
combined_unseen_input = tf.concat([unseen_tensor, cat_activations], axis=1)

# Predict the mask using the Activation Transformer
mask_pred = secondary_model(combined_unseen_input, training=False)  # Shape: (batch_size, total_perceptrons)

# Apply the predicted masks to nullify perceptron activations
masked_perceptron_dict = nullify_activations(perceptron_dict, mask_pred, layer_units, null_value=0.001)

# Re-run the baseline model with masked activations
x = unseen_tensor
for layer in baseline_model.layers:
    if isinstance(layer, tf.keras.layers.InputLayer):
        continue
    elif isinstance(layer, tf.keras.layers.Dense):
        layer_name = layer.name
        x = masked_perceptron_dict[layer_name]  # Use masked activations
    else:
        x = layer(x, training=False)
baseline_out_masked = x  # Final output after masking

# Convert the masked predictions to NumPy arrays
masked_preds = baseline_out_masked.numpy()

# Decode the masked predictions
masked_preds_decoded = trim_encoder.inverse_transform(masked_preds)

# Compute and display percent differences for the first 10 test cases
masked_diffs = []
for i in range(min(10, len(true_prices))):
    true_val = float(true_prices[i][0])
    pred_val = float(masked_preds_decoded[i][0])
    diff = 100 * abs(true_val - pred_val) / (true_val if true_val != 0 else 1)
    print(f"Percent Difference for Test Case {i+1}: {diff:.2f}%")
    masked_diffs.append(diff)

print(f"Average Percent Difference (Masked Predictions): {np.mean(masked_diffs):.2f}%")

# ------------------------------------------------
# 12) Visualization: Mask Value Distribution
# ------------------------------------------------
# Plot the distribution of mask values to understand masking behavior
plt.figure(figsize=(10, 6))
plt.hist(mask_pred.numpy().flatten(), bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of Mask Values")
plt.xlabel("Mask Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()