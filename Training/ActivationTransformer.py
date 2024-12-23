




"""
Skeleton Structure-

1. Baseline model takes input
2. Baseline model predicts - perceptron activations in form of {perceptron id: activation} are saved
3. Activation transformer is fed what was the input for baseline model + activations
4. Activation transformer predicts which perceptron id(s) to nullify (this can be a different amount everytime)
5. Those perceptrons are now set to 0.001 for their output
6. Baseline model now runs again with those modfiications and the output is observed
7. That output is compared vs the true output to compute loss for the activation transformer
8. Process is repeated


"""


import numpy as np
import tensorflow as tf
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend as K

# ------------------------------------------------
# 1) Load Your Baseline Model
# ------------------------------------------------
from BaselineTransformerTraining import TransformerBlock
baseline_model_path = "/Users/sohamsane/Documents/Coding Projects/PerceptronDropout/baseline_model.keras"

baseline_model = tf.keras.models.load_model(
    baseline_model_path,
    custom_objects={"TransformerBlock": TransformerBlock}
)
# Freeze baseline so its weights do not update
for layer in baseline_model.layers:
    layer.trainable = False

# ------------------------------------------------
# 2) Prepare Data (Same as BaselineTransformerTraining script)
#    - We'll load the same data, same preprocessing.
#    - Then we'll split into train/val for the activation transformer.
# ------------------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

file_path = "/Users/sohamsane/Documents/Coding Projects/PerceptronDropout/Data/Model1_Omni/Asus_laptops.csv"
data = pd.read_csv(file_path).dropna(subset=["Price"])

input_columns = ["Brand", "Processor_Speed", "RAM_Size", "Storage_Capacity", "Screen_Size", "Weight"]
target_col = "Price"

X = data[input_columns]
y = data[target_col]

categorical_features = X.select_dtypes(include=["object"]).columns
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns

num_transformer = Pipeline([("scaler", StandardScaler())])
cat_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_features),
        ("cat", cat_transformer, categorical_features),
    ]
)
X_preprocessed = preprocessor.fit_transform(X)

# For the sake of example, treat price as a regression target
# (If it's categorical in your real script, adjust accordingly)
y = np.array(y, dtype=np.float32).reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42
)

# ------------------------------------------------
# 3) Extract Baseline Perceptron Activations
#    We'll create a small function that:
#       - Passes input through baseline
#       - Captures the intermediate "Dense" layer outputs
#    (If you prefer an actual dictionary {id: activation}, we do so below.)
# ------------------------------------------------
class PerceptronExtractor(tf.keras.Model):
    """
    A wrapper around the baseline model that
    outputs both the final baseline prediction and
    a dictionary of {perceptron_id: activation}.
    """
    def __init__(self, baseline_model):
        super().__init__()
        self.baseline_model = baseline_model
        self.perceptron_ids = []
        # Gather references to the Dense layers for a stable ID
        for layer in baseline_model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                self.perceptron_ids.append(layer.name)

    def call(self, inputs, training=False):
        """
        Return (final_output, perceptron_activations)
        where perceptron_activations is a concatenated vector
        or dictionary with all Dense-layer activations.
        """
        x = inputs
        # We'll build a dictionary keyed by layer_name, each layer's activation
        perceptron_dict = {}
        for layer in self.baseline_model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            # If it's a Dense layer, capture the input and output for that layer
            if isinstance(layer, tf.keras.layers.Dense):
                # This is the activation
                x = layer(x)
                # shape = (batch_size, units)
                # We can store each perceptron's activation in a dictionary
                # or just flatten them as a single vector
                perceptron_dict[layer.name] = x
            else:
                # For other layers (Dropout, Flatten, TransformerBlock, etc.)
                # just feed forward
                x = layer(x, training=training)
        return x, perceptron_dict

# Build the extractor
extractor = PerceptronExtractor(baseline_model)

# Example usage:
# outputs, activations_dict = extractor(X_train_tensor)
# But we'll do it in a custom loop below.

# ------------------------------------------------
# 4) Define the Activation Transformer (Secondary Model)
#    Input = [ original input features (6 after one-hot?), + concatenated perceptron activations ]
#    Output = a mask indicating which perceptrons to nullify
# ------------------------------------------------
def build_activation_transformer(num_input_features, total_perceptrons):
    """
    Example: If baseline has 2 Dense layers with 64 and 32 units each,
    total_perceptrons = 96. 
    We'll feed the (num_input_features + total_perceptrons) into a model that outputs
    a binary mask of size 'total_perceptrons'.
    """
    inputs_ = Input(shape=(num_input_features + total_perceptrons,))
    x = Dense(128, activation="relu")(inputs_)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    mask_output = Dense(total_perceptrons, activation="sigmoid")(x)
    model = Model(inputs=inputs_, outputs=mask_output, name="ActivationTransformer")
    return model

# We must figure out how many total perceptrons we have
# E.g., sum of units across all Dense layers in the baseline model
total_perceptrons = 0
for layer in baseline_model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        total_perceptrons += layer.units

# We also need the dimension of X_train (post-preprocessing)
num_input_features = X_train.shape[1]  # e.g. 6 + one-hot expansions

secondary_model = build_activation_transformer(num_input_features, total_perceptrons)

# ------------------------------------------------
# 5) Nullify Function
# ------------------------------------------------
def nullify_activations(perceptron_dict, mask, layer_units, null_value=0.001):
    """
    perceptron_dict: {layer_name: (batch_size, n_units) Tensor}
    mask: (batch_size, total_perceptrons) for each sample
    layer_units: list of layer_units in the order we encountered them
    null_value: value to set if nullified
    Return a new dictionary with the same keys but with masked activation values.
    """
    # mask shape = (batch_size, sum(layer_units))
    # We need to slice the mask according to each layer's #units
    # E.g. if layer_units = [64, 32, 110], we slice mask accordingly
    # Then apply that slice to each layer's activation.
    # The function returns a new dictionary of masked activations
    new_dict = {}
    start_idx = 0

    for (layer_name, activation), units in zip(perceptron_dict.items(), layer_units):
        end_idx = start_idx + units
        # slice mask for this layer
        layer_mask = mask[:, start_idx:end_idx]  # shape = (batch_size, units)
        start_idx = end_idx

        # shape = (batch_size, units)
        # if layer_mask[i,j] = 1 => nullify
        # we do something like: new_activation = activation*(1-mask) + mask*null_value
        # But we must broadcast shapes if needed
        layer_mask = tf.cast(layer_mask, dtype=activation.dtype)
        kept = activation * (1.0 - layer_mask)
        nulled = null_value * layer_mask
        new_activation = kept + nulled
        new_dict[layer_name] = new_activation

    return new_dict

# ------------------------------------------------
# 6) Build a Custom Training Loop
# ------------------------------------------------
mse_loss_fn = MeanSquaredError()
optimizer = Adam(learning_rate=1e-3)

@tf.function
def train_step(x_batch, y_batch):
    """
    x_batch: (batch_size, num_input_features) preprocessed baseline input
    y_batch: (batch_size, 1) true price
    """
    with tf.GradientTape() as tape:
        # 1. Extract baseline perceptron activations
        final_out, perceptron_dict = extractor(x_batch, training=False)
        # perceptron_dict is e.g. {"dense_0": (batch_size, 64), "dense_1": (batch_size, 32), ...}

        # 2. Flatten them into a single activation vector per sample
        #    We also keep track of the #units in each layer for slicing
        #    layer_units example: [64, 32, ...]
        layer_units = []
        activation_vecs = []
        for layer in baseline_model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                layer_units.append(layer.units)

        # Build a single vector [ a1, a2, a3, ... ] for each sample
        all_activations = []
        for (layer_name, activation) in perceptron_dict.items():
            # shape = (batch_size, n_units)
            all_activations.append(activation)
        # Concatenate them on axis=1 => shape=(batch_size, sum of n_units)
        cat_activations = tf.concat(all_activations, axis=1)  # (batch_size, total_perceptrons)

        # 3. Feed [x_batch, cat_activations] to the secondary model
        combined_input = tf.concat([x_batch, cat_activations], axis=1)
        mask_pred = secondary_model(combined_input, training=True)  # shape=(batch_size, total_perceptrons)

        # 4. Nullify the perceptron_dict using the predicted mask
        masked_perceptron_dict = nullify_activations(perceptron_dict, mask_pred, layer_units, null_value=0.001)

        # 5. Re-run the baseline forward pass with masked activations
        #    We must manually re-inject the masked activations into the baseline's next layers
        #    => We'll replicate the baseline's forward pass but substitute the masked activations
        #    for each Dense layer output. So we need a custom pass:
        x = x_batch
        layer_idx = 0
        for layer in baseline_model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            elif isinstance(layer, tf.keras.layers.Dense):
                # Instead of layer(x), we skip to the masked activation for this layer
                # Then pass that forward
                # Retrieve masked activation from masked_perceptron_dict
                layer_name = layer.name
                x = masked_perceptron_dict[layer_name]
            else:
                x = layer(x, training=False)

        # x is now the final baseline output after nullification
        # shape = (batch_size, 1) or (batch_size, 206) if classification
        baseline_out_masked = x

        # 6. Compute MSE between baseline_out_masked and y_batch
        #    If you have a classification problem, you might do cross-entropy, etc.
        loss_value = mse_loss_fn(y_batch, baseline_out_masked)

    # 7. Backprop into secondary model
    grads = tape.gradient(loss_value, secondary_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, secondary_model.trainable_variables))

    return loss_value

def train_activation_transformer(X_train_np, y_train_np, epochs=5, batch_size=32):
    dataset_size = X_train_np.shape[0]
    for epoch in range(epochs):
        # Shuffle indices
        idx = np.arange(dataset_size)
        np.random.shuffle(idx)
        X_train_shuffled = X_train_np[idx]
        y_train_shuffled = y_train_np[idx]

        batch_losses = []
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            x_batch_np = X_train_shuffled[start:end]
            y_batch_np = y_train_shuffled[start:end]

            # Convert to TF tensors
            x_batch = tf.convert_to_tensor(x_batch_np, dtype=tf.float32)
            y_batch = tf.convert_to_tensor(y_batch_np, dtype=tf.float32)

            batch_loss = train_step(x_batch, y_batch)
            batch_losses.append(batch_loss.numpy())

        epoch_loss = np.mean(batch_losses)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# ------------------------------------------------
# 7) Run the Training
# ------------------------------------------------
# Convert X_train, y_train to float32
X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)

# Train for some epochs
train_activation_transformer(X_train, y_train, epochs=10, batch_size=32)

# ------------------------------------------------
# 8) Evaluate or Test
#    - We can do a quick check on X_val to see if masked baseline
#      predictions are better or not
# ------------------------------------------------
@tf.function
def evaluate_step(x_batch, y_batch):
    """
    Returns the MSE of the baseline with predicted nullification
    """
    final_out, perceptron_dict = extractor(x_batch, training=False)

    # Concat
    layer_units = []
    for layer in baseline_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer_units.append(layer.units)
    all_activs = []
    for (layer_name, activation) in perceptron_dict.items():
        all_activs.append(activation)
    cat_activations = tf.concat(all_activs, axis=1)

    combined_input = tf.concat([x_batch, cat_activations], axis=1)
    mask_pred = secondary_model(combined_input, training=False)

    # Nullify
    masked_dict = nullify_activations(perceptron_dict, mask_pred, layer_units, null_value=0.001)

    # Rerun baseline
    x = x_batch
    for layer in baseline_model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        elif isinstance(layer, tf.keras.layers.Dense):
            x = masked_dict[layer.name]
        else:
            x = layer(x, training=False)

    baseline_out_masked = x
    loss_val = mse_loss_fn(y_batch, baseline_out_masked)
    return loss_val

def evaluate_activation_transformer(X_val_np, y_val_np, batch_size=32):
    dataset_size = X_val_np.shape[0]
    all_losses = []
    for start in range(0, dataset_size, batch_size):
        end = start + batch_size
        x_batch_np = X_val_np[start:end]
        y_batch_np = y_val_np[start:end]
        x_batch = tf.convert_to_tensor(x_batch_np, dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_batch_np, dtype=tf.float32)

        val_loss = evaluate_step(x_batch, y_batch)
        all_losses.append(val_loss.numpy())
    return np.mean(all_losses)

val_loss = evaluate_activation_transformer(X_val, y_val)
print(f"Validation Loss with Masking: {val_loss:.4f}")