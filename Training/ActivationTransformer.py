import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from BaselineTransformerTraining import TransformerBlock

# ------------------------------------------------
# 1) Load JSON and prepare (input, target) data
# ------------------------------------------------
with open("/Users/sohamsane/Documents/Coding Projects/PerceptronDropout/Data/GroupedPerceptronDataWithInputs.json", "r") as f:
    activation_data = json.load(f)

inputs = []
targets = []

for entry in activation_data:
    input_activations = [data["output"] for data in entry["perceptrons"].values()]
    
    # Example: "ground-truth" mask is 1 if perceptron_id in ["Dense_Embed_0","Dense_Embed_1"]
    target_nullify = [
        1 if perceptron_id in ["Dense_Embed_0", "Dense_Embed_1"] else 0
        for perceptron_id in entry["perceptrons"].keys()
    ]
    
    inputs.append(input_activations)
    targets.append(target_nullify)

inputs = np.array(inputs, dtype=np.float32)
targets = np.array(targets, dtype=np.float32)

X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# ------------------------------------------------
# 2) Define a function to nullify certain activations
# ------------------------------------------------
def nullify_activations(original_activations, nullify_mask):
    """
    Nullify specific perceptrons (1 = nullify).
    """
    kept_values     = original_activations * (1.0 - nullify_mask)
    nullified_values = 0.00001 * nullify_mask
    return kept_values + nullified_values

# ------------------------------------------------
# 3) Load and freeze the baseline model
# ------------------------------------------------
baseline_model = tf.keras.models.load_model(
    '/Users/sohamsane/Documents/Coding Projects/PerceptronDropout/baseline_model.keras',
    custom_objects={'TransformerBlock': TransformerBlock}
)
# Freeze baseline so its weights won't update
for layer in baseline_model.layers:
    layer.trainable = False

# (Optional) Inspect the baseline's input shape
print("Baseline model input shape:", baseline_model.input_shape)
print("Baseline model output shape:", baseline_model.output_shape)

# ------------------------------------------------
# 4) Build the secondary model
# ------------------------------------------------
def build_secondary_model(input_dim, output_dim):
    inputs_ = Input(shape=(input_dim,))
    x = Dense(128, activation="relu")(inputs_)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    # Final layer: Sigmoid in [0,1], so we can threshold at 0.5 or treat it as probabilities
    x = Dense(output_dim, activation="sigmoid")(x)
    return Model(inputs=inputs_, outputs=x)

secondary_model = build_secondary_model(
    input_dim=inputs.shape[1],  # e.g. 70
    output_dim=targets.shape[1] # also e.g. 70
)

# We'll create our own optimizer, or reuse Keras default
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# ------------------------------------------------
# 5) Custom Training Loop
# ------------------------------------------------

# Let's define a custom loss that measures how close the secondary model's
# predicted mask is to the "target" mask, *plus* maybe something about baseline's output.
# For simplicity, let's do standard binary crossentropy vs. y_train.
loss_fn = tf.keras.losses.BinaryCrossentropy()

# If you want to incorporate the baseline's final output into the loss,
# you could define your own function, e.g.:
# def custom_loss_fn(baseline_out, desired_baseline_out, predicted_mask, true_mask):
#    # combine these terms however you want
#    return some_value

@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        # 1) Predict the mask
        mask_pred = secondary_model(x_batch, training=True)  # shape (batch_size, input_dim)

        # 2) Nullify
        masked_batch = nullify_activations(x_batch, mask_pred)
        
        # 3) Pass through baseline (training=False so it remains frozen)
        baseline_out = baseline_model(masked_batch, training=False)
        
        # Right now, we don't do anything with baseline_out in this example.
        # We'll just measure how close mask_pred is to y_batch using BCE:
        loss_value = loss_fn(y_batch, mask_pred)

        # If you want to incorporate the baseline's output into the loss, you can do it here,
        # e.g. by measuring difference from an original baseline or from some ground truth.
        # For example:
        #
        #   some_baseline_loss = tf.reduce_mean( tf.square(baseline_out - some_label) )
        #   total_loss = loss_value + 0.001 * some_baseline_loss
        #
        # Then "loss_value = total_loss"
        #
        # For demonstration, we'll keep it as is:
        
    grads = tape.gradient(loss_value, secondary_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, secondary_model.trainable_variables))
    return loss_value

def train_secondary_custom(X_train, y_train, epochs=10, batch_size=32):
    num_samples = X_train.shape[0]
    for epoch in range(epochs):
        # Shuffle indices if you like
        indices = tf.random.shuffle(tf.range(num_samples))
        X_train_shuffled = tf.gather(X_train, indices)
        y_train_shuffled = tf.gather(y_train, indices)

        batch_losses = []
        for i in range(0, num_samples, batch_size):
            x_batch = X_train_shuffled[i : i+batch_size]
            y_batch = y_train_shuffled[i : i+batch_size]
            loss_value = train_step(x_batch, y_batch)
            batch_losses.append(loss_value.numpy())
        
        epoch_loss = np.mean(batch_losses)
        print(f"Epoch {epoch+1}, Loss = {epoch_loss:.4f}")

# ------------------------------------------------
# 6) Run the custom training loop
# ------------------------------------------------
train_secondary_custom(X_train, y_train, epochs=10, batch_size=32)

# After training, you can evaluate on X_val, or run your feedback loop, etc.