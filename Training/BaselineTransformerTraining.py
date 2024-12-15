import pandas as pd
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, LayerNormalization, MultiHeadAttention, Reshape
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay


# Suppress all warnings
warnings.filterwarnings('ignore')

#--------------------------------------------------------------------------------------------------------------------------------------------------

"""
Description-- Latest

"""

#--------------------------------------------------------------------------------------------------------------------------------------------------
# PREPROCESSING & ENCODING

# Load the training dataset
file_path = '/Users/sohamsane/Documents/Coding Projects/PerceptronDropout/Data/Model1_Omni/Asus_laptops.csv'
data = pd.read_csv(file_path)

# Drop rows with missing values for the output column
data = data.dropna(subset=['Price'])

# Identify relevant features and target
target_trim = 'Price'
input_columns = ['Brand', 'Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']  

# Split the training data into features and target
X = data[input_columns]
y_trim = data[target_trim]

# Define preprocessing for numerical and categorical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create a preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit and transform the features for the training dataset
X_preprocessed = preprocessor.fit_transform(X)

# Encode the target variable
trim_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
y_trim_encoded = trim_encoder.fit_transform(y_trim.values.reshape(-1, 1))

# Split the data into training and test sets
X_train, X_test, y_trim_train, y_trim_test = train_test_split(
    X_preprocessed, y_trim_encoded, test_size=0.2, random_state=42)

# Load the unseen test dataset and handle missing values by filling them with zero
file_path_unseen = '/Users/sohamsane/Documents/Coding Projects/PerceptronDropout/Data/Model2_Split/Acer_Lenovo_laptops.csv'
unseen_data = pd.read_csv(file_path_unseen)
unseen_data.fillna(0, inplace=True)

# Transform the unseen test data with the same preprocessing pipeline
unseen_preprocessed = preprocessor.transform(unseen_data[input_columns])

# Ensure there are no NaNs in the input data
X_train[np.isnan(X_train)] = 0
X_test[np.isnan(X_test)] = 0


#--------------------------------------------------------------------------------------------------------------------------------------------------
# TRANSFORMER ARCHITECTURE 

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)  # Assumes self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

def build_transformer_model(input_shape, embed_dim, num_heads, ff_dim, num_transformer_blocks, dropout_rate):
    inputs = Input(shape=(input_shape,))
    x = Dense(embed_dim, kernel_regularizer=l2(0.05), name="Dense_Embed")(inputs)
    x = Reshape((1, embed_dim))(x)  # Ensure embedding dimension is properly set

    for block_id in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate, name=f"TransformerBlock_{block_id}")(x)
    
    x = Flatten()(x)  # Create a 2D tensor [batch_size, features]
    x = Dropout(dropout_rate, name="Dropout_Final")(x)
    output_layer = Dense(y_trim_encoded.shape[1], activation='softmax', kernel_regularizer=l2(0.01), name="Output")(x)

    model = Model(inputs=inputs, outputs=output_layer, name="TransformerModel")
    return model

# Build the transformer model
input_shape = X_train.shape[1]
embed_dim = 64  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer
num_transformer_blocks = 2  # Number of transformer blocks
dropout_rate = 0.1  # Increased dropout rate

model = build_transformer_model(input_shape, embed_dim, num_heads, ff_dim, num_transformer_blocks, dropout_rate)
model.summary()

#--------------------------------------------------------------------------------------------------------------------------------------------------
# TRANSFORMER TRAINING & BUILD

# Compile the model with the specified optimizer and loss functions
model.compile(
    optimizer=Adam(learning_rate=ExponentialDecay(
        initial_learning_rate=0.0025, 
        decay_steps=100000,
        decay_rate=0.98,
        staircase=True
    )), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)  
history = model.fit(X_train, y_trim_train, 
                    epochs=125, 
                    batch_size=64,  
                    validation_split=0.05,
                    callbacks=[early_stopping])

# Save the model
model.save('/Users/sohamsane/Documents/Coding Projects/PerceptronDropout/baseline_model.keras')

# Load the saved model
loaded_model = tf.keras.models.load_model('/Users/sohamsane/Documents/Coding Projects/PerceptronDropout/baseline_model.keras', custom_objects={'TransformerBlock': TransformerBlock})

#------------------------------------------------------------------------------------------------------------------------------------
# DEBUGGING TRAINING OUTPUT 

# Get predictions
y_trim_pred = loaded_model.predict(X_test) #Test During Training

# Decode the trim predictions
y_trim_pred_decoded = trim_encoder.inverse_transform(y_trim_pred)

# Print the true and predicted trims for the first 10 inputs
true_trim_decoded = trim_encoder.inverse_transform(y_trim_test[:10])
print("True Price:", true_trim_decoded.flatten())
print("Predicted Price:", y_trim_pred_decoded[:10].flatten())

# Visualize the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------
# TESTING UNSEEN DATA 

# Get predictions
y_trim_pred = loaded_model.predict(unseen_preprocessed)

# Decode the trim predictions
y_trim_pred_decoded = trim_encoder.inverse_transform(y_trim_pred)

# Extract true prices from the last column of the unseen data
true_prices = unseen_data.iloc[:, -1].values.reshape(-1, 1)

# Print the true and predicted trims for the first 10 inputs
print("UNSEEN DATA SET REGULAR TRANSFORMER PREDICTED PRICES: ")
print("True Price:", true_prices[:10])
print("Predicted Price:", y_trim_pred_decoded[:10].flatten())

avg_perc_diff = []

for i in range(10):
    percent_diff = 100*(abs(true_prices[i] - y_trim_pred_decoded[i].flatten())/true_prices[i])
    print(f"Percent Difference for Test Case {i+1}: {percent_diff}%")
    avg_perc_diff.append(percent_diff)

print(f"Average Percent Difference: {np.mean(avg_perc_diff)}%")

#--------------------------------------------------------------------------------------------------------------------------------------------------


#RECORDING PERCEPTRON ACTIVATIONS - POST TRAINING THRU A RUN OF THE DATASET

# Step 1: Create a hook for perceptron outputs
class PerceptronOutputExtractor:
    def __init__(self, model):
        self.model = model

    def extract_outputs(self, inputs):
        """
        Extract perceptron inputs and outputs from the model for the given inputs.
        Returns a dictionary with perceptron IDs as keys and their inputs/outputs as values.
        """
        layer_data = {}  # Store inputs and outputs for each perceptron
        x = inputs

        for layer in self.model.layers:
            try:
                # Skip Input layers
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue

                if isinstance(layer, Dense):
                    # Store perceptron inputs (before activation)
                    perceptron_inputs = x.numpy().flatten()
                    
                    # Forward pass
                    x = layer(x)

                    # Store perceptron outputs (after activation)
                    perceptron_outputs = x.numpy().flatten()
                    layer_name = layer.name

                    # Record data for each perceptron
                    for perceptron_id, (input_value, output_value) in enumerate(zip(perceptron_inputs, perceptron_outputs)):
                        key = f"{layer_name}_{perceptron_id}"
                        layer_data[key] = {
                            "input": input_value,
                            "output": output_value
                        }

                elif isinstance(layer, Dropout):
                    x = layer(x, training=False)

                elif isinstance(layer, MultiHeadAttention):
                    x = layer(x, x)  # Self-attention for simplicity

                elif isinstance(layer, Flatten):
                    x = layer(x)

                elif isinstance(layer, Reshape):
                    x = layer(x)

                elif isinstance(layer, LayerNormalization):
                    x = layer(x)

                else:
                    x = layer(x)

            except Exception as e:
                print(f"Error processing layer {layer.name}: {e}")
                raise e

        return layer_data
    
# Initialize the extractor with the trained model
extractor = PerceptronOutputExtractor(loaded_model)

# Step 2: Generate perceptron outputs for training data
def generate_perceptron_outputs(extractor, model, X_train):
    """
    Run the training data through the extractor and gather inputs, outputs, and perceptron activations.
    """
    grouped_outputs = []

    for i in range(len(X_train)):
        single_input = np.expand_dims(X_train[i], axis=0)  # Prepare a single input
        perceptron_data = extractor.extract_outputs(tf.constant(single_input, dtype=tf.float32))

        # Get the model's prediction for this input
        model_output = model.predict(single_input, verbose=0).flatten()

        # Create a grouped entry
        group = {
            "input": single_input.flatten().tolist(),  # Convert input to a list
            "output": model_output.tolist(),          # Convert output to a list
            "perceptrons": perceptron_data            # Dictionary of perceptron inputs/outputs
        }

        grouped_outputs.append(group)

    return grouped_outputs

# Step 3: Save outputs to CSV
def convert_to_serializable(data):
    """
    Recursively convert TensorFlow and NumPy types to Python native types.
    """
    if isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(v) for v in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert NumPy arrays to Python lists
    elif isinstance(data, (np.float32, np.float64, np.int32, np.int64)):
        return data.item()  # Convert NumPy scalar types to Python scalars
    elif isinstance(data, tf.Tensor):
        return data.numpy().tolist()  # Convert TensorFlow tensors to Python lists
    elif hasattr(data, 'dtype') and isinstance(data.dtype, tf.dtypes.DType):
        return float(data)  # Convert TensorFlow scalar to Python float
    else:
        return data  # Return as-is if already serializable

def save_grouped_data_to_json(grouped_outputs, output_file):
    """
    Save grouped data (inputs, outputs, and perceptron data) to a JSON file.
    """
    serializable_data = convert_to_serializable(grouped_outputs)  # Convert to serializable format
    with open(output_file, "w") as f:
        json.dump(serializable_data, f, indent=4)


# Generate & Save the outputs to a JSON file
grouped_outputs = generate_perceptron_outputs(extractor, model, X_train)
output_file = "/Users/sohamsane/Documents/Coding Projects/PerceptronDropout/Data/GroupedPerceptronDataWithInputs.json"
save_grouped_data_to_json(grouped_outputs, output_file)

print(f"Grouped perceptron data with inputs saved to {output_file}")


#--------------------------------------------------------------------------------------------------------------------------------------------------