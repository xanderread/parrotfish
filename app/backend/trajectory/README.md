What you need to run:

- Pickle file for model
- A numpy array that specifies the previous trajectory of the vessel
- Number of steps in the future to predict
- 

Load model:

from runner import load_model, predict_trajectory
model, scaler = load_model("path/to/saved/model/")

and then make predictions!

predictions = predict_trajectory(model, initial_sequence, 10, scaler)

The scaler is crucial for correct predictions. Always save and load it with the model.
Initial sequence length must match what the model was trained with.
Coordinates should be in the same format as training data (lat/lon).
The scaler ensures predictions are in the same scale as your original data.