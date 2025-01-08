import asyncio
import random
from typing import Dict, Any, Tuple
import numpy as np

# Define model registry
MODELS = {
    "model_v1": {"latency": 100, "accuracy": 0.85, "reward": 0.0},  # Reward is updated dynamically
    "model_v2": {"latency": 200, "accuracy": 0.9, "reward": 0.0},
    "model_v3": {"latency": 150, "accuracy": 0.88, "reward": 0.0},
}

# Mock inference for models
async def infer(model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simulates model inference by introducing a delay and returning mock results."""
    latency = MODELS[model_name]["latency"]
    await asyncio.sleep(latency / 1000)  # Simulate inference latency
    return {"model": model_name, "output": f"Processed: {input_data}", "latency": latency}

# Reinforcement feedback simulator
def get_feedback(response: Dict[str, Any]) -> float:
    """Simulates user feedback for reinforcement learning."""
    # Mock feedback: randomize reward based on model accuracy
    model_name = response["model"]
    base_reward = MODELS[model_name]["accuracy"]
    noise = np.random.normal(0, 0.05)  # Add noise for variability
    return max(0.0, min(1.0, base_reward + noise))  # Reward is clipped between 0 and 1

# Update model rewards based on feedback
def update_model_reward(model_name: str, reward: float, alpha: float = 0.1):
    """Updates the model's reward using reinforcement learning."""
    MODELS[model_name]["reward"] = (1 - alpha) * MODELS[model_name]["reward"] + alpha * reward

# Model selection logic
def select_model(input_data: Dict[str, Any]) -> str:
    """Selects the best model based on a weighted combination of reward and latency."""
    # Calculate score: higher reward is better, lower latency is better
    scores = {
        model: (model_data["reward"] / (1 + model_data["latency"]))
        for model, model_data in MODELS.items()
    }
    # Select the model with the highest score
    selected_model = max(scores, key=scores.get)
    return selected_model

# Main inferencing pipeline
async def inference_pipeline(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handles the complete inference pipeline with dynamic model selection."""
    # Step 1: Select the optimal model
    selected_model = select_model(input_data)
    print(f"Selected model: {selected_model}")

    # Step 2: Perform inference
    response = await infer(selected_model, input_data)

    # Step 3: Collect feedback
    reward = get_feedback(response)
    print(f"Feedback (reward): {reward}")

    # Step 4: Update model reward
    update_model_reward(selected_model, reward)

    # Step 5: Return response
    response["reward"] = reward
    return response

# Example Usage
async def main():
    # Simulate multiple requests
    for i in range(10):
        input_data = {"query": f"Input data {i}"}
        response = await inference_pipeline(input_data)
        print(f"Response: {response}")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())

