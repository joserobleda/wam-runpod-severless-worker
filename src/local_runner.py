import time
from predict import Predictor

# This script is for local testing on a Mac or CPU.
# It simulates the process of running a prediction job.

if __name__ == "__main__":
    print("🚀 Starting local video generation test...")

    # 1. Initialize the predictor
    # This will download the models, which will take a long time on the first run.
    print("   - Initializing predictor (this may take a while)...")
    start_setup = time.time()
    predictor = Predictor()
    predictor.setup()
    end_setup = time.time()
    print(f"   ✅ Predictor initialized in {end_setup - start_setup:.2f} seconds.")

    # 2. Define the input prompt for the prediction
    # You can change these parameters to test different inputs.
    job_input = {
        "prompt": "a beautiful sunset over the mountains",
        "size": "512x512",
        "num_frames": 16,
        "num_inference_steps": 15,
        "fps": 8,
        "seed": 42
    }
    print(f"   - Starting prediction with prompt: '{job_input['prompt']}'")

    # 3. Run the prediction
    start_predict = time.time()
    output_path = predictor.predict(**job_input)
    end_predict = time.time()

    print(f"   ✅ Prediction finished in {end_predict - start_predict:.2f} seconds.")
    print(f"   🎬 Video saved to: {output_path}")
    print("\n🎉 Local test complete.") 