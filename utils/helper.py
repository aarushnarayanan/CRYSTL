import time
import random
import sys

# Simulated training parameters
NUM_EPOCHS = 10
BATCHES_PER_EPOCH = 50  # Simulating 50 batches per epoch

# Function to simulate model training with live updates
def train_model():
    print("Initializing model training...\n")
    time.sleep(1)  # Simulate startup delay
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0
        start_time = time.time()

        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        for batch in range(1, BATCHES_PER_EPOCH + 1):
            batch_loss = round(random.uniform(0.1, 2.5), 4)  # Simulated loss value
            epoch_loss += batch_loss
            
            # Simulating progress bar
            progress = int((batch / BATCHES_PER_EPOCH) * 30)  # 30-length bar
            bar = "[" + "=" * progress + " " * (30 - progress) + "]"
            sys.stdout.write(f"\rBatch {batch}/{BATCHES_PER_EPOCH} {bar} Loss: {batch_loss:.4f}")
            sys.stdout.flush()
            
            time.sleep(random.uniform(0.05, 0.2))  # Simulated computation time

        avg_loss = epoch_loss / BATCHES_PER_EPOCH
        elapsed_time = time.time() - start_time
        print(f"\nEpoch {epoch} completed - Avg Loss: {avg_loss:.4f} - Time: {elapsed_time:.2f}s\n")
    
    print("Testing complete! Model ran with " + str(random.randint(50, 60)) + "." + str(random.randint(1, 9)) + " percent accuracy'.")

if __name__ == "__main__":
    train_model()
