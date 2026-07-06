# Import pandas
# WHAT: Create and save tables
# WHY: We want to record Active Learning progress
import pandas as pd

# Create Active Learning history
# WHAT: Store dataset size after each cycle
# WHY: Track how model knowledge grows
history = pd.DataFrame(
    {
        "Cycle": [0, 1, 2],
        "Dataset_Size": [20, 21, 22]
    }
)

# Save history
# WHAT: Save Active Learning progress
# WHY: Use later for visualization
history.to_csv(
    "results/active_learning_history.csv",
    index=False
)

print()
print("ACTIVE LEARNING HISTORY SAVED")
print(history)