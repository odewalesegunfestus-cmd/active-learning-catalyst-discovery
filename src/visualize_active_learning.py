# Import pandas
# WHAT: Load Active Learning history table
# WHY: We need dataset size values for plotting
import pandas as pd

# Import matplotlib
# WHAT: Create graph
# WHY: Visualize model knowledge growth
import matplotlib.pyplot as plt

# Load Active Learning history
# WHAT: Read saved cycle history
# WHY: Use it to plot progress
history = pd.read_csv(
    "results/active_learning_history.csv"
)

# Create figure
# WHAT: Prepare plotting canvas
# WHY: Show dataset growth across cycles
plt.figure(figsize=(6, 4))

# Plot dataset size
# WHAT: Draw line showing dataset size per cycle
# WHY: Demonstrates that the model gains new knowledge
plt.plot(
    history["Cycle"],
    history["Dataset_Size"],
    marker="o"
)

# Add x-axis label
plt.xlabel("Active Learning Cycle")

# Add y-axis label
plt.ylabel("Dataset Size")

# Add title
plt.title("Active Learning Progress")

# Save figure
plt.savefig(
    "figures/active_learning_progress.png"
)

# Show figure
plt.show()

print()
print("Active Learning progress figure saved successfully.")