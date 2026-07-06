# Import pandas
# WHAT: Library for working with tables and datasets
# WHY: We need to load and manipulate catalyst data
import pandas as pd

# Import Gaussian Process Regression
# WHAT: Machine learning model that predicts values and uncertainty
# WHY: Active Learning needs both predictions and uncertainty estimates
from sklearn.gaussian_process import GaussianProcessRegressor

# Import Latin Hypercube Sampling
# WHAT: Tool for generating well-spread virtual catalysts
# WHY: Active Learning needs unexplored catalyst candidates
from scipy.stats import qmc

# Load catalyst dataset
# WHAT: Read catalyst data from CSV file
# WHY: This is our starting experimental knowledge
data = pd.read_csv(
    "data/catalyst_data.csv"
)

# Display dataset
# WHAT: Show first few catalyst records
# WHY: Verify dataset loaded correctly
print()
print("INITIAL CATALYST DATA")
print(
    data.head()
)

# Display dataset size
# WHAT: Count number of catalysts
# WHY: Active Learning starts with this knowledge
print()
print("INITIAL DATASET SIZE")
print(
    len(data)
)

# Define features
# WHAT: Input variables used for prediction
# WHY: Catalyst composition influences overpotential
X = data[
    [
        "Ni_fraction",
        "Co_fraction",
        "Fe_fraction"
    ]
]

# Define target
# WHAT: Property we want to predict
# WHY: Overpotential measures catalyst performance
y = data[
    "Overpotential_mV"
]

# Create GPR model
# WHAT: Initialize Gaussian Process Regression
# WHY: Active Learning requires prediction uncertainty
gpr = GaussianProcessRegressor(
    normalize_y=True,
    random_state=42
)

# Train GPR model
# WHAT: Learn catalyst-performance relationship
# WHY: Model needs initial knowledge before exploration
gpr.fit(
    X,
    y
)

# Display confirmation
# WHAT: Confirm successful training
# WHY: Verify model is ready
print()
print("GPR MODEL TRAINED")

# Create LHS sampler
# WHAT: Generate virtual catalyst compositions
# WHY: Active Learning needs new candidates to explore
sampler = qmc.LatinHypercube(
    d=3,
    seed=42
)

# Generate 100 virtual catalysts
# WHAT: Create 100 unexplored catalyst compositions
# WHY: The model will evaluate these candidates
samples = sampler.random(
    n=100
)

# Create empty DataFrame
# WHAT: Store virtual catalyst compositions
# WHY: Easier to inspect and manipulate
virtual_catalysts = pd.DataFrame()

# Store Ni fraction
# WHAT: First LHS column
# WHY: Represents Ni composition
virtual_catalysts["Ni_fraction"] = samples[:, 0]

# Store Co fraction
# WHAT: Second LHS column
# WHY: Represents Co composition
virtual_catalysts["Co_fraction"] = samples[:, 1]

# Store Fe fraction
# WHAT: Third LHS column
# WHY: Represents Fe composition
virtual_catalysts["Fe_fraction"] = samples[:, 2]

# Calculate total composition
# WHAT: Sum Ni, Co and Fe
# WHY: Needed to normalize composition
total = (
    virtual_catalysts["Ni_fraction"]
    + virtual_catalysts["Co_fraction"]
    + virtual_catalysts["Fe_fraction"]
)

# Normalize Ni
# WHAT: Divide Ni by total
# WHY: Ensure valid catalyst composition
virtual_catalysts["Ni_fraction"] = (
    virtual_catalysts["Ni_fraction"] / total
)

# Normalize Co
# WHAT: Divide Co by total
# WHY: Ensure valid catalyst composition
virtual_catalysts["Co_fraction"] = (
    virtual_catalysts["Co_fraction"] / total
)

# Normalize Fe
# WHAT: Divide Fe by total
# WHY: Ensure valid catalyst composition
virtual_catalysts["Fe_fraction"] = (
    virtual_catalysts["Fe_fraction"] / total
)

# Display first few virtual catalysts
# WHAT: Show generated candidates
# WHY: Verify LHS generation worked
print()
print("VIRTUAL CATALYSTS GENERATED")
print(
    virtual_catalysts.head()
)

# Display total virtual catalysts
# WHAT: Count generated catalysts
# WHY: Confirm 100 candidates exist
print()
print("TOTAL VIRTUAL CATALYSTS")
print(
    len(virtual_catalysts)
)

# Predict virtual catalysts
# WHAT: Use the trained GPR model to evaluate all virtual catalysts
# WHY: Active Learning must estimate performance before choosing experiments
prediction, uncertainty = gpr.predict(
    virtual_catalysts,
    return_std=True
)

# Store predicted overpotential
# WHAT: Save model predictions
# WHY: We need to rank catalyst candidates
virtual_catalysts["Predicted_Overpotential"] = prediction

# Store uncertainty
# WHAT: Save GPR uncertainty estimates
# WHY: Active Learning uses uncertainty to decide what to learn next
virtual_catalysts["Uncertainty"] = uncertainty

# Display first few predictions
# WHAT: Show predicted catalyst performance
# WHY: Verify prediction step worked correctly
print()
print("FIRST 10 PREDICTIONS")
print(
    virtual_catalysts[
        [
            "Ni_fraction",
            "Co_fraction",
            "Fe_fraction",
            "Predicted_Overpotential",
            "Uncertainty"
        ]
    ].head(10)
)

# Save predictions
# WHAT: Store all virtual catalyst predictions
# WHY: Results can be reused later without rerunning the model
virtual_catalysts.to_csv(
    "results/virtual_catalyst_predictions.csv",
    index=False
)

# Display confirmation
# WHAT: Confirm successful save
# WHY: Verify results file was created
print()
print("VIRTUAL CATALYST PREDICTIONS SAVED")

# Create Active Learning score
# WHAT: Combine uncertainty and performance into one score
# WHY: We want catalysts that are promising and informative
virtual_catalysts["AL_score"] = (
    -virtual_catalysts["Predicted_Overpotential"]
    + virtual_catalysts["Uncertainty"]
)

# Rank catalysts
# WHAT: Sort catalysts by Active Learning score
# WHY: Highest score becomes the next experiment
ranked = virtual_catalysts.sort_values(
    by="AL_score",
    ascending=False
)

# Display top candidates
# WHAT: Show the most interesting catalysts
# WHY: These are the catalysts the model wants to learn from
print()
print("TOP 10 ACTIVE LEARNING CANDIDATES")
print(
    ranked.head(10)
)

# Select best candidate
# WHAT: Take the highest-ranked catalyst
# WHY: This becomes the next laboratory experiment
best_candidate = ranked.iloc[0]

print()
print("BEST NEXT EXPERIMENT")
print(best_candidate)

# Save ranking
# WHAT: Store ranked candidates
# WHY: Preserve results for future cycles
ranked.to_csv(
    "results/active_learning_ranking.csv",
    index=False
)

print()
print("ACTIVE LEARNING RANKING SAVED")

# Simulate experimental measurement
# WHAT: Pretend the laboratory tested the selected catalyst
# WHY: Active Learning learns from new experiments
experimental_overpotential = 205

# Create new experiment row
# WHAT: Store the new catalyst and measured result
# WHY: This new knowledge will be added to the dataset
new_experiment = pd.DataFrame(
    [
        {
            "Catalyst_ID": "AL_1",
            "Ni_fraction": best_candidate["Ni_fraction"],
            "Co_fraction": best_candidate["Co_fraction"],
            "Fe_fraction": best_candidate["Fe_fraction"],
            "Overpotential_mV": experimental_overpotential
        }
    ]
)

# Display new experiment
# WHAT: Show the catalyst that was tested
# WHY: Verify the experiment record
print()
print("NEW EXPERIMENT")
print(new_experiment)

# Add new experiment to dataset
# WHAT: Combine original catalyst data with the newly tested catalyst
# WHY: Active Learning improves by learning from new experimental results
updated_data = pd.concat(
    [
        data,
        new_experiment
    ],
    ignore_index=True
)

# Display updated dataset size
# WHAT: Count catalysts after adding new experiment
# WHY: Confirm dataset increased from 20 to 21
print()
print("UPDATED DATASET SIZE")
print(
    len(updated_data)
)

# Save updated dataset
# WHAT: Store the new dataset after one Active Learning cycle
# WHY: This updated dataset will be used for retraining
updated_data.to_csv(
    "results/updated_dataset_after_AL_1.csv",
    index=False
)

print()
print("UPDATED DATASET SAVED")

# Define updated features
# WHAT: Extract catalyst compositions from updated dataset
# WHY: Retraining requires all available knowledge
X_updated = updated_data[
    [
        "Ni_fraction",
        "Co_fraction",
        "Fe_fraction"
    ]
]

# Define updated target
# WHAT: Extract measured overpotential values
# WHY: These are the values the model learns from
y_updated = updated_data[
    "Overpotential_mV"
]

# Create new GPR model
# WHAT: Initialize a fresh model
# WHY: We want to retrain using the expanded dataset
gpr_updated = GaussianProcessRegressor(
    normalize_y=True,
    random_state=42
)

# Retrain model
# WHAT: Learn from 21 catalysts instead of 20
# WHY: New experiment should improve model knowledge
gpr_updated.fit(
    X_updated,
    y_updated
)

print()
print("MODEL RETRAINED WITH NEW KNOWLEDGE")

# Create cycle summary
# WHAT: Store the key result of this Active Learning cycle
# WHY: Makes the cycle result clear and reusable
cycle_summary = pd.DataFrame(
    [
        {
            "Cycle": 1,
            "Initial_dataset_size": len(data),
            "Updated_dataset_size": len(updated_data),
            "Selected_Ni_fraction": best_candidate["Ni_fraction"],
            "Selected_Co_fraction": best_candidate["Co_fraction"],
            "Selected_Fe_fraction": best_candidate["Fe_fraction"],
            "Predicted_Overpotential": best_candidate["Predicted_Overpotential"],
            "Uncertainty": best_candidate["Uncertainty"],
            "Simulated_Experimental_Overpotential": experimental_overpotential
        }
    ]
)

# Save cycle summary
# WHAT: Save Active Learning cycle information
# WHY: Documents what happened in Cycle 1
cycle_summary.to_csv(
    "results/active_learning_cycle_summary.csv",
    index=False
)

print()
print("ACTIVE LEARNING CYCLE SUMMARY SAVED")

# Generate Cycle 2 predictions
# WHAT: Use the retrained model to predict the same virtual catalyst pool again
# WHY: After learning AL_1, the model should make updated predictions
cycle2_prediction, cycle2_uncertainty = gpr_updated.predict(
    virtual_catalysts[
        [
            "Ni_fraction",
            "Co_fraction",
            "Fe_fraction"
        ]
    ],
    return_std=True
)

# Store Cycle 2 predicted overpotential
# WHAT: Save updated predictions after retraining
# WHY: We need new predictions to choose the next experiment
virtual_catalysts["Cycle2_Predicted_Overpotential"] = cycle2_prediction

# Store Cycle 2 uncertainty
# WHAT: Save updated uncertainty after retraining
# WHY: Active Learning uses uncertainty to select informative catalysts
virtual_catalysts["Cycle2_Uncertainty"] = cycle2_uncertainty

# Create Cycle 2 Active Learning score
# WHAT: Combine prediction and uncertainty again
# WHY: Select the next best catalyst after learning from AL_1
virtual_catalysts["Cycle2_AL_score"] = (
    -virtual_catalysts["Cycle2_Predicted_Overpotential"]
    + virtual_catalysts["Cycle2_Uncertainty"]
)

# Rank Cycle 2 candidates
# WHAT: Sort catalysts by Cycle 2 AL score
# WHY: Highest score becomes the second Active Learning experiment
cycle2_ranked = virtual_catalysts.sort_values(
    by="Cycle2_AL_score",
    ascending=False
)

# Display top Cycle 2 candidates
# WHAT: Show the highest-ranked catalysts
# WHY: Verify ranking before selecting the best experiment
print()
print("TOP 10 CYCLE 2 CANDIDATES")

print(
    cycle2_ranked[
        [
            "Ni_fraction",
            "Co_fraction",
            "Fe_fraction",
            "Cycle2_Predicted_Overpotential",
            "Cycle2_Uncertainty",
            "Cycle2_AL_score"
        ]
    ].head(10)
)

# Select Cycle 2 best candidate
# WHAT: Take the first catalyst from the Cycle 2 ranking
# WHY: This is the next experiment suggested by the updated model
cycle2_best_candidate = cycle2_ranked.iloc[0]

print()
print("CYCLE 2 BEST NEXT EXPERIMENT")
print(cycle2_best_candidate)

# Save Cycle 2 ranking
# WHAT: Store Cycle 2 ranked candidates
# WHY: Keep results for documentation and analysis
cycle2_ranked.to_csv(
    "results/active_learning_cycle2_ranking.csv",
    index=False
)

print()
print("CYCLE 2 RANKING SAVED")

# Simulate Cycle 2 experimental measurement
# WHAT: Pretend the laboratory tested the Cycle 2 selected catalyst
# WHY: Active Learning needs a new experimental result to learn again
cycle2_experimental_overpotential = 204

# Create Cycle 2 experiment row
# WHAT: Store the Cycle 2 catalyst composition and measured result
# WHY: This becomes new knowledge for the model
cycle2_new_experiment = pd.DataFrame(
    [
        {
            "Catalyst_ID": "AL_2",
            "Ni_fraction": cycle2_best_candidate["Ni_fraction"],
            "Co_fraction": cycle2_best_candidate["Co_fraction"],
            "Fe_fraction": cycle2_best_candidate["Fe_fraction"],
            "Overpotential_mV": cycle2_experimental_overpotential
        }
    ]
)

# Add Cycle 2 experiment to dataset
# WHAT: Combine updated dataset with AL_2
# WHY: Dataset grows from 21 to 22 catalysts
cycle2_updated_data = pd.concat(
    [
        updated_data,
        cycle2_new_experiment
    ],
    ignore_index=True
)

# Save Cycle 2 updated dataset
# WHAT: Store dataset after two Active Learning cycles
# WHY: Preserve the learning progress
cycle2_updated_data.to_csv(
    "results/updated_dataset_after_AL_2.csv",
    index=False
)

print()
print("CYCLE 2 UPDATED DATASET SAVED")
print(cycle2_updated_data.tail())

# Create Cycle 2 summary
# WHAT: Record the important results from the second Active Learning cycle
# WHY: Document what the model learned and selected
cycle2_summary = pd.DataFrame(
    [
        {
            "Cycle": 2,
            "Dataset_Size_After_Cycle": len(cycle2_updated_data),
            "Selected_Ni_fraction": cycle2_best_candidate["Ni_fraction"],
            "Selected_Co_fraction": cycle2_best_candidate["Co_fraction"],
            "Selected_Fe_fraction": cycle2_best_candidate["Fe_fraction"],
            "Predicted_Overpotential": cycle2_best_candidate["Cycle2_Predicted_Overpotential"],
            "Uncertainty": cycle2_best_candidate["Cycle2_Uncertainty"],
            "Simulated_Experimental_Overpotential": cycle2_experimental_overpotential
        }
    ]
)

# Save Cycle 2 summary
# WHAT: Store Cycle 2 results permanently
# WHY: Keep a record of the second Active Learning iteration
cycle2_summary.to_csv(
    "results/active_learning_cycle2_summary.csv",
    index=False
)

print()
print("CYCLE 2 SUMMARY SAVED")

# Display Cycle 2 summary
# WHAT: Show summary table
# WHY: Verify saved results
print(cycle2_summary)