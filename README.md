# Active Learning Catalyst Discovery

This repository demonstrates an Active Learning workflow for AI-driven catalyst discovery.

## Workflow

Initial Catalyst Data  
↓  
Train GPR Model  
↓  
Generate Virtual Catalysts  
↓  
Predict Overpotential and Uncertainty  
↓  
Select Best Next Experiment  
↓  
Simulate Experimental Result  
↓  
Add New Data  
↓  
Retrain Model  
↓  
Repeat Learning Cycle  

## Key Outputs

- `results/virtual_catalyst_predictions.csv`
- `results/active_learning_ranking.csv`
- `results/updated_dataset_after_AL_1.csv`
- `results/updated_dataset_after_AL_2.csv`
- `results/active_learning_history.csv`
- `figures/active_learning_progress.png`

## Tools Used

- Python
- Pandas
- SciPy
- Scikit-learn
- Matplotlib

## Author

Odewale Segun Festus
