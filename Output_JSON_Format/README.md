# JSON output Standardisation
We standaridised the JSON output both for worst_case and LIRA attacks where possible. A generic JSON output strcture is presented as under:

## General Structure
Key components of JSON output across attacks will be:

````
log_id: Log identifier - a random unique id for each entry 
log_time: the time when the log was created
metadata: standardised variables related to a specific attack type
attack_experiment_logger: Attack experiment logger - keeps instances of metric across iterations
````

### Worst-Case Attack
A worst case attack will have a following components in their metadata component of JSON output.
````
metadata:
    experiment_details: 
        n_reps: 

````