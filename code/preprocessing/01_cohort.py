import marimo

__generated_with = "0.14.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # ICU Mortality Model - Cohort Generation

    This notebook generates the ICU cohort for mortality prediction modeling following the PRD requirements.

    ## Objective
    Generate a cohort table containing:
    - `hospitalization_id`
    - `start_dttm`: ICU admission timestamp
    - `hour_24_start_dttm`: first ICU hour (may equal start_dttm)
    - `hour_24_end_dttm`: end of the first 24 hours
    - `disposition`: binary outcome (1 = expired, 0 = survived)

    ## Cohort Criteria
    - First 24 hours of first ICU stay
    - Exclude re-admissions and ICU readmissions
    - ICU-OR-ICU sequences treated as continuous ICU stay
    - Minimum 24-hour ICU stay
    - Adults (≥18 years)
    - 2020-2021 data
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""## Setup and Configuration""")
    return


@app.cell
def _():
    import sys
    import os
    sys.path.append(os.path.join('..', 'src'))

    import pandas as pd
    import numpy as np
    from pyclif import CLIF
    import json
    import warnings
    warnings.filterwarnings('ignore')

    print("=== ICU Mortality Model - Cohort Generation ===")
    print("Setting up environment...")
    return CLIF, json, np, os, pd


@app.cell
def _(json, os):
    def load_config():
        """Load configuration from config.json"""
        # Try top-level config_demo.json first (new location)
        config_path = os.path.join("..", "..", "config_demo.json")
        
        # If running from project root, adjust path
        if not os.path.exists(config_path):
            config_path = "config_demo.json"

        if not os.path.exists(config_path):
            # Fallback to local config_demo.json
            config_path = "config_demo.json"

        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = json.load(file)
            print(f"✅ Loaded configuration from {config_path}")
        else:
            raise FileNotFoundError(f"Configuration file not found. Tried: {config_path}")

        return config

    # Load configuration
    config = load_config()
    print(f"Site: {config['site']}")
    print(f"Data path: {config['clif2_path']}")
    print(f"File type: {config['filetype']}")
    return (config,)


@app.cell
def _(CLIF, config):
    # Initialize pyCLIF
    clif = CLIF(
        data_dir=config['clif2_path'],
        filetype=config['filetype'],
        timezone="US/Eastern"
    )

    print("✅ pyCLIF initialized successfully")
    return (clif,)


@app.cell
def _(mo):
    mo.md(r"""## Data Loading and Preparation""")
    return


@app.cell
def _(clif):
    # Load required tables using pyCLIF
    print("Loading required tables...")
    clif.initialize(["adt", "hospitalization", "patient"])

    # Load ADT data
    adt_df = clif.adt.df.copy()
    print(f"ADT data loaded: {len(adt_df)} records")

    # Load hospitalization data
    hosp_df = clif.hospitalization.df.copy()
    print(f"Hospitalization data loaded: {len(hosp_df)} records")

    # Load patient data
    patient_df = clif.patient.df.copy()
    print(f"Patient data loaded: {len(patient_df)} records")
    return adt_df, hosp_df, patient_df


@app.cell
def _(adt_df):
    adt_df.location_category.value_counts()
    return


@app.cell
def _(adt_df, hosp_df, pd):
    # Prepare data for cohort generation
    print("Preparing data for cohort generation...")

    # Merge ADT with hospitalization data
    icu_data = pd.merge(
        adt_df[['hospitalization_id', 'location_category', 'in_dttm', 'out_dttm']],
        hosp_df[['patient_id', 'hospitalization_id', 'age_at_admission', 'discharge_category', 'admission_dttm']],
        on='hospitalization_id',
        how='left'
    )

    print(f"Merged data: {len(icu_data)} records")

    # Convert datetime columns
    datetime_cols = ['in_dttm', 'out_dttm', 'admission_dttm']
    for col in datetime_cols:
        icu_data[col] = pd.to_datetime(icu_data[col])

    # Handle location categories (convert procedural to OR as in Inference_py.ipynb)
    icu_data.loc[icu_data['location_category'] == 'procedural', 'location_category'] = 'OR'
    icu_data['location_category'] = icu_data['location_category'].str.upper()

    print("✅ Data preparation completed")
    return icu_data, pd


@app.cell
def _(mo):
    mo.md(r"""## ICU Cohort Selection""")
    return


@app.cell
def _():
    #icu_data.head()
    return


@app.cell
def _(icu_data, pd):
    # Apply initial filters
    print("Applying initial cohort filters...")

    # Filter for ICU admissions within 48 hours of hospital admission
    icu_48hr_check = icu_data[
        (icu_data['location_category'] == 'ICU') &
        (icu_data['in_dttm'] >= icu_data['admission_dttm']) &
        (icu_data['in_dttm'] <= icu_data['admission_dttm'] + pd.Timedelta(hours=48)) &
       # (icu_data['admission_dttm'].dt.year >= 2020) & (icu_data['admission_dttm'].dt.year <= 2021) &
        (icu_data['age_at_admission'] >= 18) & (icu_data['age_at_admission'].notna())
    ]['hospitalization_id'].unique()

    print(f"Hospitalizations with ICU within 48hr: {len(icu_48hr_check)}")

    # Filter to relevant encounters and extend to 72 hours for location tracking
    icu_data_filtered = icu_data[
        icu_data['hospitalization_id'].isin(icu_48hr_check) &
        (icu_data['in_dttm'] <= icu_data['admission_dttm'] + pd.Timedelta(hours=72))
    ].reset_index(drop=True)

    print(f"Filtered data for processing: {len(icu_data_filtered)} records")
    return icu_data_filtered, pd


@app.cell
def _(icu_data_filtered, pd):
    # Process ICU-OR-ICU sequences (treat as continuous ICU)
    print("Processing ICU-OR-ICU sequences...")

    # Sort by admission time and create ranking
    icu_data_sorted = icu_data_filtered.sort_values(by=['in_dttm']).reset_index(drop=True)
    icu_data_sorted["RANK"] = icu_data_sorted.sort_values(by=['in_dttm'], ascending=True).groupby("hospitalization_id")["in_dttm"].rank(method="first", ascending=True).astype(int)

    # Find minimum ICU rank for each hospitalization
    min_icu = icu_data_sorted[icu_data_sorted['location_category'] == 'ICU'].groupby('hospitalization_id')['RANK'].min()
    icu_data_sorted = pd.merge(icu_data_sorted, pd.DataFrame(zip(min_icu.index, min_icu.values), columns=['hospitalization_id', 'min_icu']), on='hospitalization_id', how='left')

    # Filter to locations from first ICU onward
    icu_data_sorted = icu_data_sorted[icu_data_sorted['RANK'] >= icu_data_sorted['min_icu']].reset_index(drop=True)

    # Convert OR to ICU for continuity (ICU-OR-ICU treated as continuous ICU)
    icu_data_sorted.loc[icu_data_sorted['location_category'] == 'OR', 'location_category'] = 'ICU'

    print(f"After ICU-OR-ICU processing: {len(icu_data_sorted)} records")
    return icu_data_sorted, pd


@app.cell
def _(icu_data_sorted):
    # Group consecutive ICU locations
    print("Grouping consecutive ICU locations...")

    # Create groups for consecutive locations
    icu_data_sorted['group_id'] = (icu_data_sorted.groupby('hospitalization_id')['location_category'].shift() != icu_data_sorted['location_category']).astype(int)
    icu_data_sorted['group_id'] = icu_data_sorted.sort_values(by=['in_dttm'], ascending=True).groupby('hospitalization_id')['group_id'].cumsum()

    # Aggregate by groups
    icu_data_grouped = icu_data_sorted.sort_values(by=['in_dttm'], ascending=True).groupby(['patient_id', 'hospitalization_id', 'location_category', 'group_id']).agg(
        min_in_dttm=('in_dttm', 'min'),
        max_out_dttm=('out_dttm', 'max'),
        admission_dttm=('admission_dttm', 'first'),
        age=('age_at_admission', 'first'),
        dispo=('discharge_category', 'first')
    ).reset_index()

    print(f"Grouped data: {len(icu_data_grouped)} records")
    return icu_data_grouped, pd


@app.cell
def _(icu_data_grouped, pd):
    # Apply final cohort criteria
    print("Applying final cohort criteria...")

    # Find minimum ICU group for each hospitalization
    min_icu_group = icu_data_grouped[icu_data_grouped['location_category'] == 'ICU'].groupby('hospitalization_id')['group_id'].min()
    icu_data_with_groups = pd.merge(icu_data_grouped, pd.DataFrame(zip(min_icu_group.index, min_icu_group.values), columns=['hospitalization_id', 'min_icu_group']), on='hospitalization_id', how='left')

    # Filter to first ICU stay with minimum 24-hour duration
    icu_data_final = icu_data_with_groups[
        (icu_data_with_groups['min_icu_group'] == icu_data_with_groups['group_id']) &
        (icu_data_with_groups['max_out_dttm'] - icu_data_with_groups['min_in_dttm'] >= pd.Timedelta(hours=24))
    ].reset_index(drop=True)

    print(f"Final cohort before demographics: {len(icu_data_final)} records")

    # Add 24-hour endpoint
    icu_data_final['after_24hr'] = icu_data_final['min_in_dttm'] + pd.Timedelta(hours=24)

    # Select required columns
    icu_data_final = icu_data_final[['patient_id', 'hospitalization_id', 'min_in_dttm', 'max_out_dttm', 'after_24hr', 'age', 'dispo']]

    print("✅ ICU cohort criteria applied")
    return icu_data_final, pd


@app.cell
def _(mo):
    mo.md(r"""## Add Demographics and Create Final Cohort""")
    return


@app.cell
def _(icu_data_final, patient_df, pd):
    # Add patient demographics
    print("Adding patient demographics...")

    # Rename columns for consistency with CLIF 2.0
    patient_df_clean = patient_df.rename(columns={
        'race_category': 'race',
        'ethnicity_category': 'ethnicity',
        'sex_category': 'sex'
    })

    # Merge with patient data
    icu_data_demo = pd.merge(
        icu_data_final,
        patient_df_clean[['patient_id', 'sex', 'ethnicity', 'race']],
        on='patient_id',
        how='left'
    )

    # Filter out records with missing sex (data quality)
    icu_data_demo = icu_data_demo[~icu_data_demo['sex'].isna()].reset_index(drop=True)

    print(f"Final cohort with demographics: {len(icu_data_demo)} records")
    return icu_data_demo, pd


@app.cell
def _():
    #icu_data.head()
    return


@app.cell
def _(icu_data_demo):
    # Create final cohort table with required columns
    print("Creating final cohort table...")

    # Create disposition binary variable (1 = expired, 0 = survived)
    icu_data_demo['disposition'] = (icu_data_demo['dispo'].fillna('Other').str.contains('dead|expired|death|died', case=False, regex=True)).astype(int)

    # Create final cohort with PRD required columns
    cohort_final = icu_data_demo[[
        'hospitalization_id',
        'min_in_dttm',     # start_dttm
        'after_24hr',      # hour_24_end_dttm
        'disposition'
    ]].rename(columns={
        'min_in_dttm': 'start_dttm',
        'after_24hr': 'hour_24_end_dttm'
    })

    # Add hour_24_start_dttm (same as start_dttm for our cohort)
    cohort_final['hour_24_start_dttm'] = cohort_final['start_dttm']

    # Reorder columns as per PRD
    cohort_final = cohort_final[[
        'hospitalization_id',
        'start_dttm',
        'hour_24_start_dttm',
        'hour_24_end_dttm',
        'disposition'
    ]]

    print(f"✅ Final cohort created: {len(cohort_final)} hospitalizations")
    print(f"Mortality rate: {cohort_final['disposition'].mean():.3f}")
    return cohort_final,


@app.cell
def _(cohort_final):
    cohort_final['disposition'].value_counts()*100/cohort_final.shape[0]
    return


@app.cell
def _(mo):
    mo.md(r"""## Cohort Summary and Validation""")
    return


@app.cell
def _(cohort_final):
    # Display cohort summary
    print("=== ICU Cohort Summary ===")
    print(f"Total hospitalizations: {len(cohort_final):,}")
    print(f"Mortality rate: {cohort_final['disposition'].mean():.3f} ({cohort_final['disposition'].sum():,} deaths)")
    print(f"Survival rate: {1 - cohort_final['disposition'].mean():.3f} ({(cohort_final['disposition'] == 0).sum():,} survivors)")

    # Time range analysis
    print(f"\n=== Time Range Analysis ===")
    print(f"Cohort start date: {cohort_final['start_dttm'].min()}")
    print(f"Cohort end date: {cohort_final['start_dttm'].max()}")
    print(f"24-hour window duration: {(cohort_final['hour_24_end_dttm'] - cohort_final['hour_24_start_dttm']).iloc[0]}")

    # Validation checks
    print(f"\n=== Validation Checks ===")
    print(f"All 24-hour windows are exactly 24 hours: {((cohort_final['hour_24_end_dttm'] - cohort_final['hour_24_start_dttm']).dt.total_seconds() == 24*3600).all()}")
    print(f"No missing hospitalization IDs: {cohort_final['hospitalization_id'].isna().sum() == 0}")
    print(f"All start times before end times: {(cohort_final['start_dttm'] <= cohort_final['hour_24_end_dttm']).all()}")
    return


@app.cell
def _(mo):
    mo.md(r"""## Save Cohort to Output Directory""")
    return


@app.cell
def _(cohort_final, json, os):
    # Save cohort to protected_outputs/preprocessing directory
    output_path = os.path.join('..', '..', 'protected_outputs', 'preprocessing', 'icu_cohort.parquet')
    
    # If running from project root, adjust path
    if not os.path.exists(os.path.dirname(output_path)):
        output_path = os.path.join('protected_outputs', 'preprocessing', 'icu_cohort.parquet')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cohort_final.to_parquet(output_path, index=False)

    print(f"✅ Cohort saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"Shape: {cohort_final.shape}")

    # Save additional metadata
    metadata = {
        'cohort_size': len(cohort_final),
        'mortality_rate': float(cohort_final['disposition'].mean()),
        'date_range': {
            'start': cohort_final['start_dttm'].min().isoformat(),
            'end': cohort_final['start_dttm'].max().isoformat()
        },
        'criteria': {
            'min_age': 18,
            'years': '2020-2021',
            'icu_window': '48_hours_from_admission',
            'min_icu_duration': '24_hours',
            'icu_or_icu_handling': 'continuous_icu'
        }
    }

    metadata_path = os.path.join('..', '..', 'protected_outputs', 'preprocessing', 'cohort_metadata.json')
    
    # If running from project root, adjust path
    if not os.path.exists(os.path.dirname(metadata_path)):
        metadata_path = os.path.join('protected_outputs', 'preprocessing', 'cohort_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata saved to: {metadata_path}")
    print("\n🎉 Cohort generation completed successfully!")
    return metadata, output_path


if __name__ == "__main__":
    app.run()
