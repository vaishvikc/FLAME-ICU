import marimo

__generated_with = "0.16.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Setup and Configuration""")
    return


@app.cell
def _():

    import os
    import sys
    sys.path.append('..')
    from config_helper import get_project_root, ensure_dir, get_output_path, load_config

    import pandas as pd
    import numpy as np
    from clifpy.tables import Adt, Hospitalization, Patient
    from clifpy.clif_orchestrator import ClifOrchestrator
    import json
    import warnings
    warnings.filterwarnings('ignore')

    print("=== ICU Mortality Model - Cohort Generation ===")
    print("Setting up environment...")
    return (
        Adt,
        ClifOrchestrator,
        Hospitalization,
        Patient,
        ensure_dir,
        get_output_path,
        json,
        load_config,
        os,
        pd,
    )


@app.cell
def _(load_config):
    # Load configuration
    config = load_config()
    print(f"Site: {config['site']}")
    print(f"Data path: {config['data_directory']}")
    print(f"File type: {config['filetype']}")
    return (config,)


@app.cell
def _(mo):
    mo.md(r"""## Data Loading and Preparation""")
    return


@app.cell
def _(Adt, Hospitalization, Patient):
    # Load required tables using clifpy config file
    print("Loading required tables...")

    # Load ADT data using config file from project root
    adt_table = Adt.from_file(config_path='clif_config.json')
    adt_df = adt_table.df.copy()
    print(f"ADT data loaded: {len(adt_df)} records")

    # Load hospitalization data using config file
    hosp_table = Hospitalization.from_file(config_path='clif_config.json')
    hosp_df = hosp_table.df.copy()
    print(f"Hospitalization data loaded: {len(hosp_df)} records")

    # Load patient data using config file
    patient_table = Patient.from_file(config_path='clif_config.json')
    patient_df = patient_table.df.copy()
    print(f"Patient data loaded: {len(patient_df)} records")
    return adt_df, hosp_df, patient_df


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
    return (icu_data,)


@app.cell
def _(mo):
    mo.md(r"""## ICU Cohort Selection""")
    return


@app.cell
def _(config, icu_data, pd):
    # Apply initial filters
    print("Applying initial cohort filters...")

    if config['site'].lower() == 'mimic': 
        # Filter for ICU admissions within 48 hours of hospital admission
        icu_48hr_check = icu_data[
            (icu_data['location_category'] == 'ICU') &
            (icu_data['in_dttm'] >= icu_data['admission_dttm']) &
            (icu_data['in_dttm'] <= icu_data['admission_dttm'] + pd.Timedelta(hours=48)) &
            #(icu_data['admission_dttm'].dt.year >= 2018) & (icu_data['admission_dttm'].dt.year <= 2024) &
            (icu_data['age_at_admission'] >= 18) & (icu_data['age_at_admission'].notna())
        ]['hospitalization_id'].unique()
    else:
        # Filter for ICU admissions within 48 hours of hospital admission
        icu_48hr_check = icu_data[
            (icu_data['location_category'] == 'ICU') &
            (icu_data['in_dttm'] >= icu_data['admission_dttm']) &
            (icu_data['in_dttm'] <= icu_data['admission_dttm'] + pd.Timedelta(hours=48)) &
            (icu_data['admission_dttm'].dt.year >= 2018) & (icu_data['admission_dttm'].dt.year <= 2024) &
            (icu_data['age_at_admission'] >= 18) & (icu_data['age_at_admission'].notna())
        ]['hospitalization_id'].unique()

    print(f"Hospitalizations with ICU within 48hr: {len(icu_48hr_check)}")

    # Filter to relevant encounters and extend to 72 hours for location tracking
    icu_data_filtered = icu_data[
        icu_data['hospitalization_id'].isin(icu_48hr_check) &
        (icu_data['in_dttm'] <= icu_data['admission_dttm'] + pd.Timedelta(hours=72))
    ].reset_index(drop=True)

    print(f"Filtered data for processing: {len(icu_data_filtered)} records")
    return (icu_data_filtered,)


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
    return (icu_data_sorted,)


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
    return (icu_data_grouped,)


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
    return (icu_data_final,)


@app.cell
def _(mo):
    mo.md(r"""## Add Demographics and Create Final Cohort""")
    return


@app.cell
def _(icu_data_final, patient_df, pd):
    # Add patient demographics
    print("Adding patient demographics...")

    # Merge with patient data (keep original column names)
    icu_data_demo = pd.merge(
        icu_data_final,
        patient_df[['patient_id', 'sex_category', 'ethnicity_category', 'race_category', 'language_category']],
        on='patient_id',
        how='left'
    )

    # Filter out records with missing demographics (data quality)
    demographic_cols = ['sex_category', 'ethnicity_category', 'race_category']
    icu_data_demo = icu_data_demo[~icu_data_demo[demographic_cols].isna().any(axis=1)].reset_index(drop=True)

    print(f"Final cohort with demographics: {len(icu_data_demo)} records")
    return (icu_data_demo,)


@app.cell
def _(icu_data_demo):
    # Create final cohort table with required columns
    print("Creating final cohort table...")

    # Create disposition binary variable (1 = expired, 0 = survived)
    icu_data_demo['disposition'] = (icu_data_demo['dispo'].fillna('Other').str.contains('dead|expired|death|died', case=False, regex=True)).astype(int)

    # Create final cohort with PRD required columns and demographics
    cohort_final = icu_data_demo[[
        'hospitalization_id',
        'min_in_dttm',     # start_dttm
        'after_24hr',      # hour_24_end_dttm
        'disposition',
        'sex_category',
        'ethnicity_category', 
        'race_category',
        'language_category'
    ]].rename(columns={
        'min_in_dttm': 'start_dttm',
        'after_24hr': 'hour_24_end_dttm'
    })

    # Add hour_24_start_dttm (same as start_dttm for our cohort)
    cohort_final['hour_24_start_dttm'] = cohort_final['start_dttm']

    # Reorder columns as per PRD with demographics
    cohort_final = cohort_final[[
        'hospitalization_id',
        'start_dttm',
        'hour_24_start_dttm',
        'hour_24_end_dttm',
        'disposition',
        'sex_category',
        'ethnicity_category',
        'race_category', 
        'language_category'
    ]]

    print(f"✅ Final cohort created: {len(cohort_final)} hospitalizations")
    print(f"Mortality prevalence: {cohort_final['disposition'].mean()*100:.1f}%")
    return (cohort_final,)


@app.cell
def _(cohort_final):
    cohort_final['disposition'].value_counts()*100/cohort_final.shape[0]
    return


@app.cell
def _(mo):
    mo.md(r"""## SOFA Score Computation""")
    return


@app.cell
def _(ClifOrchestrator):
    # Initialize ClifOrchestrator with config file
    print("Initializing ClifOrchestrator for SOFA computation...")
    co = ClifOrchestrator(config_path='clif_config.json')
    print("✅ ClifOrchestrator initialized")
    return (co,)


@app.cell
def _(cohort_final, pd):
    # Prepare cohort DataFrame for SOFA computation
    print("Preparing cohort for SOFA score computation...")

    sofa_cohort_df = pd.DataFrame({
        'hospitalization_id': cohort_final['hospitalization_id'],
        'start_time': cohort_final['start_dttm'],
        'end_time': cohort_final['hour_24_end_dttm']
    })

    print(f"SOFA cohort prepared: {len(sofa_cohort_df)} hospitalizations")
    return (sofa_cohort_df,)


@app.cell
def _(cohort_final):
    # Extract hospitalization IDs for filtering SOFA data loads
    print("Extracting hospitalization IDs for SOFA table loading...")

    sofa_cohort_ids = cohort_final['hospitalization_id'].astype(str).unique().tolist()

    print(f"Extracted {len(sofa_cohort_ids)} hospitalization IDs for SOFA data filtering")
    return (sofa_cohort_ids,)


@app.cell
def _(co, sofa_cohort_ids):
    # Load required tables for SOFA computation with cohort filtering
    print("Loading required tables for SOFA computation...")
    print("SOFA requires: Labs (creatinine, platelet_count, po2_arterial, bilirubin_total)")
    print("               Vitals (map, spo2)")
    print("               Assessments (gcs_total)")
    print("               Medications (norepinephrine, epinephrine, dopamine, dobutamine)")
    print("               Respiratory (device_category, fio2_set)")

    # Define columns to load for each table (optimize memory usage)
    sofa_columns = {
        'labs': ['hospitalization_id', 'lab_result_dttm', 'lab_category', 'lab_value', 'lab_value_numeric'],
        'vitals': ['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
        'patient_assessments': ['hospitalization_id', 'recorded_dttm', 'assessment_category', 'numerical_value'],
        'medication_admin_continuous': None,  # Load all columns
        'respiratory_support': None  # Load all columns
    }

    sofa_tables = ['labs', 'vitals', 'patient_assessments', 'medication_admin_continuous', 'respiratory_support']

    for table_name in sofa_tables:
        table_cols = sofa_columns.get(table_name)
        print(f"Loading {table_name} with {len(table_cols) if table_cols else 'all'} columns and {len(sofa_cohort_ids)} hospitalization filters...")
        co.load_table(
            table_name,
            filters={'hospitalization_id': sofa_cohort_ids},
            columns=table_cols
        )

    print("✅ All required tables loaded for SOFA computation")
    return


@app.cell
def _(co):
    # Clean medication data: remove rows with null/NaN/missing med_dose or med_dose_unit
    print("Cleaning medication data...")

    med_df = co.medication_admin_continuous.df.copy()
    initial_count = len(med_df)

    print(f"Initial medication records: {initial_count:,}")

    # Remove rows where med_dose is null, NaN, or missing
    med_df = med_df[med_df['med_dose'].notna()]
    after_dose_filter = len(med_df)
    print(f"After removing null med_dose: {after_dose_filter:,} (removed {initial_count - after_dose_filter:,})")

    # Remove rows where med_dose_unit is null, NaN, or missing
    med_df = med_df[med_df['med_dose_unit'].notna()]
    after_unit_filter = len(med_df)
    print(f"After removing null med_dose_unit: {after_unit_filter:,} (removed {after_dose_filter - after_unit_filter:,})")

    # Also remove rows where med_dose_unit is the string 'nan' (sometimes happens)
    med_df = med_df[~med_df['med_dose_unit'].astype(str).str.lower().isin(['nan', 'none', ''])]
    final_count = len(med_df)
    print(f"After removing 'nan' string values: {final_count:,} (removed {after_unit_filter - final_count:,})")

    # Update the table
    co.medication_admin_continuous.df = med_df

    print(f"✅ Medication data cleaned: {initial_count:,} → {final_count:,} records ({initial_count - final_count:,} removed, {100*(initial_count - final_count)/initial_count:.1f}% reduction)")
    return


@app.cell
def _(co):
    # Convert medication units to mcg/kg/min for SOFA computation
    print("Converting medication units to mcg/kg/min for SOFA...")

    # Define preferred units for SOFA medications
    preferred_units = {
        'norepinephrine': 'mcg/kg/min',
        'epinephrine': 'mcg/kg/min',
        'dopamine': 'mcg/kg/min',
        'dobutamine': 'mcg/kg/min'
    }

    print(f"Converting {len(preferred_units)} medications: {list(preferred_units.keys())}")

    # Convert units (uses vitals table for weight data)
    co.convert_dose_units_for_continuous_meds(
        preferred_units=preferred_units,
        override = True, 
        save_to_table=True  # Saves to co.medication_admin_continuous.df_converted
    )

    # Check conversion results
    conversion_counts = co.medication_admin_continuous.conversion_counts

    print("\n=== Conversion Summary ===")
    print(f"Total conversion records: {len(conversion_counts):,}")

    # Check for conversion failures
    success_count = conversion_counts[conversion_counts['_convert_status'] == 'success']['count'].sum()
    total_count = conversion_counts['count'].sum()

    print(f"Successful conversions: {success_count:,} / {total_count:,} ({100*success_count/total_count:.1f}%)")

    # Show any failed conversions
    failed_conversions = conversion_counts[conversion_counts['_convert_status'] != 'success']
    if len(failed_conversions) > 0:
        print(f"\n⚠️ Found {len(failed_conversions)} conversion issues:")
        for _, row in failed_conversions.head(10).iterrows():
            print(f"  {row['med_category']}: {row['_clean_unit']} → {row['_convert_status']} ({row['count']} records)")
    else:
        print("✅ All conversions successful!")

    print("\n✅ Medication unit conversion completed")
    return


@app.cell
def _(co):
    co.medication_admin_continuous.df_converted
    return


@app.cell
def _():
    return


@app.cell
def _(co, sofa_cohort_df):
    # Compute SOFA scores
    print("Computing SOFA scores...")
    sofa_scores = co.compute_sofa_scores(
        cohort_df=sofa_cohort_df,
        id_name='hospitalization_id'
    )
    print(f"✅ SOFA scores computed: {sofa_scores.shape}")
    print(f"SOFA columns: {[col for col in sofa_scores.columns if 'sofa' in col.lower()]}")
    return (sofa_scores,)


@app.cell
def _(cohort_final, pd, sofa_scores):
    # Merge SOFA scores with cohort
    print("Merging SOFA scores with cohort...")

    cohort_with_sofa = pd.merge(
        cohort_final,
        sofa_scores,
        on='hospitalization_id',
        how='left'
    )

    print(f"✅ Cohort with SOFA scores: {cohort_with_sofa.shape}")
    print(f"Total columns: {len(cohort_with_sofa.columns)}")
    return (cohort_with_sofa,)


@app.cell
def _(mo):
    mo.md(r"""## Cohort Summary and Validation""")
    return


@app.cell
def _(cohort_with_sofa):
    # Display cohort summary
    print("=== ICU Cohort Summary ===")
    print(f"Total hospitalizations: {len(cohort_with_sofa):,}")
    print(f"Mortality prevalence: {cohort_with_sofa['disposition'].mean()*100:.1f}% ({cohort_with_sofa['disposition'].sum():,} deaths)")
    print(f"Survival prevalence: {(1 - cohort_with_sofa['disposition'].mean())*100:.1f}% ({(cohort_with_sofa['disposition'] == 0).sum():,} survivors)")

    # Time range analysis
    print(f"\n=== Time Range Analysis ===")
    print(f"Cohort start date: {cohort_with_sofa['start_dttm'].min()}")
    print(f"Cohort end date: {cohort_with_sofa['start_dttm'].max()}")
    print(f"24-hour window duration: {(cohort_with_sofa['hour_24_end_dttm'] - cohort_with_sofa['hour_24_start_dttm']).iloc[0]}")

    # Validation checks
    print(f"\n=== Validation Checks ===")
    print(f"All 24-hour windows are exactly 24 hours: {((cohort_with_sofa['hour_24_end_dttm'] - cohort_with_sofa['hour_24_start_dttm']).dt.total_seconds() == 24*3600).all()}")
    print(f"No missing hospitalization IDs: {cohort_with_sofa['hospitalization_id'].isna().sum() == 0}")
    print(f"All start times before end times: {(cohort_with_sofa['start_dttm'] <= cohort_with_sofa['hour_24_end_dttm']).all()}")
    return


@app.cell
def _(mo):
    mo.md(r"""## Save Cohort to Output Directory""")
    return


@app.cell
def _(cohort_with_sofa, ensure_dir, get_output_path, json, os):
    # Save cohort using simple helper for path management
    output_path = get_output_path('preprocessing', 'icu_cohort.parquet')
    ensure_dir(output_path)

    cohort_with_sofa.to_parquet(output_path, index=False)

    print(f"✅ Cohort saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"Shape: {cohort_with_sofa.shape}")

    # Save additional metadata
    metadata = {
        'cohort_size': len(cohort_with_sofa),
        'mortality_prevalence': float(cohort_with_sofa['disposition'].mean()),
        'mortality_prevalence_percent': float(cohort_with_sofa['disposition'].mean() * 100),
        'date_range': {
            'start': cohort_with_sofa['start_dttm'].min().isoformat(),
            'end': cohort_with_sofa['start_dttm'].max().isoformat()
        }
    }

    metadata_path = get_output_path('preprocessing', 'cohort_metadata.json')
    ensure_dir(metadata_path)

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata saved to: {metadata_path}")
    print("\n🎉 Cohort generation completed successfully!")
    return


@app.cell
def _(sofa_scores):
    sofa_scores
    return


@app.cell
def _(cohort_with_sofa):
    cohort_with_sofa
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
