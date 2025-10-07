#!/usr/bin/env python3
"""
CLIF Table Validation Scanner - Focused Version
===============================================

Validates ONLY the specific columns used in the preprocessing pipeline.
Performs targeted validation for columns actually needed by 01_cohort.py and 02_feature_assmebly.py.

Key Features:
- Loads only required columns (memory efficient)
- Reports only critical errors that would break preprocessing
- Fast execution with minimal memory footprint
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import CLIF tables
from clifpy.tables import (
    Adt,
    Hospitalization,
    Patient,
    Labs,
    Vitals,
    PatientAssessments,
    MedicationAdminContinuous,
    RespiratorySupport
)


# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_colored(text: str, color: str = Colors.ENDC):
    """Print colored text to terminal."""
    print(f"{color}{text}{Colors.ENDC}")


def print_header(text: str):
    """Print a formatted header."""
    print()
    print_colored("=" * 80, Colors.HEADER)
    print_colored(text, Colors.HEADER + Colors.BOLD)
    print_colored("=" * 80, Colors.HEADER)


# Define ONLY the columns actually used in preprocessing
REQUIRED_COLUMNS = {
    'Patient': {
        'columns': ['patient_id', 'sex_category', 'race_category', 'ethnicity_category', 'language_category'],
        'critical': ['patient_id']  # Must have for pipeline to work
    },
    'Hospitalization': {
        'columns': ['patient_id', 'hospitalization_id', 'age_at_admission', 'discharge_category', 'admission_dttm'],
        'critical': ['hospitalization_id', 'admission_dttm', 'age_at_admission']
    },
    'Adt': {
        'columns': ['hospitalization_id', 'location_category', 'in_dttm', 'out_dttm'],
        'critical': ['hospitalization_id', 'location_category', 'in_dttm', 'out_dttm']
    },
    'Vitals': {
        'columns': ['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
        'critical': ['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
        'categories_used': ['heart_rate', 'map', 'sbp', 'respiratory_rate', 'spo2', 'temp_c', 'weight_kg']  # weight_kg used for med unit conversion
    },
    'Labs': {
        'columns': ['hospitalization_id', 'lab_result_dttm', 'lab_category', 'lab_value', 'lab_value_numeric'],
        'critical': ['hospitalization_id', 'lab_result_dttm', 'lab_category'],
        'categories_used': [
            'albumin', 'alt', 'ast', 'bicarbonate', 'bilirubin_total', 'bun', 'chloride',
            'creatinine', 'inr', 'lactate', 'platelet_count', 'po2_arterial', 'potassium',
            'pt', 'ptt', 'sodium', 'wbc'  # From 02_feature_assmebly.py category_filters
        ]
    },
    'PatientAssessments': {
        'columns': ['hospitalization_id', 'recorded_dttm', 'assessment_category', 'numerical_value'],
        'critical': ['hospitalization_id', 'assessment_category'],
        'categories_used': ['gcs_total']
    },
    'MedicationAdminContinuous': {
        'columns': None,  # Load all - needed for unit conversion
        'critical': ['hospitalization_id', 'admin_dttm', 'med_category'],
        'categories_used': [
            'norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin',
            'dopamine', 'dobutamine', 'milrinone', 'isoproterenol'
        ]
    },
    'RespiratorySupport': {
        'columns': None,  # Load all - multiple columns used
        'critical': ['hospitalization_id', 'recorded_dttm', 'device_category'],
        'categories_used': ['device_category', 'fio2_set', 'peep_set']
    }
}


def validate_required_columns(df: pd.DataFrame, table_name: str) -> Tuple[List[str], List[str]]:
    """
    Validate that required columns exist in the dataframe.

    Returns:
        Tuple of (missing_critical_columns, missing_optional_columns)
    """
    config = REQUIRED_COLUMNS.get(table_name, {})
    required_cols = config.get('columns', [])
    critical_cols = config.get('critical', [])

    if required_cols is None:  # Table loads all columns
        critical_cols = config.get('critical', [])
        missing_critical = [col for col in critical_cols if col not in df.columns]
        return missing_critical, []

    missing_critical = [col for col in critical_cols if col not in df.columns]
    missing_optional = [col for col in required_cols if col not in df.columns and col not in critical_cols]

    return missing_critical, missing_optional


def validate_categories(df: pd.DataFrame, table_name: str) -> Dict[str, List[str]]:
    """
    Check if required category values exist in the data.

    Returns:
        Dict with missing and available categories
    """
    config = REQUIRED_COLUMNS.get(table_name, {})
    categories_used = config.get('categories_used', [])

    if not categories_used:
        return {}

    # Determine the category column name
    if 'vital_category' in df.columns:
        cat_col = 'vital_category'
    elif 'lab_category' in df.columns:
        cat_col = 'lab_category'
    elif 'assessment_category' in df.columns:
        cat_col = 'assessment_category'
    elif 'med_category' in df.columns:
        cat_col = 'med_category'
    else:
        return {}

    # Get unique categories in the data
    available_categories = df[cat_col].dropna().unique().tolist()

    # Find missing categories
    missing_categories = [cat for cat in categories_used if cat not in available_categories]

    return {
        'missing': missing_categories,
        'available': available_categories,
        'required': categories_used
    }


def validate_table_focused(table_class, table_name: str, config_path: str = 'clif_config.json') -> Dict[str, Any]:
    """
    Load and validate a single CLIF table with focus on preprocessing requirements.

    Returns:
        Dict containing focused validation results
    """
    result = {
        'table_name': table_name,
        'status': 'not_checked',
        'critical_errors': [],
        'warnings': [],
        'stats': {}
    }

    try:
        # Get columns to load
        table_config = REQUIRED_COLUMNS.get(table_name, {})
        columns_to_load = table_config.get('columns')

        # Load table with only required columns
        print(f"  Loading {table_name}", end='')
        if columns_to_load:
            print(f" ({len(columns_to_load)} columns)", end='')
        else:
            print(f" (all columns)", end='')
        print("...", end='')

        table_instance = table_class.from_file(
            config_path=config_path,
            columns=columns_to_load,
            sample_size=None
        )

        if table_instance.df is None or len(table_instance.df) == 0:
            result['status'] = 'no_data'
            result['critical_errors'].append("No data loaded")
            print_colored(" ❌ No data", Colors.FAIL)
            return result

        df = table_instance.df
        result['stats']['num_records'] = len(df)
        result['stats']['num_columns'] = len(df.columns)
        print_colored(f" ✅ {len(df):,} records", Colors.OKGREEN)

        # Validate required columns
        missing_critical, missing_optional = validate_required_columns(df, table_name)

        if missing_critical:
            result['critical_errors'].append(f"Missing critical columns: {missing_critical}")
            print_colored(f"    ❌ Missing critical columns: {', '.join(missing_critical)}", Colors.FAIL)

        if missing_optional:
            result['warnings'].append(f"Missing optional columns: {missing_optional}")
            print_colored(f"    ⚠️  Missing optional columns: {', '.join(missing_optional)}", Colors.WARNING)

        # Validate categories if applicable
        category_check = validate_categories(df, table_name)
        if category_check:
            missing_cats = category_check.get('missing', [])
            if missing_cats:
                # Only critical for feature tables
                if table_name in ['Vitals', 'Labs', 'PatientAssessments']:
                    result['critical_errors'].append(f"Missing required categories: {missing_cats[:5]}")
                    print_colored(f"    ❌ Missing categories: {', '.join(missing_cats[:5])}", Colors.FAIL)
                    if len(missing_cats) > 5:
                        print_colored(f"       ... and {len(missing_cats)-5} more", Colors.FAIL)
                else:
                    result['warnings'].append(f"Missing categories: {missing_cats[:5]}")

        # Check for null values in critical columns
        critical_cols = table_config.get('critical', [])
        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                null_pct = (null_count / len(df)) * 100
                if null_pct > 50:
                    result['critical_errors'].append(f"Column '{col}' has {null_pct:.1f}% missing values")
                    print_colored(f"    ❌ {col}: {null_pct:.1f}% missing", Colors.FAIL)
                elif null_pct > 10:
                    result['warnings'].append(f"Column '{col}' has {null_pct:.1f}% missing values")

        # Determine overall status
        if result['critical_errors']:
            result['status'] = 'error'
        elif result['warnings']:
            result['status'] = 'warning'
        else:
            result['status'] = 'valid'

        # Clean up
        del table_instance

    except FileNotFoundError:
        result['status'] = 'not_found'
        result['critical_errors'].append("Table file not found")
        print_colored(" ❌ File not found", Colors.FAIL)

    except Exception as e:
        result['status'] = 'error'
        result['critical_errors'].append(f"Load error: {str(e)}")
        print_colored(f" ❌ Error: {str(e)}", Colors.FAIL)

    return result


def check_pipeline_readiness(results: List[Dict]) -> Dict[str, bool]:
    """
    Check if each preprocessing script can run based on validation results.
    """
    # Tables needed for 01_cohort.py
    cohort_tables = ['Patient', 'Hospitalization', 'Adt']
    sofa_tables = ['Labs', 'Vitals', 'PatientAssessments', 'MedicationAdminContinuous', 'RespiratorySupport']

    # Tables needed for 02_feature_assmebly.py
    feature_tables = ['Vitals', 'Labs', 'RespiratorySupport', 'MedicationAdminContinuous',
                     'PatientAssessments', 'Hospitalization']

    # Check readiness
    cohort_core_ready = all(
        any(r['table_name'] == t and r['status'] != 'error' and r['status'] != 'not_found' for r in results)
        for t in cohort_tables
    )

    sofa_ready = all(
        any(r['table_name'] == t and r['status'] != 'not_found' for r in results)
        for t in sofa_tables
    )

    features_ready = all(
        any(r['table_name'] == t and r['status'] != 'error' and r['status'] != 'not_found' for r in results)
        for t in feature_tables
    )

    return {
        'cohort_generation': cohort_core_ready,
        'sofa_calculation': sofa_ready,
        'feature_extraction': features_ready
    }


def main():
    """Main execution function."""
    # Define tables to validate
    TABLES_TO_VALIDATE = [
        (Patient, 'Patient'),
        (Hospitalization, 'Hospitalization'),
        (Adt, 'Adt'),
        (Vitals, 'Vitals'),
        (Labs, 'Labs'),
        (PatientAssessments, 'PatientAssessments'),
        (MedicationAdminContinuous, 'MedicationAdminContinuous'),
        (RespiratorySupport, 'RespiratorySupport')
    ]

    print_header("CLIF TABLE VALIDATION - PREPROCESSING FOCUSED")
    print_colored("Validating only columns required for preprocessing pipeline", Colors.OKBLUE)

    # Load config
    try:
        with open('clif_config.json', 'r') as f:
            config = json.load(f)
        print(f"\n📍 Site: {config.get('site', 'unknown')}")
        print(f"📁 Data: {config.get('data_directory', 'unknown')}")
        print(f"📄 Type: {config.get('filetype', 'unknown')}")
    except Exception as e:
        print_colored(f"Warning: Could not load config: {e}", Colors.WARNING)

    print_colored("\n" + "─" * 80, Colors.OKCYAN)

    # Validate each table
    results = []
    error_count = 0
    warning_count = 0

    for i, (table_class, table_name) in enumerate(TABLES_TO_VALIDATE, 1):
        print(f"\n[{i}/{len(TABLES_TO_VALIDATE)}] {table_name}")
        result = validate_table_focused(table_class, table_name)
        results.append(result)

        if result['critical_errors']:
            error_count += len(result['critical_errors'])
        if result['warnings']:
            warning_count += len(result['warnings'])

    # Summary
    print_header("VALIDATION SUMMARY")

    # Count statuses
    valid_tables = sum(1 for r in results if r['status'] == 'valid')
    warning_tables = sum(1 for r in results if r['status'] == 'warning')
    error_tables = sum(1 for r in results if r['status'] == 'error')
    not_found_tables = sum(1 for r in results if r['status'] == 'not_found')

    print(f"✅ Valid: {valid_tables}/{len(results)} tables")
    print(f"⚠️  Warnings: {warning_tables}/{len(results)} tables ({warning_count} issues)")
    print(f"❌ Errors: {error_tables}/{len(results)} tables ({error_count} critical issues)")

    if not_found_tables > 0:
        print(f"❓ Not Found: {not_found_tables} tables")

    # Pipeline readiness
    print_colored("\n" + "─" * 80, Colors.OKCYAN)
    print_colored("PREPROCESSING PIPELINE READINESS", Colors.BOLD)
    print()

    readiness = check_pipeline_readiness(results)

    # 01_cohort.py readiness
    if readiness['cohort_generation']:
        print_colored("✅ 01_cohort.py - Cohort Generation: READY", Colors.OKGREEN)
    else:
        print_colored("❌ 01_cohort.py - Cohort Generation: NOT READY", Colors.FAIL)
        # Show which tables are blocking
        cohort_tables = ['Patient', 'Hospitalization', 'Adt']
        for t in cohort_tables:
            for r in results:
                if r['table_name'] == t and r['status'] in ['error', 'not_found']:
                    print(f"     └─ {t}: {r['critical_errors'][0] if r['critical_errors'] else 'Not found'}")

    if readiness['sofa_calculation']:
        print_colored("✅ 01_cohort.py - SOFA Calculation: READY", Colors.OKGREEN)
    else:
        print_colored("⚠️  01_cohort.py - SOFA Calculation: PARTIAL", Colors.WARNING)
        print("     └─ SOFA scores may be incomplete due to missing tables")

    # 02_feature_assmebly.py readiness
    if readiness['feature_extraction']:
        print_colored("✅ 02_feature_assmebly.py - Feature Extraction: READY", Colors.OKGREEN)
    else:
        print_colored("❌ 02_feature_assmebly.py - Feature Extraction: NOT READY", Colors.FAIL)
        # Show which tables are blocking
        feature_tables = ['Vitals', 'Labs', 'RespiratorySupport', 'MedicationAdminContinuous', 'PatientAssessments', 'Hospitalization']
        for t in feature_tables:
            for r in results:
                if r['table_name'] == t and r['status'] in ['error', 'not_found']:
                    print(f"     └─ {t}: {r['critical_errors'][0] if r['critical_errors'] else 'Not found'}")

    # Critical issues summary
    if error_count > 0:
        print_colored("\n" + "─" * 80, Colors.OKCYAN)
        print_colored("⚠️  CRITICAL ISSUES TO RESOLVE", Colors.FAIL + Colors.BOLD)
        print()

        for r in results:
            if r['critical_errors']:
                print_colored(f"• {r['table_name']}:", Colors.FAIL)
                for error in r['critical_errors'][:3]:  # Show max 3 errors per table
                    print(f"  - {error}")

    print_colored("\n" + "=" * 80, Colors.HEADER)

    # Return exit code
    if error_tables > 0 or not_found_tables > 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    sys.exit(main())