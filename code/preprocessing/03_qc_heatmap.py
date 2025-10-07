import marimo

__generated_with = "0.16.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # ICU 24-Hour Missing Data Heatmap

    Simple missing data analysis showing % of hospitalizations WITHOUT data for each hour-feature combination.

    Missing % = 100 - [(Unique hospitalizations with ≥1 value) / (Total hospitalizations) × 100]

    Red intensity indicates severity of missingness (darker red = more missing data).
    """
    )
    return


@app.cell
def _():
    import sys
    sys.path.append('code')  # For config_helper when running from project root
    from config_helper import get_project_root, get_output_path

    import polars as pl
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    print("=== ICU 24-Hour Missing Data Analysis ===")
    return get_output_path, get_project_root, np, os, pl, plt, sns


@app.cell
def _(get_output_path, get_project_root, os):
    # Setup paths using config_helper (same as 02_feature_assembly.py)
    data_path = get_output_path('preprocessing', '')
    project_root = get_project_root()
    output_path = os.path.join(project_root, 'share_to_box', 'qc')

    os.makedirs(output_path, exist_ok=True)

    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    return data_path, output_path


@app.cell
def _(data_path, os, pl):
    # Load wide dataset with 24-hour windows
    wide_df_path = os.path.join(data_path, 'wide_df_24hr.parquet')

    if not os.path.exists(wide_df_path):
        raise FileNotFoundError(f"Wide dataset not found at {wide_df_path}. Run 02_feature_assembly.py first.")

    df = pl.read_parquet(wide_df_path)

    print(f"✅ Loaded dataset: {df.shape}")
    print(f"Hospitalizations: {df['hospitalization_id'].n_unique()}")

    # Show available columns
    print(f"Columns: {len(df.columns)}")
    print("Sample columns:", df.columns[:10])
    return (df,)


@app.cell
def _(data_path, os, pl):
    # Load cohort to get time window columns
    cohort_path = os.path.join(data_path, 'icu_cohort.parquet')

    if not os.path.exists(cohort_path):
        raise FileNotFoundError(f"Cohort not found at {cohort_path}. Run 01_cohort.py first.")

    cohort_df = pl.read_parquet(cohort_path)

    print(f"✅ Loaded cohort: {len(cohort_df)} hospitalizations")
    print(f"Cohort columns: {cohort_df.columns[:10]}")
    return (cohort_df,)


@app.cell
def _(cohort_df, df, pl):
    # Join time window columns from cohort
    df_with_times = df.join(
        cohort_df.select(['hospitalization_id', 'hour_24_start_dttm', 'hour_24_end_dttm']),
        on='hospitalization_id',
        how='left'
    )

    print(f"✅ Joined time window columns from cohort")
    print(f"Shape after join: {df_with_times.shape}")

    # Verify all events are within 24-hour window (strip timezone for comparison)
    df_check = df_with_times.with_columns([
        pl.col('event_time').dt.replace_time_zone(None).alias('event_time_tz_stripped'),
        pl.col('hour_24_start_dttm').dt.replace_time_zone(None).alias('start_tz_stripped'),
        pl.col('hour_24_end_dttm').dt.replace_time_zone(None).alias('end_tz_stripped')
    ]).with_columns([
        (pl.col('event_time_tz_stripped') >= pl.col('start_tz_stripped')).alias('after_start'),
        (pl.col('event_time_tz_stripped') <= pl.col('end_tz_stripped')).alias('before_end')
    ])

    outside_window = df_check.filter(
        ~pl.col('after_start') | ~pl.col('before_end')
    )

    print(f"Events outside 24hr window: {len(outside_window):,} / {len(df_with_times):,}")
    if len(outside_window) > 0:
        print("⚠️  Warning: Some events are outside the 24-hour window")
        print(f"  Events before start: {(~df_check['after_start']).sum()}")
        print(f"  Events after end: {(~df_check['before_end']).sum()}")
    else:
        print("✅ All events are within their 24-hour windows")

    return (df_with_times,)


@app.cell
def _(df_with_times, pl):
    # Calculate hour from ICU admission (0-23)
    df_hourly = df_with_times.with_columns([
        # Drop timezone if present to avoid mismatch
        pl.col('event_time').dt.replace_time_zone(None).alias('event_time_clean'),
        pl.col('hour_24_start_dttm').dt.replace_time_zone(None).alias('start_time_clean')
    ]).with_columns([
        # Calculate hours from start of 24-hour window
        ((pl.col('event_time_clean') - pl.col('start_time_clean')).dt.total_seconds() / 3600)
        .floor()
        .cast(pl.Int8)
        .alias('hour')
    ]).filter(
        pl.col('hour').is_between(0, 23)
    )

    print(f"✅ Added hour column: {df_hourly['hour'].min()} to {df_hourly['hour'].max()}")
    print(f"Events within 24 hours: {len(df_hourly):,}")
    return (df_hourly,)


@app.cell
def _(df_hourly):
    # Identify feature columns (exclude metadata and static demographics)
    exclude_cols = [
        # Identifiers
        'hospitalization_id', 'patient_id',
        # Timestamps
        'event_time', 'hour_24_start_dttm', 'hour_24_end_dttm',
        'event_time_clean', 'start_time_clean',
        # Temporal
        'hour',
        # Outcomes
        'disposition',
        # Static demographics (not time-varying)
        'age_at_admission', 'sex_category', 'ethnicity_category',
        'race_category', 'language_category'
    ]

    # Get numeric features only
    feature_cols = [
        col for col in df_hourly.columns
        if col not in exclude_cols and df_hourly[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
    ]

    print(f"Found {len(feature_cols)} numeric features for analysis")
    print(f"Sample features: {feature_cols[:10]}")
    return exclude_cols, feature_cols


@app.cell
def _(df_hourly, feature_cols, pl):
    # Calculate missing percentage for each feature-hour combination
    print("Calculating missing data % for each hour-feature combination...")

    total_hosp = df_hourly['hospitalization_id'].n_unique()
    print(f"Total hospitalizations: {total_hosp}")

    missing_data = []

    for _feat in feature_cols[:]:  # Process all features
        # For each hour, count unique hospitalizations with non-null data
        for _hr in range(24):
            hour_data = df_hourly.filter(pl.col('hour') == _hr)

            # Count unique hospitalizations with non-null values
            n_with_data = hour_data.filter(
                pl.col(_feat).is_not_null()
            )['hospitalization_id'].n_unique()

            coverage_pct = (n_with_data / total_hosp) * 100
            missing_pct = 100 - coverage_pct  # Flip to missing percentage

            missing_data.append({
                'feature': _feat,
                'hour': _hr,
                'missing_pct': missing_pct,
                'coverage_pct': coverage_pct,  # Keep for reference
                'n_with_data': n_with_data,
                'n_missing': total_hosp - n_with_data,
                'total_hosp': total_hosp
            })

        # Add TOTAL: across all 24 hours, how many have at least one value?
        all_hours_data = df_hourly.filter(
            pl.col(_feat).is_not_null()
        )
        n_with_data_total = all_hours_data['hospitalization_id'].n_unique()
        coverage_pct_total = (n_with_data_total / total_hosp) * 100
        missing_pct_total = 100 - coverage_pct_total

        missing_data.append({
            'feature': _feat,
            'hour': 24,  # Use 24 to sort after 0-23
            'missing_pct': missing_pct_total,
            'coverage_pct': coverage_pct_total,
            'n_with_data': n_with_data_total,
            'n_missing': total_hosp - n_with_data_total,
            'total_hosp': total_hosp
        })

    missing_df = pl.DataFrame(missing_data)

    print(f"✅ Calculated missing data % for {missing_df['feature'].n_unique()} features")
    print(f"Total records: {len(missing_df):,} (25 per feature: 24 hourly + 1 overall)")
    return missing_df, total_hosp


@app.cell
def _(missing_df, pl):
    # Simple feature categorization
    def categorize_feature(feat):
        feat_lower = feat.lower()

        # Vitals
        if any(k in feat_lower for k in ['heart_rate', 'map', 'sbp', 'respiratory_rate', 'spo2', 'temp_c']):
            return 'vitals'
        # Labs
        elif any(k in feat_lower for k in ['albumin', 'alt', 'ast', 'bicarbonate', 'bilirubin', 'bun',
                                           'chloride', 'creatinine', 'hemoglobin', 'inr', 'lactate',
                                           'platelet', 'po2', 'potassium', 'pt', 'ptt', 'sodium', 'wbc']):
            return 'labs'
        # Medications
        elif any(k in feat_lower for k in ['norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin',
                                           'dopamine', 'dobutamine', 'propofol', 'fentanyl', 'midazolam']):
            return 'medications'
        # Respiratory
        elif any(k in feat_lower for k in ['fio2', 'peep', 'device']):
            return 'respiratory'
        else:
            return 'other'

    # Add category column
    missing_with_cat = missing_df.with_columns([
        pl.col('feature').map_elements(categorize_feature, return_dtype=pl.Utf8).alias('category')
    ])

    # Show category distribution
    category_counts = missing_with_cat['category'].value_counts().sort('count', descending=True)
    print("\nFeature categories:")
    for _cat_row in category_counts.to_dicts():
        print(f"  {_cat_row['category']}: {_cat_row['count'] // 25} features")  # Divide by 25 (24 hours + Total)

    return categorize_feature, missing_with_cat


@app.cell
def _(mo):
    mo.md(r"""## Missing Data Heatmaps by Category""")
    return


@app.cell
def _(missing_with_cat, np, os, output_path, pl, plt, sns):
    # Create heatmap for each category using Seaborn
    categories = ['vitals', 'labs', 'medications', 'respiratory', 'other']

    heatmap_files = []
    for _cat in categories:
        cat_data = missing_with_cat.filter(pl.col('category') == _cat).to_pandas()

        if len(cat_data) == 0:
            print(f"⚠️  No features in category: {_cat}")
            continue

        # Sort features by average missing percentage (most complete → most missing)
        feat_order = cat_data.groupby('feature')['missing_pct'].mean().sort_values().index.tolist()

        # Pivot data for heatmap (features x hours)
        heatmap_data = cat_data.pivot(index='feature', columns='hour', values='missing_pct')
        heatmap_data = heatmap_data.reindex(feat_order)

        # Determine figure size and annotation font size
        n_features = len(feat_order)
        fig_height = max(6, n_features * 0.4)  # Scale with number of features
        fig_width = 15  # Slightly wider to fit 25 columns (0-23 + Total)

        # Adjust annotation font size based on number of features
        if n_features <= 10:
            annot_fontsize = 8
        elif n_features <= 20:
            annot_fontsize = 6
        else:
            annot_fontsize = 5

        # Create figure with dark background
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='#1a1a1a')
        plt.subplots_adjust(left=0.25)  # Add space for y-axis labels
        ax.set_facecolor('#2d2d2d')  # Dark background for plot area

        # Create heatmap with red gradient
        sns.heatmap(
            heatmap_data,
            annot=True,  # Show percentages in cells
            fmt='.1f',   # Format to 1 decimal place
            cmap='Reds',  # Red gradient (white → dark red)
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Missing %', 'shrink': 0.8},
            linewidths=0.5,
            linecolor='#404040',  # Dark gray gridlines
            annot_kws={'fontsize': annot_fontsize},
            ax=ax
        )

        # Update colorbar colors for dark theme
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')

        # Styling with white text for dark theme
        ax.set_xlabel('Hour from ICU Admission', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel(f'{_cat.capitalize()} Features', fontsize=12, fontweight='bold', color='white')
        ax.set_title(f'{_cat.capitalize()}: Missing Data % (by Hour)',
                     fontsize=14, fontweight='bold', pad=15, color='white')

        # Update tick labels to white with custom x-labels (0-23, Total)
        x_labels = [str(i) for i in range(24)] + ['Total']
        ax.set_xticklabels(x_labels, rotation=0, ha='center', color='white')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9, color='white')
        ax.tick_params(axis='both', colors='white')  # Tick marks white

        # Tight layout
        plt.tight_layout()

        # Save to JPG
        output_file = os.path.join(output_path, f'{_cat}_missing_heatmap.jpg')
        fig.savefig(output_file, format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)  # Close to free memory

        print(f"✅ Saved {_cat} heatmap: {output_file}")
        heatmap_files.append(output_file)

    return categories, heatmap_files


@app.cell
def _(missing_with_cat, os, output_path):
    # Save missing data to CSV for further analysis
    missing_csv_path = os.path.join(output_path, 'hourly_missing_data.csv')
    missing_with_cat.to_pandas().to_csv(missing_csv_path, index=False)

    print(f"\n✅ Saved missing data: {missing_csv_path}")
    print(f"File size: {os.path.getsize(missing_csv_path) / 1024:.1f} KB")
    return (missing_csv_path,)


@app.cell
def _(missing_with_cat, pl):
    # Summary statistics
    print("\n=== Missing Data Summary ===")

    # Overall stats
    overall_stats = missing_with_cat.select([
        pl.col('missing_pct').mean().alias('mean_missing'),
        pl.col('missing_pct').median().alias('median_missing'),
        pl.col('missing_pct').min().alias('min_missing'),
        pl.col('missing_pct').max().alias('max_missing')
    ])

    print("Overall missing data statistics:")
    for _stat in overall_stats.to_dicts()[0].items():
        print(f"  {_stat[0]}: {_stat[1]:.1f}%")

    # By category
    print("\nMean missing % by category:")
    cat_means = missing_with_cat.group_by('category').agg([
        pl.col('missing_pct').mean().alias('mean_missing')
    ]).sort('mean_missing', descending=False)  # Sort ascending (least missing first)

    for _cat_mean_row in cat_means.to_dicts():
        print(f"  {_cat_mean_row['category']}: {_cat_mean_row['mean_missing']:.1f}%")

    # Features with best/worst missingness
    feat_missing = missing_with_cat.group_by('feature').agg([
        pl.col('missing_pct').mean().alias('mean_missing')
    ]).sort('mean_missing', descending=False)  # Sort ascending

    print("\nTop 5 features (least missing - best data availability):")
    for _top_feat_row in feat_missing.head(5).to_dicts():
        print(f"  {_top_feat_row['feature']}: {_top_feat_row['mean_missing']:.1f}% missing")

    print("\nBottom 5 features (most missing - worst data availability):")
    for _bot_feat_row in feat_missing.tail(5).to_dicts():
        print(f"  {_bot_feat_row['feature']}: {_bot_feat_row['mean_missing']:.1f}% missing")

    print("\n✅ Missing data analysis complete!")
    return cat_means, feat_missing, overall_stats


if __name__ == "__main__":
    app.run()