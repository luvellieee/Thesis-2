import pandas as pd

def clean_job_data(csv_file):
    """
    Clean job descriptions data from CSV by removing rows where:
    - Industry (Column D) is 'other' or 'error'
    - Rate Type (Column P) is 'hourly'
    
    Outputs cleaned data as TSV file.
    
    Args:
        csv_file: Path to the CSV file
    """
    
    # Read the CSV file
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Display initial information
    print(f"\nInitial data shape: {df.shape}")
    print(f"Total rows before cleaning: {len(df)}")
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns):
        print(f"  Column {chr(65+i)} (index {i}): {col}")
    
    # Get column names for Column D (index 3) and Column P (index 15)
    if len(df.columns) > 3:
        industry_col = df.columns[3]  # Column D
        print(f"\nColumn D (Industry): '{industry_col}'")
        print(f"Unique values: {df[industry_col].unique()}")
    else:
        print("\nError: CSV doesn't have enough columns for Column D")
        return None
    
    if len(df.columns) > 15:
        rate_type_col = df.columns[15]  # Column P
        print(f"\nColumn P (Rate Type): '{rate_type_col}'")
        print(f"Unique values: {df[rate_type_col].unique()}")
    else:
        print("\nError: CSV doesn't have enough columns for Column P")
        return None
    
    # Count rows to be removed
    industry_filter = df[industry_col].astype(str).str.lower().isin(['other', 'error'])
    rate_type_filter = df[rate_type_col].astype(str).str.lower() == 'hourly'
    
    rows_with_other_error = industry_filter.sum()
    rows_with_hourly = rate_type_filter.sum()
    rows_with_both = (industry_filter & rate_type_filter).sum()
    
    print(f"\nRows with Industry = 'other' or 'error': {rows_with_other_error}")
    print(f"Rows with Rate Type = 'hourly': {rows_with_hourly}")
    print(f"Rows with both conditions: {rows_with_both}")
    
    # Clean the data - remove rows where:
    # 1. Industry is 'other' or 'error' (case-insensitive)
    # 2. Rate Type is 'hourly' (case-insensitive)
    df_cleaned = df[~industry_filter & ~rate_type_filter]
    
    # Display cleaning results
    rows_removed = len(df) - len(df_cleaned)
    print(f"\nTotal rows removed: {rows_removed}")
    print(f"Total rows after cleaning: {len(df_cleaned)}")
    print(f"Cleaned data shape: {df_cleaned.shape}")
    print(f"Percentage retained: {(len(df_cleaned)/len(df)*100):.2f}%")
    
    # Save to TSV file
    output_file = csv_file.replace('.csv', '_cleaned.tsv')
    df_cleaned.to_csv(output_file, sep='\t', index=False)
    
    print(f"\n✓ Cleaned data saved to: {output_file}")
    
    return df_cleaned


# Main execution
if __name__ == "__main__":
    # Your CSV file
    csv_file = "Manual Data Clean _ Pivot Table_ Gantt Chart - freelancerresult.csv"
    
    try:
        cleaned_df = clean_job_data(csv_file)
        
        if cleaned_df is not None:
            print("\n" + "="*50)
            print("CLEANING COMPLETE!")
            print("="*50)
            print(f"\nCleaned data preview (first 5 rows):")
            print(cleaned_df.head())
        
    except FileNotFoundError:
        print(f"\nError: File '{csv_file}' not found.")
        print("Please make sure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()