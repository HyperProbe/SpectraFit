import pandas as pd
import os


def process_biopsy_excel(excel_path: str, output_csv_path: str) -> None:
    """
    Parses the biopsy Excel file, processes it, and saves it as a CSV file.

    Parameters
    ----------
    excel_path : str
        Path to the input Excel file (biopsy2.0.xlsx).
    output_csv_path : str
        Path to save the processed CSV file.

    Returns
    -------
    None
    """
    try:
        # Read the first sheet of the Excel file
        df = pd.read_excel(excel_path, sheet_name=0)

        # Define the columns to keep and their new names
        columns_map = {
            "HP Sample Name": "id",
            "age": "age",
            "sex": "sex",
            "Tipo ": "type_of_tumor",  # Added trailing space
            "Grading sec WHO 2021": "grading",
            "Additional info": "additional_info",
            "HISTHOLOGY ": "histology",  # Added trailing space
            "ki 67 indice proliferativo ": "Ki-67-index",  # Added trailing space
        }

        # Select and rename columns
        # First, filter out columns that might not exist in the DataFrame to avoid KeyErrors
        existing_columns_to_select = {
            k: v for k, v in columns_map.items() if k in df.columns
        }
        print(
            f"Processing columns: {existing_columns_to_select.keys()} from the Excel sheet."
        )
        missing_original_columns = set(columns_map.keys()) - set(df.columns)

        if missing_original_columns:
            print(
                f"Warning: The following original columns were not found in the Excel sheet and will be skipped: {missing_original_columns}"
            )

        processed_df = df[list(existing_columns_to_select.keys())].copy()
        processed_df.rename(columns=existing_columns_to_select, inplace=True)

        # Ensure all target columns are present, fill with NaN if source was missing
        for original_col, new_col_name in columns_map.items():
            if new_col_name not in processed_df.columns:
                processed_df[new_col_name] = pd.NA  # Or use np.nan if preferred

        # Reorder columns to match the requested order, handling cases where some might be missing
        final_column_order = [
            "id",
            "age",
            "sex",
            "type_of_tumor",
            "grading",
            "additional_info",
            "histology",
            "Ki-67-index",
        ]
        processed_df = processed_df.reindex(columns=final_column_order)

        # Create the directory for the output CSV if it doesn't exist
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        # Save the processed DataFrame to a CSV file
        processed_df.to_csv(output_csv_path, index=False)
        print(f"Successfully processed '{excel_path}' and saved to '{output_csv_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{excel_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Define file paths
    # Assuming the script is in src/ and data is in ../data/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_excel_path = os.path.join(base_dir, "data", "raw", "biopsy2505.xlsx")
    processed_csv_path = os.path.join(
        base_dir, "data", "processed", "biopsy_metadata2505.csv"
    )

    process_biopsy_excel(raw_excel_path, processed_csv_path)
