import os
from dotenv import load_dotenv
from src.data.etl.extract import extract_data, save_state
from src.data.etl.transform import transform_data
from src.data.etl.load import load_data


load_dotenv()


def run_pipeline(batch_size=1000):
    api_url = os.getenv("API")
    if not api_url:
        print("Error: API URL not found in .env")
        return False
    
    try:
        # extract
        print("Fetching data from API....")
        raw_records, offset = extract_data(api_url, batch_size)

        if not raw_records:
            print("No new data to process")
            return False
        
        # transform
        print("Transforming data")
        df_transformed = transform_data(raw_records)

        if df_transformed.empty:
            print("No records after transformation. Pipeline finished.")
            return False
        clean_records = df_transformed.to_dict(orient='records')

        # load
        print("Loading into database...")
        # success = load_data(raw_records)
        success = load_data(clean_records)        
        
        if success:
            new_offset = offset + len(raw_records)
            print(f"Batch Complete! New Offset Saved: {new_offset}")
            return True
        else:
            print("failed during load phase")
            return False
            
    except Exception as e:
        print(f"Batch failed with error: {e}")
        return False
    
def run_full_ingestion(total_records: int = 100000, batch_size: int = 1000):
    num_batches = total_records // batch_size
    

    print(f"Hospital Readmission ETL Pipeline - FULL INGESTION")
    print(f"Total records: {total_records} | Batch size: {batch_size}")
    print(f"Number of batches: {num_batches}")

    
    successful_batches = 0
    failed_batches = 0
    
    for i in range(num_batches):
        print(f"Batch {i+1}/{num_batches}")

        
        success = run_pipeline(batch_size)
        
        if success:
            successful_batches += 1
        else:
            failed_batches += 1
            print(f"Batch {i+1} failed. Continuing...")
    
    print("FULL INGESTION SUMMARY")

    print(f"Successful batches: {successful_batches}/{num_batches}")
    print(f"Failed batches: {failed_batches}/{num_batches}")
    print(f"Total records ingested: ~{successful_batches * batch_size}")



if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--full":
            total = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
            batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
            run_full_ingestion(total_records=total, batch_size=batch)
        else:
            batch_size = int(sys.argv[1])
            run_pipeline(batch_size=batch_size)
    else:
        run_pipeline(batch_size=1000)