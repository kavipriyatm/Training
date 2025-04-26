import threading
import logging
import pandas as pd

logging.basicConfig(
    filename='processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_csv(file_path, column_name, result_dict, key):
    logging.info(f"Started processing {file_path}")
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            logging.error(f"Column '{column_name}' not found in {file_path}")
            return
        mean_value = df[column_name].mean()
        result_dict[key] = mean_value
        logging.info(f"Finished processing {file_path}, Mean: {mean_value}")
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")

file1 = r"D:\DS-DAY3\housing.csv"
file2 = r"D:\DS-DAY3\USA Housing Dataset.csv"

results = {}

thread1 = threading.Thread(target=process_csv, args=(file1, 'median_house_value', results, "mean1"))
thread2 = threading.Thread(target=process_csv, args=(file2, 'price', results, "mean2"))

thread1.start()
thread2.start()

thread1.join()
thread2.join()

print(f"Mean1 ({file1}): {results.get('mean1')}")
print(f"Mean2 ({file2}): {results.get('mean2')}")
print("✅ Processing complete! Check 'processing.log' for details.")
