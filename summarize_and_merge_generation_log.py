import csv
import os
from collections import defaultdict

# Initialize dictionaries for tracking and calculations
model_count_total = defaultdict(int)
model_count = defaultdict(int)
model_details = defaultdict(lambda: defaultdict(float))
model_item_count = defaultdict(int)

directory = os.path.dirname(os.path.realpath(__file__))
log_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.startswith('generation_log') and f.endswith('.csv')]
log_files.sort()

# Initialize the num_merged right after identifying log_files to ensure it's always defined
num_merged = len(log_files)

# Pre-process to count all huggingface_models total before removing any row
total_models = 0
skipped_rows = 0

for log_file in log_files:
    log_file_path = os.path.join(directory, log_file)
    with open(log_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        for row in csvreader:
            if len(row) > 14:  # Ensure the row has at least 15 columns
                model_count_total[row[14]] += 1
                total_models += 1
            else:
                # Simply increment the skipped_rows variable without writing to a log file
                skipped_rows += 1
                

temp_csv_path = os.path.join(directory, 'temp_generation_log.csv')
merged_csv_path = os.path.join(directory, 'merged_generation_log.csv')

rows_kept = 0
total_rows = 0
header_written = False

if len(log_files) > 1:
    num_merged = len(log_files)  # Define num_merged based on the number of log files found

    if len(log_files) > 1:
        print(f"Merging {num_merged} generation_log CSV files...")

    with open(temp_csv_path, mode='w', newline='', encoding='utf-8') as temp_csvfile:
        temp_writer = csv.writer(temp_csvfile)
        
        for log_file in log_files:
            log_file_path = os.path.join(directory, log_file)
        
            with open(log_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
            
                for i, row in enumerate(csvreader):
                    if i == 0 and not header_written:
                        temp_writer.writerow(row)
                        header_written = True
                    elif i > 0:
                        temp_writer.writerow(row)
            
            os.remove(log_file_path)

    # If merged_csv_path exists, delete it before renaming
    if os.path.exists(merged_csv_path):
        os.remove(merged_csv_path)
    os.rename(temp_csv_path, merged_csv_path)
    csv_file_path = merged_csv_path
else:
    csv_file_path = os.path.join(directory, log_files[0])

# Ensure to process only if there's at least one log file
if log_files:
    file_names = set([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile, open(temp_csv_path, mode='w', newline='', encoding='utf-8') as temp_csvfile:
        csvreader = csv.reader(csvfile)
        csvwriter = csv.writer(temp_csvfile)
        header = next(csvreader)
        csvwriter.writerow(header)
        
        for row in csvreader:
            total_rows += 1 
            new_file_name = row[1]
            
            if new_file_name in file_names:
                current_model = row[14]
                # Update counts and sums
                model_count[current_model] += 1
                model_item_count[current_model] += 1
                model_details[current_model]['identitynet_strength_sum'] += float(row[2])
                model_details[current_model]['adapter_strength_sum'] += float(row[3])
                model_details[current_model]['num_inference_steps_sum'] += int(row[4])
                model_details[current_model]['guidance_scale_sum'] += float(row[5])
                rows_kept += 1
                csvwriter.writerow(row)

    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)
    os.rename(temp_csv_path, csv_file_path)
    
    # Calculate combined averages across all models
    combined_identitynet_strength_avg = sum([details['identitynet_strength_sum'] for details in model_details.values()]) / rows_kept
    combined_adapter_strength_avg = sum([details['adapter_strength_sum'] for details in model_details.values()]) / rows_kept
    combined_num_inference_steps_avg = sum([details['num_inference_steps_sum'] for details in model_details.values()]) / rows_kept
    combined_guidance_scale_avg = sum([details['guidance_scale_sum'] for details in model_details.values()]) / rows_kept

    # Calculate the number of removed rows
    rows_removed = total_rows - rows_kept
    # Calculate the percentage of rows kept
    percentage_kept = (rows_kept / total_rows) * 100 if total_rows > 0 else 0

    # Calculate and print overall and per model averages
    sorted_models = sorted(model_count.items(), key=lambda item: item[1], reverse=True)

    # Now, use sorted_models for printing detailed stats for each model
    print("===INDIVIDUAL MODEL STATS ON KEPT FILES===")
    for model, _ in sorted_models:  # We ignore the usage count here as it's not directly used
        details = model_details[model]
        identitynet_strength_avg = details['identitynet_strength_sum'] / model_item_count[model]
        adapter_strength_avg = details['adapter_strength_sum'] / model_item_count[model]
        num_inference_steps_avg = details['num_inference_steps_sum'] / model_item_count[model]
        guidance_scale_avg = details['guidance_scale_sum'] / model_item_count[model]

        print(f"===MODEL: {model}===")
        print(f"Usage Count: {model_count[model]}")
        print(f"Average IdentityNet Strength: {identitynet_strength_avg:.5f}")
        print(f"Average Adapter Strength: {adapter_strength_avg:.5f}")
        print(f"Average Number of Inference Steps: {num_inference_steps_avg:.1f}")
        print(f"Average Guidance Scale: {guidance_scale_avg:.5f}\n")

    print("===COMBINED STATS===")
    print(f"Average IdentityNet Strength: {combined_identitynet_strength_avg:.5f}")
    print(f"Average Adapter Strength: {combined_adapter_strength_avg:.5f}")
    print(f"Average Number of Inference Steps: {combined_num_inference_steps_avg:.1f}")
    print(f"Average Guidance Scale: {combined_guidance_scale_avg:.5f}\n")

    # Printing summary and model choice percentages
    print("===CSV CLEANUP STATS===")
    print(f"Cleanup completed on {'merged' if num_merged > 1 else 'single'} generation_log CSV file.")
    print(f"Images kept: {rows_kept}")
    print(f"Images removed: {rows_removed}")
    print(f"Percentage kept: {percentage_kept:.2f}%\n")

    print("===MODEL CHOICE PERCENTAGES OF KEPT FILES===")
    if rows_kept > 0:
        for model, _ in sorted_models:  # Reusing sorted_models for order consistency
            percentage_of_total = (model_count[model] / rows_kept) * 100 if rows_kept > 0 else 0
            # Now, also show the count along with the percentage
            print(f"Model {model}: {model_count[model]} out of {rows_kept} ({percentage_of_total:.2f}%)")
    else:
        print("No images were kept; unable to calculate model choice percentages.\n")

# Displaying the counts for each model before proceeding with merging or filtering
print("\n===MODEL CHOICE DISTRIBUTION COUNTS ACROSS ALL FILES===")
print(f"Skipped rows due to insufficient columns: {skipped_rows}")
for model, count in sorted(model_count_total.items(), key=lambda item: item[1], reverse=True):
    percentage_of_total = (count / total_models) * 100 if total_models > 0 else 0
    print(f"Model {model}: {count} of {total_models} models, ({percentage_of_total:.2f}% of time)")

print("\n===SUGGESTED FINDINGS===")
# Summary of what PEI represents
print("\n=== PEI INDEX SUMMARY ===")

print("The Preference Efficiency Index (PEI) quantifies how effectively each model's outputs met your criteria for being kept relative to how often you used each model. A higher PEI indicates a model not only was used but its outputs were deemed satisfactory and kept at a higher rate.")

print("\n===PEI RESULTS===")
# Calculate PEI for each model and store in a list for sorting
pei_list = []
for model, usage in model_count_total.items():
    kept_times = model_count.get(model, 0)
    pei = (kept_times / usage) * 100
    pei_list.append((model, pei, kept_times, usage))  

# Sort the list by PEI, highest first
pei_list.sort(key=lambda x: x[1], reverse=True)

# Print the sorted list with PEI values and calculation shown
for model, pei, kept_times, usage in pei_list:
    calculation = f"({kept_times} / {usage}) * 100"  # How PEI was calculated
    print(f"Model {model}: PEI = {pei:.7f} = {calculation}")
print("\n=== === === === ===")

print("\n=== MEI INDEX SUMMARY ===")
print("The Model Efficiency Index (MEI) score signifies a model's ability to deliver outputs that align with your criteria both effectively and efficiently. In essence, models with higher MEI scores offer more value ('more bang for your buck') by consistently producing desired outcomes with minimized computational effort.")

# Initialize an empty list to store model MEI
mei_list = []
scaling_factor = 1000  # Defines the scale for MEI values

for model, usage in model_count_total.items():
    kept_times = model_count.get(model, 0)
    if kept_times > 0 and model in model_item_count:
        avg_inference_steps = model_details[model]['num_inference_steps_sum'] / model_item_count[model]
        percentage_kept = (kept_times / usage) * 100
        # Apply scaling factor to MEI calculation
        mei = (percentage_kept / avg_inference_steps) * scaling_factor
        mei_list.append((model, mei, percentage_kept, avg_inference_steps))
    else:
        # For models with no kept images or missing data, we set MEI to 0
        mei_list.append((model, 0, 0, 0))

# Sort the list by MEI, highest first
mei_list.sort(key=lambda x: x[1], reverse=True)

# Print the sorted list with MEI values and contextual information
print("\n===MEI RESULTS===")
for model, mei, percentage_kept, avg_inference_steps in mei_list:
    print(f"Model {model}: MEI = {mei:.2f} (Kept: {percentage_kept:.2f}%, Avg Steps: {avg_inference_steps:.1f})")

else:
    # As a final print out after your existing code structure
    print("\n===LOG FILES INGESTED===")
    if log_files:
        print(f"Number of generation_log CSV files merged: {num_merged}")
    else:
        print("No generation_log CSV files found to process.")