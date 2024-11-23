import csv
import random
from tqdm import tqdm
from collections import Counter, defaultdict

label_file = "index_label_to_hierarchical.csv"
id_file = "index_image_to_landmark_008.csv"
output_file = "result.csv"
filtered_output_file = "filtered_result.csv"
final_output_file = "final_result.csv"
MAX_PER_CLASS = 500

# Step 1: Load `label` as a dictionary for faster lookup
label_dict = {}
with open(label_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:  # Ensure the row has enough columns
            # Replace "castle / fort" with "castle" in the dictionary values
            label_dict[row[0]] = row[1].replace("castle / fort", "castle")  # Mapping

# Step 2: Open `id` and append matching label
with open(id_file, "r") as f, open(output_file, "w", newline="") as out_f:
    reader = csv.reader(f)
    writer = csv.writer(out_f)

    # Progress bar setup
    id_rows = list(reader)
    with tqdm(total=len(id_rows)) as pbar:
        for row in id_rows:
            if (
                len(row) > 1 and row[1] in label_dict
            ):  # Ensure sufficient columns and match exists
                row.append(label_dict[row[1]])  # Append the matching label
            writer.writerow(row)  # Write the updated row to output file
            pbar.update(1)

# Step 3: Filter rows based on multiple conditions
filter_conditions = [
    lambda label: label != "",
    lambda label: label != "stone",
    lambda label: label != "stairs",
    lambda label: label != "tree",
    lambda label: label != "artwork",
    lambda label: label != "agricultural land",
    lambda label: label != "air transportation",
    lambda label: label != "aqueduct",
    lambda label: label != "bath",
    lambda label: label != "boat",
    lambda label: label != "canal",
    lambda label: label != "cave",
    lambda label: label != "city",
    lambda label: label != "cross",
    lambda label: label != "dam",
    lambda label: label != "gate",
    lambda label: label != "government building",
    lambda label: label != "hindu temple",
    lambda label: label != "hospital",
    lambda label: label != "house",
    lambda label: label != "island",
    lambda label: label != "libaray",
    lambda label: label != "memorial",
    lambda label: label != "monastery",
    lambda label: label != "mosque",
    lambda label: label != "observatory",
    lambda label: label != "power plant",
    lambda label: label != "pyramid",
    lambda label: label != "restaurant",
    lambda label: label != "swimming pool",
    lambda label: label != "synagogue",
    lambda label: label != "theatre",
    lambda label: label != "underground infrastructure",
    lambda label: label != "village",
    lambda label: label != "wetland",
    lambda label: label != "volcano",
    lambda label: label != "school",
    lambda label: label != "factory",
    lambda label: label != "prison",
    lambda label: label != "mine",
    lambda label: label != "road",
    lambda label: label != "cliff",
    lambda label: label != "shopping",
    lambda label: label != "ocean area",
    lambda label: label != "festival",
    lambda label: label != "trail",
    lambda label: label != "ruins",
    lambda label: label != "windmill",
    lambda label: label != "cable transportation",
    lambda label: label != "fountain",
]

with open(output_file, "r") as f, open(filtered_output_file, "w", newline="") as out_f:
    reader = csv.reader(f)
    writer = csv.writer(out_f)

    rows = list(reader)
    if rows:
        writer.writerow(rows[0])  # Write header row
        for row in rows[1:]:
            if len(row) > 2:
                label = row[-1]
                # Check if the label passes all filter conditions
                if all(condition(label) for condition in filter_conditions):
                    # Replace "castle / fort" with "castle" in rows if needed
                    if label == "castle / fort":
                        row[-1] = "castle"
                    writer.writerow(row)

# Step 4: Read the filtered data and group rows by hierarchical_label
class_rows = defaultdict(list)

with open(filtered_output_file, "r") as f:
    reader = csv.reader(f)
    rows = list(reader)

    if rows:
        header = rows[0]  # Extract header
        label_index = header.index("hierarchical_label")  # Find the column index

        for row in rows[1:]:  # Skip header
            if len(row) > label_index:
                label = row[label_index]
                class_rows[label].append(row)

# Step 5: Reduce the rows for each class to a maximum of MAX_PER_CLASS
final_rows = []

for label, rows in class_rows.items():
    if len(rows) > MAX_PER_CLASS:
        # Randomly sample MAX_PER_CLASS rows
        sampled_rows = random.sample(rows, MAX_PER_CLASS)
        final_rows.extend(sampled_rows)
    else:
        final_rows.extend(rows)

# Step 6: Write the final filtered rows to a new output file
with open(final_output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)  # Write header
    writer.writerows(final_rows)  # Write rows

# Step 7: Print the final distribution of classes
final_label_counts = Counter(row[label_index] for row in final_rows)

print("Final Hierarchical Label Distribution:")
for label, count in final_label_counts.most_common():
    print(f"{label}: {count}")


