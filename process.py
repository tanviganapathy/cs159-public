import pickle
import pandas as pd

# Open the file in read-binary mode
with open('info.pkl', 'rb') as f:
    missclassified_idxs_1, wrong_answers_1, none_idxs_1 = pickle.load(f)

with open('info_v2.pkl', 'rb') as f:
    missclassified_idxs_2, wrong_answers_2, none_idxs_2 = pickle.load(f)

with open('info_o4_mini.pkl', 'rb') as f:
    missclassified_idxs_3, wrong_answers_3, none_idxs_3 = pickle.load(f)

with open('info_v3.pkl', 'rb') as f:
    missclassified_idxs_4, wrong_answers_4, none_idxs_4 = pickle.load(f)

print(f"D1: {len(missclassified_idxs_1)}")
# print(missclassified_idxs_1)
# print(wrong_answers_1)
df_1 = pd.DataFrame({
    "Index": missclassified_idxs_1,
    "Wrong Answer Turbo 1": wrong_answers_1
})

# print(df_1)
# exit()

print(f"D2: {len(missclassified_idxs_2)}")
df_2 = pd.DataFrame({
    "Index": missclassified_idxs_2,
    "Wrong Answer Turbo 2": wrong_answers_2
})
print(f"D3: {len(missclassified_idxs_3)}")
df_3 = pd.DataFrame({
    "Index": missclassified_idxs_3,
    "Wrong Answer o4-mini": wrong_answers_3
})

print(f"D4: {len(missclassified_idxs_4)}")
df_4 = pd.DataFrame({
    "Index": missclassified_idxs_4,
    "Wrong Answer Turbo 3": wrong_answers_4
})

combined_wrong = set(missclassified_idxs_1 + missclassified_idxs_2 + missclassified_idxs_3 + missclassified_idxs_4)
print(f"C: {len(combined_wrong)}")
merged_df = df_1.merge(df_2, on='Index', how='outer').merge(df_3, on='Index', how='outer').merge(df_4, on='Index', how='outer')
print(merged_df)
print("at least 1 got wrong (same as combined):", len(merged_df))
print()

# Convert lists to sets
set1 = set(missclassified_idxs_1)
set2 = set(missclassified_idxs_2)
set3 = set(missclassified_idxs_3)
set4 = set(missclassified_idxs_4)

all_overlapping_elements = set1 & set2 & set3 & set4
print(f"LENGTH OF ALL OVERLAPPING: {len(all_overlapping_elements)}")
print(all_overlapping_elements)
all_wrong = merged_df.dropna(subset=['Wrong Answer Turbo 1', 'Wrong Answer Turbo 2', 'Wrong Answer o4-mini', 'Wrong Answer Turbo 3'])
print(all_wrong)
print("all 4 got wrong:", len(all_wrong))
all_wrong.to_pickle("all_wrong.pkl")

exit()

# Elements unique to each list
unique_to_list1 = set1 - set2
unique_to_list2 = set2 - set1

# Elements common to both lists
overlapping_elements = set1 & set2

# Printing results
print(f"D1: {len(data_list1)}")
print(f"D2: {len(data_list2)}")
print(f"Unique to D1: {len(unique_to_list1)}")
print(f"Unique to D2: {len(unique_to_list2)}")
print(f"Overlap: {len(overlapping_elements)}")
print(f"Total Unique Combined (correct): {len(set1 | set2)}")
print(overlapping_elements)

# data_list1.extend(data_list2)
# print(len(set(data_list1)))