from glob import glob
import pandas as pd
import sys

# Where are the grade files?

# Find input and template files.

template_file = glob(sys.argv[1])[0]
input_file = glob(sys.argv[2])[0]
output_file = sys.argv[3]

#template_file = glob(base_path + "/CS_5740_*.csv")[0]
#input_file    = glob(base_path + "/Class_*.xlsx")[0]
#output_file   = base_path + "/grades.csv"

# Read in CMS grading template, remove blank columns.

template = pd.read_csv(template_file).set_index("NetID")
del template["Total"]
del template["Add Comments"]

# Read in grade file.

grades = pd.read_excel(input_file, skiprows = range(6))

# Eliminate unused rows and columns (e.g., class total at the bottom):

grades = grades[["Student ID", "Total Score (0 - 100)"]]
grades = grades.loc[range(grades.shape[0] - 1)]

# Adjustments to fit CMS template:

grades.columns = ["NetID", "Total"]
grades["NetID"] = [x.lower() for x in grades["NetID"]]

grades["Add Comments"] = ""
grades = grades.set_index("NetID")

# Join grades with template to ensure everyone enrolled receives a grade.

output = template.join(grades).fillna("")
output.to_csv(output_file)

# Sanity check.

extra = set(grades.index) - set(output.index)
missing = set(output.index) - set(grades.index)

print("Sanity check:")
print("Took quiz, but NetID not in gradebook:")
print("\n".join(extra))
print("NetID in gradebook, but didn't take quiz:")
print("\n".join(missing))
