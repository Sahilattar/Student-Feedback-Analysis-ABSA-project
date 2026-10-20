# Dataset Analysis and Reshaping Report

## 1. Overview and Detection
- **Original File**: `student-data.xlsx`
- **Original Dimensions**: 222 rows × 13 columns
- **Format**: Wide format (each row represents a single student survey response containing 6 feedback aspects)
- **Aspects Evaluated (6 total)**:
  1. `teaching` & `teachingl`
  2. `coursecontent` & `coursecontentl`
  3. `examination` & `examinationl`
  4. `labwork` & `labworkl`
  5. `library_facilities` & `library_facilitiesl` (detected leading space in column name `' library_facilities'`)
  6. `extracurricular` & `extracurricularl`

## 2. Raw Data Inspection
- **Missing Values in Raw Data**:
| Column | Missing Values |
|---|---|
| `teachingl` | 0 |
| `teaching` | 0 |
| `coursecontentl` | 1 |
| `coursecontent` | 0 |
| `examinationl` | 3 |
| `Examination` | 2 |
| `labworkl` | 1 |
| `labwork` | 1 |
| `library_facilitiesl` | 3 |
| ` library_facilities` | 0 |
| `extracurricularl` | 0 |
| `extracurricular` | 0 |
| `student_id` | 0 |


- **Duplicate Rows in Wide Data**: 10 identical survey responses detected.
- **Label Encoding**:
  - `-1`: Negative sentiment
  - `0`: Neutral sentiment
  - `1`: Positive sentiment

## 3. Data Cleaning and Reshaping Logic
1. **Cleaning Rules**:
   - Column names matched case-insensitively with leading/trailing whitespace trimmed.
   - Text entries stripped of whitespace; non-string or empty representations (`nan`, `None`, whitespace) treated as missing.
   - Label values converted to numeric, keeping only valid labels in `{-1, 0, 1}`.
   - Rows with either missing text or missing label were dropped from the training set (LONG format).
2. **Reshaped Long Format (`data/processed/feedback_long.csv`)**:
   - Total Valid Feedback Instances: **1324** rows
   - Columns: `student_id`, `aspect`, `text`, `label`, `sentiment`
3. **Reshaped Wide Format (`data/processed/feedback_wide.csv`)**:
   - Total Students: **222** rows
   - Added `mean_aspect_score`: average of numeric labels for available aspects per student.
   - Added `overall_sentiment`:
     - `mean_aspect_score <= -0.34` → `negative`
     - `mean_aspect_score >= 0.34` → `positive`
     - `-0.34 < mean_aspect_score < 0.34` → `neutral`

## 4. Class Distribution Overall (Long Format)
Total Records: 1324
- **Positive**: 803 (60.65%)
- **Neutral**: 235 (17.75%)
- **Negative**: 286 (21.60%)

## 5. Class Distribution Per Aspect
| Aspect | Total Responses | Positive (%) | Neutral (%) | Negative (%) |
|---|---|---|---|---|
| teaching | 222 | 133 (59.9%) | 42 (18.9%) | 47 (21.2%) |
| coursecontent | 221 | 132 (59.7%) | 41 (18.6%) | 48 (21.7%) |
| examination | 219 | 129 (58.9%) | 45 (20.5%) | 45 (20.5%) |
| labwork | 221 | 130 (58.8%) | 32 (14.5%) | 59 (26.7%) |
| library_facilities | 219 | 125 (57.1%) | 43 (19.6%) | 51 (23.3%) |
| extracurricular | 222 | 154 (69.4%) | 32 (14.4%) | 36 (16.2%) |

### Imbalance and Low Frequency Class Check
- **Threshold**: Warning if any sentiment class is under 5.0% of the aspect or overall data.
- **Detected Warnings**:
  - None. All classes in every aspect represent >= 5.0% of the sample.

## 6. Student-Level Overall Sentiment (Wide Format)
- **Positive Students**: 128 (57.66%)
- **Neutral Students**: 65 (29.28%)
- **Negative Students**: 29 (13.06%)

## 7. Assumptions & Project Constraints
1. **Absence of Metadata**: The dataset contains no timestamps, semester info, course codes, or instructor identifiers. Thus, analysis focuses purely on aspect-based feedback semantics and student-level sentiment correlation.
2. **Student Disjoint Splits**: Because each wide row represents a distinct student survey response, model validation splits must group by `student_id` to prevent data leakage between training and testing sets.
3. **Aspect Conditioning**: For unified cross-aspect sentiment classification models, the aspect name is prepended to the feedback string (`<aspect>: <text>`) to provide explicit context.
4. **Class Weighting**: Because positive feedback predominates (~60%), cost-sensitive learning (balanced class weights) is required during model training.
