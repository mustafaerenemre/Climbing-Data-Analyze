import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load the dataset
df = pd.read_csv('climber_df.csv')

# Load grade conversion mapping
grades_df = pd.read_csv('grades_conversion_table.csv')
grade_map = dict(zip(grades_df['grade_id'], grades_df['grade_fra']))

def format_grade_ticks(ax, axis='x'):
    """Replaces numeric grade ticks with French grades like '7a', '6b+'"""
    ticks = ax.get_yticks() if axis == 'y' else ax.get_xticks()
    valid_ticks = [t for t in ticks if int(t) in grade_map and t >= 0]
    labels = [grade_map[int(t)] for t in valid_ticks]
    if axis == 'y':
        ax.set_yticks(valid_ticks)
        ax.set_yticklabels(labels)
    else:
        ax.set_xticks(valid_ticks)
        ax.set_xticklabels(labels)

print("--- Dataset Formatting ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# 2. Duplicate data analysis and cleaning
duplicated_count = df.duplicated().sum()
print(f"\n--- Duplicate Data Check ---")
print(f"Duplicate row count: {duplicated_count}")

if duplicated_count > 0:
    df = df.drop_duplicates()
    print(f"Duplicate rows deleted. New shape: {df.shape}")

# 3. Null data analysis
print("\n--- Missing (Null) Data Check ---")
null_counts = df.isnull().sum()
null_cols = null_counts[null_counts > 0]

if len(null_cols) > 0:
    print("Columns with missing data and their counts:")
    print(null_cols)
    print("\nMissing data percentages (%):")
    print((null_cols / len(df)) * 100)
else:
    print("No missing data found in the dataset.")

# 4. Outlier Analysis
print("\n--- Basic Statistics for Numeric Data ---")
# Exclude categorical/identifier columns from statistical summary
cols_to_exclude_stats = ['user_id', 'sex', 'year_first', 'year_last']
stats_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in cols_to_exclude_stats]
stats_summary = df[stats_cols].describe().drop(index=['count']).round(2)
print(stats_summary)

print("\n--- Extreme Outliers identifying by IQR Method (Multiplier: 3.0) ---")
# Do not search for outliers in identifier and categorical columns like user_id and sex
cols_to_exclude = ['user_id', 'sex']
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in cols_to_exclude]

# Increase the multiplier from 1.5 to 3 to stretch the limits significantly
iqr_multiplier = 3.0 

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    if not outliers.empty:
        print(f"Column '{col}': {len(outliers)} extreme values detected. (Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f})")

print("\n--- Logical Boundary Control and Cleaning Based on Human Physiology ---")
# Filtering conditions based on human realities rather than statistics
suspicious_cond = (
    (df['weight'] < 35) | (df['weight'] > 120) |
    (df['height'] < 130) | (df['height'] > 220) |
    (df['age'] < 8) | (df['age'] > 85)
)

print(f"Suspicious Physical Data (Removed from dataset): {suspicious_cond.sum()} people")

# Remove outliers and update the data (so models don't learn garbage data)
df = df[~suspicious_cond].copy()
print(f"Dataset shape after cleaning: {df.shape}")

# 5. Calculating BMI (Body Mass Index) from Height and Weight
# Since faulty/outlier height and weight values are deleted, BMI will be calculated more accurately!
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

# --- VISUALIZATION PHASE ---

# Step 2: Correlation Matrix and Heatmap (Relationship with grades_max only)
print("\nCharts are being displayed. (New ones will open as you close the windows)")
# Sort features by correlation with grades_max to show most influential ones on top (Exclude comparison with itself)
corr_target = df[['age', 'height', 'weight', 'bmi', 'years_cl', 'grades_max']].corr()[['grades_max']].drop(index='grades_max').sort_values(by='grades_max', ascending=False)

plt.figure(figsize=(4, 6))
# Set vmin and vmax to -1 and 1 so the color scale fits standard correlation
sns.heatmap(corr_target, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation of Features with grades_max')
plt.show()

# Step 3: Experience vs Performance Relationship (Scatter Plot - Density Correction)
plt.figure(figsize=(10, 6))
# Prevent points from strictly overlapping with jitter
# Increased transparency by lowering alpha to 0.05, decreased scatter size (s).
sns.regplot(data=df, x='years_cl', y='grades_max', 
            x_jitter=0.4, y_jitter=0.4,
            scatter_kws={'alpha':0.09, 's':15}, 
            line_kws={'color':'red', 'linewidth':3})
plt.title('Relationship Between Climbing Experience (Years) and Maximum Grade')
plt.xlabel('Climbing Years (years_cl)')
plt.ylabel('Maximum Grade (grades_max)')
format_grade_ticks(plt.gca(), axis='y')
plt.show()

# Step 4: Grade Distribution by Gender
# Convert 0 and 1 to text for clear reading in the legend
df_plot = df.copy()
df_plot['Gender'] = df_plot['sex'].map({0: 'Male', 1: 'Female'})

# --- PLOT 4.1: Absolute Distribution ---
plt.figure(figsize=(10, 6))
# element="step" prevents bars from overlapping. palette gives visual contrast.
sns.histplot(data=df_plot, x='grades_max', hue='Gender', kde=True, bins=30, 
             palette=['#1f77b4', '#ff7f0e'], alpha=0.3, element="step", hue_order=['Male', 'Female'])
plt.title('1: Climbing Grade Distribution by Gender (By Population Size)')
plt.xlabel('Maximum Grade (grades_max)')
plt.ylabel('Number of People (Absolute Frequency)')
format_grade_ticks(plt.gca(), axis='x')
plt.show()

# --- PLOT 4.2: Normalized Distribution (Within Itself) ---
plt.figure(figsize=(10, 6))
# stat="percent" and common_norm=False sets Male total to 100% and Female total to 100%
sns.histplot(data=df_plot, x='grades_max', hue='Gender', kde=True, bins=30, 
             stat="percent", common_norm=False, 
             palette=['#1f77b4', '#ff7f0e'], alpha=0.3, element="step", hue_order=['Male', 'Female'])
plt.title('2: Proportional Climbing Grade Distribution by Gender (Normalized to Own Group)')
plt.xlabel('Maximum Grade (grades_max)')
plt.ylabel('Percentage (%) - Proportion within Own Gender')
format_grade_ticks(plt.gca(), axis='x')
plt.show()

# --- CLASSIFICATION MODELS ---
print("\n=== Classification Model Training Started ===")

# 1. Categorizing the Target Variable (Binning)
def categorize_grade(grade):
    if grade < 45:
        return 0  # Beginner/Intermediate
    elif grade <= 60:
        return 1  # Advanced
    else:
        return 2  # Elite

df['level'] = df['grades_max'].apply(categorize_grade)

# Clean missing values (For features to be used)
features = ['age', 'height', 'weight', 'bmi', 'years_cl', 'sex']
df_model = df.dropna(subset=features + ['level'])

# 2. Select Inputs (X) and Target Variable (y)
X = df_model[features]
y = df_model['level']

# 3. Split Data into Train and Test (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale Data (Important for KNN and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# 6. Train and Evaluate Models
for name, model in models.items():
    print(f"\n=== {name} Model Results ===")
    
    # Use scaled data for Logistic Reg and KNN, normal data for Random Forest
    if name == "Random Forest":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    
    # Print requested metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Visualize Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=['Beginner/Inter.', 'Advanced', 'Elite'],
                yticklabels=['Beginner/Inter.', 'Advanced', 'Elite'])
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()




