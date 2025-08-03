import pandas as pd

# Load the comprehensive data
df = pd.read_csv('powerbi/comprehensive_data.csv')

# Filter for STEM students only
stem_data = df[df['is_stem'] == True]

print("=== STEM PERFORMANCE KPIs ===")
print(f"Total STEM Students: {len(stem_data):,}")
print(f"STEM Excellence Rate: {stem_data['stem_excellence'].mean()*100:.1f}%")
print(f"STEM Success Rate: {stem_data['stem_success'].mean()*100:.1f}%")
print(f"Average Excellence Probability: {stem_data['excellence_probability'].mean()*100:.1f}%")
print(f"Average Success Probability: {stem_data['success_probability'].mean()*100:.1f}%")

print("\n=== RISK LEVEL DISTRIBUTION ===")
risk_dist = stem_data['risk_level'].value_counts(normalize=True)*100
for level, pct in risk_dist.items():
    print(f"{level}: {pct:.1f}%")

print("\n=== SUBJECT BREAKDOWN ===")
subject_dist = stem_data['subject_name'].value_counts()
for subject, count in subject_dist.items():
    print(f"{subject}: {count} students")

print("\n=== EDUCATION LEVEL IMPACT ===")
education_impact = stem_data.groupby('highest_education')['excellence_probability'].mean().sort_values(ascending=False)
for education, prob in education_impact.items():
    print(f"{education}: {prob*100:.1f}% excellence probability") 