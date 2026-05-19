import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_health_analytics_pipeline(data_source="conversation_analysis_results.csv"):
    df = pd.read_csv(data_source)
    
    # Set aesthetics
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # INSIGHT 1: Urgency vs. Barriers (The Equity Map)
    plt.subplot(1, 2, 1)
    barrier_counts = df[df['urgency_level'] == 'HIGH']['sdoh_indicators.economic_barrier'].value_counts()
    plt.pie(barrier_counts, labels=['Economic Barrier', 'No Barrier'], autopct='%1.1f%%', colors=['#e74c3c', '#3498db'])
    plt.title('High Urgency Cases: Economic Barriers')

    # INSIGHT 2: Intent Distribution by Clinical Category
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, y='clinical_category', hue='intent', palette='viridis')
    plt.title('Patient Intent by Clinical Category')
    
    plt.tight_layout()
    plt.show()

    # INSIGHT 3: Literacy Gaps (For your LinkedIn Post)
    literacy_pivot = df.groupby('clinical_category')['health_literacy_level'].value_counts(normalize=True).unstack()
    literacy_pivot.plot(kind='barh', stacked=True, figsize=(10, 6), color=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Health Literacy Distribution across Categories')
    plt.xlabel('Proportion of Conversations')
    plt.show()

# Example Call
# run_health_analytics_pipeline()