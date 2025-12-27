import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from load_data import load_census_data, preprocess_for_clustering

# Find optimal number of clusters using elbow method and silhouette score.
def find_optimal_clusters(X, max_k=10, random_state=42):

    print("Finding optimal number of clusters...")
    
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))
        
        print(f"  k={k}: Silhouette={silhouette_scores[-1]:.4f}, "
              f"Davies-Bouldin={davies_bouldin_scores[-1]:.4f}")
    
    # Find optimal k (highest silhouette score)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    scores = {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'optimal_k': optimal_k
    }
    
    print(f"\nOptimal number of clusters: {optimal_k}")
    print(f"  (based on highest silhouette score: {max(silhouette_scores):.4f})")
    
    return optimal_k, scores

# Perform K-means clustering.
def perform_clustering(X, n_clusters=5, random_state=42):

    print(f"\nPerforming K-means clustering with {n_clusters} clusters...")
    
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    
    labels = model.fit_predict(X)
    
    # Calculate metrics
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
    
    return model, labels

#Analyze cluster characteristics.
def analyze_clusters(df_original, labels, feature_names, n_clusters):
    print("\nAnalyzing cluster characteristics...")
    
    # Add cluster labels to dataframe
    df_analysis = df_original.copy()
    df_analysis['cluster'] = labels
    
    # Analyze key features for each cluster
    analysis_results = []
    
    # Key features to analyze
    key_features = ['age', 'education', 'marital stat', 'sex', 'race', 
                    'capital gains', 'capital losses', 'weeks worked in year',
                    'wage per hour', 'label']
    
    for cluster_id in range(n_clusters):
        cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
        
        cluster_info = {'cluster': cluster_id, 'size': len(cluster_data)}
        
        for feature in key_features:
            if feature in cluster_data.columns:
                if feature in ['age', 'capital gains', 'capital losses', 
                              'weeks worked in year', 'wage per hour']:
                    # Numerical features
                    cluster_info[f'{feature}_mean'] = cluster_data[feature].mean()
                    cluster_info[f'{feature}_median'] = cluster_data[feature].median()
                else:
                    # Categorical features - get most common value
                    most_common = cluster_data[feature].mode()
                    if len(most_common) > 0:
                        cluster_info[f'{feature}_mode'] = most_common.iloc[0]
                    else:
                        cluster_info[f'{feature}_mode'] = 'N/A'
        
        # Income distribution
        if 'label' in cluster_data.columns:
            income_dist = cluster_data['label'].value_counts(normalize=True)
            cluster_info['pct_high_income'] = income_dist.get('50000+', 0)
            cluster_info['pct_low_income'] = income_dist.get('- 50000', 0)
        
        analysis_results.append(cluster_info)
    
    cluster_analysis = pd.DataFrame(analysis_results)
    
    return cluster_analysis, df_analysis



#Create marketing insights based on cluster analysis.
def create_marketing_insights(cluster_analysis, save_path='marketing_insights.txt'):
    print("\nGenerating marketing insights...")
    
    insights = []
    insights.append("="*80)
    insights.append("MARKETING SEGMENTATION INSIGHTS")
    insights.append("="*80)
    insights.append("")
    
    for idx, row in cluster_analysis.iterrows():
        cluster_id = int(row['cluster'])
        size = int(row['size'])
        pct_high_income = row.get('pct_high_income', 0)
        
        insights.append(f"{'='*80}")
        insights.append(f"SEGMENT {cluster_id} (Size: {size:,} customers, {size/cluster_analysis['size'].sum()*100:.1f}% of population)")
        insights.append(f"{'='*80}")
        
        # Key characteristics
        if 'age_mean' in row:
            insights.append(f"Average Age: {row['age_mean']:.1f} years")
        if 'education_mode' in row and pd.notna(row['education_mode']):
            insights.append(f"Most Common Education: {row['education_mode']}")
        if 'marital stat_mode' in row and pd.notna(row['marital stat_mode']):
            insights.append(f"Most Common Marital Status: {row['marital stat_mode']}")
        if 'sex_mode' in row and pd.notna(row['sex_mode']):
            insights.append(f"Most Common Gender: {row['sex_mode']}")
        if 'race_mode' in row and pd.notna(row['race_mode']):
            insights.append(f"Most Common Race: {row['race_mode']}")
        if 'wage per hour_mean' in row:
            insights.append(f"Average Wage per Hour: ${row['wage per hour_mean']:.2f}")
        if 'weeks worked in year_mean' in row:
            insights.append(f"Average Weeks Worked: {row['weeks worked in year_mean']:.1f}")
        
        insights.append(f"High Income (>$50k) Percentage: {pct_high_income*100:.1f}%")
        
        # Marketing recommendations
        insights.append("")
        insights.append("MARKETING RECOMMENDATIONS:")
        if pct_high_income > 0.3:
            insights.append("  - Target for premium products/services")
            insights.append("  - Focus on high-value offerings")
            insights.append("  - Emphasize quality and exclusivity")
        elif pct_high_income > 0.15:
            insights.append("  - Target for mid-range products/services")
            insights.append("  - Balance quality and affordability")
            insights.append("  - Consider value-based messaging")
        else:
            insights.append("  - Target for budget-friendly products/services")
            insights.append("  - Emphasize affordability and value")
            insights.append("  - Consider promotional offers and discounts")
        
        insights.append("")
    
    insights_text = "\n".join(insights)
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write(insights_text)
    
    print(f"Marketing insights saved to: {save_path}")
    print("\n" + insights_text)


def main():
    """Main function to run segmentation model."""
    print("="*60)
    print("CENSUS BUREAU CUSTOMER SEGMENTATION MODEL")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = load_census_data()
    print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Sample data for clustering (for computational efficiency)
    # Use 20,000 samples for faster processing
    sample_size = min(20000, len(df))
    print(f"\nUsing {sample_size:,} samples for clustering...")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_scaled, feature_names, label_encoders, scaler, labels, df_sampled = \
        preprocess_for_clustering(df, n_samples=sample_size)
    
    print(f"Feature matrix shape: {X_scaled.shape}")
    
    # Find optimal number of clusters
    optimal_k, cluster_scores = find_optimal_clusters(X_scaled, max_k=8)
    
    # Perform clustering
    model, cluster_labels = perform_clustering(X_scaled, n_clusters=optimal_k)
    
    # Analyze clusters
    cluster_analysis, df_with_clusters = analyze_clusters(
        df_sampled, cluster_labels, feature_names, optimal_k
    )
    
    # Print cluster analysis
    print("\n" + "="*60)
    print("CLUSTER ANALYSIS SUMMARY")
    print("="*60)
    print(cluster_analysis.to_string(index=False))
    
    # Save cluster analysis
    cluster_analysis.to_csv('cluster_analysis.csv', index=False)
    print("\nCluster analysis saved to: cluster_analysis.csv")
    
    
    # Create marketing insights
    create_marketing_insights(cluster_analysis)
    
    # Save model and preprocessing objects
    print("\n" + "="*60)
    print("Saving segmentation model...")
    print("="*60)
    joblib.dump(model, 'segmentation_model.pkl')
    joblib.dump(label_encoders, 'segmentation_label_encoders.pkl')
    joblib.dump(scaler, 'segmentation_scaler.pkl')
    joblib.dump(feature_names, 'segmentation_feature_names.pkl')
    print("Segmentation model saved to: segmentation_model.pkl")
    
    # Save cluster labels
    df_with_clusters.to_csv('data_with_clusters.csv', index=False)
    print("Data with cluster labels saved to: data_with_clusters.csv")
    
    print("\n" + "="*60)
    print("SEGMENTATION MODEL COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

