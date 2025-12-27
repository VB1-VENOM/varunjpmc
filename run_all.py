
import sys

def main():
    #Run both classification and segmentation models.
    print("="*80)
    print("CENSUS BUREAU DATA SCIENCE PROJECT")
    print("="*80)
    print("\nThis script will run:")
    print("1. Classification Model (Income Prediction)")
    print("2. Segmentation Model (Customer Segmentation)")
    print("\n" + "="*80 + "\n")
    
    # Run classification model
    print("\n" + "="*80)
    print("PART 1: CLASSIFICATION MODEL")
    print("="*80)
    try:
        import classification_model
        classification_model.main()
    except Exception as e:
        print(f"Error running classification model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run segmentation model
    print("\n\n" + "="*80)
    print("PART 2: SEGMENTATION MODEL")
    print("="*80)
    try:
        import segmentation_model
        segmentation_model.main()
    except Exception as e:
        print(f"Error running segmentation model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n\n" + "="*80)
    print("ALL MODELS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  - classification_model.pkl: Trained classification model")
    print("  - classification_results.csv: Classification performance metrics")
    print("  - feature_importance.csv: Feature importance rankings")
    print("  - segmentation_model.pkl: Trained segmentation model")
    print("  - cluster_analysis.csv: Cluster characteristics")
    print("  - marketing_insights.txt: Marketing recommendations")
    print("  - data_with_clusters.csv: Data with cluster labels")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

