from src.analysis.publisher_analyzer import run_publisher_analysis

if __name__ == "__main__":
    results = run_publisher_analysis()
    print("🏢 Publisher Analysis:")
    print(f"   Total Publishers: {results['publisher_activity']['total_publishers']}")
    print(f"   Top 10 Concentration: {results['publisher_activity']['concentration']}")
    print(f"   Email Domains: {len(results['email_domains'])}")