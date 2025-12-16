
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_and_preprocess_data
import os

def analyze_and_visualize_nodes():
    """
    Analyzes node availability per timestamp and generates a plot.
    """
    print("--- Starting Node Analysis ---")
    
    # Load data
    attribute_df, node_df = load_and_preprocess_data(base_path="../..")

    # Get all possible node IDs
    all_node_ids = set(node_df['ComputerNo'].unique())
    print(f"Total number of unique nodes (stations): {len(all_node_ids)}")
    print(f"Node IDs: {sorted(list(all_node_ids))}")

    # Get nodes per timestamp
    nodes_per_timestamp = attribute_df.groupby('TimeStamp')['Compno'].apply(set)
    
    print("\n--- Node Availability per Timestamp ---")
    # Analyze missing nodes for each timestamp
    for timestamp, present_nodes in nodes_per_timestamp.items():
        missing_nodes = all_node_ids - present_nodes
        if missing_nodes:
            print(f"Timestamp: {timestamp.strftime('%Y-%m-%d')}")
            print(f"  - Present nodes: {len(present_nodes)}")
            print(f"  - Missing nodes ({len(missing_nodes)}): {sorted(list(missing_nodes))}")
        else:
            print(f"Timestamp: {timestamp.strftime('%Y-%m-%d')}")
            print(f"  - All {len(present_nodes)} nodes are present.")

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Timestamp': nodes_per_timestamp.index,
        'NodeCount': nodes_per_timestamp.apply(len)
    }).sort_values('Timestamp')

    # Create the plot
    plt.figure(figsize=(15, 7))
    plt.bar(plot_df['Timestamp'], plot_df['NodeCount'], width=20)
    plt.xlabel("Timestamp")
    plt.ylabel("Number of Nodes")
    plt.title("Number of Nodes per Timestamp")
    plt.grid(axis='y', linestyle='--')
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), "nodes_per_timestamp.png")
    plt.savefig(output_path)
    print(f"\n--- Plot saved to {output_path} ---")

if __name__ == '__main__':
    analyze_and_visualize_nodes()
