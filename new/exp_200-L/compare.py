import pandas as pd
import os
import numpy as np
from collections import defaultdict

class TransitionNetworkAnalyzer:
    def __init__(self, main_folder_path):
        self.main_folder_path = main_folder_path
        self.ground_truth_folder = 'a-hardcoded-loc-None'
        
    def categorize_transition(self, row):
        is_special_state = row['from_state'] == "(0, 0, 0, 0, 0, 1)"
        is_box2_state = row['from_state'] == "(0, 0, 1, 0, 0, 0)"
        is_to_self = row['from_state'] == row['to_state']
        vision = row['vision']
        
        if is_special_state and is_to_self and vision == 0:
            return "uninformed"
        elif vision == 0:
            return "misinformed"
        elif vision == 1:
            return "informed"
        else:
            raise ValueError(f"Unknown vision value: {vision}")
    
    def categorize_transition_box2(self, row):
        is_special_state = row['from_state'] == "(0, 0, 0, 0, 0, 1)"
        is_to_self = row['from_state'] == row['to_state']
        is_to_box2 = row['to_state'] == "(0, 0, 1, 0, 0, 0)"
        vision = row['vision']
        
        if (is_special_state and is_to_self and vision == 0) or (is_to_box2 and vision == 0):
            return "uninformed"
        elif vision == 0:
            return "misinformed"
        elif vision == 1:
            return "informed"
        else:
            raise ValueError(f"Unknown vision value: {vision}")
    
    def load_transitions(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df['from_state'] = df['from_state'].astype(str).str.strip()
            df['to_state'] = df['to_state'].astype(str).str.strip()
            df['treat_state'] = df['treat_state'].astype(str).str.strip()
            df['vision'] = df['vision'].astype(int)
            df['count'] = df['count'].fillna(1).astype(int)
            df['category'] = df.apply(self.categorize_transition, axis=1)
            df['category_box2'] = df.apply(self.categorize_transition_box2, axis=1)
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def load_losses(self, file_path):
        try:
            df = pd.read_csv(file_path)
            if len(df) == 0:
                return None, None
            
            final_row = df.iloc[-1]
            accuracy = final_row.get('Accuracy', None)
            novel_accuracy = final_row.get('Novel_Accuracy', None)
            
            return accuracy, novel_accuracy
        except Exception as e:
            print(f"Error loading losses {file_path}: {e}")
            return None, None
    
    def aggregate_by_category(self, df, use_box2=False):
        categories = {'uninformed': {}, 'misinformed': {}, 'informed': {}}
        category_col = 'category_box2' if use_box2 else 'category'
        
        for category in categories.keys():
            category_data = df[df[category_col] == category]
            edge_counts = defaultdict(int)
            
            for _, row in category_data.iterrows():
                edge_key = f"{row['from_state']}->{row['to_state']}"
                edge_counts[edge_key] += row['count']
            
            categories[category] = dict(edge_counts)
        
        return categories
    
    def jaccard_similarity(self, graph1, graph2, normalize_box2=False):
        if normalize_box2:
            normalized_graph1 = {}
            normalized_graph2 = {}
            
            for edge, count in graph1.items():
                if edge.endswith("->(0, 0, 1, 0, 0, 0)"):
                    normalized_edge = "(0, 0, 0, 0, 0, 1)->(0, 0, 0, 0, 0, 1)"
                    normalized_graph1[normalized_edge] = normalized_graph1.get(normalized_edge, 0) + count
                else:
                    normalized_graph1[edge] = count
                    
            for edge, count in graph2.items():
                if edge.endswith("->(0, 0, 1, 0, 0, 0)"):
                    normalized_edge = "(0, 0, 0, 0, 0, 1)->(0, 0, 0, 0, 0, 1)"
                    normalized_graph2[normalized_edge] = normalized_graph2.get(normalized_edge, 0) + count
                else:
                    normalized_graph2[edge] = count
            
            edges1 = set(normalized_graph1.keys())
            edges2 = set(normalized_graph2.keys())
        else:
            edges1 = set(graph1.keys())
            edges2 = set(graph2.keys())
        
        intersection = edges1.intersection(edges2)
        union = edges1.union(edges2)
        
        return len(intersection) / len(union) if len(union) > 0 else 1.0

    def calculate_vision0_similarity(self, df1, df2, use_box2=True):
        vision0_df1 = df1[df1['vision'] == 0].copy()
        vision0_df2 = df2[df2['vision'] == 0].copy()
        
        edges1 = {}
        edges2 = {}
        
        for _, row in vision0_df1.iterrows():
            edge_key = f"{row['from_state']}->{row['to_state']}"
            if use_box2 and row['to_state'] == "(0, 0, 1, 0, 0, 0)":
                edge_key = "(0, 0, 0, 0, 0, 1)->(0, 0, 0, 0, 0, 1)"
            edges1[edge_key] = edges1.get(edge_key, 0) + row['count']
            
        for _, row in vision0_df2.iterrows():
            edge_key = f"{row['from_state']}->{row['to_state']}"
            edges2[edge_key] = edges2.get(edge_key, 0) + row['count']
        
        edges1_set = set(edges1.keys())
        edges2_set = set(edges2.keys())
        
        intersection = edges1_set.intersection(edges2_set)
        union = edges1_set.union(edges2_set)
        
        return len(intersection) / len(union) if len(union) > 0 else 1.0
    
    def analyze_all_folders(self):
        ground_truth_path = os.path.join(self.main_folder_path, self.ground_truth_folder, 'transitions-0.csv')
        ground_truth_df = self.load_transitions(ground_truth_path)
        ground_truth_graphs = self.aggregate_by_category(ground_truth_df, use_box2=False)
        ground_truth_graphs_box2 = self.aggregate_by_category(ground_truth_df, use_box2=False)
        
        print("Ground truth categories:")
        for cat, graph in ground_truth_graphs.items():
            print(f"  {cat}: {len(graph)} edges, total weight: {sum(graph.values())}")
            if len(graph) > 0:
                print(f"    Sample edges: {list(graph.items())[:3]}")
        
        results = []
        
        for folder_name in os.listdir(self.main_folder_path):
            folder_path = os.path.join(self.main_folder_path, folder_name)
            
            if not os.path.isdir(folder_path) or folder_name == self.ground_truth_folder:
                continue
            
            transition_files = []
            for file_name in os.listdir(folder_path):
                if file_name.startswith('transitions-') and file_name.endswith('.csv'):
                    try:
                        number_part = file_name[12:-4]
                        transition_num = int(number_part)
                        transition_files.append((transition_num, file_name))
                    except ValueError:
                        continue
            
            transition_files.sort()
            
            if not transition_files:
                print(f"No transitions-*.csv files found in {folder_name}")
                continue
            
            for transition_num, file_name in transition_files:
                transitions_file = os.path.join(folder_path, file_name)
                losses_file = os.path.join(folder_path, f"losses-{transition_num}.csv")
                
                df = self.load_transitions(transitions_file)
                if df.empty:
                    continue
                
                accuracy, novel_accuracy = self.load_losses(losses_file)
                
                graphs = self.aggregate_by_category(df, use_box2=False)
                graphs_box2 = self.aggregate_by_category(df, use_box2=True)
                
                print(f"\n{folder_name} - {file_name} categories:")
                for cat, graph in graphs.items():
                    print(f"  {cat}: {len(graph)} edges, total weight: {sum(graph.values())}")
                    if len(graph) > 0:
                        print(f"    Sample edges: {list(graph.items())[:3]}")
                
                print(f"  box2 categories:")
                for cat, graph in graphs_box2.items():
                    print(f"    {cat}: {len(graph)} edges, total weight: {sum(graph.values())}")
                    if len(graph) > 0:
                        print(f"      Sample edges: {list(graph.items())[:3]}")
                
                print(f"  DEBUG: Transitions with vision=0:")
                vision0_transitions = df[df['vision'] == 0]
                print(f"    Total vision=0 transitions: {len(vision0_transitions)}")
                
                vision0_to_box2 = df[(df['vision'] == 0) & (df['to_state'] == "(0, 0, 1, 0, 0, 0)")]
                print(f"    Vision=0 transitions TO (0,0,1,0,0,0): {len(vision0_to_box2)}")
                
                vision0_self_excluded = df[(df['vision'] == 0) & (df['from_state'] == df['to_state']) & (df['from_state'] != "(0, 0, 0, 0, 0, 1)")]
                print(f"    Vision=0 self-transitions (excluded from box2): {len(vision0_self_excluded)}")
                
                if len(vision0_self_excluded) > 0:
                    print(f"      Sample excluded: {vision0_self_excluded[['from_state', 'to_state', 'count']].head(3).to_dict('records')}")
                
                included_for_box2 = len(vision0_transitions) - len(vision0_self_excluded)
                print(f"    Transitions included in box2 similarity: {included_for_box2}")
                
                uninformed_similarity = self.jaccard_similarity(
                    graphs['uninformed'], 
                    ground_truth_graphs['uninformed']
                )
                
                misinformed_similarity = self.jaccard_similarity(
                    graphs['misinformed'], 
                    ground_truth_graphs['misinformed']
                )
                
                informed_similarity = self.jaccard_similarity(
                    graphs['informed'], 
                    ground_truth_graphs['informed']
                )
                
                uninformed_box2_similarity = self.calculate_vision0_similarity(
                    df, ground_truth_df, use_box2=True
                )
                
                print(f"  SIMILARITIES:")
                print(f"    uninformed: {uninformed_similarity:.3f}")
                print(f"    misinformed: {misinformed_similarity:.3f}")
                print(f"    informed: {informed_similarity:.3f}")
                print(f"    uninformed_box2: {uninformed_box2_similarity:.3f}")
                
                if accuracy is not None or novel_accuracy is not None:
                    print(f"  ACCURACIES:")
                    if accuracy is not None:
                        print(f"    accuracy: {accuracy:.3f}")
                    if novel_accuracy is not None:
                        print(f"    novel_accuracy: {novel_accuracy:.3f}")
                
                results.append({
                    'train': folder_name,
                    'rep': transition_num,
                    'uninformed': round(uninformed_similarity, 3),
                    'misinformed': round(misinformed_similarity, 3),
                    'informed': round(informed_similarity, 3),
                    'uninformed+box2': round(uninformed_box2_similarity, 3),
                    'accuracy': round(accuracy, 3) if accuracy is not None else None,
                    'novel_accuracy': round(novel_accuracy, 3) if novel_accuracy is not None else None
                })
        
        return results
    
    def save_results(self, results, output_file='similarity_results.csv'):
        df = pd.DataFrame(results)
        output_path = os.path.join(self.main_folder_path, output_file)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        return df

if __name__ == "__main__":
    main_folder = "."
    analyzer = TransitionNetworkAnalyzer(main_folder)
    results = analyzer.analyze_all_folders()
    df_results = analyzer.save_results(results)
    
    print("\nSimilarity Results:")
    print(df_results.to_string(index=False))