import os
import numpy as np
from build_hetero_graph import HeteroGraphBuilder, parse_arguments

def main():
    args = parse_arguments()
    args.output_dir = f"graphs_hetero_batch"
    os.makedirs(args.output_dir, exist_ok=True)
    builder = HeteroGraphBuilder(
        dataset_name=args.dataset_name,
        k_shot=args.k_shot,
        embedding_type=args.embedding_type,
        edge_policy=args.edge_policy,
        k_neighbors=args.k_neighbors,
        partial_unlabeled=args.partial_unlabeled,
        sample_unlabeled_factor=args.sample_unlabeled_factor,
        output_dir=args.output_dir,
        seed=args.seed,
        pseudo_label=args.pseudo_label,
        pseudo_label_cache_path=args.pseudo_label_cache_path,
        multi_view=args.multi_view,
        enable_dissimilar=args.enable_dissimilar if hasattr(args, 'enable_dissimilar') else False,
        interaction_embedding_field=args.interaction_embedding_field if hasattr(args, 'interaction_embedding_field') else "interaction_embeddings_list",
        interaction_tone_field=args.interaction_tone_field if hasattr(args, 'interaction_tone_field') else "interaction_tones_list",
        interaction_edge_mode=args.interaction_edge_mode if hasattr(args, 'interaction_edge_mode') else "edge_type",
        ensure_test_labeled_neighbor=args.ensure_test_labeled_neighbor if hasattr(args, 'ensure_test_labeled_neighbor') else False,
    )
    builder.load_dataset()
    test_indices = np.arange(len(builder.test_data))
    batch_size = args.batch_size
    for i, start in enumerate(range(0, len(test_indices), batch_size)):
        batch_indices = test_indices[start:start+batch_size]
        print(f"\n==== Building batch {i+1} ({start}~{start+len(batch_indices)-1}) ====")
        hetero_graph = builder.build_hetero_graph(test_batch_indices=batch_indices)
        print("hetero_graph.x.shape:", hetero_graph['news'].x.shape)
        print("hetero_graph.train_labeled_mask.shape:", hetero_graph['news'].train_labeled_mask.shape)
        print("hetero_graph.train_unlabeled_mask.shape:", hetero_graph['news'].train_unlabeled_mask.shape)
        print("hetero_graph.test_mask.shape:", hetero_graph['news'].test_mask.shape)
        builder.analyze_hetero_graph(hetero_graph)
        builder.save_graph(hetero_graph, batch_id=i+1)

if __name__ == "__main__":
    main() 