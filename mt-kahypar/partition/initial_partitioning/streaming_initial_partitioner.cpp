#include "mt-kahypar/partition/initial_partitioning/streaming_initial_partitioner.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/initial_partitioning/policies/pseudo_peripheral_start_nodes.h"
#include "mt-kahypar/utils/randomize.h"

namespace mt_kahypar {

template <typename TypeTraits>
void StreamingInitialPartitioner<TypeTraits>::partitionImpl() {
  if (_ip_data.should_initial_partitioner_run(
          InitialPartitioningAlgorithm::streaming)) {
    HighResClockTimepoint start = std::chrono::high_resolution_clock::now();
    PartitionedHypergraph& hg = _ip_data.local_partitioned_hypergraph();


    LOG << "Computing a" << _context.partition.k << "-way partition!";


    // alpha and gamma taken from Fennel
    double gamma = 1.5;
    double alpha = std::sqrt(hg.k()) * hg.initialNumEdges() / (std::pow(hg.initialNumNodes(), 1.5));

    for (const auto& node : hg.nodes()) {

        if (hg.isFixed(node)) {
            continue;
        }

        PartitionID new_part;
        double max_objective = std::numeric_limits<double>::min();

        // Calculate the objective for each part
        for (PartitionID part = 0; part < hg.k(); part++) {     
            std::size_t block_score = compute_block_score(node, part); 
            HypernodeWeight part_weight = hg.partWeight(part); 

            double current_objective = block_score - alpha * gamma * std::pow(part_weight, 0.5); 
            
            if (current_objective > max_objective) {
                new_part = part;
                max_objective = current_objective;
            }            
        }

        PartitionID old_part = hg.partID(node);
        hg.changeNodePart(node, old_part, new_part);
    }


    HighResClockTimepoint end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end - start).count();
    _ip_data.commit(InitialPartitioningAlgorithm::streaming, _rng, _tag, time);
  }
}

template<typename TypeTraits>
[[nodiscard]]
std::size_t StreamingInitialPartitioner<TypeTraits>::compute_block_score(HypernodeID node, PartitionID part) {
    PartitionedHypergraph& hg = _ip_data.local_partitioned_hypergraph();

    // if there is an artificial node in an edge
    // then the (up until now) highest degree vertex 
    // of that edge was assigned to the block 
    // the artificial node is fixed to.
    std::size_t result = 0;
    for (const auto&  incident_edge : hg.incidentEdges(node)) {
        for (const auto& pin : hg.pins(incident_edge)) {
            if (hg.isFixed(pin) && hg.partID(pin) == part) {
                result++;
            }
        }
    }

    return result;
}

INSTANTIATE_CLASS_WITH_TYPE_TRAITS(StreamingInitialPartitioner)

} // namespace mt_kahypar
