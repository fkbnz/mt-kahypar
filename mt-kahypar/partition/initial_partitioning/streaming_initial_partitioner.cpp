#include <algorithm>

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

    for (const auto& node : hg.nodes()) {

        if (hg.isFixed(node)) { continue; }

        PartitionID new_part = selectPart(node, computeObjectiveForAllParts(node));
        hg.setNodePart(node, new_part);
        updatePartitionHistory(node, new_part);
    }

    HighResClockTimepoint end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end - start).count();
    _ip_data.commit(InitialPartitioningAlgorithm::streaming, _rng, _tag, time);
  }
}

template<typename TypeTraits>
[[nodiscard]]
std::vector<std::pair<int, double>> StreamingInitialPartitioner<TypeTraits>::computeObjectiveForAllParts(HypernodeID node) {

    PartitionedHypergraph& hg = _ip_data.local_partitioned_hypergraph();

    // alpha and gamma taken from Fennel
    double gamma = 1.5;
    double alpha = std::sqrt(hg.k()) * hg.initialNumEdges() / (std::pow(hg.initialNumNodes(), 1.5));

    std::vector<std::pair<int, double>> objectives(hg.k());
    for (PartitionID part = 0; part < hg.k(); part++) {
        std::size_t block_score = computeBlockScore(node, part); 
        HypernodeWeight part_weight = hg.partWeight(part); 

        double current_objective = block_score - alpha * gamma * std::pow(part_weight, 0.5);
        objectives[part] = {part, current_objective};
    }

    return objectives;
}

template<typename TypeTraits>
[[nodiscard]]
std::size_t StreamingInitialPartitioner<TypeTraits>::computeBlockScore(HypernodeID node, PartitionID part) {
    PartitionedHypergraph& hg = _ip_data.local_partitioned_hypergraph();

    std::size_t result = 0;
    for (const auto&  incident_edge : hg.incidentEdges(node)) {

        // if there is an artificial node in an edge
        // then the (up until now) highest degree vertex 
        // of that edge was assigned to the block 
        // the artificial node is fixed to.
        // Note that fixed nodes represent assignments of 
        // PREVIOUS batches.
        for (const auto& pin : hg.pins(incident_edge)) {
            if (hg.isFixed(pin) && hg.partID(pin) == part) {
                result++;
            }
        }

        // if the entry in the partition history is 
        // the current part id then the (up until now) highest degree 
        // vertex of THIS batch was assignt to that node.
        if (_partition_history[incident_edge].first == part) {
            result++;
        } 
    }

    return result;
}

template<typename TypeTraits>
PartitionID StreamingInitialPartitioner<TypeTraits>::selectPart(
    HypernodeID node, 
    std::vector<std::pair<int, double>> objectives_for_parts
) {
    PartitionedHypergraph& hg = _ip_data.local_partitioned_hypergraph();
    auto compare = [](const std::pair<int, double>& lhs, const std::pair<int, double>& rhs) {
        return lhs.second > rhs.second;
    };

    std::sort(std::begin(objectives_for_parts), std::end(objectives_for_parts), compare);
     
    for (const auto& [part, objective] : objectives_for_parts) {
        if (fitsIntoBlock(hg, node, part)) {
            return part;
        }
    }

    return objectives_for_parts[0].first;
}

template<typename TypeTraits>
void StreamingInitialPartitioner<TypeTraits>::updatePartitionHistory(HypernodeID node, PartitionID part) {
    PartitionedHypergraph& hg = _ip_data.local_partitioned_hypergraph();

    std::size_t node_degree = hg.nodeDegree(node);
    for (const auto& incident_edge : hg.incidentEdges(node)) {
        if (node_degree > _partition_history[incident_edge].second) {
            _partition_history[incident_edge] = {part, node_degree};
        }
    } 
}

INSTANTIATE_CLASS_WITH_TYPE_TRAITS(StreamingInitialPartitioner)

} // namespace mt_kahypar
