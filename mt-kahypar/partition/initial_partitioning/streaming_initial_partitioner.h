#pragma once
#include <tuple>
#include <vector>


#include "mt-kahypar/parallel/stl/scalable_queue.h"
#include "mt-kahypar/partition/initial_partitioning/i_initial_partitioner.h"
#include "mt-kahypar/partition/initial_partitioning/initial_partitioning_data_container.h"

namespace mt_kahypar {

template <typename TypeTraits>
class StreamingInitialPartitioner : public IInitialPartitioner {
  static constexpr bool debug = false;

  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;

public:
  StreamingInitialPartitioner(const InitialPartitioningAlgorithm,
                              ip_data_container_t *ip_data,
                              const Context &context, const int seed,
                              const int tag)
      : _ip_data(ip::to_reference<TypeTraits>(ip_data)), _context(context),
        _rng(seed), _tag(tag), 
        _partition_history(_ip_data.local_partitioned_hypergraph().initialNumEdges(), -1) { }

private:
  void partitionImpl() final;

  bool fitsIntoBlock(PartitionedHypergraph& hypergraph,
                     const HypernodeID hn,
                     const PartitionID block) const {
    ASSERT(block != kInvalidPartition && block < _context.partition.k);
    return hypergraph.partWeight(block) + hypergraph.nodeWeight(hn) <=
      _context.partition.perfect_balance_part_weights[block];
  }

  std::size_t computeBlockScore(HypernodeID node, PartitionID part);


 std::vector<std::pair<int, double>> computeObjectiveForAllParts(HypernodeID node); 
  void updatePartitionHistory(HypernodeID node, PartitionID part);
  PartitionID selectPart(HypernodeID node, std::vector<std::pair<int, double>> objectives_for_parts);

  double balanced_objective(std::size_t block_score, HypernodeWeight part_weight) {
      PartitionedHypergraph& hg = _ip_data.local_partitioned_hypergraph();

      // alpha and gamma taken from Fennel
      constexpr double gamma = 1.5; 
      const double alpha = (std::sqrt(hg.k()) * hg.initialNumEdges()) / (std::pow(hg.initialNumNodes(), gamma));

      return block_score - alpha * gamma * std::sqrt(part_weight);
  }
   double greedy_objective(std::size_t block_score) {
      return static_cast<double>(block_score);
  }

  InitialPartitioningDataContainer<TypeTraits> &_ip_data;
  const Context &_context;
  std::mt19937 _rng;
  const int _tag;
  std::vector<PartitionID> _partition_history;

};

} // namespace mt_kahypar
