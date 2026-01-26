/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2019 Tobias Heuer <tobias.heuer@kit.edu>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#include "mt-kahypar/partition/initial_partitioning/streaming_initial_partitioner.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/refinement/gains/cut/cut_attributed_gains.h"
#include "mt-kahypar/partition/initial_partitioning/policies/pseudo_peripheral_start_nodes.h"
#include "mt-kahypar/partition/initial_partitioning/policies/gain_computation_policy.h"
#include "mt-kahypar/utils/randomize.h"

namespace mt_kahypar {

template<typename TypeTraits>
void StreamingInitialPartitioner<TypeTraits>::partitionImpl() {
  if ( _ip_data.should_initial_partitioner_run(InitialPartitioningAlgorithm::label_propagation) ) {
    HighResClockTimepoint start = std::chrono::high_resolution_clock::now();
    PartitionedHypergraph& hg = _ip_data.local_partitioned_hypergraph();

    _ip_data.reset_unassigned_hypernodes(_rng);
    _ip_data.preassignFixedVertices(hg);

    vec<vec<HypernodeID>> start_nodes =
      PseudoPeripheralStartNodes<TypeTraits>::computeStartNodes(_ip_data, _context, kInvalidPartition, _rng);

    for ( PartitionID block = 0; block < _context.partition.k; ++block ) {
      size_t i = 0;
      for ( ; i < std::min(start_nodes[block].size(),
        _context.initial_partitioning.lp_initial_block_size); ++i ) {
        const HypernodeID hn = start_nodes[block][i];
        if ( hg.partID(hn) == kInvalidPartition && fitsIntoBlock(hg, hn, block) ) {
          hg.setNodePart(hn, block);
        } else {
          std::swap(start_nodes[block][i--], start_nodes[block][start_nodes[block].size() - 1]);
          start_nodes[block].pop_back();
        }
      }

      // Remove remaining unassigned seed nodes
      for ( ; i < start_nodes[block].size(); ++i ) {
        start_nodes[block].pop_back();
      }

      if ( start_nodes[block].size() == 0 ) {
        // There has been no seed node assigned to the block
        // => find an unassigned node and assign it to the block
        const HypernodeID hn = _ip_data.get_unassigned_hypernode();
        if ( hn != kInvalidHypernode ) {
          hg.setNodePart(hn, block);
          start_nodes[block].push_back(hn);
        }
      }
    }

    // Each block is extended with 5 additional vertices which are adjacent
    // to their corresponding seed vertices. This should prevent that block
    // becomes empty after several label propagation rounds.
    for ( PartitionID block = 0; block < _context.partition.k; ++block ) {
      if ( !start_nodes[block].empty() && start_nodes[block].size() <
            _context.initial_partitioning.lp_initial_block_size ) {
        extendBlockToInitialBlockSize(hg, start_nodes[block], block);
      }
    }

    bool converged = false;
    for ( size_t i = 0; i < _context.initial_partitioning.lp_maximum_iterations && !converged; ++i ) {
      converged = true;

      for ( const HypernodeID& hn : hg.nodes() ) {
        if (hg.nodeDegree(hn) > 0 && !hg.isFixed(hn)) {
          // Assign vertex to the block where FM gain is maximized
          MaxGainMoveStreaming max_gain_move = computeMaxGainMove(hg, hn);

          const PartitionID to = max_gain_move.block;
          if ( to != kInvalidPartition ) {
            const PartitionID from = hg.partID(hn);
            if ( from == kInvalidPartition ) {
              ASSERT(fitsIntoBlock(hg, hn, to));

              // disable for streaming ?
              HEAVY_INITIAL_PARTITIONING_ASSERT([&] {
                Gain expected_gain = CutGainPolicy<TypeTraits>::calculateGain(hg, hn, to);
                if ( expected_gain != max_gain_move.gain ) {
                  LOG << V(hn);
                  LOG << V(from);
                  LOG << V(to);
                  LOG << V(max_gain_move.gain);
                  LOG << V(expected_gain);
                }
                return true;
              }(), "Gain calculation failed");

              converged = false;
              hg.setNodePart(hn, to);
            } else if ( from != to ) {
              ASSERT(fitsIntoBlock(hg, hn, to));
              converged = false;

              #ifndef KAHYPAR_ENABLE_HEAVY_INITIAL_PARTITIONING_ASSERTIONS
              hg.changeNodePartNoSync(hn, from, to);
              #else
              Gain expected_gain = 0;
              auto cut_delta = [&](const HyperedgeID he,
                                  const HyperedgeWeight edge_weight,
                                  const HypernodeID,
                                  const HypernodeID pin_count_in_from_part_after,
                                  const HypernodeID pin_count_in_to_part_after) {
                HypernodeID adjusted_edge_size = 0;
                for ( const HypernodeID& pin : hg.pins(he) ) {
                  if ( hg.partID(pin) != kInvalidPartition ) {
                    ++adjusted_edge_size;
                  }
                }
                expected_gain -= CutAttributedGains::gain(
                  he, edge_weight, adjusted_edge_size,
                  pin_count_in_from_part_after, pin_count_in_to_part_after);
              };
              hg.changeNodePart(hn, from, to, cut_delta);
              ASSERT(expected_gain == max_gain_move.gain, "Gain calculation failed"
                << V(expected_gain) << V(max_gain_move.gain));
              #endif
            }
          }

        } else if ( !hg.isFixed(hn) ) {
          // In case vertex hn is a degree zero vertex we assign it
          // to the block with minimum weight
          assignVertexToBlockWithMinimumWeight(hg, hn);
        } 
      }
    }
    hg.resetEdgeSynchronization();

    // If there are still unassigned vertices left, we assign them to the
    // block with minimum weight.
    while ( _ip_data.get_unassigned_hypernode() != kInvalidHypernode ) {
      const HypernodeID unassigned_hn = _ip_data.get_unassigned_hypernode();
      assignVertexToBlockWithMinimumWeight(hg, unassigned_hn);
    }

    HighResClockTimepoint end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end - start).count();
    _ip_data.commit(InitialPartitioningAlgorithm::label_propagation, _rng, _tag, time);
  }
}

template<typename TypeTraits>
MaxGainMoveStreaming StreamingInitialPartitioner<TypeTraits>::computeMaxGainMoveForUnassignedVertex(PartitionedHypergraph& hypergraph,
                                                                                                  const HypernodeID hn) {
  ASSERT(hypergraph.partID(hn) == kInvalidPartition);
  ASSERT(std::all_of(_tmp_scores.begin(), _tmp_scores.end(), [](Gain i) { return i == 0; }),
          "Temp gain array not initialized properly");

  for (const HyperedgeID& he : hypergraph.incidentEdges(hn)) {
    const HyperedgeWeight he_weight = hypergraph.edgeWeight(he);
    for (PartitionID to : hypergraph.connectivitySet(he)) {
        _tmp_scores[to] += he_weight;
    }
  }

  return findMaxGainMove(hypergraph, hn);
}

template<typename TypeTraits>
MaxGainMoveStreaming StreamingInitialPartitioner<TypeTraits>::computeMaxGainMoveForAssignedVertex(PartitionedHypergraph& hypergraph,
                                                                                                const HypernodeID hn) {
  ASSERT(hypergraph.partID(hn) != kInvalidPartition);
  ASSERT(std::all_of(_tmp_scores.begin(), _tmp_scores.end(), [](Gain i) { return i == 0; }),
          "Temp gain array not initialized properly");

  const PartitionID from = hypergraph.partID(hn);
  for (const HyperedgeID& he : hypergraph.incidentEdges(hn)) {
    const HyperedgeWeight he_weight = hypergraph.edgeWeight(he);

    for (PartitionID to : hypergraph.connectivitySet(he)) { 

        // if there is only one pin in `from` we can reduce 
        // connectivity by moving the node `hn` to one of  
        // the blocks of the other pins
        if (from == to && 
            hypergraph.pinCountInPart(he, from) <= 1) {
            continue;
        }

        _tmp_scores[to] += he_weight;
    }
  }

  return findMaxGainMove(hypergraph, hn);
}

template<typename TypeTraits>
MaxGainMoveStreaming StreamingInitialPartitioner<TypeTraits>::findMaxGainMove(PartitionedHypergraph& hypergraph,
                                                                            const HypernodeID hn) {

  constexpr static double gamma = 1.5;
  const double alpha = (std::sqrt(_context.partition.k) * 
                       _context.streaming.inputNumEdges) / (std::pow(_context.streaming.inputNumNodes, gamma));

  const PartitionID from = hypergraph.partID(hn);
  PartitionID best_block = from;
  Gain best_score = 0;

  if (from == kInvalidPartition) {
    best_score = -std::numeric_limits<Gain>::max();
  } else { 
    double from_penalty = hypergraph.nodeWeight(hn) * alpha * gamma *
                          std::sqrt(hypergraph.partWeight(from) - hypergraph.nodeWeight(hn));
    _tmp_scores[from] -= from_penalty;
    best_score = _tmp_scores[from];
  } 

  for (PartitionID block = 0; block < _context.partition.k; ++block) {
    // only considering valid blocks might 
    // prevent the fennel penalty to be considered
    if (from != block) {

      double fennel_penalty = hypergraph.nodeWeight(hn) * alpha * gamma * std::sqrt(hypergraph.partWeight(block));
      _tmp_scores[block] -= fennel_penalty;

      // Since we perform size-constraint label propagation, the move to the
      // corresponding block is only valid, if it fullfils the balanced constraint.
      if (fitsIntoBlock(hypergraph, hn, block) && _tmp_scores[block] > best_score) {
        best_score = _tmp_scores[block];
        best_block = block;
      }
    } 
  }
  
  std::fill(std::begin(_tmp_scores), std::end(_tmp_scores), 0);
  return MaxGainMoveStreaming { best_block, best_score };
}

template<typename TypeTraits>
void StreamingInitialPartitioner<TypeTraits>::extendBlockToInitialBlockSize(PartitionedHypergraph& hypergraph,
                                                                                   const vec<HypernodeID>& seed_vertices,
                                                                                   const PartitionID block) {
  ASSERT(seed_vertices.size() > 0);
  size_t block_size = seed_vertices.size();

  // We search for _context.initial_partitioning.lp_initial_block_size vertices
  // around the seed vertex to extend the corresponding block
  for ( const HypernodeID& seed_vertex : seed_vertices ) {
    for ( const HyperedgeID& he : hypergraph.incidentEdges(seed_vertex) ) {
      for ( const HypernodeID& pin : hypergraph.pins(he) ) {
        if ( hypergraph.partID(pin) == kInvalidPartition &&
             fitsIntoBlock(hypergraph, pin, block) ) {
          hypergraph.setNodePart(pin, block);
          block_size++;
          if ( block_size >= _context.initial_partitioning.lp_initial_block_size ) break;
        }
      }
      if ( block_size >= _context.initial_partitioning.lp_initial_block_size ) break;
    }
    if ( block_size >= _context.initial_partitioning.lp_initial_block_size ) break;
  }

  // If there are less than _context.initial_partitioning.lp_initial_block_size
  // adjacent vertices to the seed vertex, we find a new seed vertex and call
  // this function recursive
  while ( block_size < _context.initial_partitioning.lp_initial_block_size ) {
    const HypernodeID seed_vertex = _ip_data.get_unassigned_hypernode();
    if ( seed_vertex != kInvalidHypernode && fitsIntoBlock(hypergraph, seed_vertex, block)  ) {
      hypergraph.setNodePart(seed_vertex, block);
      block_size++;
    } else {
      break;
    }
  }
}

template<typename TypeTraits>
void StreamingInitialPartitioner<TypeTraits>::assignVertexToBlockWithMinimumWeight(PartitionedHypergraph& hypergraph,
                                                                                          const HypernodeID hn) {
  // ASSERT(hypergraph.partID(hn) == kInvalidPartition);
  PartitionID minimum_weight_block = kInvalidPartition;
  HypernodeWeight minimum_weight = std::numeric_limits<HypernodeWeight>::max();
  for ( PartitionID block = 0; block < _context.partition.k; ++block ) {
    HypernodeWeight block_weight = hypergraph.partWeight(block);
    if (block == hypergraph.partID(hn)) {
        block_weight -= hypergraph.nodeWeight(hn);
    }

    if ( block_weight < minimum_weight ) {
      minimum_weight = block_weight;
      minimum_weight_block = block;
    }
  }
  ASSERT(minimum_weight_block != kInvalidPartition);
  if (hypergraph.partID(hn) != minimum_weight_block) {
    if (hypergraph.partID(hn) == kInvalidPartition) {
        hypergraph.setNodePart(hn, minimum_weight_block);
    } else {
        hypergraph.changeNodePart(hn, hypergraph.partID(hn), minimum_weight_block);
    }
  }
}

INSTANTIATE_CLASS_WITH_TYPE_TRAITS(StreamingInitialPartitioner)

} // namespace mt_kahypar
