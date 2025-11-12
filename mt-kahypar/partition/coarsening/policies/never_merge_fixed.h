#pragma once

#include "kahypar-resources/meta/policy_registry.h"
#include "kahypar-resources/meta/typelist.h"

#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/datastructures/fixed_vertex_support.h"
#include "mt-kahypar/datastructures/hypergraph_common.h"
#include "mt-kahypar/macros.h"


namespace mt_kahypar {

class NoMergeFixedPolicy final : public kahypar::meta::PolicyBase {
public:

// This function decides if contracting v onto u is allowed if the hypergraph contains fixed vertices.
template<typename Hypergraph>
MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE static bool acceptContraction(const Hypergraph& hypergraph,
                                                                 const ds::FixedVertexSupport<Hypergraph>& fixed_vertices,
                                                                 const Context& context,
                                                                 const HypernodeID u,
                                                                 const HypernodeID v) 
{
    // Never allow any fixed vertices to be contracted
    return !(fixed_vertices.isFixed(u) || fixed_vertices.isFixed(v));
}

};

} // namespace mt_kahypar
