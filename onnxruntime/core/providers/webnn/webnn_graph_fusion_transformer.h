// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <unordered_map>
#include "core/optimizer/graph_transformer.h"
#include "core/framework/execution_providers.h"


namespace onnxruntime
{
class WebNNGraphFusionTransformer : public onnxruntime::GraphTransformer
{
public:
    WebNNGraphFusionTransformer(
        const std::string& name,
        const onnxruntime::IExecutionProvider* provider
    );

public:
    static inline const char* const WEBNN_GRAPH_FUSION_NODE_NAME_PREFIX = "WebNNFusedNode_";
    static inline const char* const WEBNN_GRAPH_FUSION_NODE_DOMAIN = "WebNNFusedNodeDomain";

private:
    onnxruntime::common::Status ApplyImpl(onnxruntime::Graph& graph,
                                            bool& modified,
                                            int graph_level,
                                            const onnxruntime::logging::Logger& logger) const final;
};
}
