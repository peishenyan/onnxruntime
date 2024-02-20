// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.



#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/reshape_helper.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class EinsumOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

// Helper functions, thanks for DML EP's OperatorHelper.
enum class RecognizedOperatorType {
    None,
    Identity,
    Multiply,
    OuterProduct,
    MatMul,
    MatMulTransposeA,
    MatMulTransposeB,
    MatMulNhcw,
    MatMulNhcwTransposeA,
    MatMulNhcwTransposeB,
    ReduceSum,
    Transpose,
    Total,
};

struct RecognizedOperatorInfo {
  RecognizedOperatorType recognized_operator_type;
  std::initializer_list<uint32_t> component_ranks;
  std::initializer_list<uint32_t> label_indices;
};

struct Component {
  uint32_t label_idx_begin;
  uint32_t label_idx_end;

  uint32_t GetDimensionCount() const noexcept {
    return label_idx_end - label_idx_begin;
  }
  gsl::span<const uint32_t> GetLabels(gsl::span<const uint32_t> labels) const {
    return labels.subspan(label_idx_begin, label_idx_end - label_idx_begin);
  };
};

bool ParseEquationComponents(const InitializedTensorSet& initializers,
                     const Node& node, const std::string& equation,
                     std::vector<uint32_t>& m_label_indices,
                     std::vector<Component>& m_components,
                     std::vector<uint32_t>& m_output_dimensions,
                     uint32_t& num_labels,
                     const logging::Logger& logger) {

  // Parse the equation and mapping each axis into numeric indices
  std::map<char, uint32_t> label_maps;
  std::set<char> repeated_labels;

  num_labels = 0;
  Component current_component = {};
  bool at_output = false;
  bool end_flag = false;

  // Parsing inputs and output
  for (const char* it = equation.data(); !end_flag; ++it) { // std::string.data() promises the end of the string is '\0'
    char ch = *it;

    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')) {
      const auto [i, inserted] = label_maps.insert({ch, num_labels});
      if (inserted) {
        if (at_output) {
          LOGS(logger, VERBOSE) << "Found label in equation output not matching any label from inputs.";
          return false;
        }
        ++num_labels;
      }
      else if (!at_output) {
        repeated_labels.insert(ch);
      }
      m_label_indices.push_back(i->second);
    }
    else if (ch == ' ') {
      continue;
    }
    else {
      current_component.label_idx_end = static_cast<uint32_t>(m_label_indices.size());
      m_components.push_back(current_component);
      current_component.label_idx_begin = current_component.label_idx_end;

      switch (ch) {
        case ',':
          break;

        case '-':
          ++it;
          if (*it != '>') {
            LOGS(logger, VERBOSE) << "Expected '->' for output.";
            return false;
          }
          if (at_output) {
            LOGS(logger, VERBOSE) << "Only one output arrow '->' is valid.";
            return false;
          }
          at_output = true;
          break;

        case '.':
          // Ellipsis is unsupported
          LOGS(logger, VERBOSE) << "Ellipsis is unsupported.";
          return false;

        case '\0':
          end_flag = true;
          break; // End of string.

        default:
          LOGS(logger, VERBOSE) << "Unsupported character in equation string.";
          return false;
      }
    }
  }

  // No explicit output was given
  if (!at_output) {
    for (auto i : label_maps) {
      if (repeated_labels.count(i.first) == 0) {
        m_label_indices.push_back(i.second);
      }
    }

    current_component.label_idx_end = static_cast<uint32_t>(m_label_indices.size());
    m_components.push_back(current_component);
  }
  return true;
}


void PairwiseOperandProcess(const std::vector<uint32_t>& m_label_indices,
                     const std::vector<Component>& m_components,
                     const std::vector<uint32_t>& m_output_dimensions,
                     const uint32_t& num_lables
                     const logging::Logger& logger) {

  auto input_a_labels = m_components[0].GetLabels(m_label_indices);
  auto input_b_labels = m_components[1].GetLabels(m_label_indices);
  auto output_labels = m_components[2].GetLabels(m_label_indices);

  std::map<uint32_t, int32_t> input_a_axes_map, input_b_axes_map, output_axes_map;

  for (uint32_t i = 0; i < num_lables; ++i) {
    input_a_axes_map[i] = input_b_axes_map[i] = output_axes_map[i] = -1;
  }
  int32_t index = 0;
  for (auto axis : input_a_labels) {
    input_a_axes_map[axis] = index;
    index++;
  }
  index = 0;
  for (auto axis : input_b_labels) {
    input_b_axes_map[axis] = index;
    index++;
  }
  index = 0;
  for (auto axis : output_labels) {
    output_axes_map[axis] = index;
    index++;
  }

  // Inputs Reshape
  std::vector<uint32_t> a_1, a_2, a_3, b_1, b_2, b_3;
  uint32_t a_idx = input_a_labels.size();
  uint32_t b_idx = input_b_labels.size();
  bool a_flag = false;
  bool b_flag = false;
  uint32_t a_lack, b_lack;

  for (uint32_t i = 0; i < num_lables; ++i) {
    if (input_a_axes_map[i] != -1) {
      if (input_b_axes_map[i] != -1) {
        if (output_axes_map[i] != -1) { // (1,1,1) push back in the front
          a_1.push_back(i);
          b_1.push_back(i)
        }
        else { // (1,1,0) push back in the middle for b and end for a
          a_3.push_back(i);
          b_2.push_back(i);
        }
      }
      else { // (1,0,x) push back in the middle for a. If more than one, push back in the front for a b
        input_b_axes_map[i] = b_idx;
        b_idx++;
        if (a_flag) {
          a_1.push_back(i);
          b_1.push_back(i);
        }
        else {
          a_2.push_back(i);
          b_lack = i;
          a_flag = true;
        }
      }
    }
    else {
      input_a_axes_map[i] = a_idx;
      a_idx++;
      if (input_b_axes_map[i] != -1) { // (0,1,x) push back in the end for b.  If more than one, push back in the front for a b
        if (b_flag) {
          a_1.push_back(i);
          b_1.push_back(i);
        }
        else {
          b_3.push_back(i);
          a_lack = i;
          b_flag = true;
        }
      }
    }
  }

  if (!a_flag) {
    a_2.push_back(a_lack);
  }
  if (!b_flag) {
    b_3.push_back(b_lack);
  }

  if (a_3.empty()) {
    a_3.push_back(a_idx);
    b_2.push_back(b_idx);
    input_a_axes_map[a_idx] = a_idx;
    input_b_axes_map[b_idx] = b_idx;
  }

  a_1.insert(a_1.end(), a_2.begin(), a_2.end());
  a_1.insert(a_1.end(), a_3.begin(), a_3.end());
  b_1.insert(b_1.end(), b_2.begin(), b_2.end());
  b_1.insert(b_1.end(), b_3.begin(), b_3.end());

  std::vector<uint32_t> permutation_a, permutation_b;

  for (uint32_t i = 0; i < a_1.size(); ++i) {
    permutation_a.push_back(static_cast<uint32_t>(input_a_axes_map[a_1[i]]));
    permutation_b.push_back(static_cast<uint32_t>(input_b_axes_map[b_1[i]]));
  }

  const auto& input_defs = node.InputDefs();
  emscripten::val input_a = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val input_b = model_builder.GetOperand(input_defs[1]->Name());
  if (input_a_labels.size() < a_1.size()) {
    std::vector<int64_t> input_a_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_a_shape, logger), "Cannot get shape");
    std::vector<uint32_t> new_a_shape = input_a_shape;
    for (uint32_t index = 0; index < a_1.size() - input_a_labels.size(); ++index) {
      new_a_shape.push_back(SafeInt<int32_t>(1));
    }
    input_a = model_builder.GetBuilder().call<emscripten::val>("reshape", input_a, emscripten::val::array(new_a_shape));
  }
  if (input_b_labels.size() < b_1.size()) {
    std::vector<int64_t> input_b_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[1], input_b_shape, logger), "Cannot get shape");
    std::vector<uint32_t> new_b_shape = input_b_shape;
    for (uint32_t index = 0; index < b_1.size() - input_b_labels.size(); ++index) {
      new_b_shape.push_back(SafeInt<int32_t>(1));
    }
    input_b = model_builder.GetBuilder().call<emscripten::val>("reshape", input_b, emscripten::val::array(new_b_shape));
  }

  // Inputs Transpose
  emscripten::val options = emscripten::val::object();
  options.set("permutation", emscripten::val::array(permutation_a));
  input_a = model_builder.GetBuilder().call<emscripten::val>("transpose", input_a, options);
  options.set("permutation", emscripten::val::array(permutation_b));
  input_b = model_builder.GetBuilder().call<emscripten::val>("transpose", input_b, options);

  // Matmul
  emscripten::val output = emscripten::val::object();
  output = model_builder.GetBuilder().call<emscripten::val>("matmul", input_a, input_b);
  std::vector<uint32_t> output_indices = a_1;
  output_indices.pop_back();
  output_indices.push_back(b_1.back());

  // Output Transpose
  std::vector<uint32_t> target_output_indices = output_labels;
  uint32_t p = target_output_indices.size();
  std::vector<uint32_t> s(output_indices.size(), -1), t(output_indices.size(), -1), v(output_indices.size(), -1);
  for (uint32_t i = 0; i < output_indices.size(); ++i) {
    s[output_indices[i]] = i;
    if (i < target_output_indices.size()) {
      t[target_output_indices[i]] = i;
    }
  }
  for (uint32_t i = 0; i < output_indices.size(); ++i) {
    if (t[i] == -1) {
      t[i] = p++;
    }
    v[s[i]] = t[i];
  }

  options.set("permutation", emscripten::val::array(v));
  output = model_builder.GetBuilder().call<emscripten::val>("transpose", output, options);

  // Output ReduceSum
  std::vector<int32_t> axes_data;
  for (uint32_t i = output_labels.size(); i < output_indices.size(); ++i) {
    axes_data.push_back(SafeInt<int32_t>(i));
  }
  emscripten::val options_reduce = emscripten::val::object();
  options_reduce.set("axes", emscripten::val::array(axes_data));
  output = model_builder.GetBuilder().call<emscripten::val>("reduceSum", output, options_reduce);

}

RecognizedOperatorType DetermineRecognizedOperatorType(const std::vector<uint32_t>& m_label_indices,
                     const std::vector<Component>& m_components,
                     const std::vector<uint32_t>& m_output_dimensions) {
  if (m_components.empty()) return RecognizedOperatorType::None;

  auto equals = [](gsl::span<const uint32_t> a, gsl::span<const uint32_t> b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
  };

  auto as_span = [](std::initializer_list<uint32_t> il) {
    return gsl::make_span(il.begin(), il.size());
  };

  std::array<uint32_t, 3> component_ranks;
  if (m_components.size() > component_ranks.size()) {
    // So far, not support for more than two inputs and one output.
    return RecognizedOperatorType::None;
  }
  else if (m_components.size() == 2) { // one input
    auto input_labels = m_components[0].GetLabels(m_label_indices);
    auto output_labels = m_components[1].GetLabels(m_label_indices);
    if (input_labels.size() == output_labels.size()) {
      if (equals(input_labels, output_labels)) { // identity
        return RecognizedOperatorType::Identity;
      }
      else {
        return RecognizedOperatorType::Transpose;
      }
    }
    else if (output_labels.empty()) { // scalar output, reduce
      return RecognizedOperatorType::ReduceSum;
    }

  }
  else if (m_components.size() == 3) { // two inputs
    auto input_A_labels = m_components[0].GetLabels(m_label_indices);
    auto input_B_labels = m_components[1].GetLabels(m_label_indices);
    auto output_labels = m_components[2].GetLabels(m_label_indices);
    if (equals(input_A_labels, output_labels) && equals(input_B_labels, output_labels)) { // element-wise product
      return RecognizedOperatorType::Multiply;
    }
  }

  const RecognizedOperatorInfo recognized_operators[] = {
    {RecognizedOperatorType::MatMul,                {2,2,2},{0,1, 1,2, 0,2}}, // ik,kj->ij
    {RecognizedOperatorType::MatMul,                {3,3,3},{0,1,2, 0,2,3, 0,1,3}}, // bik,bkj->bij
    {RecognizedOperatorType::MatMul,                {4,4,4},{0,1,2,3, 0,1,3,4, 0,1,2,4}}, // abik,abkj->abij
    {RecognizedOperatorType::OuterProduct,          {1,1,2},{0, 1, 0,1}}, // i,j->ij
    {RecognizedOperatorType::MatMulTransposeA,      {2,2,2},{0,1, 0,2, 1,2}}, // ji,jk->ik
    {RecognizedOperatorType::MatMulTransposeA,      {3,3,3},{0,1,2, 0,1,3, 0,2,3}}, // bji,bjk->bik
    {RecognizedOperatorType::MatMulTransposeA,      {4,4,4},{0,1,2,3, 0,1,2,4, 0,1,3,4}}, // abji,abjk->abik
    {RecognizedOperatorType::MatMulTransposeB,      {2,2,2},{0,1, 2,1, 0,2}}, // ij,kj->ik
    {RecognizedOperatorType::MatMulTransposeB,      {3,3,3},{0,1,2, 0,3,2, 0,1,3}}, // bij,bkj->bik
    {RecognizedOperatorType::MatMulTransposeB,      {4,4,4},{0,1,2,3, 0,1,4,3, 0,1,2,4}}, // abij,abkj->abik
    {RecognizedOperatorType::MatMulNhcw,            {4,4,4},{0,1,2,3, 0,3,2,4, 0,1,2,4}}, // aibj,ajbk->aibk
    {RecognizedOperatorType::MatMulNhcwTransposeA,  {4,4,4},{0,1,2,3, 0,1,2,4, 0,3,2,4}}, // ajbi,ajbk->aibk
    {RecognizedOperatorType::MatMulNhcwTransposeB,  {4,4,4},{0,1,2,3, 0,4,2,3, 0,1,2,4}}, // aibj,akbj->aibk
    {RecognizedOperatorType::ReduceSum,             {2,1  },{0,1, 0}}, //ij->i
    {RecognizedOperatorType::ReduceSum,             {2,1  },{0,1, 1}}, //ij->j
  };


  for (auto& recognized_operator: recognized_operators) {
    if (equals(m_label_indices, as_span(recognized_operator.label_indices))
    && m_components.size() == recognized_operator.component_ranks.size()) {
      for (size_t i = 0; i < m_components.size(); ++i) {
        component_ranks[i] = m_components[i].GetDimensionCount();
      }

      if (equals(gsl::make_span(component_ranks.data(), m_components.size()), as_span(recognized_operator.component_ranks))) {
        return recognized_operator.recognized_operator_type;
      }
    }
  }

  return RecognizedOperatorType::None;
}


// Add operator related.

void EinsumOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());
}

Status EinsumOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                              const Node& node,
                                              const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& initializers(model_builder.GetInitializerTensors());
  emscripten::val output = emscripten::val::object();


  NodeAttrHelper helper(node);
  const auto equation = helper.Get("equation", std::string(" "));

  std::vector<uint32_t> m_label_indices;
  std::vector<Component> m_components;
  std::vector<uint32_t> m_output_dimensions;
  uint32_t num_labels;
  ORT_RETURN_IF_NOT(ParseEquationComponents(initializers, node, equation, m_label_indices,
      m_components, m_output_dimensions, num_labels, logger), "Error parsing equation components.");

  if (m_components.size() == 2) { // one input
    auto input_labels = m_components[0].GetLabels(m_label_indices);
    auto output_labels = m_components[1].GetLabels(m_label_indices);
    if (input_labels.size() == output_labels.size()) {
      if (equals(input_labels, output_labels)) { // identity
        return RecognizedOperatorType::Identity;
      }
      else {
        return RecognizedOperatorType::Transpose;
      }
    }
    else if (output_labels.empty()) { // scalar output, reduce
      return RecognizedOperatorType::ReduceSum;
    }

  }
  else if (m_components.size() == 3) { // two inputs
    auto input_A_labels = m_components[0].GetLabels(m_label_indices);
    auto input_B_labels = m_components[1].GetLabels(m_label_indices);
    auto output_labels = m_components[2].GetLabels(m_label_indices);

  }


  RecognizedOperatorType recognized_operator_type = DetermineRecognizedOperatorType(m_label_indices, m_components, m_output_dimensions);

  switch(recognized_operator_type) {
    case RecognizedOperatorType::Multiply: {
      const size_t a_idx = 0, b_idx = 1;
      emscripten::val a = model_builder.GetOperand(node.InputDefs()[a_idx]->Name());
      emscripten::val b = model_builder.GetOperand(node.InputDefs()[b_idx]->Name());
      output = model_builder.GetBuilder().call<emscripten::val>("mul", a, b);
    }
    break;

    case RecognizedOperatorType::OuterProduct: {
      const size_t a_idx = 0, b_idx = 1;
      emscripten::val a = model_builder.GetOperand(node.InputDefs()[a_idx]->Name());
      emscripten::val b = model_builder.GetOperand(node.InputDefs()[b_idx]->Name());

      std::vector<int64_t> a_shape, b_shape;
      ORT_RETURN_IF_NOT(GetShape(*input_defs[0], a_shape, logger), "Cannot get shape");
      ORT_RETURN_IF_NOT(GetShape(*input_defs[1], b_shape, logger), "Cannot get shape");


      std::vector<int64_t> new_a_shape = a_shape;
      new_a_shape.push_back(static_cast<uint32_t>(1));
      std::vector<int64_t> new_b_shape = b_shape;
      new_b_shape.insert(new_b_shape.begin(), static_cast<uint32_t>(1));

      emscripten::val new_a = model_builder.GetBuilder().call<emscripten::val>("reshape",
                  a, emscripten::val::array(new_a_shape));
      emscripten::val new_b = model_builder.GetBuilder().call<emscripten::val>("reshape",
                  b, emscripten::val::array(new_b_shape));

      emscripten::val options = emscripten::val::object();

      output = model_builder.GetBuilder().call<emscripten::val>("gemm", a, b);
    }
    break;

    case RecognizedOperatorType::MatMulTransposeA:
    case RecognizedOperatorType::MatMulTransposeB:
    case RecognizedOperatorType::MatMul: {
      const size_t a_idx = 0, b_idx = 1;
      emscripten::val a = model_builder.GetOperand(node.InputDefs()[a_idx]->Name());
      emscripten::val b = model_builder.GetOperand(node.InputDefs()[b_idx]->Name());

      if (recognized_operator_type == RecognizedOperatorType::MatMulTransposeA) {
        std::vector<int64_t> input_shape;
        ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
        auto input_dims = static_cast<int64_t>(input_shape.size());

        std::vector<uint32_t> permutation;

        for (uint32_t i = 0; i < input_dims-2; ++i)
          permutation.push_back(i);

        permutation.push_back(static_cast<uint32_t>(input_dims-1));
        permutation.push_back(static_cast<uint32_t>(input_dims-2));

        emscripten::val options = emscripten::val::object();
        options.set("permutation", emscripten::val::array(permutation));
        a = model_builder.GetBuilder().call<emscripten::val>("transpose", a, options);
      }
      else if (recognized_operator_type == RecognizedOperatorType::MatMulTransposeB) {
        std::vector<int64_t> input_shape;
        ORT_RETURN_IF_NOT(GetShape(*input_defs[1], input_shape, logger), "Cannot get shape");
        auto input_dims = input_shape.size();

        std::vector<int64_t> permutation;

        for (int64_t i = 0; i < input_dims-2; ++i)
          permutation.push_back(i);

        permutation.push_back(input_dims-1);
        permutation.push_back(input_dims-2);

        emscripten::val options = emscripten::val::object();
        options.set("permutation", emscripten::val::array(permutation));
        b = model_builder.GetBuilder().call<emscripten::val>("transpose", b, options);
      }

      output = model_builder.GetBuilder().call<emscripten::val>("matmul", a, b);
    }
    break;

    case RecognizedOperatorType::MatMulNhcw:
    case RecognizedOperatorType::MatMulNhcwTransposeA:
    case RecognizedOperatorType::MatMulNhcwTransposeB: {
      const size_t a_idx = 0, b_idx = 1;
      emscripten::val a = model_builder.GetOperand(node.InputDefs()[a_idx]->Name());
      emscripten::val b = model_builder.GetOperand(node.InputDefs()[b_idx]->Name());

      emscripten::val options = emscripten::val::object();
      std::vector<int64_t> permutation = {0,2,1,3};
      std::vector<int64_t> permutation_a = {0,2,1,3};
      std::vector<int64_t> permutation_b = {0,2,1,3};
      if (recognized_operator_type == RecognizedOperatorType::MatMulNhcwTransposeA) {
        permutation_a = {0,2,3,1};
      }
      else if (recognized_operator_type == RecognizedOperatorType::MatMulNhcwTransposeB) {
        permutation_b = {0,2,3,1};
      }

      options.set("permutation", emscripten::val::array(permutation_a));
      a = model_builder.GetBuilder().call<emscripten::val>("transpose", a, options);
      options.set("permutation", emscripten::val::array(permutation_b));
      b = model_builder.GetBuilder().call<emscripten::val>("transpose", b, options);
      output = model_builder.GetBuilder().call<emscripten::val>("matmul", a, b);

      options.set("permutation", emscripten::val::array(permutation));
      output = model_builder.GetBuilder().call<emscripten::val>("transpose", output, options);
    }
    break;

    case RecognizedOperatorType::ReduceSum: {
      auto kept_axes = m_components.back().GetLabels(m_label_indices);
      assert(kept_axes.size() <= 1);
      std::vector<uint32_t> reduced_axes;
      uint32_t kept_axes_mask = 0;
      for (auto axis : kept_axes) {
          kept_axes_mask |= (1 << axis);
      }
      std::vector<int64_t> input_shape;
      ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
      for (uint32_t axis = 0, axis_count = static_cast<uint32_t>(input_shape.size()); axis < axis_count; ++axis) {
          if (~kept_axes_mask & (1<<axis)) {
              reduced_axes.push_back(axis);
          }
      }

      emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
      emscripten::val options = emscripten::val::object();
      options.set("keepDimensions", false);

      const auto input_rank = input_shape.size();
      std::vector<int32_t> axes_data;
      std::transform(
          reduced_axes.begin(), reduced_axes.end(), std::back_inserter(axes_data),
          [input_rank](int64_t axis) -> int32_t { return SafeInt<int32_t>(HandleNegativeAxis(axis, input_rank)); });
      options.set("axes", emscripten::val::array(axes_data));

      output = model_builder.GetBuilder().call<emscripten::val>("reduceSum", input, options);
    }
    break;

    case RecognizedOperatorType::Transpose: {
      emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
      // Transpose via input strides. The output tensor is not strided.
      assert(m_components.front().GetDimensionCount() == m_components.back().GetDimensionCount());
      // Remap transposed strides using the component labels from input to output.
      auto label_indices = m_components.back().GetLabels(m_label_indices);

      std::vector<uint32_t> permutation{label_indices.begin(), label_indices.end()};
      emscripten::val options = emscripten::val::object();
      options.set("permutation", emscripten::val::array(permutation));
      output = model_builder.GetBuilder().call<emscripten::val>("transpose", input, options);
    }
    break;
    case RecognizedOperatorType::Identity: {
      emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
      output = model_builder.GetBuilder().call<emscripten::val>("identity", input);
    }
    break;
    default:
    break;
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool EinsumOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                        const Node& node,
                                        const WebnnDeviceType device_type,
                                        const logging::Logger& logger) const {


  emscripten::val console = emscripten::val::global("console");
  console.call<void>("log", emscripten::val("log from Einsum..."));
  const auto& input_defs = node.InputDefs();

  NodeAttrHelper helper(node);
  const auto equation = helper.Get("equation", std::string(" "));
  std::vector<uint32_t> m_label_indices;
  std::vector<Component> m_components;
  std::vector<uint32_t> m_output_dimensions;
  uint32_t num_labels;

  if (!ParseEquationComponents(initializers, node, equation, m_label_indices,
      m_components, m_output_dimensions, num_labels, logger)) {

    LOGS(logger, VERBOSE) << "EinSum input equation is illegal.";
    return false;
  }

  if (static_cast<uint32_t>(input_defs.size()) + 1 != m_components.size()) {
    LOGS(logger, VERBOSE) << "EinSum input tensor count is inconsistent with the equation component count.";
    return false;
  }

  RecognizedOperatorType recognized_operator_type = DetermineRecognizedOperatorType(m_label_indices, m_components, m_output_dimensions);
  if (recognized_operator_type == RecognizedOperatorType::None) {
    LOGS(logger, VERBOSE) << "The equation is not supported in Einsum.";
    return false;
  }

  if (recognized_operator_type == RecognizedOperatorType::ReduceSum && device_type == WebnnDeviceType::CPU) {
    LOGS(logger, VERBOSE) << "Einsum is not supported for cpu in WebNN EP. ReduceSum is not supported in XNNPACK.";
    return false;
  }

  return true;
}

void CreateEinsumOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<EinsumOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
