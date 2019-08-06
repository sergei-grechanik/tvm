/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2019 by Contributors
 * \file remove_unused_dims.cc
 */
// Remove unused dimensions from a DAG of tensors.

#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>
#include "../op/op_util.h"
#include <vector>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace ir {

class RemoveUnusedDimsMutator : public IRMutator {
 public:
  explicit RemoveUnusedDimsMutator(
                std::unordered_map<Tensor, std::pair<Tensor, std::vector<int>>>* transformed)
      : transformed_(transformed) {}

  Expr Mutate_(const Call* call, const Expr& e) {
    if (call->call_type == Call::CallType::Halide) {
      if (const ComputeOpNode* op = call->func.as<ComputeOpNode>()) {
        Tensor callee = GetRef<Operation>(op).output(call->value_index);
        auto callee_it = transformed_->find(callee);
        if (callee_it != transformed_->end()) {
          if (callee_it->second.first != callee_it->first ||
              callee_it->second.second.size() != call->args.size()) {
            Array<Expr> new_args;
            for (int index : callee_it->second.second) {
              new_args.push_back(Mutate(call->args[index]));
            }
            Tensor new_callee = callee_it->second.first;
            return Call::make(call->type, call->name, new_args, call->call_type,
                              new_callee->op, new_callee->value_index);
          }
        }
      }
    }

    // Try to transform the arguments
    return IRMutator::Mutate_(call, e);
  }

 private:
  std::unordered_map<Tensor, std::pair<Tensor, std::vector<int>>>* transformed_;
};

void RemoveUnusedDimsRecursivelyImpl(
        Tensor tensor,
        std::unordered_map<Tensor, Tensor>* semitransformed,
        std::unordered_map<Tensor, std::pair<Tensor, std::vector<int>>>* transformed) {
  if (semitransformed->count(tensor)) {
    return;
  }

  for (const Tensor& subtensor : tensor->op->InputTensors()) {
    RemoveUnusedDimsRecursivelyImpl(subtensor, semitransformed, transformed);
  }

  if (const ComputeOpNode* op = tensor->op.as<ComputeOpNode>()) {
    RemoveUnusedDimsMutator mut(transformed);
    Expr new_body = mut.Mutate(op->body[tensor->value_index]);

    if (new_body.same_as(op->body[tensor->value_index])) {
      // If the body didn't change then we can use the same tensor
      (*semitransformed)[tensor] = tensor;
    } else {
      (*semitransformed)[tensor] =
        op::TensorFromExpr(new_body, op->axis, op->name, op->tag, op->attrs);
    }

    Array<IterVar> new_axis;
    std::vector<int> retained_dims;
    for (size_t i = 0; i < op->axis.size(); ++i) {
      if (ExprUseVar(new_body, op->axis[i]->var)) {
        new_axis.push_back(op->axis[i]);
        retained_dims.push_back(i);
      }
    }

    if (new_body.same_as(op->body[tensor->value_index]) && new_axis.size() == op->axis.size()) {
      (*transformed)[tensor] = make_pair(tensor, retained_dims);
    } else {
      (*transformed)[tensor] =
        make_pair(op::TensorFromExpr(new_body, new_axis, op->name, op->tag, op->attrs),
                  retained_dims);
    }
  } else {
    Operation new_op = tensor->op->ReplaceInputs(tensor->op, *semitransformed);
    if (new_op.same_as(tensor->op)) {
      (*semitransformed)[tensor] = tensor;
    } else {
      (*semitransformed)[tensor] =
        TensorNode::make(tensor->shape, tensor->dtype, new_op, tensor->value_index);
    }
  }
}

Array<Tensor> RemoveUnusedDimsRecursively(const Array<Tensor> tensors) {
  std::unordered_map<Tensor, Tensor> semitransformed;
  std::unordered_map<Tensor, std::pair<Tensor, std::vector<int>>> transformed;
  Array<Tensor> res;
  for (const Tensor& t : tensors) {
    RemoveUnusedDimsRecursivelyImpl(t, &semitransformed, &transformed);
    res.push_back(semitransformed[t]);
  }
  return res;
}

}  // namespace ir
}  // namespace tvm
