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
                std::unordered_map<Operation, std::pair<Operation, std::vector<int>>>* transformed)
      : transformed_(transformed) {}

  Expr Mutate_(const Call* call, const Expr& e) {
    if (call->call_type == Call::CallType::Halide) {
      if (const ComputeOpNode* op = call->func.as<ComputeOpNode>()) {
        auto callee_it = transformed_->find(GetRef<Operation>(op));
        if (callee_it != transformed_->end()) {
          if (callee_it->second.first != callee_it->first ||
              callee_it->second.second.size() != call->args.size()) {
            Array<Expr> new_args;
            for (int index : callee_it->second.second) {
              new_args.push_back(Mutate(call->args[index]));
            }
            Operation new_callee = callee_it->second.first;
            return Call::make(call->type, call->name, new_args, call->call_type,
                              new_callee, call->value_index);
          }
        }
      }
    }

    // Try to transform the arguments
    return IRMutator::Mutate_(call, e);
  }

 private:
  std::unordered_map<Operation, std::pair<Operation, std::vector<int>>>* transformed_;
};

void RemoveUnusedDimsRecursivelyImpl(
        Operation oper,
        std::unordered_map<Operation, Operation>* semitransformed,
        std::unordered_map<Operation, std::pair<Operation, std::vector<int>>>* transformed) {
  if (semitransformed->count(oper)) {
    return;
  }

  for (const Tensor& subtensor : oper->InputTensors()) {
    RemoveUnusedDimsRecursivelyImpl(subtensor->op, semitransformed, transformed);
  }

  if (const ComputeOpNode* op = oper.as<ComputeOpNode>()) {
    RemoveUnusedDimsMutator mut(transformed);
    Array<Expr> new_body;
    bool changed = false;
    if (op->body[0].as<Reduce>()) {
      Expr e = mut.Mutate(op->body[0]);
      changed = changed || !e.same_as(op->body[0]);
      new_body.push_back(e);
    } else {
      for (const Expr& b : op->body) {
        Expr e = mut.Mutate(b);
        changed = changed || !e.same_as(b);
        new_body.push_back(e);
      }
    }

    if (!changed) {
      // If the body didn't change then we can use the same operation
      (*semitransformed)[oper] = oper;
    } else {
      (*semitransformed)[oper] =
        op::ComputeOpFromExprs(new_body, op->axis, op->name, op->tag, op->attrs);
    }

    Array<IterVar> new_axis;
    std::vector<int> retained_dims;
    for (size_t i = 0; i < op->axis.size(); ++i) {
      bool used = false;
      for (const Expr e : new_body) {
        used = used || ExprUseVar(e, op->axis[i]->var);
      }
      if (used) {
        new_axis.push_back(op->axis[i]);
        retained_dims.push_back(i);
      }
    }

    if (!changed && new_axis.size() == op->axis.size()) {
      (*transformed)[oper] = make_pair(oper, retained_dims);
    } else {
      (*transformed)[oper] =
        make_pair(op::ComputeOpFromExprs(new_body, new_axis, op->name, op->tag, op->attrs),
                  retained_dims);
    }
  } else {
    std::unordered_map<Tensor, Tensor> subst;
    for (const Tensor& inp : oper->InputTensors()) {
      if (semitransformed->count(inp->op)) {
        subst[inp] = (*semitransformed)[inp->op].output(inp->value_index);
      }
    }
    Operation new_op = oper->ReplaceInputs(oper, subst);
    if (new_op.same_as(oper)) {
      (*semitransformed)[oper] = oper;
    } else {
      (*semitransformed)[oper] = new_op;
    }
  }
}

Array<Tensor> RemoveUnusedDimsRecursively(const Array<Tensor> tensors) {
  std::unordered_map<Operation, Operation> semitransformed;
  std::unordered_map<Operation, std::pair<Operation, std::vector<int>>> transformed;
  Array<Tensor> res;
  for (const Tensor& t : tensors) {
    RemoveUnusedDimsRecursivelyImpl(t->op, &semitransformed, &transformed);
    res.push_back(semitransformed[t->op].output(t->value_index));
  }
  return res;
}

}  // namespace ir
}  // namespace tvm
