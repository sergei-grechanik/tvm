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
 * \brief Utility to make loop nest.
 * \file op_util.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>
#include <tvm/ir_mutator.h>
#include <string>
#include "op_util.h"
#include "../schedule/message_passing.h"
#include "../arithmetic/compute_expr.h"

namespace tvm {
namespace op {

using namespace arith;
using namespace ir;

std::vector<std::vector<Stmt> >
MakeLoopNest(const Stage& stage,
             const std::unordered_map<IterVar, Range>& dom_map,
             size_t begin_iter_pos,
             bool new_loop_var,
             const std::unordered_set<IterVar>& skip_iter,
             std::unordered_map<IterVar, Expr>* p_value_map,
             bool debug_keep_trivial_loop) {
  auto leaf_iter_vars = stage->leaf_iter_vars;
  Stmt no_op = Evaluate::make(0);
  // create the loop nest
  std::vector<std::vector<Stmt> > nest;
  nest.resize(leaf_iter_vars.size() + 1);
  std::unordered_map<IterVar, Expr>& value_map = *p_value_map;

  for (size_t i = begin_iter_pos; i < leaf_iter_vars.size(); ++i) {
    auto iv = leaf_iter_vars[i];
    if (skip_iter.count(iv) || iv->iter_type == kOpaque) {
      // skip this iteration.
      value_map[iv] = iv->var;
      continue;
    }
    // Bind iv could be another thread.
    IterVar bind_iv = iv;
    if (stage->iter_var_attrs.count(iv)) {
      IterVar bind_thread = stage->iter_var_attrs[iv]->bind_thread;
      if (bind_thread.defined()) bind_iv = bind_thread;
    }

    Range dom = dom_map.at(iv);

    // initialize the offset and loop_level
    Var var = bind_iv->var;

    // Mark the iter var in the IR, to remember the point
    if (bind_iv->thread_tag.length() == 0) {
      // Only generate new loop if we're not bound to a thread.
      if (new_loop_var) {
        var = Var(iv->var->name_hint + ".init", bind_iv->var.type());
      }

      ForType for_type = ForType::Serial;
      IterVarAttr it_attr;
      if (stage->iter_var_attrs.count(iv)) {
        it_attr = stage->iter_var_attrs[iv];
      }
      if (it_attr.defined()) {
        switch (it_attr->iter_type) {
          case kUnrolled: for_type = ForType::Unrolled; break;
          case kVectorized: for_type = ForType::Vectorized; break;
          case kParallelized: for_type = ForType::Parallel; break;
          case kDataPar: break;
          case kTensorized: break;
          default: LOG(FATAL) << "Unknown iter type"
                              << it_attr->iter_type
                              << " in the iter_var_attrs";
        }
        CHECK_EQ(it_attr->pragma_keys.size(), it_attr->pragma_values.size());
        for (size_t k = 0; k < it_attr->pragma_keys.size(); ++k) {
          const std::string& pkey = it_attr->pragma_keys[k].as<StringImm>()->value;
          Expr pvalue = it_attr->pragma_values[k];
          if (!pvalue.defined()) {
            pvalue = make_const(Int(32), 1);
          }
          nest[i + 1].emplace_back(
              AttrStmt::make(iv, ir::attr::pragma_scope_prefix + pkey, pvalue, no_op));
        }
      }
      if (!debug_keep_trivial_loop && is_one(dom->extent)) {
        nest[i + 1].emplace_back(
            LetStmt::make(var, dom->min, no_op));
        value_map[iv] = dom->min;
      } else if (is_zero(dom->min)) {
        nest[i + 1].emplace_back(
            For::make(var, 0, dom->extent,
                      for_type, DeviceAPI::None, no_op));
        value_map[iv] = var;
      } else {
        Var idx(bind_iv->var->name_hint + ".idx", bind_iv->var.type());
        nest[i + 1].emplace_back(
            For::make(idx, 0, dom->extent,
                      for_type, DeviceAPI::None, no_op));
        Expr new_value = dom->min + idx;
        value_map[iv] = new_value;
        nest[i + 1].emplace_back(
            LetStmt::make(var, new_value, no_op));
      }
      if (it_attr.defined() && it_attr->prefetch_data.size() != 0) {
        CHECK(!is_one(dom->extent))
            << "Cannot prefetch on trivial loop with extent=1";
        CHECK_EQ(it_attr->prefetch_data.size(),
                 it_attr->prefetch_offset.size());
        for (size_t j = 0; j < it_attr->prefetch_data.size(); ++j) {
          nest[i + 1].emplace_back(
              AttrStmt::make(it_attr->prefetch_data[j],
                             ir::attr::prefetch_scope,
                             it_attr->prefetch_offset[j], no_op));
        }
      }
    } else if (bind_iv->thread_tag == "vthread" ||
               bind_iv->thread_tag == "cthread") {
      // virtual thread
      // Always restrict threaded IterVar to starts from 0.
      CHECK(is_zero(dom->min));
      CHECK(is_positive_const(dom->extent));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmt::make(bind_iv, ir::attr::virtual_thread, dom->extent, no_op));
      value_map[iv] = var;
    } else if (bind_iv->thread_tag == "pipeline") {
      // pipeline marker.
      CHECK(is_zero(dom->min));
      CHECK(is_one(dom->extent));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmt::make(bind_iv, ir::attr::pipeline_exec_scope, dom->extent, no_op));
      value_map[iv] = dom->min;
    } else {
      // Always restrict threaded IterVar to starts from 0.
      CHECK(is_zero(dom->min));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmt::make(bind_iv, ir::attr::thread_extent, dom->extent, no_op));
      if (!debug_keep_trivial_loop && is_one(dom->extent)) {
        value_map[iv] = dom->min;
      } else {
        value_map[iv] = var;
      }
    }
    // annotate the extent of the IterVar
    if (!new_loop_var) {
      nest[i + 1].emplace_back(
          AttrStmt::make(iv, attr::loop_scope, iv->var, no_op));
    }
  }
  // message passing to get offset of root iter vars.
  schedule::PassUpIndex(stage, dom_map, &value_map);
  return nest;
}

std::vector<Stmt> MakeIfNest(const std::vector<Expr>& predicates) {
  Stmt no_op = Evaluate::make(0);
  std::vector<Stmt> nest;
  for (const Expr& cond : predicates) {
    nest.emplace_back(IfThenElse::make(cond, no_op));
  }
  return nest;
}

// replacer to replace tensors
class TensorReplacer : public ir::IRMutator {
 public:
  explicit TensorReplacer(const std::unordered_map<Tensor, Tensor>& vmap)
      : vmap_(vmap) {}

  Expr Mutate_(const ir::Call* op, const Expr& e) {
    if (op->call_type == ir::Call::Halide) {
      Tensor t = Downcast<Operation>(op->func).output(op->value_index);
      auto it = vmap_.find(t);
      if (it != vmap_.end()) {
        Expr ret = ir::Call::make(
            op->type, it->second->op->name, op->args,
            op->call_type, it->second->op, it->second->value_index);
        found = true;
        return IRMutator::Mutate_(ret.as<ir::Call>(), ret);
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  // whether it is found.
  bool found{false};

 private:
  const std::unordered_map<Tensor, Tensor>& vmap_;
};

Stmt ReplaceTensor(Stmt stmt,
                   const std::unordered_map<Tensor, Tensor>& replace) {
  TensorReplacer repl(replace);
  Stmt ret = repl.Mutate(stmt);
  return repl.found ? ret : stmt;
}
Expr ReplaceTensor(Expr expr,
                   const std::unordered_map<Tensor, Tensor>& replace) {
  TensorReplacer repl(replace);
  Expr ret = repl.Mutate(expr);
  return repl.found ? ret : expr;
}


void ReplaceTensorRecursivelyImpl(Tensor tensor,
                                  std::unordered_map<Tensor, Tensor>* replace) {
  if (!replace->count(tensor)) {
    for (const Tensor& subtensor : tensor->op->InputTensors()) {
      ReplaceTensorRecursivelyImpl(subtensor, replace);
    }
    Operation new_op = tensor->op->ReplaceInputs(tensor->op, *replace);
    if (new_op.same_as(tensor->op)) {
      (*replace)[tensor] = tensor;
    } else {
      (*replace)[tensor] =
        TensorNode::make(tensor->shape, tensor->dtype, new_op, tensor->value_index);
    }
  }
}

Array<Tensor> ReplaceTensorRecursively(Array<Tensor> tensors,
                                       const std::unordered_map<Tensor, Tensor>& replace) {
  auto new_replace = replace;
  Array<Tensor> res;
  for (const Tensor& t : tensors) {
    ReplaceTensorRecursivelyImpl(t, &new_replace);
    res.push_back(new_replace[t]);
  }
  return res;
}


Stmt Substitute(Stmt s,
                const std::unordered_map<IterVar, Expr>& value_map) {
  std::unordered_map<const Variable*, Expr> init;
  for (const auto& kv : value_map) {
    init[kv.first->var.get()] = kv.second;
  }
  return ir::Substitute(s, init);
}

IterVarType ForTypeToIterVarType(ir::ForType for_type) {
  switch (for_type) {
  case ForType::Serial:
    return kDataPar;
  case ForType::Parallel:
    return kParallelized;
  case ForType::Vectorized:
    return kVectorized;
  case ForType::Unrolled:
    return kUnrolled;
  default:
    return kDataPar;
  }
}

ir::ForType IterVarTypeToForType(IterVarType iter_type) {
  switch (iter_type) {
  case kDataPar:
    return ForType::Serial;
  case kParallelized:
    return ForType::Parallel;
  case kVectorized:
    return ForType::Vectorized;
  case kUnrolled:
    return ForType::Unrolled;
  default:
    return ForType::Serial;
  }
}

Expr CloneReduction(const Expr& expr) {
  if (const Reduce* red = expr.as<Reduce>()) {
    Array<IterVar> new_axis;
    Map<Var, Expr> vmap;
    std::tie(new_axis, vmap) = CloneIterVars(red->axis);

    Array<Expr> src_with_newaxis;
    for (const auto& src : red->source) {
      src_with_newaxis.push_back(ir::Substitute(src, vmap));
    }

    return Reduce::make(red->combiner, src_with_newaxis,
        new_axis, ir::Substitute(red->condition, vmap), red->value_index);
  } else {
    return expr;
  }
}

Operation ComputeOpFromExprs(const Array<Expr>& exprs, const Array<IterVar>& axis,
                             const std::string& name, const std::string& tag,
                             const Map<std::string, NodeRef>& attrs,
                             bool clone_axis) {
  if (clone_axis) {
    Array<IterVar> new_axis = axis;
    Map<Var, Expr> vmap;
    std::tie(new_axis, vmap) = CloneIterVars(axis);
    Array<Expr> new_exprs;
    for (const Expr& e : exprs) {
      new_exprs.push_back(ir::Substitute(CloneReduction(e), vmap));
    }
    return ComputeOpFromExprs(new_exprs, new_axis, name, tag, attrs, false);
  }

  Array<Expr> new_exprs;

  // If this is a reduction then we have to replicate it
  if (const Reduce* red = exprs[0].as<Reduce>()) {
    for (size_t i = 0; i < red->source.size(); ++i) {
      Expr ith_red = Reduce::make(red->combiner, red->source, red->axis, red->condition, i);
      new_exprs.push_back(ith_red);
    }
  } else {
    new_exprs = exprs;
  }

  return ComputeOpNode::make(name, tag, attrs, axis, new_exprs);
}

Tensor TensorFromExpr(const Expr& expr, const Array<IterVar>& axis,
                      const std::string& name, const std::string& tag,
                      const Map<std::string, NodeRef>& attrs,
                      bool clone_axis) {
  int new_value_index = 0;
  if (const Reduce* red = expr.as<Reduce>()) {
    new_value_index = red->value_index;
  }
  return ComputeOpFromExprs({expr}, axis, name, tag, attrs, clone_axis).output(new_value_index);
}

Tensor TransformBody(const Tensor& tensor,
                     std::function<Expr(const Expr&, const Array<IterVar>&)> func) {
  if (const ComputeOpNode* op = tensor->op.as<ComputeOpNode>()) {
    // Transform only one body
    Expr new_body = func(op->body[tensor->value_index], op->axis);

    // If the body didn't change then we can return the same tensor
    if (new_body.same_as(op->body[tensor->value_index])) {
      return tensor;
    }

    return TensorFromExpr(new_body, op->axis, op->name, op->tag, op->attrs);
  } else {
    return tensor;
  }
}

Tensor TransformBody(const Tensor& tensor, std::function<Expr(const Expr&)> func) {
  return TransformBody(tensor, [func](const Expr& e, const Array<IterVar>&) { return func(e); });
}

}  // namespace op
}  // namespace tvm
