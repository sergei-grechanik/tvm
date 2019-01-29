/*!
 *  Copyright (c) 2018 by Contributors
 * \file zero_elimination.h
 * \brief Transform tensors in such a way as to eliminate summation over zeros.
 */
#ifndef TVM_PASS_ZERO_ELIMINATION_H_
#define TVM_PASS_ZERO_ELIMINATION_H_

#include <tvm/ir.h>
#include <tvm/tensor.h>

#include <string>

namespace tvm {
namespace ir {

/*!
 * \brief Clone the reduction by cloning its iteration variables.
 */
Expr CloneReduction(const Expr& expr);

/*!
 * \brief Check if the given combiner represents summation.
 */
EXPORT bool IsSumCombiner(const CommReducer& combiner);

/*!
 * \brief Check if zero may be factored out of a reduction with this combiner when it is in
 *  the \p value_index position.
 *
 *  For example, if the combiner works on tuples of two elements and `value_index = 1`,
 *  check that `(a, 0) combine (b, 0) = (c, 0)` for any a, b and some c.
 *  Note that all combiners generated by autodiff have this property.
 */
EXPORT bool CanFactorZeroFromCombiner(const CommReducer& combiner, int value_index);

/*!
 * \brief Transform the expression into `c ? e : 0`, that is lift the condition of being
 *  possible to be non-zero to the top level.
 */
EXPORT Expr LiftNonzeronessCondition(const Expr& expr);

/*!
 * \brief If the body of the tensor consists of a single tensor call (indexing) expression,
 *  inline it.
 */
EXPORT Tensor InlineTailCall(const Tensor& tensor);

/*!
 * \brief Inline tensors recursively.
 *
 *  This function will inline tensors recursively until it reaches a tensor which is impossible to
 *  inline (a reduction if \p inline_reductions is false, a non-compute tensor, a tensor which is
 *  not from \p inlineable). It won't descend into non-inlinable tensors' bodies.
 *
 * \param expr The expression to transform.
 * \param inlineable A list of tensors which are allowed to be inlined. If empty, try
 *  to inline all tensors.
 * \param inline_reductions Whether to inline reductions (this may result in top-level reduction
 *  nodes).
 */
EXPORT Expr InlineTensors(const Expr& expr,
                          const Array<Tensor>& inlineable = Array<Tensor>(),
                          bool inline_reductions = false);

/*!
 * \brief Inline tensors recursively.
 *
 *  This function will inline tensors recursively until it reaches a tensor which is impossible to
 *  inline (a reduction if \p inline_reductions is false, a non-compute tensor, a tensor which is
 *  not from \p inlineable). It won't descend into non-inlinable tensors' bodies.
 *
 * \param tensor The tensor whose body to transform.
 * \param inlineable A list of tensors which are allowed to be inlined. If empty, try
 *  to inline all tensors.
 * \param inline_reductions Whether to inline reductions (this may result in top-level reduction
 *  nodes).
 */
EXPORT Tensor InlineTensors(const Tensor& tensor,
                            const Array<Tensor>& inlineable = Array<Tensor>(),
                            bool inline_reductions = false);


/*!
 * \brief A struct representing a set of inequalities describing bounds of a variable.
 *
 *  Given a variable x, this struct represents the following (in)equalities:
 *  - `coef*x >= low` for each `low` in `lower`
 *  - `coef*x == eq` for each `eq` in `equal`
 *  - `coef*x <= upp` for each `upp` in `upper`
 *
 *  Note that every array is supposed to be sorted in the order of increasing expression
 *  complexity.
 */
struct VarBounds {
  Expr coef;
  Array<Expr> lower;
  Array<Expr> equal;
  Array<Expr> upper;

 /*!
  * \brief Perform substitution on all components of the struct.
  */
  VarBounds substitute(const Map<Var, Expr>& subst) const;
};

/*!
 * \brief A struct representing a system of inequalities resulted from Fourier-Motzkin elimination.
 */
struct SolveSystemOfInequalitiesResult {
  Array<Var> variables;
  std::unordered_map<const Variable*, VarBounds> bounds;
  Array<Expr> other_conditions;

  /*!
   * \brief Combine the information into an array of (in)equalities.
   */
  Array<Expr> as_conditions() const;
};

/*!
 * \brief Rewrite the system of inequalities using Fourier-Motzkin elimination.
 *
 *  This function takes an array of (in)equalities and an array of variables, and essentially
 *  rewrites the (in)equalities into an array of (in)equalities of the following form:
 *
 *      x0 >= f0(x1, x2, ..., xn)
 *      x0 <= g0(x1, x2, ..., xn)
 *      x1 >= f1(x2, ..., xn)
 *      x1 <= g1(x2, ..., xn)
 *      ...
 *      xn >= fn()  // just a constant
 *      xn <= gn()  // just a constant
 *
 *  This array is represented in a more structural way using SolveSystemOfInequalitiesResult.
 *
 *  Note that the algorithm is extremely slow, it is super-exponential, so please provide variable
 *  ranges to aid the removal of redundant inequalities.
 *
 * \param inequalities The original (in)equalities.
 * \param variables The variables x0, ..., xn
 * \param vranges A map from variables to the corresponding value ranges. Extremely important for
 *   efficiency.
 */
EXPORT SolveSystemOfInequalitiesResult SolveSystemOfInequalities(
    const Array<Expr>& inequalities, const Array<Var>& variables, const Map<Var, Range>& vranges);

/*!
 * \brief A struct representing a result of domain simplification. It is basically
 *  a new array of variables, the information about their ranges, and a new condition together with
 *  substitutions from the old variables to the new ones and from the new ones to the old ones.
 */
struct DomainSimplificationResult {
  Array<Expr> conditions;
  Array<Var> axis;
  Map<Var, Range> ranges;
  Map<Var, Expr> old_to_new;
  Map<Var, Expr> new_to_old;
};

/*!
 * \brief Simplify an iteration domain.
 *
 *  An iteration domain is basically an array of variables and a condition. The function will do the
 *  following:
 *  - Replace div and mod operations with new variables (optional).
 *  - Extract (in)equalities from the condition.
 *  - Perform Fourier-Motzkin elimination.
 *  - Shear the domain of iteration (e.g. if `y <= x <= y + 2` then x will be replaced with `y + d`
 *    where `d` is a new variable such that `0 <= d <= 2`).
 *  - Remove redundant variables.
 *  - Infer new variable ranges (hopefully more precise).
 *
 * \param cond The condition of the original domain.
 * \param axis The variables of the original domain.
 * \param vranges A map from variables (both domain and outer) to their value ranges.
 * \param eliminate_div_mod Whether to eliminate div and mod by introducing new variables.
 */
EXPORT DomainSimplificationResult SimplifyDomain(const Expr& cond,
                                                 const Array<Var>& axis,
                                                 Map<Var, Range> vranges,
                                                 bool eliminate_div_mod = true);


/*!
 * \brief Simplify the iteration domain of a reduction expression using SimplifyDomain.
 */
EXPORT Expr SimplifyReductionDomain(const Expr& expr, const Map<Var, Range>& outer_vranges);

/*!
 * \brief Extract the given expression under the given condition as a separate tensor if the volume
 *  of the extracted tensor will be less than the volume of the \p outer_axis.
 *
 * \param expr The expression to extract.
 * \param cond A condition which is assumed to be true.
 * \param outer_axis Some variables, usually input variables of the enclosing tensor.
 * \param vranges Information about ranges of variables.
 * \return Either a call to an extracted tensor or the original expression.
 */
EXPORT Expr ExtractAsTensorMaybe(const Expr& expr, const Expr& cond,
                                 const Array<Var>& outer_axis,
                                 const Map<Var, Range>& vranges);

/*!
 * \brief Extract reductions as separate tensors. This may be needed when non-top-level reductions
 *  are created.
 *
 * \param expr The expression from which to extract reductions.
 * \param vranges Information about ranges of variables.
 * \return An expression without non-top-level reductions.
 */
EXPORT Expr ExtractReductions(const Expr& expr, const Map<Var, Range>& vranges);

/*!
 * \brief Extract reductions as separate tensors, but if the expr itself is a reduction, leave it
 *  intact.
 *
 * \param expr The expression from which to extract reductions.
 * \param vranges Information about ranges of variables.
 * \return An expression without non-top-level reductions.
 */
EXPORT Expr ExtractNonTopReductions(const Expr& expr, const Map<Var, Range>& vranges);

/*!
 * \brief Perform lifting of conditions of being possible to be non-zero together with
 *  applying some transformations like simplifying the reduction domain. Works only with
 *  this particular tensor's body, i.e. doesn't perform inlining.
 */
EXPORT Tensor OptimizeAndLiftNonzeronessConditions(const Tensor& tensor);

/*!
 * \brief Pretty print the tensor with all its dependencies.
 */
EXPORT std::string PrintTensorRecursively(const Tensor& tensor);

/*!
 * \brief Pretty print the tensors with all their dependencies.
 */
EXPORT std::string PrintTensorsRecursively(const Array<Tensor>& tensor);

}  // namespace ir
}  // namespace tvm
#endif  // TVM_PASS_ZERO_ELIMINATION_H_