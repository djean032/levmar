#pragma once

#include <cstddef>
#include <expected>
#include <mdspan>
#include <span>
#include <string>

namespace levmar {

using Index = std::size_t;

enum class ErrorCode { InvalidProblem, UserFunctionError, NumericalFailure };

struct Error {
  ErrorCode code;
  std::string message;
};

template <class T> using ErrorOr = std::expected<T, Error>;

using ErrorOrVoid = ErrorOr<void>;

struct NoJacobian {};

enum class TerminationReason {
  NotTerminated,
  SmallStep,
  SmallGradient,
  SmallCostReduction,
  MaxIterations,
  MaxFunctionEvaluations,
  NumericalFailure,
  TrustRegionTooSmall,
  DampingLimit
};

template <Index N, class Scalar = double>
using VectorView = std::span<Scalar, N>;

template <Index N, class Scalar = double>
using ConstVectorView = std::span<const Scalar, N>;

template <Index M, Index N, class Scalar = double>
using MatrixView =
    std::mdspan<Scalar, std::extents<Index, M, N>, std::layout_left>;

template <Index M, Index N, class Scalar = double>
using ConstMatrixView =
    std::mdspan<const Scalar, std::extents<Index, M, N>, std::layout_left>;

} // namespace levmar
