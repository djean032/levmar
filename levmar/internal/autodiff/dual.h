#pragma once

#include <array>
#include <cassert>
#include <cmath>

#include <levmar/internal/core.h>

struct DualEvaluationState {
  bool pow_domain_error = false;
};

template <Index N> struct Dual {
  double value = 0.0;
  std::array<double, N> derivatives{};
  DualEvaluationState *state = nullptr;

  constexpr Dual() = default;
  constexpr Dual(double value_) : value(value_) {}
};

template <Index N>
constexpr DualEvaluationState *propagate_state(const Dual<N> &a,
                                               const Dual<N> &b) {
  assert(a.state == nullptr || b.state == nullptr || a.state == b.state);
  return a.state != nullptr ? a.state : b.state;
}

template <Index N>
constexpr Dual<N> operator+(const Dual<N> &a, const Dual<N> &b) {
  Dual<N> result;
  result.value = a.value + b.value;
  result.state = propagate_state(a, b);

  for (Index j = 0; j < N; ++j) {
    result.derivatives[j] = a.derivatives[j] + b.derivatives[j];
  }

  return result;
}

template <Index N> constexpr Dual<N> operator+(const Dual<N> &a, double b) {
  return a + Dual<N>{b};
}

template <Index N> constexpr Dual<N> operator+(double a, const Dual<N> &b) {
  return Dual<N>{a} + b;
}

template <Index N>
constexpr Dual<N> operator-(const Dual<N> &a, const Dual<N> &b) {
  Dual<N> result;
  result.value = a.value - b.value;
  result.state = propagate_state(a, b);

  for (Index j = 0; j < N; ++j) {
    result.derivatives[j] = a.derivatives[j] - b.derivatives[j];
  }

  return result;
}

template <Index N> constexpr Dual<N> operator-(const Dual<N> &a, double b) {
  return a - Dual<N>{b};
}

template <Index N> constexpr Dual<N> operator-(double a, const Dual<N> &b) {
  return Dual<N>{a} - b;
}

template <Index N> constexpr Dual<N> operator-(const Dual<N> &a) {
  Dual<N> result;
  result.value = -a.value;
  result.state = a.state;

  for (Index j = 0; j < N; ++j) {
    result.derivatives[j] = -a.derivatives[j];
  }

  return result;
}

template <Index N>
constexpr Dual<N> operator*(const Dual<N> &a, const Dual<N> &b) {
  Dual<N> result;
  result.value = a.value * b.value;
  result.state = propagate_state(a, b);

  for (Index j = 0; j < N; ++j) {
    result.derivatives[j] =
        a.value * b.derivatives[j] + b.value * a.derivatives[j];
  }
  return result;
}

template <Index N> constexpr Dual<N> operator*(const Dual<N> &a, double b) {
  return a * Dual<N>{b};
}

template <Index N> constexpr Dual<N> operator*(double a, const Dual<N> &b) {
  return Dual<N>{a} * b;
}

template <Index N>
constexpr Dual<N> operator/(const Dual<N> &a, const Dual<N> &b) {
  Dual<N> result;
  result.value = a.value / b.value;
  result.state = propagate_state(a, b);

  auto tmp = b.value * b.value;
  for (Index j = 0; j < N; ++j) {
    result.derivatives[j] =
        (a.derivatives[j] * b.value - a.value * b.derivatives[j]) / tmp;
  }
  return result;
}

template <Index N> constexpr Dual<N> operator/(const Dual<N> &a, double b) {
  return a / Dual<N>{b};
}

template <Index N> constexpr Dual<N> operator/(double a, const Dual<N> &b) {
  return Dual<N>{a} / b;
}

template <Index N> constexpr Dual<N> exp(const Dual<N> &a) {
  Dual<N> result;
  result.value = std::exp(a.value);
  result.state = a.state;

  for (Index j = 0; j < N; ++j) {
    result.derivatives[j] = result.value * a.derivatives[j];
  }
  return result;
}

template <Index N> constexpr Dual<N> atan2(const Dual<N> &a, const Dual<N> &b) {
  Dual<N> result;
  result.value = std::atan2(a.value, b.value);
  result.state = propagate_state(a, b);

  auto a_sq = a.value * a.value;
  auto b_sq = b.value * b.value;
  for (Index j = 0; j < N; ++j) {
    result.derivatives[j] =
        (b.value * a.derivatives[j] - b.derivatives[j] * a.value) /
        (b_sq + a_sq);
  }
  return result;
}

template <Index N> constexpr Dual<N> atan2(const Dual<N> &a, double b) {
  return atan2(a, Dual<N>{b});
}

template <Index N> constexpr Dual<N> atan2(double a, const Dual<N> &b) {
  return atan2(Dual<N>{a}, b);
}

template <Index N> constexpr Dual<N> log(const Dual<N> &a) {
  Dual<N> result;
  result.value = std::log(a.value);
  result.state = a.state;

  for (Index j = 0; j < N; ++j) {
    result.derivatives[j] = a.derivatives[j] / a.value;
  }
  return result;
}

template <Index N> constexpr Dual<N> log1p(const Dual<N> &a) {
  Dual<N> result;
  result.value = std::log1p(a.value);
  result.state = a.state;

  for (Index j = 0; j < N; ++j) {
    result.derivatives[j] = a.derivatives[j] / (1 + a.value);
  }
  return result;
}

template <Index N> constexpr Dual<N> expm1(const Dual<N> &a) {
  Dual<N> result;
  result.value = std::expm1(a.value);
  result.state = a.state;

  auto tmp = std::exp(a.value);
  for (Index j = 0; j < N; ++j) {
    result.derivatives[j] = tmp * a.derivatives[j];
  }
  return result;
}

template <Index N> constexpr Dual<N> sin(const Dual<N> &a) {
  Dual<N> result;
  result.value = std::sin(a.value);
  result.state = a.state;

  auto tmp = std::cos(a.value);
  for (Index j = 0; j < N; ++j) {
    result.derivatives[j] = tmp * a.derivatives[j];
  }
  return result;
}

template <Index N> constexpr Dual<N> cos(const Dual<N> &a) {
  Dual<N> result;
  result.value = std::cos(a.value);
  result.state = a.state;

  auto tmp = -std::sin(a.value);
  for (Index j = 0; j < N; ++j) {
    result.derivatives[j] = tmp * a.derivatives[j];
  }
  return result;
}

template <Index N> constexpr Dual<N> tan(const Dual<N> &a) {
  Dual<N> result;
  result.value = std::tan(a.value);
  result.state = a.state;

  auto cosine = std::cos(a.value);
  auto tmp = cosine * cosine;
  for (Index j = 0; j < N; ++j) {
    result.derivatives[j] = a.derivatives[j] / tmp;
  }
  return result;
}

template <Index N> constexpr Dual<N> sqrt(const Dual<N> &a) {
  Dual<N> result;
  result.value = std::sqrt(a.value);
  result.state = a.state;

  for (Index j = 0; j < N; ++j) {
    result.derivatives[j] = a.derivatives[j] / (2 * result.value);
  }
  return result;
}

template <Index N> constexpr Dual<N> pow(const Dual<N> &a, const Dual<N> &b) {
  Dual<N> result;
  result.state = propagate_state(a, b);
  if (a.value <= 0.0) {
    if (result.state != nullptr) {
      result.state->pow_domain_error = true;
    }
    return result;
  }
  result.value = std::pow(a.value, b.value);

  auto tmp = std::log(a.value);
  for (Index j = 0; j < N; ++j) {
    result.derivatives[j] =
        result.value *
        (b.value * a.derivatives[j] / a.value + tmp * b.derivatives[j]);
  }
  return result;
}

template <Index N> constexpr Dual<N> pow(const Dual<N> &a, double b) {
  return pow(a, Dual<N>{b});
}

template <Index N> constexpr Dual<N> pow(double a, const Dual<N> &b) {
  return pow(Dual<N>{a}, b);
}
