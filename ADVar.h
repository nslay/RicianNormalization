/*-
 * Copyright (c) 2018 Nathan Lay (enslay@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#ifndef ADVAR_H
#define ADVAR_H

// A lightweight header-only forward-mode automatic differentiation library...
// NOTE: Not all functions are implemented!

#include <cmath>
#include <array>
#include <algorithm>
#include <functional>

template<typename RealTypeT, unsigned int NumIndependents>
class ADVar {
public:
  typedef RealTypeT RealType;
  typedef std::array<RealType, NumIndependents> GradientType;

  static constexpr unsigned int GetNumIndependents() { return NumIndependents; }

  ADVar() { m_gradient.fill(RealType()); }

  explicit ADVar(const RealType &value)
  : m_value(value) {
    m_gradient.fill(RealType());
  }

  ADVar(const RealType &value, unsigned int uiIndex, const RealType &dvalue = RealType(1))
  : m_value(value) {
    m_gradient.fill(RealType());
    m_gradient[uiIndex] = dvalue;
  }

  ADVar(const ADVar &) = default;
  ADVar(ADVar &&) = default;

  void SetIndependent(const RealType &value, unsigned int uiIndex, const RealType &dvalue = RealType(1)) {
    m_value = value;
    m_gradient.fill(RealType());
    m_gradient[uiIndex] = dvalue;
  }

  void SetConstant(const RealType &value) {
    m_value = value;
    m_gradient.fill(RealType());
  }

  RealType & Value() { return m_value; }
  const RealType & Value() const { return m_value; }
  GradientType & Gradient() { return m_gradient; }
  const GradientType & Gradient() const { return m_gradient; }

  // Unary operators
  ADVar operator-() const {
    ADVar clResult;
    clResult.m_value = -m_value;
    std::transform(m_gradient.begin(), m_gradient.end(), clResult.m_gradient.begin(), std::negate<RealType>());
    return clResult;
  }

  // Binary operators
  ADVar operator+(const ADVar &clOther) const {
    ADVar clResult;
    clResult.m_value = m_value + clOther.m_value;
    std::transform(m_gradient.begin(), m_gradient.end(), clOther.m_gradient.begin(), clResult.m_gradient.begin(), std::plus<RealType>());
    return clResult;
  }

  ADVar operator+(const RealType &other) const { 
    ADVar clResult(*this);
    clResult.m_value += other;
    return clResult;
  }

  ADVar operator-(const ADVar &clOther) const {
    ADVar clResult;
    if (this != &clOther) {
      clResult.m_value = m_value - clOther.m_value;
      std::transform(m_gradient.begin(), m_gradient.end(), clOther.m_gradient.begin(), clResult.m_gradient.begin(), std::minus<RealType>());
    }
    return clResult;
  }

  ADVar operator-(const RealType &other) const { 
    ADVar clResult(*this);
    clResult.m_value -= other;
    return clResult;
  }

  ADVar operator*(const ADVar &clOther) const {
    ADVar clResult;
    if (this == &clOther) {
      clResult.m_value = m_value * m_value;
      const RealType tmp = RealType(2)*m_value;
      std::transform(m_gradient.begin(), m_gradient.end(), clResult.m_gradient.begin(),
        [&tmp](const RealType &dThis) -> RealType {
          return tmp * dThis;
        });
    }
    else {
      clResult.m_value = m_value * clOther.m_value;
      std::transform(m_gradient.begin(), m_gradient.end(), clOther.m_gradient.begin(), clResult.m_gradient.begin(),
        [this, &clOther](const RealType &dThis, const RealType &dOther) -> RealType {
          return this->m_value * dOther + clOther.m_value * dThis;
        });
    }
    return clResult;
  }

  ADVar operator*(const RealType &other) const {
    ADVar clResult;
    clResult.m_value = m_value * other;
    std::transform(m_gradient.begin(), m_gradient.end(), clResult.m_gradient.begin(),
      [&other](const RealType &dThis) -> RealType {
        return dThis * other;
      });
    return clResult;
  }

  ADVar operator/(const ADVar &clOther) const {
    ADVar clResult;
    if (this == &clOther) {
      if (m_value != RealType(0)) {
        clResult.m_value = RealType(1);
      }
      else {
        clResult.m_value /= m_value; // Trigger a NaN or signal/exception
        clResult.m_gradient.fill(clResult.m_value);
      }
    }
    else {
      clResult.m_value = m_value / clOther.m_value;
      std::transform(m_gradient.begin(), m_gradient.end(), clOther.m_gradient.begin(), clResult.m_gradient.begin(),
        [this, &clOther](const RealType &dThis, const RealType &dOther) -> RealType {
          return (clOther.m_value * dThis - this->m_value * dOther)/(clOther.m_value * clOther.m_value);
        });
    }
    return clResult;
  }

  ADVar operator/(const RealType &other) const {
    ADVar clResult;
    clResult.m_value = m_value / other;
    std::transform(m_gradient.begin(), m_gradient.end(), clResult.m_gradient.begin(),
      [&other](const RealType &dThis) -> RealType {
        return dThis / other;
      });
    return clResult;
  }

  // Assignment operators

  ADVar & operator+=(const ADVar &clOther) {
    m_value += clOther.m_value;
    std::transform(m_gradient.begin(), m_gradient.end(), clOther.m_gradient.begin(), m_gradient.begin(), std::plus<RealType>());
    return *this;
  }

  ADVar & operator+=(const RealType &other) {
    m_value += other;
    return *this;
  }

  ADVar & operator-=(const ADVar &clOther) {
    if (this == &clOther) {
      m_value = RealType();
      m_gradient.fill(RealType());
    }
    else {
      m_value -= clOther.m_value;
      std::transform(m_gradient.begin(), m_gradient.end(), clOther.m_gradient.begin(), m_gradient.begin(), std::minus<RealType>());
    }
    return *this;
  }

  ADVar & operator-=(const RealType &other) {
    m_value -= other;
    return *this;
  }

  ADVar & operator*=(const ADVar &clOther) {
    if (this == &clOther) {
      const RealType tmp = RealType(2)*m_value;
      std::transform(m_gradient.begin(), m_gradient.end(), m_gradient.begin(),
        [&tmp](const RealType &dThis) -> RealType {
          return tmp * dThis;
        });
      m_value *= m_value;
    }
    else {
      std::transform(m_gradient.begin(), m_gradient.end(), clOther.m_gradient.begin(), m_gradient.begin(),
        [this, &clOther](const RealType &dThis, const RealType &dOther) -> RealType {
          return this->m_value * dOther + clOther.m_value * dThis;
        });
      m_value *= clOther.m_value;
    }
    return *this;
  }

  ADVar & operator*=(const RealType &other) {
    std::transform(m_gradient.begin(), m_gradient.end(), m_gradient.begin(),
      [&other](const RealType &dThis) -> RealType {
        return dThis * other;
      });
    m_value *= other;
    return *this;
  }

  ADVar & operator/=(const ADVar &clOther) {
    if (this == &clOther) {
      if (m_value != RealType(0)) {
        m_value = RealType(1);
        m_gradient.fill(RealType(0));
      }
      else {
        m_value /= m_value; // Make this into NaN and/or cause a signal/exception
        m_gradient.fill(m_value);
      }
    }
    else {
      std::transform(m_gradient.begin(), m_gradient.end(), clOther.m_gradient.begin(), m_gradient.begin(),
        [this, &clOther](const RealType &dThis, const RealType &dOther) -> RealType {
          return (clOther.m_value * dThis - this->m_value * dOther)/(clOther.m_value * clOther.m_value);
        });
      m_value /= clOther.m_value;
    }
    return *this;
  }

  ADVar & operator/=(const RealType &other) {
    std::transform(m_gradient.begin(), m_gradient.end(), m_gradient.begin(),
      [&other](const RealType &dThis) -> RealType {
        return dThis / other;
      });
    m_value /= other;
    return *this;
  }

  ADVar & operator=(const ADVar &clOther) = default;
  ADVar & operator=(ADVar &&) = default;

  ADVar & operator=(const RealType &other) {
    m_value = other;
    m_gradient.fill(RealType());
    return *this;
  }

  // Logic and comparison operators
  bool operator!() const { return !m_value; }
  explicit operator bool() const { return m_value != RealType(); }

  bool operator==(const RealType &other) const { return m_value == other; }
  bool operator!=(const RealType &other) const { return m_value != other; }

  bool operator>(const RealType &other) const { return m_value > other; }
  bool operator>=(const RealType &other) const { return m_value >= other; }
  bool operator<(const RealType &other) const { return m_value < other; }
  bool operator<=(const RealType &other) const { return m_value <= other; }

private:
  RealType m_value = RealType();
  GradientType m_gradient;
};

// Other variants of operators

template<typename RealType, unsigned int NumIndependents>
ADVar<RealType, NumIndependents> operator+(const RealType &a, const ADVar<RealType, NumIndependents> &clB) { return clB + a; }

template<typename RealType, unsigned int NumIndependents>
ADVar<RealType, NumIndependents> operator-(const RealType &a, const ADVar<RealType, NumIndependents> &clB) { 
  ADVar<RealType, NumIndependents> clC;
  clC.Value() = a - clB.Value();
  std::transform(clB.Gradient().begin(), clB.Gradient().end(), clC.Gradient().begin(), std::negate<RealType>());
  return clC;
}

template<typename RealType, unsigned int NumIndependents>
ADVar<RealType, NumIndependents> operator*(const RealType &a, const ADVar<RealType, NumIndependents> &clB) { return clB * a; }

template<typename RealType, unsigned int NumIndependents>
ADVar<RealType, NumIndependents> operator/(const RealType &a, const ADVar<RealType, NumIndependents> &clB) { 
  ADVar<RealType, NumIndependents> clC;

  clC.Value() = a / clB.Value();
  const RealType tmp = -clC.Value() / clB.Value();
  
  std::transform(clB.Gradient().begin(), clB.Gradient().end(), clC.Gradient().begin(), 
    [&tmp](const RealType &dB) -> RealType {
      return tmp * dB;
    });

  return clC;
}

template<typename RealType, unsigned int NumIndependents>
bool operator==(const RealType &a, const ADVar<RealType, NumIndependents> &clB) { return clB == a; }

template<typename RealType, unsigned int NumIndependents>
bool operator!=(const RealType &a, const ADVar<RealType, NumIndependents> &clB) { return clB != a; }

template<typename RealType, unsigned int NumIndependents>
bool operator>(const RealType &a, const ADVar<RealType, NumIndependents> &clB) { return clB < a; }

template<typename RealType, unsigned int NumIndependents>
bool operator>=(const RealType &a, const ADVar<RealType, NumIndependents> &clB) { return clB <= a; }

template<typename RealType, unsigned int NumIndependents>
bool operator<(const RealType &a, const ADVar<RealType, NumIndependents> &clB) { return clB > a; }

template<typename RealType, unsigned int NumIndependents>
bool operator<=(const RealType &a, const ADVar<RealType, NumIndependents> &clB) { return clB >= a; }

// Unary functions

template<typename RealType, unsigned int NumIndependents>
ADVar<RealType, NumIndependents> sqrt(const ADVar<RealType, NumIndependents> &clX) {
  ADVar<RealType, NumIndependents> clResult;

  clResult.Value() = std::sqrt(clX.Value());

  const RealType tmp = RealType(1)/(RealType(2)*clResult.Value());

  std::transform(clX.Gradient().begin(), clX.Gradient().end(), clResult.Gradient().begin(),
    [&tmp](const RealType &dX) -> RealType {
      return tmp * dX;
    });

  return clResult;
}

template<typename RealType, unsigned int NumIndependents>
ADVar<RealType, NumIndependents> exp(const ADVar<RealType, NumIndependents> &clX) {
  ADVar<RealType, NumIndependents> clResult;

  clResult.Value() = std::exp(clX.Value());

  std::transform(clX.Gradient().begin(), clX.Gradient().end(), clResult.Gradient().begin(),
    [&clResult](const RealType &dX) -> RealType {
      return clResult.Value() * dX;
    });
  
  return clResult;
}

template<typename RealType, unsigned int NumIndependents>
ADVar<RealType, NumIndependents> log(const ADVar<RealType, NumIndependents> &clX) {
  ADVar<RealType, NumIndependents> clResult;

  clResult.Value() = std::log(clX.Value());

  const RealType tmp = RealType(1)/clX.Value();

  std::transform(clX.Gradient().begin(), clX.Gradient().end(), clResult.Gradient().begin(),
    [&tmp](const RealType &dX) -> RealType {
      return tmp * dX;
    });

  return clResult;
}

template<typename RealType, unsigned int NumIndependents>
ADVar<RealType, NumIndependents> abs_weak(const ADVar<RealType, NumIndependents> &clX) { 
  return clX.Value() < 0.0 ? -clX : clX; 
}

template<typename RealType, unsigned int NumIndependents>
ADVar<RealType, NumIndependents> huber_loss(const ADVar<RealType, NumIndependents> &clX) { 
  if (std::abs(clX.Value()) <= RealType(1))
    return RealType(0.5)*(clX * clX);

  return abs_weak(clX) - RealType(0.5);
}

// Binary functions
template<typename RealType, unsigned int NumIndependents>
ADVar<RealType, NumIndependents> pow(const ADVar<RealType, NumIndependents> &clX, int iExp) {
  switch (iExp) {
  case 0:
    {
      ADVar<RealType, NumIndependents> clResult;
      if (clX.Value() != RealType(0)) {
        clResult.Value() = RealType(1);
      }
      else {
        clResult.Value() = std::pow(clX.Value(), iExp); // Trigger NaN, signal/exception
        clResult.Gradient().fill(clResult.Value());
      }
      return clResult;
    }
  case 1:
    return clX;
  default:
    {
      ADVar<RealType, NumIndependents> clResult;

      RealType tmp = std::pow(clX.Value(), iExp-1);

      clResult.Value() = tmp * clX.Value();

      tmp *= RealType(iExp);

      std::transform(clX.Gradient().begin(), clX.Gradient().end(), clResult.Gradient().begin(),
        [&tmp](const RealType &dX) -> RealType {
          return tmp * dX;
        });

      return clResult;
    }
  }

  return ADVar<RealType, NumIndependents>(); // Not reached
}

#endif // ADVAR_H
