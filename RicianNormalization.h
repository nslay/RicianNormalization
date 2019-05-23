/*-
 * Copyright (c) 2019 Nathan Lay (enslay@gmail.com)
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

#ifndef RICIANNORMALIZATION_H
#define RICIANNORMALIZATION_H

#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>
#include <vector>
#include <type_traits>
#include <numeric>
#include "ADVar.h"

#include "vnl/vnl_cost_function.h"
#include "vnl/algo/vnl_lbfgsb.h"

// Normalization scheme based on this paper:
//
// Lemaître, Guillaume, et al. "Normalization of t2w-mri prostate images using rician a priori." Medical Imaging 2016: Computer-Aided Diagnosis. Vol. 9785. International Society for Optics and Photonics, 2016.
//

template<typename RealType>
class RiceDistribution;

template<typename ImageType>
class RicianNormalization : public vnl_cost_function {
public:
  typedef vnl_cost_function SuperType;
  typedef typename ImageType::PixelType PixelType;
  typedef ADVar<double, 2> ADVarType;

  static_assert(std::is_convertible<PixelType, double>::value, "Pixel type must be convertible to double.");

  RicianNormalization()
  : SuperType(2) { }

  virtual ~RicianNormalization() = default;

  bool Normalize(ImageType *p_clImage, double dScale = 1.0) {
    if (p_clImage == nullptr)
      return false;

    m_p_clImage = p_clImage;

    vnl_vector<double> x = InitialGuess();

    if (x.size() != 2)
      return false;

    vnl_lbfgsb clSolver(*this);

    vnl_vector<long> clBoundSelection(2);
    clBoundSelection[0] = 1; // Only has lower bound
    clBoundSelection[1] = 1; // Ditto

    vnl_vector<double> clLower(2, 5e-4);

    clSolver.set_bound_selection(clBoundSelection);
    clSolver.set_lower_bound(clLower);

    clSolver.set_trace(true);
    clSolver.set_verbose(true);

    std::cout << "\nInitial x = " << x << std::endl;

    clSolver.minimize(x);

    // Now do normalization...
    const double dMean = RiceDistribution<double>::Mean(x[0], x[1]);
    const double dStd = std::sqrt(RiceDistribution<double>::Variance(x[0], x[1]));

    std::cout << "\nFinal x = " << x << ", rice mean = " << dMean << " rice std = " << dStd << std::endl;

    PixelType * const p_buffer = m_p_clImage->GetBufferPointer();
    const size_t numPixels = m_p_clImage->GetBufferedRegion().GetNumberOfPixels();

    if (std::is_signed<PixelType>::value) {
      for (size_t i = 0; i < numPixels; ++i) {
        p_buffer[i] = PixelType(dScale*((double)p_buffer[i] - dMean)/dStd);
      }
    }
    else {
      for (size_t i = 0; i < numPixels; ++i) {
        p_buffer[i] = PixelType(std::max(0.0, dScale*(((double)p_buffer[i] - dMean)/dStd + 4.0))); // Offset by 4 standard deviations
      }
    }

    return true;
  }

  virtual void compute(const vnl_vector<double> &x, double *f, vnl_vector<double> *g) override {
    ADVarType clNu(x[0], 0), clSigma(x[1], 1), clLoss(0.0);

    const PixelType * const p_buffer = m_p_clImage->GetBufferPointer();
    const size_t numPixels = m_p_clImage->GetBufferedRegion().GetNumberOfPixels();

    for (size_t i = 0; i < numPixels; ++i) {
      ADVarType clTmp((double)p_buffer[i]);

      clTmp = RiceDistribution<double>::Pdf(clTmp, clNu, clSigma);
      if (clTmp.Value() > 0.0) // The 0 pdf pixels wouldn't contribute to the optimization anyway (pdf derivative is 0)
        clLoss -= log(RiceDistribution<double>::Pdf(clTmp, clNu, clSigma));
    }

    clLoss /= (double)numPixels;

    if (f != nullptr)
      *f = clLoss.Value();

    if (g != nullptr) {
      g->set_size(2);
      g->copy_in(clLoss.Gradient().data());

      std::cout << "loss = " << clLoss.Value() << ", x = " << x << ", grad = " << *g << std::endl;
    }
  }

private:
  typename ImageType::Pointer m_p_clImage;

  vnl_vector<double> InitialGuess() const {
    constexpr const double tol = 1e-5;

    if (!m_p_clImage)
      return vnl_vector<double>();

    // Let's compute an initial guess
    //
    // u1 = s*sqrt(pi/2)*L_{1/2}(-v^2/(2*s^2))
    // u2 = 2*s^2 + v^2
    //
    // We can calculate the sample first and second moment...
    // Now let K = v^2/(2*s^2), with some rewriting we have
    //
    // u1 = s*sqrt(pi/2)*L_{1/2}(-K)
    // u2 = 2*s^2 + v^2
    //
    // More rewriting...
    // u2/(2*s^2) = 1 + v^2/(2*s^2) = 1 + K
    //
    // K = u2/(2*s^2) - 1
    // u1 = s*sqrt(pi/2)*L_{1/2}(1 - u2/(2*s^2))
    //
    // So we have a single equation for one unknown (s)!
    // We can bound the unknown using the fact that
    //
    // s > 0 and 1 - u2/(2*s^2) <= 0
    //
    // 0 < s <= sqrt(u2/2)
    //
    // L_{1/2}(x) is a monotonically decreasing function. Let's use bisection method!
    //

    // Calculate samle moments
    double u1 = 0.0, u2 = 0.0;

    const PixelType * const p_buffer = m_p_clImage->GetBufferPointer();
    const size_t numPixels = m_p_clImage->GetBufferedRegion().GetNumberOfPixels();

    if (p_buffer == nullptr || numPixels == 0)
      return vnl_vector<double>();

    for (size_t i = 0; i < numPixels; ++i) {
      const double value = (double)p_buffer[i];

      if (value < 0.0) // Not Rice distribution
        return vnl_vector<double>();

      double delta = value - u1;
      u1 += delta/(i+1.0);

      delta = value*value - u2;
      u2 += delta/(i+1.0);
    }

    if (u1 < 0.0 || u2 <= 0.0) // The latter probably shouldn't happen?
      return vnl_vector<double>();

    auto fnObjective = [&](double s) -> double {
      return s*std::sqrt(M_PI_2)*RiceDistribution<double>::HalfLaguerre(1.0 - u2/(2.0*s*s)) - u1;
    };

    // Now do bisection method
    double a = 1e-1, b = std::sqrt(0.5*u2);
    double fa = fnObjective(a), fb = fnObjective(b);
    double s = 0.0;

    std::cout << "Solving for initial guess with bisection method..." << std::endl;
    std::cout << "fa = " << fa << ", fb = " << fb << std::endl;

    if (std::abs(fb) < tol) {
      s = b;
    }
    else if (std::abs(fa) < tol) {
      s = a;
    }
    else {
      for (int i = 0; i < 21; ++i) {
        s = 0.5*(a + b);

        const double fs = fnObjective(s);

        std::cout << "s = " << s << ", fs = " << fs << std::endl;

        if (std::abs(fs) < 1e-5)
          break;

        if ((fb < 0.0) ^ (fs < 0.0)) {
          a = s;
          fa = fs;
        }
        else if ((fa < 0.0) ^ (fs < 0.0)) {
          b = s;
          fb = fs;
        }
        else {
          // Uhh?
          break;
        }
      }
    }

    const double v = M_SQRT2*s*std::sqrt(u2/(2.0*s*s) - 1.0);

    std::cout << "\nInitial nu = " << v << ", sigma = " << s << std::endl;
    std::cout << "Sample mean = " << u1 << ", rice mean = " << RiceDistribution<double>::Mean(v, s) << std::endl;
    std::cout << "Sample 2nd moment = " << u2 << ", rice 2nd moment = " << 2*s*s + v*v << std::endl;

    vnl_vector<double> x(2);
    x[0] = v;
    x[1] = s;

    return x;
  }
};

template<typename RealType>
class RiceDistribution {
  static constexpr unsigned int MaxTerms = 20; // Really, this is completely overkill for the values we will likely need... pessimistic napkin math confirms!
public:
  // OK, but not how to sample it...
  RiceDistribution(const RealType &nu, const RealType &sigma)
  : m_nu(nu), m_sigma(sigma) { }

  static RealType Pdf(const RealType &x, const RealType &nu, const RealType &sigma) {
    const RealType sigma2 = sigma*sigma; // Just to be concise...
    return x/sigma2 * std::exp(-(x*x + nu*nu)/(RealType(2)*sigma2)) * ModifiedBessel0(x*nu/sigma2);
  }

  template<unsigned int NumIndependents>
  static ADVar<RealType, NumIndependents> Pdf(const ADVar<RealType, NumIndependents> &x, const ADVar<RealType, NumIndependents> &nu, const ADVar<RealType, NumIndependents> &sigma) {
    const ADVar<RealType, NumIndependents> sigma2 = sigma*sigma;
    return x/sigma2 * exp(-(x*x + nu*nu)/(RealType(2)*sigma2)) * ModifiedBessel0(x*nu/sigma2);
  }

  static RealType Mean(const RealType &nu, const RealType &sigma) {
    return sigma*std::sqrt(RealType(M_PI_2))*HalfLaguerre(-std::pow(nu/sigma, 2)/RealType(2));
  }

  static RealType Variance(const RealType &nu, const RealType &sigma) {
    return RealType(2)*sigma*sigma + nu*nu - 
      RealType(M_PI_2)*std::pow(sigma*HalfLaguerre(-std::pow(nu/sigma, 2)/RealType(2)), 2);
  }

  const RealType & Nu() const { return m_nu; }
  const RealType & Sigma() const { return m_sigma; }

  RealType Pdf(const RealType &x) const { return Pdf(x, Nu(), Sigma()); }

  // This doesn't seem quite useful but here for consistency...
  template<unsigned int NumIndependents>
  ADVar<RealType, NumIndependents> Pdf(const ADVar<RealType, NumIndependents> &x) const {
    const ADVar<RealType, NumIndependents> clNu(Nu()), clSigma(Sigma());
    return Pdf(x, clNu, clSigma);
  }

  RealType Mean() const { return Mean(Nu(), Sigma()); }
  RealType Variance() const { return Variance(Nu(), Sigma()); }

  // Don't use this junk outside of calculating the pdf!
  //
  // http://mathworld.wolfram.com/ModifiedBesselFunctionoftheFirstKind.html
  // From (7):
  static RealType ModifiedBessel0(const RealType &x) {
    if (x == RealType(0))
      return RealType(1);

    // Let z = 1/4*x*x
    const RealType z = x*x/RealType(4);
    RealType term = RealType(1);
    RealType sum = term; // First term is 1

    for (unsigned int k = 1; k < MaxTerms; ++k) {
      term *= z/RealType(k*k);
      sum += term;
    }

    return sum;
  }

  template<unsigned int NumIndependents>
  static ADVar<RealType, NumIndependents> ModifiedBessel0(const ADVar<RealType, NumIndependents> &x) {
    ADVar<RealType, NumIndependents> y;

    y.Value() = ModifiedBessel0(x.Value());

    const RealType tmp = ModifiedBessel1(x.Value());

    std::transform(x.Gradient().begin(), x.Gradient().end(), y.Gradient().begin(),
      [&tmp](const RealType &dx) -> RealType {
        return tmp*dx;
      });

    return y;
  }

  // From (6):
  static RealType ModifiedBessel1(const RealType &x) {
    // http://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html
    // T_1(x) = x and hence T_1(d/dx) = d/dx
    // ModifiedBessel1(x) = d/dx ModifiedBessel0(x)

    if (x == RealType(0))
      return RealType(0);

    // Let z = 1/4*x*x
    const RealType z = x*x/RealType(4);
    RealType term = RealType(1);
    RealType sum = term; // First term is 0, second term is 1

    for (unsigned int k = 2; k < MaxTerms; ++k) {
      term *= z/RealType(k*k);
      sum += RealType(k)*term;
    }

    sum *= x/RealType(2); // Chain rule x^2/4 --> x/2
    
    return sum;
  }

  // https://en.wikipedia.org/wiki/Rice_distribution
  // From section "Moments"
  static RealType HalfLaguerre(const RealType &x) {
    return std::exp(x/RealType(2))*((RealType(1) - x)*ModifiedBessel0(-x/RealType(2)) - x*ModifiedBessel1(-x/RealType(2)));
  }

private:
  RealType m_sigma;
  RealType m_nu;
};

#endif // !RICIANNORMALIZATION_H
