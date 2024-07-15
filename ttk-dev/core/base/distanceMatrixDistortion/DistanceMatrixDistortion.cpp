#include <DistanceMatrixDistortion.h>
#include <Geometry.h> // To check wheter a double is zero.
#include <Os.h>
#include <random>
#include <vector>

ttk::DistanceMatrixDistortion::DistanceMatrixDistortion() {
  this->setDebugMsgPrefix("DistanceMatrixDistortion");
}

int ttk::DistanceMatrixDistortion::execute(
  const std::vector<double *> &highDistMatrix,
  const std::vector<double *> &lowDistMatrix,
  double &distortionValue,
  double *distortionVerticesValues) const {
  ttk::Timer timer;
  auto n = highDistMatrix.size();

  if(lowDistMatrix.size() != n) {
    this->printErr(" Sizes mismatch: the high distance matrix has "
                   + std::to_string(n)
                   + " rows and the low distance matrix has "
                   + std::to_string(lowDistMatrix.size()) + " rows.");
    return 0;
  }

  /* The computation, which is optimised for performance here, can be decomposed
   * as follows: compute for each (x,y) delta(x,y) =
   * (dist_low(x,y)-dist_high(x,y))^2. Compute maxi = maximum (delta(x,y)) over
   * all (x,y). Compute delta2(x,y) = 1-delta(x,y) for each (x,y). The sim value
   * is the mean of the n^2 values of delta2(x,y). The distortion for a vertex
   * x0 is the mean of delta2(x0,y) over all y's.
   */

  double maxi = 0;
  if(distortionVerticesValues == nullptr) {
    this->printErr(
      " The output pointer to the distortionValues must be non NULL. "
      "It must point to an allocated array of the right size.");
    return 1;
  }

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(this->threadNumber_) reduction(max     \
                                                                    : maxi) \
  schedule(dynamic)
#endif // TTK_ENABLE_OPENMP
  for(size_t i = 0; i < n; i++) {
    for(size_t j = i + 1; j < n; j++) {
      const double diff = lowDistMatrix[i][j] - highDistMatrix[i][j];
      maxi = std::max(maxi, diff * diff);
    }
  }
  const double EPS = ttk::Geometry::powIntTen(-DBL_DIG);
  if(maxi <= EPS) // We consider maxi is equal to zero.
  {
    this->printMsg(
      "The two distance matrices provided for SIM computation are equal.");
    maxi = 1;
  }

  double totalSum = 0;

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(this->threadNumber_), reduction(+:totalSum)
#endif // TTK_ENABLE_OPENMP
  for(size_t i = 0; i < n; i++) {
    double sum = 0;
    for(size_t j = 0; j < n; j++) {
      const double diff = lowDistMatrix[i][j] - highDistMatrix[i][j];
      const double diff2 = diff * diff;
      sum += diff2;
    }
    const double sumNormalized = (this->DoNotNormalize ? sum : sum / maxi);
    distortionVerticesValues[i] = 1 - sumNormalized / n;
    totalSum += 1 - sumNormalized / n;
  }

  distortionValue = totalSum / n;

  this->printMsg("Size of output in ttk/base = " + std::to_string(n));

  this->printMsg("Computed distortion value: "
                 + std::to_string(distortionValue));
  this->printMsg(ttk::debug::Separator::L2); // horizontal '-' separator
  this->printMsg("Complete", 1, timer.getElapsedTime());
  this->printMsg(ttk::debug::Separator::L1); // horizontal '=' separator

  return 0;
}
