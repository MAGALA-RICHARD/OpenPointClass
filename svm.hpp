//
// Created by rmagala on 9/14/2026.
//

#ifndef SVM_HPP
#define SVM_HPP

#include <memory>
#include <string>
#include <vector>

#include "vendor/libsvm/svm.h"

#include "classifier.hpp"
#include "features.hpp"
#include "labels.hpp"
#include "constants.hpp"
#include "point_io.hpp"


namespace svm {


// -----------------------------------------------------------------------------
// Supported SVM kernels
// -----------------------------------------------------------------------------

enum class Kernel {
  Linear,
  RBF,
  Polynomial,
  Sigmoid
};


// Convert command-line/string value to Kernel
Kernel kernelFromString(const std::string &kernelName);


// Convert our Kernel enum to LIBSVM kernel constant
int kernelToLibSVM(Kernel kernel);


// Convert Kernel enum to readable string
std::string kernelToString(Kernel kernel);


// -----------------------------------------------------------------------------
// Model memory management
// -----------------------------------------------------------------------------

struct SVMDeleter {

  void operator()(::svm_model *model) const;

};


using SVM = std::unique_ptr<::svm_model, SVMDeleter>;


// -----------------------------------------------------------------------------
// SVM hyperparameters
// -----------------------------------------------------------------------------

struct SVMParams {

  Kernel kernel = Kernel::RBF;

  double C = 1.0;
  double gamma = 0.1;
  double degree = 3.0;
  double coef0 = 0.0;

  bool probability = true;
};


// -----------------------------------------------------------------------------
// Training
// -----------------------------------------------------------------------------

SVM train(
    const std::vector<std::string> &filenames,
    double *startResolution,
    int numScales,
    double radius,
    int maxSamples,
    const std::vector<int> &classes,
    const std::vector<std::string> &excludedFeatures,

    Kernel kernel = Kernel::RBF,
    double C = 1.0,
    double gamma = 0.1,
    double degree = 3.0,
    double coef0 = 0.0
);


// -----------------------------------------------------------------------------
// Model IO
// -----------------------------------------------------------------------------

SVM loadSVM(
    const std::string &modelFilename
);


void saveSVM(
    const SVM &model,
    const std::string &modelFilename
);


// -----------------------------------------------------------------------------
// Model parameters
// -----------------------------------------------------------------------------

SVMParams extractSVMParams(
    const SVM &model
);


// -----------------------------------------------------------------------------
// Classification
// -----------------------------------------------------------------------------

void classify(
    PointSet &pointSet,
    const SVM &model,
    const std::vector<Feature *> &features,
    const std::vector<Label> &labels,

    Regularization regularization = Regularization::None,
    double regRadius = 2.5,
    bool useColors = false,
    bool unclassifiedOnly = false,
    bool evaluate = false,

    const std::vector<int> &skip = {},
    const std::string &statsFile = ""
);


} // namespace svm

#endif // SVM_HPP