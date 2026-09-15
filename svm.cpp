//
// Created by rmagala on 9/14/2026.
//

#include "svm.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cstdlib>
#include <new>

static void detachSupportVectors(::svm_model* model)
{
  size_t n = 0;

  for (int i = 0; i < model->l; ++i) {
    auto* p = model->SV[i];
    do { ++n; } while ((p++)->index != -1);
  }

  auto* data = static_cast<::svm_node*>(
      std::malloc(n * sizeof(::svm_node))
  );

  if (!data)
    throw std::bad_alloc{};

  size_t k = 0;

  for (int i = 0; i < model->l; ++i) {
    auto* src = model->SV[i];
    model->SV[i] = data + k;

    do {
      data[k++] = *src;
    } while ((src++)->index != -1);
  }

  model->free_sv = 1;
}
namespace svm {
void SVMDeleter::operator()(::svm_model *model) const
{
  if (model != nullptr) {

    ::svm_model *ptr = model;

    ::svm_free_and_destroy_model(
        &ptr
    );
  }
}

Kernel kernelFromString(
    const std::string &kernelName
)
{
  std::string name = kernelName;

  std::transform(
      name.begin(),
      name.end(),
      name.begin(),
      [](unsigned char c) {
        return static_cast<char>(
            std::tolower(c)
        );
      }
  );


  if (name == "linear") {
    return Kernel::Linear;
  }

  if (name == "rbf" ||
      name == "radial") {
    return Kernel::RBF;
  }


  if (
      name == "poly" ||
      name == "polynomial"
  ) {
    return Kernel::Polynomial;
  }


  if (name == "sigmoid") {
    return Kernel::Sigmoid;
  }


  throw std::invalid_argument(
      "Unsupported SVM kernel: " +
      kernelName
  );
}


int kernelToLibSVM(
    Kernel kernel
)
{
  switch (kernel) {

  case Kernel::Linear:
    return LINEAR;

  case Kernel::RBF:
    return RBF;

  case Kernel::Polynomial:
    return POLY;

  case Kernel::Sigmoid:
    return SIGMOID;
  }


  throw std::invalid_argument(
      "Unknown SVM kernel."
  );
}


std::string kernelToString(Kernel kernel){
  switch (kernel) {
  case Kernel::Linear:
    return "linear";
  case Kernel::RBF:
    return "rbf";
  case Kernel::Polynomial:
    return "polynomial";
  case Kernel::Sigmoid:
    return "sigmoid";
  }
  return "unknown";
}


// -----------------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------------

namespace {
Kernel kernelFromLibSVM(int kernel){
  switch (kernel) {
  case LINEAR:
    return Kernel::Linear;
  case RBF:
    return Kernel::RBF;
  case POLY:
    return Kernel::Polynomial;
  case SIGMOID:
    return Kernel::Sigmoid;
  default:
    throw std::runtime_error(
        "Unsupported kernel in LIBSVM model."
    );
  }
}


// Convert OpenPointClass feature values into LIBSVM nodes
std::vector<::svm_node> makeNodes(
    const std::vector<Feature *> &features,
    std::size_t pointIndex
)
{
  std::vector<::svm_node> nodes;
  nodes.reserve(
      features.size() + 1
  );
  for (
      std::size_t f = 0;
      f < features.size();
      ++f) {

    const double value =
        features[f]->getValue(
            pointIndex
        );


    if (!std::isfinite(value)) {
      throw std::runtime_error(
          "Non-finite feature value encountered during SVM training."
      );
    }


    /*
         * LIBSVM uses sparse vectors.
         *
         * Zero-valued features do not need to be stored.
     */
    if (value != 0.0) {

      ::svm_node node;

      /*
             * LIBSVM feature indices start at 1.
       */
      node.index =
          static_cast<int>(f + 1);

      node.value = value;

      nodes.push_back(node);
    }
  }


  /*
     * Every LIBSVM sample must end with index = -1.
   */
  ::svm_node terminator;

  terminator.index = -1;
  terminator.value = 0.0;

  nodes.push_back(
      terminator
  );


  return nodes;
}


} // anonymous namespace

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
    Kernel kernel,
    double C,
    double gamma,
    double degree,
    double coef0
)
{
  if (filenames.empty()) {
    throw std::invalid_argument(
        "No SVM training files were provided."
    );
  }
  if (startResolution == nullptr) {
    throw std::invalid_argument(
        "startResolution cannot be null."
    );
  }


  if (numScales <= 0) {
    throw std::invalid_argument(
        "numScales must be greater than zero."
    );
  }
  if (radius <= 0.0) {
    throw std::invalid_argument(
        "radius must be greater than zero."
    );
  }


  if (maxSamples <= 0) {
    throw std::invalid_argument(
        "maxSamples must be greater than zero."
    );
  }


  if (C <= 0.0) {
    throw std::invalid_argument(
        "SVM C must be greater than zero."
    );
  }


  if (gamma < 0.0) {
    throw std::invalid_argument(
        "SVM gamma cannot be negative."
    );
  }


  std::cout
      << "Training Support Vector Machine"
      << std::endl;

  std::cout
      << "Kernel: "
      << kernelToString(kernel)
      << std::endl;

  std::cout
      << "C: "
      << C
      << std::endl;

  std::cout
      << "Gamma: "
      << gamma
      << std::endl;


  if (kernel == Kernel::Polynomial) {

    std::cout
        << "Degree: "
        << degree
        << std::endl;
  }


  if (
      kernel == Kernel::Polynomial ||
      kernel == Kernel::Sigmoid
  ) {

    std::cout
        << "Coef0: "
        << coef0
        << std::endl;
  }


  // -------------------------------------------------------------------------
  // Training storage
  // -------------------------------------------------------------------------

  std::size_t featureCount = 0;


  std::vector<
      std::vector<::svm_node>
      > trainingSamples;


  std::vector<double>
      trainingLabels;


  // -------------------------------------------------------------------------
  // Called by getTrainingData() once features are known
  // -------------------------------------------------------------------------

  auto init =
      [&](
          std::size_t numFeatures,
          std::size_t /*numLabels*/
      )
  {
    featureCount =
        numFeatures;
  };


  // -------------------------------------------------------------------------
  // Store each balanced training sample
  // -------------------------------------------------------------------------

  auto storeFeatures =
      [&](
          const std::vector<Feature *> &features,
          std::size_t idx,
          int label
      )
  {
    /*
         * This protects against cases where init() was not called,
         * for example if an earlier training file had no labels.
     */
    if (featureCount == 0) {

      featureCount =
          features.size();
    }


    trainingSamples.push_back(
        makeNodes(
            features,
            idx
            )
    );


    /*
         * label is already the internal OpenPointClass
         * training-class index.
     */
    trainingLabels.push_back(
        static_cast<double>(
            label
            )
    );
  };


  // -------------------------------------------------------------------------
  // Reuse existing OpenPointClass training pipeline
  // -------------------------------------------------------------------------

  getTrainingData(
      filenames,
      startResolution,
      numScales,
      radius,
      maxSamples,
      classes,
      excludedFeatures,
      storeFeatures,
      init
  );


  if (trainingSamples.empty()) {

    throw std::runtime_error(
        "No SVM training samples were generated."
    );
  }


  if (featureCount == 0) {

    throw std::runtime_error(
        "No SVM features were generated."
    );
  }


  std::cout
      << "Training samples: "
      << trainingSamples.size()
      << std::endl;


  std::cout
      << "Features: "
      << featureCount
      << std::endl;


  // -------------------------------------------------------------------------
  // Build LIBSVM problem
  // -------------------------------------------------------------------------

  std::vector<::svm_node *>
      samplePointers(
          trainingSamples.size()
      );


  for (
      std::size_t i = 0;
      i < trainingSamples.size();
      ++i
  ) {

    samplePointers[i] =
        trainingSamples[i].data();
  }


  ::svm_problem problem{};

  problem.l =
      static_cast<int>(
          trainingSamples.size()
      );

  problem.y =
      trainingLabels.data();

  problem.x =
      samplePointers.data();


  // -------------------------------------------------------------------------
  // Configure LIBSVM
  // -------------------------------------------------------------------------

  ::svm_parameter param{};


  param.svm_type =C_SVC;


  param.kernel_type =kernelToLibSVM(
          kernel);


  param.degree = static_cast<int>(
          std::round(degree));


  /*
     * gamma = 0 means automatically use 1 / featureCount.
   */
  if (gamma == 0.0) {
    param.gamma =
        1.0 /
        static_cast<double>(
            featureCount
        );

  }
  else {

    param.gamma =
        gamma;
  }


  param.coef0 = coef0;
  param.cache_size =512.0;
  param.eps =1e-3;
  param.C = C;
  param.nr_weight =0;
  param.weight_label = nullptr;
  param.weight = nullptr;
  param.nu = 0.5;
  param.p = 0.1;
  param.shrinking =1;


  /*
     * Very important for OpenPointClass.
     *
     * classifyData() expects per-class scores/probabilities,
     * especially when LocalSmooth is enabled.
   */
  param.probability =1;


  // -------------------------------------------------------------------------
  // Validate parameters
  // -------------------------------------------------------------------------

  const char *error =
      ::svm_check_parameter(
          &problem,
          &param
      );


  if (error != nullptr) {

    throw std::runtime_error(
        std::string(
            "Invalid LIBSVM parameters: "
            ) +
        error
    );
  }


  // -------------------------------------------------------------------------
  // Train
  // -------------------------------------------------------------------------

  auto* rawModel = ::svm_train(&problem, &param);

  if (!rawModel) {
    ::svm_destroy_param(&param);
    throw std::runtime_error("SVM training failed");
  }

  detachSupportVectors(rawModel);
  ::svm_destroy_param(&param);

  return SVM(rawModel);

  if (rawModel == nullptr) {

    throw std::runtime_error(
        "LIBSVM training failed."
    );
  }


  std::cout
      << "SVM training completed."
      << std::endl;


  return SVM(
      rawModel
  );
}


// -----------------------------------------------------------------------------
// Model IO
// -----------------------------------------------------------------------------

SVM loadSVM(
    const std::string &modelFilename
)
{
  if (modelFilename.empty()) {

    throw std::invalid_argument(
        "SVM model filename cannot be empty."
    );
  }


  ::svm_model *rawModel =
      ::svm_load_model(
          modelFilename.c_str()
      );


  if (rawModel == nullptr) {

    throw std::runtime_error(
        "Unable to load SVM model: " +
        modelFilename
    );
  }


  return SVM(
      rawModel
  );
}


void saveSVM(
    const SVM &model,
    const std::string &modelFilename
)
{
  if (!model) {

    throw std::invalid_argument(
        "Cannot save an empty SVM model."
    );
  }


  if (modelFilename.empty()) {

    throw std::invalid_argument(
        "SVM model filename cannot be empty."
    );
  }


  const int result =
      ::svm_save_model(
          modelFilename.c_str(),
          model.get()
      );


  if (result != 0) {

    throw std::runtime_error(
        "Unable to save SVM model: " +
        modelFilename
    );
  }
}


// -----------------------------------------------------------------------------
// Extract model parameters
// -----------------------------------------------------------------------------

SVMParams extractSVMParams(
    const SVM &model
)
{
  if (!model) {

    throw std::invalid_argument(
        "Cannot extract parameters from an empty SVM model."
    );
  }


  SVMParams params;


  params.kernel =
      kernelFromLibSVM(
          model->param.kernel_type
      );


  /*
     * NOTE:
     *
     * LIBSVM does not store C in its saved prediction model because
     * C is only needed during training.
     *
     * This value is reliable for a model immediately returned by train(),
     * but should not be relied upon after saving and loading.
   */
  params.C =
      model->param.C;


  params.gamma =
      model->param.gamma;


  params.degree =
      static_cast<double>(
          model->param.degree
      );


  params.coef0 =
      model->param.coef0;


  params.probability =
      ::svm_check_probability_model(
          model.get()
              ) != 0;


  return params;
}


// -----------------------------------------------------------------------------
// Classification
// -----------------------------------------------------------------------------

void classify(
    PointSet &pointSet,
    const SVM &model,
    const std::vector<Feature *> &features,
    const std::vector<Label> &labels,
    Regularization regularization,
    double regRadius,
    bool useColors,
    bool unclassifiedOnly,
    bool evaluate,
    const std::vector<int> &skip,
    const std::string &statsFile
)
{
  if (!model) {

    throw std::invalid_argument(
        "Cannot classify using an empty SVM model."
    );
  }


  if (features.empty()) {

    throw std::invalid_argument(
        "No features were provided to SVM classification."
    );
  }


  if (labels.empty()) {

    throw std::invalid_argument(
        "No labels were provided to SVM classification."
    );
  }


  const int numModelClasses =
      ::svm_get_nr_class(
          model.get()
      );


  if (numModelClasses <= 0) {

    throw std::runtime_error(
        "Invalid number of classes in SVM model."
    );
  }


  // -------------------------------------------------------------------------
  // LIBSVM stores its own label ordering.
  // -------------------------------------------------------------------------

  std::vector<int>
      modelLabels(
          numModelClasses
      );


  ::svm_get_labels(
      model.get(),
      modelLabels.data()
  );


  /*
     * Validate model labels before entering the OpenMP region.
   */
  for (const int label : modelLabels) {

    if (
        label < 0 ||
        static_cast<std::size_t>(label) >= labels.size()
    ) {

      throw std::runtime_error(
          "SVM model contains an invalid OpenPointClass training label."
      );
    }
  }


  const bool hasProbability =
      ::svm_check_probability_model(
          model.get()
              ) != 0;


  if (!hasProbability) {

    std::cout
        << "Warning: SVM model has no probability estimates. "
        << "Predictions will use one-hot class scores."
        << std::endl;
  }


  const std::size_t featureCount =
      features.size();


  const std::size_t labelCount =
      labels.size();


  // -------------------------------------------------------------------------
  // Evaluation callback used by classifyData()
  // -------------------------------------------------------------------------

  auto evaluateFunc =
      [&](
          const double *ft,
          double *probs
      )
  {
    // Clear OpenPointClass output probabilities
    std::fill(
        probs,
        probs + labelCount,
        0.0
    );


    // Build LIBSVM sparse input row
    std::vector<::svm_node> nodes;

    nodes.reserve(
        featureCount + 1
    );


    for (
        std::size_t f = 0;
        f < featureCount;
        ++f
    ) {

      const double value =
          ft[f];


      /*
             * Sparse LIBSVM representation:
             * zero features can be omitted.
       */
      if (
          value != 0.0 &&
          std::isfinite(value)
      ) {

        ::svm_node node;

        node.index =
            static_cast<int>(
                f + 1
            );

        node.value =
            value;

        nodes.push_back(
            node
        );
      }
    }


    ::svm_node terminator;

    terminator.index = -1;
    terminator.value = 0.0;

    nodes.push_back(
        terminator
    );


    // ---------------------------------------------------------------------
    // Probability prediction
    // ---------------------------------------------------------------------

    if (hasProbability) {

      std::vector<double>
          svmProbabilities(
              numModelClasses,
              0.0
          );


      ::svm_predict_probability(
          model.get(),
          nodes.data(),
          svmProbabilities.data()
      );


      /*
             * IMPORTANT:
             *
             * LIBSVM's probability array follows LIBSVM's internal
             * model-label ordering.
             *
             * OpenPointClass expects:
             *
             * probs[trainingClass]
             *
             * therefore probabilities must be remapped.
       */
      for (
          int i = 0;
          i < numModelClasses;
          ++i
      ) {

        const int trainingClass =
            modelLabels[i];


        probs[trainingClass] =
            svmProbabilities[i];
      }
    }

    // ---------------------------------------------------------------------
    // Hard-class fallback
    // ---------------------------------------------------------------------

    else {

      const double prediction =
          ::svm_predict(
              model.get(),
              nodes.data()
          );


      const int predictedClass =
          static_cast<int>(
              std::llround(
                  prediction
                  )
          );


      if (
          predictedClass >= 0 &&
          static_cast<std::size_t>(
              predictedClass
              ) < labelCount
      ) {

        probs[predictedClass] =
            1.0;
      }
    }
  };


  // -------------------------------------------------------------------------
  // Reuse existing OpenPointClass classifier pipeline
  // -------------------------------------------------------------------------

  classifyData<double>(
      pointSet,
      evaluateFunc,
      features,
      labels,
      regularization,
      regRadius,
      useColors,
      unclassifiedOnly,
      evaluate,
      skip,
      statsFile
  );
}


} // namespace svm