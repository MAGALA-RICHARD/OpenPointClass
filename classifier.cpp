#include "classifier.hpp"

Regularization parseRegularization(const std::string &regularization) {
    if (regularization == "none") return None;
    if (regularization == "local_smooth") return LocalSmooth;
    throw std::runtime_error("Invalid regularization value: " + regularization + ". value should be either: none or local_smooth");
}
ClassifierType fingerprint(const std::string& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open " + file);

    char magic[4];
    if (!in.read(magic, 4))
      throw std::runtime_error("Invalid model file: " + file);

    const std::string_view id(magic, 4);
    std::cout << "Fingerprint: [" << id << "]\n";
    if (id == "tree") return GradientBoostedTrees;
    if (id == "svm_") return SupportVectorMachine;
    return RandomForest;


}