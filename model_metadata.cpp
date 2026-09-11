#include "model_metadata.hpp"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>

namespace fs = std::filesystem;


// ---------------------------------------------------------
// Construct metadata path
// ---------------------------------------------------------
fs::path getMetadataPath(const std::string& classifier)
{
    // Directory containing model_metadata.cpp
    const fs::path sourceDir =
        fs::path(__FILE__).parent_path();

    // Metadata directory
    const fs::path metadataDir =
        sourceDir / "metadata";

    // Take only the filename in case classifier is a full path
    const fs::path classifierPath(classifier);

    const std::string classifierName =
        classifierPath.filename().string();

    // Example:
    // rf_model.bin -> rf_model.bin.features
    return metadataDir /
           (classifierName + ".features");
}


// ---------------------------------------------------------
// Save excluded features
// ---------------------------------------------------------
void saveExcludedFeatures(
    const std::string& model_file_name,
    const std::vector<std::string>& features
)
{
     const std::string featuresFile = model_file_name + ".features";

    std::cout
        << "[DEBUG] Saving "
        << features.size()
        << " excluded features to: "
        << featuresFile
        << std::endl;

    std::ofstream out(featuresFile);

    if (!out)
    {
        throw std::runtime_error(
            "Could not create features file: "
            + featuresFile
        );
    }

    for (const auto& feature : features)
    {
        std::cout
            << "  - "
            << feature
            << std::endl;

        out << feature << '\n';
    }
}


// ---------------------------------------------------------
// Load excluded features
// ---------------------------------------------------------
std::vector<std::string> loadExcludedFeatures(
    const std::string& model_file_name
)
{
    std::vector<std::string> features;

    const std::string featuresFile = model_file_name + ".features";

    std::cout
        << "[DEBUG] Looking for exclusions file: "
        << featuresFile
        << std::endl;

    std::ifstream in(featuresFile);

    if (!in)
    {
        std::cerr
            << "[WARNING] Exclusions file not found: "
            << featuresFile
            << std::endl;

        return features;
    }

    std::string feature;

    while (std::getline(in, feature))
    {
        if (!feature.empty())
        {
            features.push_back(feature);
        }
    }

    std::cout
        << "[DEBUG] Loaded "
        << features.size()
        << " excluded features:"
        << std::endl;

    for (const auto& feature : features)
    {
        std::cout
            << "  - "
            << feature
            << std::endl;
    }

    return features;
}