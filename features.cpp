#include "features.hpp"
#include <algorithm>
#include <cctype>
#include <iostream>

static std::string baseFeatureName(const std::string &name) {
    const size_t pos = name.find_last_of('_');

    if (pos == std::string::npos)
        return name;

    const std::string suffix = name.substr(pos + 1);

    // Feature::setName() appends the scale ID:
    // omnivariance -> omnivariance_1
    // order_1_axis_1 -> order_1_axis_1_1
    if (!suffix.empty() &&
        std::all_of(
            suffix.begin(),
            suffix.end(),
            [](unsigned char c) {
                return std::isdigit(c);
            }
        )) {
        return name.substr(0, pos);
    }

    return name;
}

static bool isExcluded(
    const std::string &name,
    const std::vector<std::string> &exclude
) {
    const std::string baseName = baseFeatureName(name);

    for (const auto &item : exclude) {
        // Base name removes feature from every scale.
        // Full name can remove one particular scale.
        if (item == baseName || item == name)
            return true;
    }

    return false;
}

std::vector<Feature *> getFeatures(const std::vector<Scale *> &scales,
    const std::vector<std::string>& excludedFeatures) {
    std::vector<Feature *> feats;

    for (size_t i = 0; i < scales.size(); i++) {
        // Covariance
        feats.push_back(reinterpret_cast<Feature *>(new Omnivariance(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new Eigenentropy(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new Anisotropy(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new Planarity(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new Linearity(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new SurfaceVariation(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new Scatter(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new Verticality(scales[i])));

        // Moments
        feats.push_back(reinterpret_cast<Feature *>(new OrderAxis(scales[i], 1, 1)));
        feats.push_back(reinterpret_cast<Feature *>(new OrderAxis(scales[i], 1, 2)));
        feats.push_back(reinterpret_cast<Feature *>(new OrderAxis(scales[i], 2, 1)));
        feats.push_back(reinterpret_cast<Feature *>(new OrderAxis(scales[i], 2, 2)));

        // Height
        feats.push_back(reinterpret_cast<Feature *>(new VerticalRange(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new HeightBelow(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new HeightAbove(scales[i])));

        // Color (using data from first scale only)
        for (size_t c = 0; c < 3; c++) {
            feats.push_back(reinterpret_cast<Feature *>(new PointColor(scales[0], c)));
            feats.push_back(reinterpret_cast<Feature *>(new NeighborhoodColors(scales[0], c)));
        }
       feats.push_back(reinterpret_cast<Feature *>(new GreenLeafIndex(scales[0])));
       feats.push_back(reinterpret_cast<Feature *>(new FlowerIndex(scales[0])));
    }
    //std::cout << "Excluded features received: " << excludedFeatures.size() << std::endl;
    std::vector<Feature *> selected;

    for (Feature *feature : feats) {

        if (isExcluded(feature->getName(), excludedFeatures)) {
        std::cout << "Removing feature: "
                  << feature->getName()
                  << std::endl;
            delete feature;
        }
        else {
            selected.push_back(feature);
        }
    }

    if (selected.empty()) {
        throw std::runtime_error(
            "All features were excluded"
        );
    }

    return selected;


    return selected;
}