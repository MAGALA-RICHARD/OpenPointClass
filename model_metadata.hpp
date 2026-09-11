#ifndef MODEL_METADATA_H
#define MODEL_METADATA_H

#include <string>
#include <vector>

void saveExcludedFeatures(
    const std::string &modelFile,
    const std::vector<std::string> &features
);

std::vector<std::string> loadExcludedFeatures(
    const std::string &modelFile
);

#endif