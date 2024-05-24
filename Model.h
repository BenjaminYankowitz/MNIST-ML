#ifndef _MODEL__H__
#define _MODEL__H__
#include "Eigen/Dense"
#include <array>
#include <fstream>

template <class ActFunc>
concept LegalActFunc = requires(double a) {
    { ActFunc::app(a) } -> std::same_as<double>;
    { ActFunc::dir(a) } -> std::same_as<double>;
};

template <LegalActFunc ActFunc, int NumLayers, std::array<int, NumLayers> LayerSizes>
class Model {
    static_assert(NumLayers >= 2);

public:
    Model() {
        construct();
    }
    Model(const std::string &fileName) {
        std::ifstream input(fileName, std::ios::in | std::ios::binary);
        if (!input.is_open()) {
            construct();
            return;
        }
        for (size_t i = 0; i < NumLayers - 1; i++) {
            connectionsMod[i] = Eigen::MatrixXd::Zero(LayerSizes[i + 1], LayerSizes[i]);
            biasessMod[i] = Eigen::VectorXd::Zero(LayerSizes[i + 1]);
            Eigen::MatrixXd &cM = connections[i];
            Eigen::VectorXd &cB = biases[i];
            cM = Eigen::MatrixXd::Zero(LayerSizes[i + 1], LayerSizes[i]);
            cB = Eigen::VectorXd::Zero(LayerSizes[i + 1]);
            input.read(reinterpret_cast<char *>(cM.data()), sizeof(cM.data()[0]) * cM.size());
            input.read(reinterpret_cast<char *>(cB.data()), sizeof(cB.data()[0]) * cB.size());
        }
        input.close();
    }

    void writeTo(const std::string &fileName) {
        std::ofstream output(fileName, std::ios::out | std::ios::binary);
        for (size_t i = 0; i < NumLayers - 1; i++) {
            const Eigen::MatrixXd &cM = connections[i];
            const Eigen::VectorXd &cB = biases[i];
            output.write(reinterpret_cast<const char *>(cM.data()), sizeof(cM.data()[0]) * cM.size());
            output.write(reinterpret_cast<const char *>(cB.data()), sizeof(cB.data()[0]) * cB.size());
        }
        output.close();
    }
    Eigen::VectorXd runModel(Eigen::VectorXd && input) const{
        assert(input.size() == LayerSizes.front());
        Eigen::VectorXd cLayer = std::move(input);
        for(size_t i = 0; i < connections.size()-1; i++){
            cLayer = (connections[i]*cLayer+biases[i]).unaryExpr(&ActFunc::app);
        }
        cLayer = connections.back()*cLayer+biases.back();
        return cLayer;
    }
    template <bool calcError = false>
    double trainModel(Eigen::VectorXd &&input, const std::array<double, LayerSizes.back()> &answer) {
        assert(input.size() == LayerSizes.front());
        double error = -1;
        std::array<Eigen::VectorXd, NumLayers> layerValues;
        layerValues[0] = std::move(input);
        for (size_t i = 0; i < connections.size(); i++) {
            layerValues[i + 1] = (connections[i] * layerValues[i] + biases[i]).unaryExpr(&ActFunc::app);
        }
        layerValues.back() = connections.back() * layerValues[layerValues.size() - 2] + biases.back();
        layerValues.back() = Eigen::Map<const Eigen::VectorXd>(answer.data(), answer.size()) - layerValues.back();
        if constexpr (calcError) {
            error = layerValues.back().squaredNorm();
        }
        layerValues.back() *= learningRate;
        biasessMod.back() += layerValues.back();
        connectionsMod.back() += layerValues.back() * layerValues[layerValues.size() - 2].transpose();
        for (int i = layerValues.size() - 2; i >= 1; i--) {
            layerValues[i] = layerValues[i].unaryExpr(&ActFunc::dir).cwiseProduct(connections[i].transpose() * layerValues[i + 1]);
            biasessMod[i - 1] += layerValues[i];
            connectionsMod[i - 1] += layerValues[i] * layerValues[i - 1].transpose();
        }
        return error;
    }

    void applyTraining(){
        for (size_t i = 0; i < NumLayers - 1; i++) {
            connections[i] += connectionsMod[i];
            biases[i]+=biasessMod[i];
            connectionsMod[i].setZero();
            biasessMod[i].setZero();
        }
    }

private:
    void construct() {
        for (size_t i = 0; i < NumLayers - 1; i++) {
            connections[i] = Eigen::MatrixXd::Random(LayerSizes[i + 1], LayerSizes[i]) / 100000.0;
            connectionsMod[i] = Eigen::MatrixXd::Zero(LayerSizes[i + 1], LayerSizes[i]);
            biases[i] = Eigen::VectorXd::Random(LayerSizes[i + 1]) / 100000.0;
            biasessMod[i] = Eigen::VectorXd::Zero(LayerSizes[i + 1]);
        }
    }
    constexpr double static learningRate = 1E-5;
    std::array<Eigen::MatrixXd, NumLayers - 1> connections;
    std::array<Eigen::MatrixXd, NumLayers - 1> connectionsMod;
    std::array<Eigen::VectorXd, NumLayers - 1> biases;
    std::array<Eigen::VectorXd, NumLayers - 1> biasessMod;
};

#endif