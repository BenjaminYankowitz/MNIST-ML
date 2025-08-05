#ifndef _MODEL__H__
#define _MODEL__H__
#include "Eigen/Core"
#include "Eigen/Dense"
#include <array>
#include <concepts>
#include <fstream>
#include <iostream>
//To do implement batch normilzation.
template <class ActFunc, class F>
concept LegalActFunc = requires(F a) {
    { ActFunc::app(a) } -> std::same_as<F>;
    { ActFunc::dir(a) } -> std::same_as<F>;
};


template <class F, LegalActFunc<F> ActFunc, std::size_t InputSize, std::array LayerSizes, bool SmoothFinal = true>
class Model {
    template<class T>
    struct IsComplexS{
        static constexpr bool value = false;
    };
    template<class T>
    struct IsComplexS<std::complex<T>>{
        static constexpr bool value = true;
    };
    static constexpr auto appF = [](F x) { return ActFunc::app(x); };
    static constexpr auto dirF = [](F x) { return ActFunc::dir(x); };
    static_assert(InputSize > 0, "Must give at least 1 input to model");
    static_assert(std::integral<typename decltype(LayerSizes)::value_type>, "LayerSizes have to be sizes");
    static constexpr std::size_t NumLayers = LayerSizes.size();
    static_assert(NumLayers >= 1, "Need at least 1 layer (output)");
    static_assert(std::ranges::all_of(LayerSizes, [](auto i) { return i>0; }), "Layer Sizes must be positive");
public:
    using FloatType = F;
    using VectorT = Eigen::VectorX<F>;
    using MatrixT = Eigen::MatrixX<F>;
    static constexpr bool IsComplex = IsComplexS<F>::value;
    void setLearningRate(double input){
        learningRate = input;
    }
    [[nodiscard]] constexpr F getLearningRate() const noexcept {
        return learningRate;
    }
    Model() {
        for (size_t i = 0; i < NumLayers; i++) {
            const std::size_t prevSize = i == 0 ? InputSize : LayerSizes[i-1];
            connections[i] = MatrixT::Random(LayerSizes[i], prevSize)/10;
            connectionsMod[i] = MatrixT::Zero(LayerSizes[i], prevSize);
            connectionsModV[i] = MatrixT::Zero(LayerSizes[i], prevSize);
            connectionsModS[i] = MatrixT::Zero(LayerSizes[i], prevSize);
            biases[i] = VectorT::Random(LayerSizes[i])/10;
            biasessMod[i] = VectorT::Zero(LayerSizes[i]);
            biasessModS[i] = VectorT::Zero(LayerSizes[i]);
            biasessModV[i] = VectorT::Zero(LayerSizes[i]);
        }
    }
    Model(const std::string &fileName) : Model(){
        std::ifstream input(fileName, std::ios::in | std::ios::binary);
        if (!input.is_open()) {
            return;
        }
        for (std::size_t i = 0; i < NumLayers; i++) {
            MatrixT &cM = connections[i];
            VectorT &cB = biases[i];
            input.read(reinterpret_cast<char *>(cM.data()), sizeof(cM.data()[0]) * cM.size());
            input.read(reinterpret_cast<char *>(cB.data()), sizeof(cB.data()[0]) * cB.size());
        }
        input.close();
    }

    void writeTo(const std::string &fileName) {
        std::ofstream output(fileName, std::ios::out | std::ios::binary);
        for (size_t i = 0; i < NumLayers; i++) {
            const MatrixT &cM = connections[i];
            const VectorT &cB = biases[i];
            output.write(reinterpret_cast<const char *>(cM.data()), sizeof(cM.data()[0]) * cM.size());
            output.write(reinterpret_cast<const char *>(cB.data()), sizeof(cB.data()[0]) * cB.size());
        }
        output.close();
    }

    [[nodiscard]] MatrixT runModel(const MatrixT &input) const {
        runForwardProp(input);
        return layerValues.back();
    }
    double trainModel(const MatrixT &input, const MatrixT &answers) {
        assert(input.cols() == answers.cols());
        assert(answers.rows()==LayerSizes.back());
        runForwardProp(input);
        double error = -1;
        if constexpr(SmoothFinal){
            auto toChange = answers - layerValues.back(); 
            error = toChange.squaredNorm();
            for(Eigen::Index i = 0; i < layerValues.back().cols(); i++){
                layerValues.back().col(i) = (toChange.col(i).array()-layerValues.back().col(i).dot(toChange.col(i))).matrix().cwiseProduct(layerValues.back().col(i));
            }
        } else {
            layerValues.back() = answers - layerValues.back();
            error = layerValues.back().squaredNorm();
        }
        biasessMod.back() += layerValues.back().rowwise().sum();
        connectionsMod.back() += layerValues.back() * layerValues[layerValues.size() - 2].transpose();
        for (int i = layerValues.size() - 1; i >= 1; i--) {
            layerValues[i-1] = layerValues[i-1].unaryExpr(dirF).cwiseProduct(connections[i].transpose() * layerValues[i]);
            biasessMod[i - 1] += layerValues[i-1].rowwise().sum();
            if(i==1){
                connectionsMod[i - 1] += layerValues[i-1] * input.transpose();
            } else {
                connectionsMod[i - 1] += layerValues[i-1] * layerValues[i - 2].transpose();
            }
        }
        return error;
    }

    void applyTraining() {
        for (size_t i = 0; i < NumLayers; i++) {
            connectionsModV[i] = connectionsMod [i]*momentumP + (1-momentumP)*connectionsMod[i];
            biasessModV    [i] = biasessModV    [i]*momentumP + (1-momentumP)*biasessMod    [i];
            connectionsModS[i] = connectionsModS[i]*sP + (1-sP)*connectionsMod[i].array().square().matrix();
            biasessModS    [i] = biasessModS    [i]*sP + (1-sP)*biasessMod   [i].array().square().matrix();
            connectionsMod[i].setZero();
            biasessMod    [i].setZero();
            connections[i] += learningRate*(connectionsModV[i].array()/(connectionsModS[i].array().sqrt()+epsilion)).matrix();
            biases     [i] += learningRate*(biasessModV    [i].array()/(biasessModS    [i].array().sqrt()+epsilion)).matrix();
        }
    }

private:
    void runForwardProp(const MatrixT &input) const{
        assert(input.rows()==InputSize);
        if constexpr (NumLayers>1){
            layerValues[0] = ((connections[0] * input).colwise() + biases[0]).unaryExpr(appF);
            for (std::size_t i = 1; i+1 < connections.size(); i++) {
                layerValues[i] = ((connections[i] * layerValues[i-1]).colwise() + biases[i]).unaryExpr(appF);
            }
        } else {
            layerValues[0] = ((connections[0] * input).colwise() + biases[0]);
        }
        layerValues.back() = (connections.back() * layerValues[layerValues.size() - 2]).colwise() + biases.back();
        if constexpr (SmoothFinal) {
            auto subV = [](auto mat){
                if constexpr(IsComplex){
                    return mat.array().abs().maxCoeff();
                }
                return mat.array().maxCoeff();
            };
            layerValues.back() = (layerValues.back().array()-subV(layerValues.back())).exp();
            for(auto col : layerValues.back().colwise()){
                col /= col.sum();
            }
        }
    }
    
    double learningRate = 1E-7;
    static constexpr double momentumP = 0.9;
    static constexpr double sP = 0.99;
    static constexpr double epsilion = 1e-8;
    std::array<MatrixT, NumLayers> connections;
    std::array<MatrixT, NumLayers> connectionsMod;
    std::array<MatrixT, NumLayers> connectionsModV;
    std::array<MatrixT, NumLayers> connectionsModS;
    std::array<VectorT, NumLayers> biases;
    std::array<VectorT, NumLayers> biasessMod;
    std::array<VectorT, NumLayers> biasessModV;
    std::array<VectorT, NumLayers> biasessModS;
    mutable std::array<MatrixT, NumLayers> layerValues;
};

#endif