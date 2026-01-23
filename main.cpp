#include <cstddef>
#include <fstream>
#include <iostream>
#include <type_traits>
#include "Model.h"

template<class T>
concept FloatingPoint = std::is_floating_point_v<T>;
struct ReLU {
    template<FloatingPoint T>
    static constexpr T app(T n) noexcept { return std::max<T>(0, n); }
    template<FloatingPoint T>
    static constexpr T dir(T n) noexcept { return n > 0 ? 1 : 0; }
    template<FloatingPoint T>
    static constexpr std::complex<T> app(std::complex<T> n){
        return std::complex<T>(app(n.real()),app(n.imag()));
    }
    template<FloatingPoint T>
    static constexpr std::complex<T> dir(std::complex<T> n){
        return std::complex<T>(dir(n.real()),dir(n.imag()));
    }
};
struct LReLU {
    template<FloatingPoint T>
    static constexpr T app(T n) noexcept {
        if (n >= 0) {
            return n;
        }
        return n * leakRate;
    }
    template<FloatingPoint T>
    static constexpr T dir(T n) noexcept{
        if (n > 0) {
            return 1;
        } else
            return leakRate;
    }
    template<FloatingPoint T>
    static constexpr std::complex<T> app(std::complex<T> n){
        return std::complex<T>(app(n.real()),app(n.imag()));
    }
    template<FloatingPoint T>
    static constexpr std::complex<T> dir(std::complex<T> n){
        return std::complex<T>(dir(n.real()),dir(n.imag()));
    }
    static constexpr double leakRate = 0.01;
};

constexpr int32_t swapByte(int32_t n) noexcept {
    return (n << 24) + (n >> 24) + ((n << 8) & 0xFF0000) + ((n >> 8) & 0xFF00);
}


constexpr std::array layerSizes = {25,15, 10};
using ModelType = Model<float,LReLU, 28 * 28, layerSizes,true>;
using FloatType = ModelType::FloatType;
using VectorT = ModelType::VectorT;
using MatrixT = ModelType::MatrixT;

struct openFiles{
    std::ifstream imageFile;
    std::ifstream labelFile;
    int rows;
    int cols;
    uint64_t numTimes;
};

openFiles setUpFiles(const std::string &imageFileName, const std::string &labelFileName){
    std::ifstream imageFile(imageFileName, std::ios::binary);
    std::ifstream labelFile(labelFileName, std::ios::binary);
    if (!imageFile.is_open() || !labelFile.is_open()) {
        std::cout << "File didn't open\n";
        exit(1);
    }
    int32_t readInt;
    char *readIntPtr = reinterpret_cast<char *>(&readInt);
    imageFile.read(readIntPtr, sizeof(int32_t));
    if (swapByte(readInt) != 2051) {
        std::cout << swapByte(readInt) << " " << readInt << '\n';
        std::cout << "Image File has wrong magic number\n";
        exit(1);
    }
    labelFile.read(readIntPtr, sizeof(int32_t));
    if (swapByte(readInt) != 2049) {
        std::cout << swapByte(readInt) << " " << readInt << '\n';
        std::cout << "Label File has wrong magic number\n";
        exit(1);
    }
    imageFile.read(readIntPtr, sizeof(int32_t));
    uint64_t numTimes = swapByte(readInt);
    labelFile.read(readIntPtr, sizeof(int32_t));
    if (((uint64_t)swapByte(readInt)) != numTimes) {
        std::cout << "Wrong number of labels\n";
        exit(1);
    }
    imageFile.read(readIntPtr, sizeof(int32_t));
    int rows = swapByte(readInt);
    if (rows != 28) {
        std::cout << "Wrong number of rows\n";
        exit(1);
    }
    imageFile.read(readIntPtr, sizeof(int32_t));
    int cols = swapByte(readInt);
    if (cols != 28) {
        std::cout << "Wrong number of cols\n";
        exit(1);
    }
    openFiles ret;
    ret.cols = cols;
    ret.rows = rows;
    ret.numTimes = numTimes;
    ret.imageFile = std::move(imageFile);
    ret.labelFile = std::move(labelFile);
    return ret;
}

void runTest(const ModelType &model, const std::vector<MatrixT>& inputs, const std::vector<MatrixT>& outputs, bool printFailures) {
    assert(inputs.size()==outputs.size());
    uint64_t correctNum = 0;
    double totalError = 0;
    std::size_t totalNum = 0;
    for (uint64_t cinput = 0; cinput < inputs.size(); cinput++) {
        const auto& input = inputs[cinput];
        const auto& output = outputs[cinput];
        assert(input.rows()==28*28);
        totalNum+=input.cols();
        auto answer = model.runModel(input);
        assert(input.cols()==output.cols());
        assert(input.cols()==answer.cols());
        assert(answer.rows()==output.rows());
        totalError+=(answer-output).squaredNorm();
        static constexpr FloatType one = FloatType(1);
        for(long cCol = 0; cCol < output.cols(); cCol++){
            std::size_t givenAnswer = 0;
            std::size_t trueAnswer = 0;
            const auto outputCol = output.col(cCol);
            const auto answerCol = answer.col(cCol);
            for(long i = 1; i < outputCol.rows(); i++){
                if (std::abs(one-answerCol(i)) < std::abs(one-answerCol(givenAnswer))) {
                    givenAnswer = i;
                }
                if (std::abs(one-outputCol(i)) < std::abs(one-outputCol(trueAnswer))) {
                    trueAnswer = i;
                }
            }
            if(trueAnswer==givenAnswer){
                correctNum++;
                continue;
            }
            if(!printFailures){
                continue;
            }
            std::cout << answerCol(0);
            for (long i = 1; i < answerCol.rows(); i++) {
                std::cout << "," << answerCol(i);
            }
            std::cout << '\n';
            std::cout << "Highest: " << givenAnswer << " Correct: " << trueAnswer << '\n';
            for (std::size_t i = 0; i < 28*28; i++) {
                if (i % 28 == 0) {
                    std::cout << '\n';
                }
                auto cVal = input(i,cCol)*255;
                if (cVal <= 52) {
                    std::cout << "█";
                } else if (cVal <= 102) {
                    std::cout << "▓";
                } else if (cVal <= 154) {
                    std::cout << "▒";
                } else if (cVal <= 205) {
                    std::cout << "░";
                } else {
                    std::cout << " ";
                }
            }
            std::cout << "\n\n";
        }
    }
    std::cout << correctNum << " correctly identifed " << (totalNum-correctNum) << " incorrectly identied " << totalNum << " total\n";
    std::cout << 100.0*correctNum/totalNum << "% correct\n";
    std::cout << totalError/totalNum << " average error\n";
}

std::size_t loadData(const std::string &imageFileName, const std::string &labelFileName, std::vector<MatrixT>& inputs, std::vector<MatrixT>& outputs){
auto info = setUpFiles(imageFileName, labelFileName);
    std::ifstream imageFile = std::move(info.imageFile);
    std::ifstream labelFile = std::move(info.labelFile);
    uint64_t numData = info.numTimes;
    constexpr std::size_t perGroup = 512;
    // if(numData!=60000){
    //     std::cout << "wrong number of elements\n";
    //     exit(1);
    // }
    constexpr int rows = 28;
    constexpr int cols = 28;
    constexpr int imageDim = rows * cols;
    std::array<VectorT, 10> answerChoices;
    for (std::size_t i = 0; i < answerChoices.size(); i++) {
        answerChoices[i] = VectorT::Constant(answerChoices.size(),0);
        answerChoices[i][i] = 1.0;
    }
    std::array<unsigned char,imageDim> imageBuffer;
    const std::size_t numGropus = (numData-1+perGroup)/perGroup;
    inputs.resize(numGropus);
    outputs.resize(numGropus);
    std::size_t rem = numData;
    for(auto& input : inputs){
        input.resize(imageDim,std::min(rem,perGroup));
        rem-=perGroup;
    }
    rem = numData;
    for(auto& output : outputs){
        output.resize(answerChoices.size(),std::min(rem,perGroup));
        rem-=perGroup;
        output.fill(0);
    }
    for (uint64_t i = 0; i < numData; i++) {
        imageFile.read(reinterpret_cast<char*>(imageBuffer.data()), imageDim);
        inputs[i/perGroup].col(i%perGroup) = Eigen::Map<Eigen::Matrix<unsigned char,-1,1>>(imageBuffer.data(), imageDim, 1).cast<FloatType>()/255.0;
        outputs[i/perGroup](labelFile.get(),i%perGroup) = 1.0;
    }
    std::cout << "Data load done\n";
    return numData;
}

void shuffle(std::vector<MatrixT>& inputs, std::vector<MatrixT>& outputs){
    const std::size_t perGroup = inputs[0].cols();
    const std::size_t total = (inputs.size()-1)*perGroup+inputs.back().cols();
    for(std::size_t i = 0; i < total; i++){
        const std::size_t i2 = rand()%(total-i)+i;
        if(i2==i) continue;
        inputs[i/perGroup].col(i%perGroup).swap(inputs[i2/perGroup].col(i2%perGroup));
        outputs[i/perGroup].col(i%perGroup).swap(outputs[i2/perGroup].col(i2%perGroup));
    }
}

double runTraining(ModelType &model, std::vector<MatrixT>& inputs, std::vector<MatrixT>& outputs, std::size_t totalTrainingExamples) {
    shuffle(inputs,outputs);
    double totalError = 0;
    for(std::size_t i = 0; i < inputs.size(); i++){
        totalError += model.trainModel(inputs[i],outputs[i]);
        model.applyTraining();
    }
    if (std::isnan(totalError)) {
        std::cout << "we got a NaN\n";
        exit(1);
    }
    model.writeTo("weights");
    return totalError / totalTrainingExamples;
}

int main(int argc, const char** argv) {
    // Eigen::setNbThreads(12);
    srand((unsigned int)time(0));
    double learningRate = 1e-7;
    if(argc>=2){
        char* endPtr = nullptr;
        learningRate = std::strtof(argv[1],&endPtr);
        if(learningRate==0){
            std::cout << "learning Rate input not understood\n";
        }
    }
    std::cout << "learning rate: " << learningRate << '\n';
    bool printFailures = false;
    if(argc>=3){
        if(argv[2][0]=='y'){
            printFailures = true;
        }
    }
    ModelType model("weights");
    model.setLearningRate(learningRate);
    std::vector<MatrixT> inputs;
    std::vector<MatrixT> outputs;
    std::vector<MatrixT> testinputs;
    std::vector<MatrixT> testoutputs;
    bool increaseLearningRate = false;
    constexpr double learningRateAdjust = 1.1;
    const std::size_t totalTrainingExamples = loadData("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", inputs,outputs);
    loadData("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", testinputs,testoutputs);
    auto getNextLearningRate = [&increaseLearningRate,&learningRate](){
        return increaseLearningRate ? learningRate*learningRateAdjust : learningRate/learningRateAdjust;
    };
    auto doTrain = [&model, &inputs, &outputs, totalTrainingExamples](std::size_t times){
        for(std::size_t i = 0; i < times; i++){
            runTraining(model,inputs,outputs,totalTrainingExamples);
        }
        return runTraining(model,inputs,outputs,totalTrainingExamples);
    };
    runTest(model,inputs, outputs,false);
    runTest(model,testinputs,testoutputs,printFailures);
    std::cout << "Error now: " << doTrain(0) << '\n';
    double prevError = doTrain(20);
    while(true){
        std::cout << "Error now: " << prevError << '\n';
        double nextError = doTrain(10);
        const double eRatio1 = nextError/prevError;
        prevError = nextError;
        std::cout << "Error multiplied by: " << eRatio1 << '\n';
        std::cout << "Error now: " << prevError << '\n';
        model.setLearningRate(getNextLearningRate());
        nextError = doTrain(10);
        double eRatio2 = nextError/prevError;
        prevError = nextError;
        std::cout << "Error multiplied by: " << eRatio2 << '\n';
        if(eRatio2>eRatio1){
            increaseLearningRate = !increaseLearningRate;
            model.setLearningRate(learningRate);
        } else {
            learningRate = getNextLearningRate();
            std::cout << "learningRate is now " << learningRate << '\n';
        }
        runTest(model,testinputs,testoutputs,printFailures);
    }
}
