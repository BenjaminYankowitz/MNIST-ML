#include <bit>
#include <fstream>
#include <iostream>

#include "Model.h"

static double lasterror = std::numeric_limits<double>::infinity();

struct ReLU {
    static double app(double n) { return std::max(0.0, n); }
    static double dir(double n) { return n >= 0; }
};
struct LRELU {
    static double app(double n) {
        if (n >= 0) {
            return n;
        }
        return n * leakRate;
    }
    static double dir(double n) {
        if (n >= 0) {
            return 1;
        } else
            return leakRate;
    }
    static constexpr double leakRate = 0.01;
};

int32_t swapByte(int32_t n) {
    return (n << 24) + (n >> 24) + ((n << 8) & 0xFF0000) + ((n >> 8) & 0xFF00);
}

constexpr std::array layerSizes = {28 * 28, 800, 10};
using ModelType = Model<LRELU, layerSizes.size(), layerSizes,true>;

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

void runTest(const ModelType &model, const std::string &imageFileName, const std::string &labelFileName) {
    auto info = setUpFiles(imageFileName, labelFileName);
    std::ifstream imageFile = std::move(info.imageFile);
    std::ifstream labelFile = std::move(info.labelFile);
    uint64_t numTimes = info.numTimes;
    constexpr int rows = 28;
    constexpr int cols = 28;
    constexpr int imageDim = rows * cols;
    unsigned char readChar;
    unsigned char imageBuffer[imageDim];
    uint64_t correctNum = 0;
    for (uint64_t i = 1; i <= numTimes; i++) {
        imageFile.read(reinterpret_cast<char *>(imageBuffer), imageDim);
        Eigen::VectorXd input = Eigen::Map<Eigen::Matrix<unsigned char, -1, 1>>(imageBuffer, imageDim, 1).cast<double>()/255.0;
        labelFile.read(reinterpret_cast<char *>(&readChar), 1);
        auto output = model.runModel(std::move(input));
        
        std::size_t mi = 0;
        for (long i = 0; i < output.rows(); i++) {
            if (output(i) > output(mi)) {
                mi = i;
            }
        }
        if(mi==readChar){
            correctNum++;
            continue;
        }
        // for (long i = 0; i < output.rows(); i++) {
        //     std::cout << i << ": " << output(i) << "\n";
        // }
        // std::cout << "Highest: " << mi << " Correct: " << ((int)readChar) << '\n';
        // for (std::size_t i = 0; i < imageDim; i++) {
        //     if (i % cols == 0) {
        //         std::cout << '\n';
        //     }
        //     if (imageBuffer[i] <= 52) {
        //         std::cout << "█";
        //     } else if (imageBuffer[i] <= 102) {
        //         std::cout << "▓";
        //     } else if (imageBuffer[i] <= 154) {
        //         std::cout << "▒";
        //     } else if (imageBuffer[i] <= 205) {
        //         std::cout << "░";
        //     } else {
        //         std::cout << " ";
        //     }
        // }
        // std::cout << "\n\n";
    }
    std::cout << correctNum << " correctly identifed " << (numTimes-correctNum) << " incorrectly identied " << numTimes << " total\n";
    std::cout << 100.0*correctNum/numTimes << "% correct\n";
    imageFile.close();
    labelFile.close();
}

void runTraining(ModelType &model, const std::string &imageFileName, const std::string &labelFileName, uint64_t numTimes) {
    auto info = setUpFiles(imageFileName, labelFileName);
    std::ifstream imageFile = std::move(info.imageFile);
    std::ifstream labelFile = std::move(info.labelFile);
    uint64_t numData = info.numTimes;
    if(numData!=60000){
        std::cout << "wrong number of elements\n";
        exit(1);
    }
    constexpr int rows = 28;
    constexpr int cols = 28;
    constexpr int imageDim = rows * cols;
    unsigned char readChar;
    std::array<std::array<double, 10>, 10> answerChoices;
    for (std::size_t i = 0; i < answerChoices.size(); i++) {
        answerChoices[i].fill(0);
        answerChoices[i][i] = 1.0;
    }
    unsigned char imageBuffer[imageDim];
    std::vector<Eigen::VectorXd> inputs;
    std::vector<char> labeles;
    for (uint64_t i = 0; i < numData; i++) {
        imageFile.read(reinterpret_cast<char *>(imageBuffer), imageDim);
        labelFile.read(reinterpret_cast<char *>(&readChar), 1);
        labeles.push_back(readChar);
        inputs.push_back(Eigen::Map<Eigen::Matrix<unsigned char, -1, 1>>(imageBuffer, imageDim, 1).cast<double>()/255.0);
    }
    imageFile.close();
    labelFile.close();
    for(uint64_t i2 = 0; i2 < numTimes; i2++){
        double totalError = 0;
        for(uint64_t i = 0; i < numData; i++){
            totalError += model.trainModel<true>(inputs[i], answerChoices[labeles[i]]);
            if(i%10000==0){
                model.applyTraining();
            }
        }
        if (std::isnan(totalError)) {
            std::cout << "we got a NaN\n";
            exit(1);
        }
        model.applyTraining();
        std::cout << totalError / numData << ",";
        if(lasterror<totalError){
            std::cout << '\n';
            model.divideLearningRate();
        } else {
            lasterror = totalError;
        }
        model.writeTo("weights");
    }
    std::cout << '\n';
}

int main() {
    srand((unsigned int)time(0));
    // rand();
    ModelType model("weights");
    for(int i = 0; i < 1000; i++){
        runTest(model, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
        runTraining(model, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte",4);
    }
    model.writeTo("weights");
}
