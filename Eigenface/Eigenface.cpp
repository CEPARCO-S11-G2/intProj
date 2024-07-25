// Eigenface.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <string>
#include <Eigen>
#include <Dense>
#include <Eigenvalues>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace cv;
using namespace Eigen;
namespace fs = std::filesystem;

extern "C" int asmNorm(uint16_t* img, uint16_t* f, uint16_t counter);
extern "C" int asmNormDiv(uint16_t * f, uint8_t counter, uint8_t* norm, uint16_t length);
extern "C" int subtNorm(uint8_t * f, uint8_t* i, uint16_t length);
extern "C" long int covMat(uint8_t * img1, uint8_t * img2, uint16_t length);
extern "C" int jacobify(double *ortho, double *orthoT, double *cov, uint8_t length);
extern "C" int asmweights(double* eV, double* imgVector, double *weights , uint8_t count, uint16_t length);
extern "C" int eigPart1(double* weights, double * img, double* wPerImgPtr, uint8_t counter, uint16_t length, int imgN);
extern "C" int eigPart2(double* weights, double* wPerImgPtr, double* eigFace, uint8_t counter, uint16_t length, int imgN);

#define trsize 4
#define width 128
#define height 128
#define precision 1e-3
#define setting true

static array<Mat, trsize> getImgtoGray()
{
    array<Mat, trsize> images;
    string path = "TrainingImages/";
    string imgpath = "";
    int count = 0;
    Mat color;
    Mat gray;
    Mat resized;
    for (const auto& entry : fs::directory_iterator(path))
    {
        if (count >= trsize)
            break;
        imgpath = (entry.path()).string();
        cout << imgpath << endl;
        color = imread(imgpath);
        resize(color, resized, Size(width, height));
        cvtColor(resized, gray, COLOR_BGR2GRAY);
        images[count] = gray;
        //imshow("grayimage", gray);
        //waitKey(0);
        resized.release();
        color.release();
        gray.release();
        count++;
    }
    //cout << count << endl;

    return images;
}

void vectorPrinting(vector<uint8_t> v)
{
    int i = 0;
    for (; i < v.size();i++)
    {
        cout << (int)v[i] << " ";
    }
    cout << endl << i;
}

void matrixPrinting(MatrixXd m)
{
    for (int r = 0; r < m.rows(); r++)
    {
        for (int c = 0; c < m.cols(); c++)
        {
            if (c == r)
                cout << "'" << m(r, c) << "'";
            else
                cout << m(r, c) << " ";
        }
        cout << endl;
    }
}

array<vector<uint8_t>,trsize> imgToVector(array<Mat,trsize> images)
{
    array<vector<uint8_t>,trsize> imgVectors;
    Mat grayImage;
    vector<uint8_t> imageVector;
    for (int i = 0; i < images.size(); i++)
    {
        grayImage = images[i];
        if (grayImage.isContinuous()) {
                imageVector.assign(grayImage.data, grayImage.data + grayImage.total() * grayImage.channels());
            }
            else {
                for (int i = 0; i < grayImage.rows; ++i) {
                    imageVector.insert(imageVector.end(), grayImage.ptr<uint8_t>(i), grayImage.ptr<uint8_t>(i) + grayImage.cols*grayImage.channels());
                }
            }
        imgVectors[i] = imageVector;
        grayImage.release();
        imageVector.clear();
    }
    return imgVectors;
}

vector<uint8_t> inputToVector(Mat image)
{
    vector<uint8_t> inputVector;
    if (image.isContinuous()) {
        inputVector.assign(image.data, image.data + image.total() * image.channels());
    }
    else {
        for (int i = 0; i < image.rows; ++i) {
            inputVector.insert(inputVector.end(), image.ptr<uint8_t>(i), image.ptr<uint8_t>(i) + image.cols * image.channels());
        }
    }

    return inputVector;
}

vector<uint8_t> asmNormalize(array<vector<uint8_t>, trsize> &imgVectors)
{
    //vector<uint8_t> imageVector;
    uint16_t* imageVector;
    uint8_t* normVector;
    vector<uint16_t> tempImage(width*height,0);
    //vector<uint16_t> currImage(imgVectors[1].begin(),imgVectors[1].end());
    uint16_t* finalImage;

    for (int i = 0; i < trsize; i++) {
        vector<uint16_t> currImage(imgVectors[i].begin(), imgVectors[i].end());
        imageVector = currImage.data();
        finalImage = tempImage.data();
        asmNorm(imageVector,finalImage,width*height);
        currImage.clear();
    }
    finalImage = tempImage.data();

    vector<uint8_t> normalized(width*height, 0);
    uint8_t* norm = normalized.data();
   /* for (int i = 0; i < width * height; i++)
    {
        norm[i] = static_cast<uint8_t>(finalImage[i] / trsize);
    }*/
    asmNormDiv(finalImage, trsize, norm, width*height);
    for (int i = 0; i < trsize; i++)
    {
        vector<uint8_t> currImage(imgVectors[i].begin(), imgVectors[i].end());
        normVector = currImage.data();
        norm = normalized.data();
        subtNorm(norm, normVector, width*height);
        imgVectors[i] = currImage;
    }

    return normalized;
}

vector<uint8_t> normalize(array<vector<uint8_t>, trsize> &imgVectors)
{
    vector<uint8_t> imageVector(width*height,0);
    vector<uint8_t> currImage;
    vector<uint16_t> tempImage(width * height, 0);
    tempImage.reserve(imageVector.size());
    int imsize = imgVectors[1].size();


    for (int i = 0; i < imsize; i++)
    {
        for (int im = 0; im < trsize; im++)
        {
            currImage = imgVectors[im];
            tempImage[i] += static_cast<uint16_t>(currImage[i]);
        }
        imageVector[i] = static_cast<uint8_t>(tempImage[i]/trsize);
        for (int im = 0; im < trsize; im++)
        {
            imgVectors[im][i] = (imgVectors[im][i] > imageVector[i]) ? (imgVectors[im][i] - imageVector[i]) : 0;
        }
    }
    return imageVector;
}

MatrixXd getCovariance(array<vector<uint8_t>,trsize> imgVectors, float &max, int &maxRow, int &maxCol)
{
    MatrixXd covarianceMatrix(trsize, trsize);
    long int currval = 0;
    int imgCount = 0;

    vector<uint8_t> imgVector1;
    vector<uint8_t> imgVector2;
    while (imgCount < trsize)
    {
        imgVector1 = imgVectors[imgCount];
        for (int i = 0; i < trsize; i++)
        {
            imgVector2 = imgVectors[i];
            for (int y = 0; y < width*height; y++)
            {
                currval += (static_cast<long int>(imgVector1[y]) * static_cast<long int>(imgVector2[y]));
            }
            currval /= width * height;
            //cout << "Currval of NormalCov at : i = " << imgCount << " and x = " << i << " : " << currval << endl;
            if (abs(currval) > max && i!=imgCount)
            {
                max = abs(currval);
                maxRow = imgCount;
                maxCol = i;
            }
            covarianceMatrix(imgCount, i) = currval;
            currval = 0;
        }

        currval = 0;
        imgCount++;
    }
    return covarianceMatrix;
}

MatrixXd asmCovariance(array<vector<uint8_t>, trsize> imgVectors,float &max, int& maxRow, int& maxCol)
{
    MatrixXd covarianceMatrix(trsize, trsize);
    uint8_t* imgVector1;
    uint8_t* imgVector2;
    long int currval=0;

    for (int i = 0; i < trsize; i++)
    {
        
        for (int x = 0; x < trsize; x++)
        {
            imgVector1 = imgVectors[i].data();
            imgVector2 = imgVectors[x].data();

            currval = covMat(imgVector1, imgVector2, width * height);
            //cout << "Currval of ASMCov at : i = " << i << " and x = " << x << " : " << currval << endl;
            if (abs(currval) > max && i != x)
            {
                max = abs(currval);
                maxRow = x;
                maxCol = i;
            }
           
            covarianceMatrix(i, x) = currval;
            currval = 0;
        }
    }

    return covarianceMatrix;
}

MatrixXd getJacobied(MatrixXd covariance, MatrixXd &eigenvectors, float &max, int &maxRow, int &maxCol)
{
    //Get Maximum Diagonal Value
    MatrixXd orthagonal = MatrixXd::Identity(trsize, trsize);
    MatrixXd jacobiedFinal = MatrixXd::Zero(trsize, trsize);

    double maxi = 0;
    int maxiRow = 0, maxiCol = 0;
    float theta =(2 * max) / (covariance(maxRow, maxRow) - covariance(maxCol, maxCol));
    theta *= M_PI / 180.0;
    theta = atan(theta) / 2;
    double cosT = cos(theta);
    double sinT = sin(theta);
    orthagonal(maxRow, maxRow) = cosT;
    orthagonal(maxCol, maxCol) = cosT;
    orthagonal(maxRow, maxCol) = sinT;
    orthagonal(maxCol, maxRow) = -sinT;

    if (eigenvectors.isZero())
        eigenvectors = orthagonal;

    /*matrixPrinting(covariance);
    matrixPrinting(orthagonal.transpose());
    waitKey(0);*/
    jacobiedFinal = orthagonal.transpose() * covariance* orthagonal;
    /*matrixPrinting(jacobiedFinal);*/
    //cout << endl << "Jacobied " << endl;
    //matrixPrinting(jacobiedFinal);
    //waitKey(0);

    eigenvectors *= orthagonal;

    for (int i = 0; i < trsize; i++) {
        for (int x = 0; x < trsize; x++) {
            if (i != x && abs(jacobiedFinal(i, x)) > maxi) {
                maxi = abs(jacobiedFinal(i, x));
                maxiRow = i;
                maxiCol = x;
            }
        }
    }

    max = maxi;
    maxRow = maxiRow;
    maxCol = maxiCol;
    return jacobiedFinal;
}

void asmJacobify(MatrixXd &covariance, MatrixXd &eigenvectors, float& max, int& maxRow, int& maxCol)
{
    MatrixXd orthagonal = MatrixXd::Identity(trsize, trsize);
    MatrixXd jacobiedPartial = MatrixXd::Zero(trsize, trsize);
    MatrixXd jacobiedFinal = MatrixXd::Zero(trsize, trsize);
    MatrixXd currEigen = eigenvectors;
    MatrixXd tempEigen = MatrixXd::Zero(trsize, trsize);
    MatrixXd covarianceR = covariance;
    
    float maxi = max;
    int maxiRow = maxRow, maxiCol = maxCol;
    double* orthPtr, *covPtr, *jF, *jP, *eV, *tV;
    if (currEigen.isZero())
        currEigen = orthagonal;
    float theta;
    double cosT, sinT;

    orthPtr = orthagonal.data();
    covPtr = covarianceR.data();
    jP = jacobiedPartial.data();
    jF = jacobiedFinal.data();
    eV = currEigen.data();
    tV = tempEigen.data();

    while (!covarianceR.isDiagonal(precision))
    {
        orthagonal = MatrixXd::Identity(trsize, trsize);
        theta = (2 * maxi) / (covarianceR(maxiRow, maxiRow) - covarianceR(maxiCol, maxiCol));
        theta *= M_PI / 180.0;
        theta = atan(theta) / 2;
        cosT = cos(theta);
        sinT = sin(theta);
        orthagonal(maxiRow, maxiRow) = cosT;
        orthagonal(maxiCol, maxiCol) = cosT;
        orthagonal(maxiRow, maxiCol) = sinT;
        orthagonal(maxiCol, maxiRow) = -sinT;

        
        maxiRow = 0;
        maxiCol = 0;
        maxi = 0;
        //cout << covarianceR.size() << " " << orthagonal.size() << " " << jacobiedPartial.size() << " " << jacobiedFinal.size() << endl;
        jacobify(covPtr, jP, orthPtr, trsize);
       // cout << covarianceR.size() << " " << orthagonal.size() << " " << jacobiedPartial.size() << " " << jacobiedFinal.size() << endl;
        jacobiedPartial.transposeInPlace();
        //cout << covarianceR.size() << " " << orthagonal.size() << " " << jacobiedPartial.size() << " " << jacobiedFinal.size() << endl;
        jacobify(jP, jF, orthPtr, trsize);
        //cout << covarianceR.size() << " " << orthagonal.size() << " " << jacobiedPartial.size() << " " << jacobiedFinal.size() << endl;
        //currEigen.transposeInPlace();
        jacobify(eV, tV, orthPtr, trsize);
        //cout << covarianceR.size() << " " << orthagonal.size() << " " << jacobiedPartial.size() << " " << jacobiedFinal.size() << endl;
        

        currEigen = tempEigen;
        covarianceR = jacobiedFinal;
        /*cout << covarianceR << endl <<endl;
        waitKey(0);*/
        for (int i = 0; i < trsize; i++) {
            for (int x = 0; x < trsize; x++) {
                if (i != x && abs(jacobiedFinal(i, x)) > maxi) {
                    maxi = abs(jacobiedFinal(i, x));
                    maxiRow = i;
                    maxiCol = x;
                }
            }
        }
        //
        // currEigen.transposeInPlace();
    }
    currEigen.transposeInPlace();
    eigenvectors = currEigen;
    covariance = jacobiedFinal;
    //return jacobiedFinal;
}

MatrixXd weightCalcs(MatrixXd eigenvectors, array<vector<uint8_t>, trsize> imgVectors)
{
    int size = imgVectors[0].size();
    cout << size << endl;
    MatrixXd weights = MatrixXd::Zero(trsize, size);
    
    chrono::time_point begin = chrono::steady_clock::now();
    for (int i = 0; i < size; i++)
    {
        for (int x = 0; x < trsize; x++)
        {
            for (int y = 0; y < trsize; y++)
            {
                weights(x, i) += static_cast<double>(imgVectors[y][i]) * eigenvectors(y, x);
                //cout << static_cast<double>(weights(x,i)) << " " << endl;
            }
        }
    }
    cout << "rows: " << weights.rows() << " cols: " << weights.cols();
    chrono::time_point end = chrono::steady_clock::now();
    cout << "Time elapsed in millisecconds: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;
    return weights;
}

MatrixXd asmWeights(MatrixXd eigenvectors, array<vector<uint8_t>, trsize> imgVectors)
{
    double* eV = eigenvectors.data(); // column major
    double* imgVector, * weightPtr;
    MatrixXd weights = MatrixXd::Zero(trsize, width*height);
    MatrixXd imgVectorsM(trsize, width * height);
    weightPtr = weights.data();
    uint16_t size = width * height;
    //Preprocessing, a sacrifice we can take because of speed up.
    for (int i = 0; i < trsize; i++)
    {
        for (int x = 0; x < size; x++)
        {
            imgVectorsM(i, x) = static_cast<double>(imgVectors[i][x]);
        }
    }
    imgVector = imgVectorsM.data();
    cout << imgVectorsM(0,0) << " " << imgVectorsM(1, 0) << " " << imgVectorsM(2, 0) << " " << imgVectorsM(3, 0) << " " << endl;

    asmweights(eV, imgVector, weightPtr, trsize,size);

    return weights;
}

MatrixXd eigenFaces(MatrixXd weights, array<vector<uint8_t>, trsize> imgVectors)
{
    int size = width * height;
    MatrixXd eigenFace = MatrixXd::Zero(trsize, size);
    vector<double> wPerImage(trsize, 1);
    vector<uint8_t> currImg;

    for (int img = 0; img < trsize; img++)
    {
        currImg = imgVectors[img];
        fill(wPerImage.begin(), wPerImage.end(), 0);
        for (int x = 0; x < trsize; x++)
        {
            for (int y = 0; y < size; y++)
            {
                wPerImage[x] += weights(x, y) * currImg[y];
            }
            wPerImage[x] /= size;
        }
        for (int x = 0; x < size; x++)
        {
            for (int y = 0; y < trsize; y++)
            {
                eigenFace(img, x) +=  weights(y, x) * wPerImage[y];
            }
            eigenFace(img, x) /= trsize;
        }
    }

    return eigenFace;
}

MatrixXd asmEigenFace(MatrixXd weights, array<vector<uint8_t>, trsize> imgVectors)
{
    uint16_t size = width * height;
    MatrixXd eigenFace = MatrixXd::Zero(size, trsize);
    vector<double> wPerImage(trsize, 1);
    MatrixXd imgVectorsM = MatrixXd::Zero(trsize, size);
    MatrixXd weightsL = weights;
    double* weightPtr;
    double* ImgPtr;
    double* wPerImgPtr = wPerImage.data();
    double* eigenF = eigenFace.data();

    for (int i = 0; i < trsize; i++)
    {
        for (int x = 0; x < size; x++)
        {
            imgVectorsM(i, x) = static_cast<double>(imgVectors[i][x]);
        }
    }
    ImgPtr = imgVectorsM.data();
    imgVectorsM.transposeInPlace();
    
    weightPtr = weightsL.data();
     
    for (int img = 0; img < trsize; img++)
    {

        weightsL.transposeInPlace();
        fill(wPerImage.begin(), wPerImage.end(), 0);
        eigPart1(weightPtr, ImgPtr, wPerImgPtr ,trsize, size, img);
        cout << wPerImage[0] << " " << wPerImage[1] << " " << wPerImage[2] << " " << wPerImage[3] << endl;
        weightsL.transposeInPlace();
        eigPart2(weightPtr, wPerImgPtr, eigenF, trsize, size, img);
        cout << eigenFace(img, 0) << " " << eigenFace(img, 1) << " " << eigenFace(img, 2) << " " << eigenFace(img, 3) << endl;
    }
    
    eigenFace.transposeInPlace();
    return eigenFace;
}

vector<double> recognition(vector<uint8_t> inputImage, MatrixXd weights)

{
    int size = inputImage.size();
    vector<double> inputEigen(size, 1);
    vector<double> inputWeight(trsize,1);
   
    for (int weightIndex = 0; weightIndex < trsize; weightIndex++)
    {
        inputWeight[weightIndex] = 0;
        for (int i = 0; i < size; i++)
        {
            inputWeight[weightIndex] +=  static_cast<double>(weights(weightIndex, i) * inputImage[i]);            
        }
        inputWeight[weightIndex] /= size;
        cout << "IP: " << inputWeight[weightIndex] << endl;
    }   
    for (int i = 0; i < size; i++)
    {   
        inputEigen[i] = 0;
        for (int x = 0; x < trsize; x++)
        {
            inputEigen[i] += static_cast<double>(weights(x, i) * inputWeight[x]);
        
        }
        //matrixMultiply(weights,inputWeight,trsize,size)
        inputEigen[i] /= trsize;
    }

    return inputEigen;
}

vector<double> asmRecognition(vector<uint8_t> inputImage, MatrixXd weights)
{
    int size = inputImage.size();
    vector<double> inputEigen(inputImage.begin(),inputImage.end());
    vector<double> inputEigFace(size, 1);
    vector<double> inputWeight(trsize, 1);
    MatrixXd weightsL = weights;
    double* weightPtr = weightsL.data();
    double* ImgPtr = inputEigen.data();
    double* wPerImgPtr = inputWeight.data();
    double* eigenF = inputEigFace.data();

    fill(inputEigFace.begin(), inputEigFace.end(), 0);
    fill(inputWeight.begin(), inputWeight.end(), 0);
    weightsL.transposeInPlace();
    eigPart1(weightPtr, ImgPtr, wPerImgPtr, trsize, size, 0);
    weightsL.transposeInPlace();
    eigPart2(weightPtr, wPerImgPtr, eigenF, trsize, size, 0);

    return inputEigFace;


}

int euclideanDistances(MatrixXd Eigenfaces, vector<double> inputEigen)
{
    array<double, trsize> eds;
    int size = width * height;
    double temp = 0;
    int id = 0;
    for (int index=0;index < trsize;index++)
    {
        for (int i = 0;i < size; i++)
        {
            temp += pow(static_cast<double>((Eigenfaces(index, i) - inputEigen[i])), 2);
        }
        eds[index] = sqrt(temp);
        cout << endl;
        temp = 0;
    }
    //for (int i = 0; i < eds.size(); i++)
    //{
    //    cout << eds[i] << endl;
    //}

    auto min = min_element(eds.begin(), eds.end());
    id = distance(eds.begin(), min);
    return id;

}

void vidCap(MatrixXd &eigenvectors, MatrixXd &eigenfaces, vector<uint8_t> normal, MatrixXd weights, bool cond)
{
    VideoCapture cap(0);
    string name = "";
    Mat img;
    int count = 0;
    Mat resized;
    Mat gray;
    vector<uint8_t> inputVector(width*height,0);
    vector<double> inputEigen(width*height,0);
    int index;
    string path;
    while (true)
    {
        //Mat img;
        //Mat resized;
        //Mat gray;
        //vector<uint8_t> inputVector;
        //vector<double> inputEigen;
        fill(inputVector.begin(), inputVector.end(), 0);
        fill(inputEigen.begin(), inputEigen.end(), 0);
        path = "TrainingImages/";
        
        if(true)
        {
           cap.read(img);// get image from vidcam
        }
        
        waitKey(30);
        resize(img, resized, Size(width, height));
        cvtColor(resized, gray, COLOR_BGR2GRAY);
        imshow("gray input", gray);
        inputVector = inputToVector(gray);
        fill(inputEigen.begin(), inputEigen.end(), 0.0);
        for (int i = 0; i < inputVector.size(); i++)
        {
            inputVector[i] -=  normal[i];
        }
        //cout << "check0";
        if (cond)
        {
            inputEigen = asmRecognition(inputVector, weights);
        }
        else
        {
            inputEigen = recognition(inputVector, weights);
        }
        //waitKey(0);
        cout << "check1";
        index = euclideanDistances(eigenfaces, inputEigen);
        count++;
        cout << "check2";
        path.append(to_string(index + 1));
        path.append(".jpg");
        cout << path << endl;
        try
        {
            img = imread(path);
        }
        catch (Exception e) { cout << e.what() << endl;continue; }
        
        cout << count << endl;
        //cout << "check3";
        imshow("Closest Image", img);
        resized.release();
        gray.release();
        img.release();
        waitKey(100);//optional stopper
    }
}

int main()
{
    //Training Phase
    //uint8_t imgVectors[trsize] = imgToVector(images);
    array<Mat, trsize> images;
    images = getImgtoGray();
    float max = 0;
    int maxRow = 0;
    int maxCol = 0;
    array<vector<uint8_t>, trsize> imgVectors = imgToVector(images);
    array<double,trsize> eigenValues;
    MatrixXd eigenvectors(trsize, trsize);
    eigenvectors.setZero();

    bool asmCond = setting;

    if (asmCond) //ASM SPEED
    {
        chrono::time_point totalBegin = chrono::steady_clock::now();
        chrono::time_point begin = chrono::steady_clock::now();
        vector<uint8_t> asmNormal = asmNormalize(imgVectors);
        chrono::time_point end = chrono::steady_clock::now();
        cout << "Time elapsed in millisecconds: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;

        cout << static_cast<int>(asmNormal[0]) << " " << static_cast<int>(asmNormal[5]) << " " << static_cast<int>(asmNormal[6]) << endl;
        Mat image2(width, height, CV_8UC1, imgVectors[0].data());
        imshow("asm image", image2);
        waitKey(20);

        cout << static_cast<int>(imgVectors[0][0]) << " " << static_cast<int>(imgVectors[0][1]) 
            << static_cast<int>(imgVectors[1][2]) << " " << static_cast<int>(imgVectors[1][3]) << endl;

        begin = chrono::steady_clock::now();
        MatrixXd asmCovarianceMat = asmCovariance(imgVectors, max, maxRow, maxCol);
        end = chrono::steady_clock::now();
        cout << "Time elapsed in millisecconds: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;
        waitKey(20);

        matrixPrinting(asmCovarianceMat);
        MatrixXd jacobied = asmCovarianceMat;
        begin = chrono::steady_clock::now();
        asmJacobify(jacobied, eigenvectors, max, maxRow, maxCol);
        end = chrono::steady_clock::now();
        cout << "Time elapsed in millisecconds: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;

        cout << endl << "Eigenvalues" << endl;
        matrixPrinting(jacobied);

        cout << endl << endl << "Eigenvectors" << endl;
        matrixPrinting(eigenvectors);

        //transfer EigenValues
        for (int i = 0; i < trsize; i++)
            eigenValues[i] = jacobied(i, i);

        begin = chrono::steady_clock::now();
        MatrixXd asmweights = asmWeights(eigenvectors, imgVectors);
        end = chrono::steady_clock::now();
        cout << "Time elapsed in millisecconds: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;

        begin = chrono::steady_clock::now();
        MatrixXd asmeigenface = asmEigenFace(asmweights, imgVectors);
        end = chrono::steady_clock::now();

        chrono::time_point totalEnd = chrono::steady_clock::now();
        cout << "Time elapsed in millisecconds: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;
        cout << "Total Time elapsed in millisecconds: " << chrono::duration_cast<chrono::milliseconds>(totalEnd - totalBegin).count() << endl;

        waitKey(0);
        vidCap(eigenvectors, asmeigenface, asmNormal, asmweights, asmCond);
    }
    else
    {
        chrono::time_point totalBegin = chrono::steady_clock::now();
        chrono::time_point begin = chrono::steady_clock::now();
        vector<uint8_t> normalImage = normalize(imgVectors); //   replace with normal/asm call and vice versa
        chrono::time_point end = chrono::steady_clock::now();
        //imgVectors mean-centered by nomralize function.
        //vectorPrinting(normalImage);
        cout << "Time elapsed in millisecconds: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;
        cout << static_cast<int>(normalImage[0]) << " " << static_cast<int>(normalImage[5]) << " " << static_cast<int>(normalImage[6]) << endl;
        Mat image(width, height, CV_8UC1,  imgVectors[0].data());

        imshow("normal image", image);
        waitKey(20);

        cout << static_cast<int>(imgVectors[0][0]) << " " << static_cast<int>(imgVectors[0][1])
            << static_cast<int>(imgVectors[1][2]) << " " << static_cast<int>(imgVectors[1][3]) << endl;

        begin = chrono::steady_clock::now();
        MatrixXd covarianceMatrix = getCovariance(imgVectors, max, maxRow, maxCol);
        end = chrono::steady_clock::now();
        cout << "Time elapsed in millisecconds: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;

        waitKey(20);
        /* PRINT COVARIANCE MATRIX
        */
        matrixPrinting(covarianceMatrix);

        MatrixXd jacobied = covarianceMatrix;
    
        begin = chrono::steady_clock::now();
        do
        {
            //matrixPrinting(eigenvalues);
            //waitKey(0);
            jacobied = getJacobied(jacobied, eigenvectors, max, maxRow, maxCol);
            /*cout << jacobied << endl;
            waitKey(0);*/
        } while (!jacobied.isDiagonal(precision));
        end = chrono::steady_clock::now();
        cout << "Time elapsed in millisecconds: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;
        /* PRINT Eigen MATRIX
    */

        cout << endl << "Eigenvalues" << endl;
        matrixPrinting(jacobied);

        cout << endl << endl << "Eigenvectors" << endl;
        matrixPrinting(eigenvectors);

        //transfer EigenValues
        for (int i = 0; i < trsize; i++)
            eigenValues[i] =jacobied(i,i);

        //Weights
        begin = chrono::steady_clock::now();
        MatrixXd weights = weightCalcs(eigenvectors, imgVectors);
        end = chrono::steady_clock::now();
        cout << "Time elapsed in millisecconds: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;

        //Eigenfaces
        begin = chrono::steady_clock::now();
        MatrixXd eigenfaces = eigenFaces(weights, imgVectors);
        end = chrono::steady_clock::now();
        chrono::time_point totalEnd = chrono::steady_clock::now();
        cout << "Time elapsed in millisecconds: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;
        cout << "Total Time elapsed in millisecconds: " << chrono::duration_cast<chrono::milliseconds>(totalEnd - totalBegin).count() << endl;
        /*cout << endl << endl << "Transformed Image Matrix" << endl;
        matrixPrinting(eigenFaces);*/
        waitKey(0);
        vidCap(eigenvectors, eigenfaces, normalImage, weights, asmCond);
    }



    return 0;
}
