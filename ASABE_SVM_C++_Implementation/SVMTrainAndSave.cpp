#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <list>
#include <array>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <fstream>


using namespace std;
using namespace cv;
using namespace cv::ml;

string type2str(int type);

int main(int argc, char const *argv[])
{
	//variant to save training set
	Mat trainData,testData,trainlabel,testlabel;
	float prop = 0.9;

	std::list<std::string> filelist;
	std::list<std::string>::iterator it;
		

	for(int i=1; i < 5; i++)
	{
		// std::string name("trainData");
		filelist.push_back("train_xml/trainData" + std::to_string(i) + ".xml");
	}
	it = filelist.begin();

	for(int i=0; it !=filelist.end();it++)
	{
		//read matrix from xml file
		Mat tempData;
		cout<<"file's name is :"<< *it << endl;
		FileStorage fs(*it,FileStorage::READ);
		fs["trainData"] >> tempData;

		int rows = tempData.rows;
		//create labels for trainData
		Mat templab(rows,1,CV_8UC1,Scalar(i+1));

		trainData.push_back(tempData(Range(0,(int)(prop*rows)),Range::all()));
		testData.push_back(tempData(Range((int)(prop*rows),rows), Range::all()));
		trainlabel.push_back(templab(Range(0,(int) (prop*rows)),Range::all()));
		testlabel.push_back(templab(Range((int)(prop*rows),rows), Range::all()));

		fs.release();
		cout << "i's value is :" << i+1 << endl;
		i++;

		cout<<"trainData's size is: "<<trainData.size()<<endl;
		cout<<"testlabel's size is: "<<trainlabel.size()<<endl;
		cout<<"------------------------------------------------"<<endl;
	}

	trainData.convertTo(trainData,CV_32FC1);
	testData.convertTo(testData,CV_32FC1);
	trainlabel.convertTo(trainlabel,CV_32SC1);
	testlabel.convertTo(testlabel,CV_32SC1);


	cout<<"trainData's size is: "<<trainData.size()<<" and type is "<<type2str(trainData.type())<<endl;
	cout<<"trainData's size is: "<<testlabel.size()<<"and type is "<<type2str(trainlabel.type()) <<endl;
	
	// Train the SVM
    //! [init]
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setC(4.0);
    svm->setKernel(SVM::RBF);
    svm->setGamma(0.01);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)10, 1e-5));
    //! [init]
    //! [train]
    svm->train(trainData, ROW_SAMPLE, trainlabel);
    
    //predict the output of the testdata and save the result into result.xml
    Mat predicted(testlabel.size(),testlabel.type());
    
    svm->predict(testData,predicted);
    cout<<"result of predicted is : "<< predicted.rowRange(50,60) << endl;
    cout<<"result of test data is : "<< testlabel.rowRange(50,60) << endl;

    predicted.convertTo(predicted,CV_8UC1);
    testlabel.convertTo(testlabel,CV_8UC1);

    string prefile = "result.xml";
    FileStorage fs(prefile,FileStorage::WRITE);
    fs << "predicted" << predicted;
    fs << "testlabel" << testlabel;
    fs.release();

    //------------------caculate accuracy of SVM model------------------------------
    float accuracy=0;
    for(int i=0; i<testlabel.rows;i++)
    {
    	const uchar *pt = testlabel.ptr<uchar>(i);
    	const uchar *pd = predicted.ptr<uchar>(i);
    	for(int j=0;j<testlabel.cols;j++)
    		if(pt[0] != pd[0])
    			accuracy += 1.0;

    }
    cout<<"precise is: "<<1.0-(accuracy/testlabel.rows)<<endl;

    //---------------------------Show support vectors------------------------------
  
    string modelname("treeClassifier.xml"); 
    svm->save(modelname);
    cout<<"model has been saved successfully! haha"<<endl;

	return 0;
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

	
	
