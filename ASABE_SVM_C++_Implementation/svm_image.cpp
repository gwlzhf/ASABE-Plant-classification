#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <sys/types.h>
#include <dirent.h>
#include "svm_image.h"


using namespace std;
using namespace cv;
using namespace cv::ml;

void read_directory(std::string & name, vector<string> &v)
{

    DIR *dirp = opendir(name.c_str());
    struct dirent *dp;
    while( (dp=readdir(dirp)) != NULL )
    {	
    	if(strcmp(dp->d_name,"..") & strcmp(dp->d_name,"."))
    		v.push_back(dp->d_name);
    }
    
    closedir(dirp);
}

// int main(int argc, char const *argv[])
// {
// 	//load image you need here
// 	std::vector<string> filenames;
//     std::string path="/home/lhc/ASABE/cnn_image/1";
//     read_directory(path,filenames);

//     std::vector<string>::iterator v = filenames.begin();
//     while(v != filenames.end())
//     {
//    		cout<<"value of v =" << *v <<endl;
//    		v++;
//     }
//     v = filenames.begin();
//     cout<<"this is the begin image;s name: "<<*v<<endl;
// 	return 0;
// }

