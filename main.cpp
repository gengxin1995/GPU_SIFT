#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <GL/gl.h>
#include <stdlib.h>
#include <vector>
#include <iostream>

#include "SiftGPU.h"

using std::vector;
using std::iostream;
using namespace cv;
using namespace std;

int main()
{
    SiftGPU *sift = new SiftGPU;
    SiftMatchGPU *matcher = new SiftMatchGPU(4096);

    vector<float> descriptors1(1), descriptors2(1);
    vector<SiftGPU::SiftKeypoint> keys1(1), keys2(1);
    int num1 = 0, num2 = 0;

    char * argv[] = {"-fo", "-1", "-v", "1","-cuda"};
    //-fo -1 starting from -1 octave
    //-v 1   only print out # feature and overall time
    //-cuda  using CUDA  

    int argc = sizeof(argv)/sizeof(char*);
    sift->ParseParam(argc, argv);

    // Create a context for computation, and SiftGPU will be initialized automatically
    // The same context can be used by SiftMatchGPU
    //if(sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) return 0;
    int support = sift->CreateContextGL();
    if ( support != SiftGPU::SIFTGPU_FULL_SUPPORTED )
    {
        cerr<<"SiftGPU is not supported!"<<endl;
        return 2;
    }

    VideoCapture vc;
    vc.open("test0.avi");
    namedWindow("SIFT");
    //Mat img1 = imread("800-1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat img2 = imread("test2.png", CV_LOAD_IMAGE_GRAYSCALE);//处理灰度图像

    if(vc.isOpened())
    {
	while(1)
	{
	    Mat frame;
	    for (int i = 0; i < 5; i++)//每隔5帧取一帧进行处理
            {
                vc.read(frame);
            }

            if (frame.empty())
            {
                break;
            }
            Mat img1;
	    cvtColor(frame,img1,CV_BGR2GRAY);
	    unsigned char* data1 = (unsigned char*) img1.data;
	    unsigned char* data2 = (unsigned char*) img2.data;

	    if(sift->RunSIFT(img1.cols, img1.rows, data1, GL_LUMINANCE, GL_UNSIGNED_BYTE)) {
		//获得特征点的数量
		num1 = sift->GetFeatureNum();
		//为关键点和描述符申请空间
		keys1.resize(num1);
		descriptors1.resize(128*num1);
		//获得关键点和描述符
		sift->GetFeatureVector(&keys1[0], &descriptors1[0]);
	    }

	    if(sift->RunSIFT(img2.cols, img2.rows, data2, GL_LUMINANCE, GL_UNSIGNED_BYTE)) {
		num2 = sift->GetFeatureNum();
		keys2.resize(num2);
		descriptors2.resize(128*num2);
		sift->GetFeatureVector(&keys2[0], &descriptors2[0]);
	    }

	    // 将SiftGPU关键点格式转换成OpenCV关键点格式
	    vector<KeyPoint> kpList1;
	    vector<KeyPoint> kpList2;
	    kpList1.resize(num1);
	    kpList2.resize(num2);

	    for (int i = 0; i < num1; i++) {
		Point2f pt(keys1[i].x, keys1[i].y);
		kpList1[i].pt = pt;
	    }

	    for (int i = 0; i < num2; i++) {
		Point2f pt(keys2[i].x, keys2[i].y);
		kpList2[i].pt = pt;
	    }

	    //Mat img_keypoints_1;
	    //Mat img_keypoints_2;
	    //drawKeypoints( img1, kpList1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	    //drawKeypoints( img2, kpList2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	    //-- Show detected (drawn) keypoints
	    //imshow("Keypoints 1", img_keypoints_1 );
	    //imshow("Keypoints 2", img_keypoints_2 );

	    //waitKey(0);
	    
	    //设置语言为CUDA
	    matcher->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA);
	    
	    matcher->VerifyContextGL(); 
	    
	    //匹配描述符，第一个参数为0或1
	    matcher->SetDescriptors(0, num1, &descriptors1[0]); 
	    matcher->SetDescriptors(1, num2, &descriptors2[0]); 

	    //使用默认的threshold进行匹配
	    int (*match_buf)[2] = new int[num1][2];
	    
	    int num_match = matcher->GetSiftMatch(num1, match_buf);
	    std::cout << num_match << " sift matches were found;\n";

	    vector<DMatch> matches;
	    matches.resize(num_match);

	    //枚举匹配的特征
	    for(int i  = 0; i < num_match; ++i) {
		matches[i].queryIdx = match_buf[i][0];
		matches[i].trainIdx = match_buf[i][1];
	    }

	    Mat img_matches;
	    drawMatches(img1, kpList1, img2, kpList2, matches, img_matches);
	    imshow("SIFT", img_matches);
	    waitKey(1);
	    //delete[] match_buf;
	}
    }
    vc.release();

    delete matcher;
    delete sift;

    return 0;
}
