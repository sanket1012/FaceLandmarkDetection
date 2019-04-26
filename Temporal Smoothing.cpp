//***************Face Landmark Detection with Temporal Smoothing****************//
/*
	1. In this intial facelandmark detections is detected similar to LandmarkDetection.cpp
	2. The landmark points for each frame is stored in vectored and used for temporal smoothing.
	3. For each frame F:
		-Optical Flow is calculated between F and next 12 frames, consecutively.
		-This Optical Flow is cummulatively summed to get Transformation relation between F and each of 12 frames
		-Performing average for the Transformation matrix gives a smooth estimate of frame F
	4. Perform such estimation and averaging for all the frame
*/
/*
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/drawLandmarks.hpp>
#include <numeric>

using namespace dlib;
using namespace std;
using namespace cv;
using namespace cv::face;

// Converting Dlib's full_object_detection to opencv Point-Vector 
void dlib_point2cv_Point(full_object_detection& S, std::vector<Point>& L, double& scale)
{
	for (unsigned int i = 0; i < S.num_parts(); ++i)
		L.push_back(Point(S.part(i).x()*(1 / scale), S.part(i).y()*(1 / scale)));
}

struct TransformParam
{
	TransformParam() {}
	TransformParam(double _dx, double _dy, double _da)
	{	dx = _dx;	dy = _dy;	da = _da; }
	double dx;
	double dy;
	double da;

	void getTransform(Mat &T)
	{
		// Reconstruct transformation matrix accordingly to new values
		T.at<double> (0, 0) = cos(da);
		T.at<double> (0, 1) = -sin(da);
		T.at<double> (1, 0) = sin(da);
		T.at<double> (1, 1) = cos(da);

		T.at<double> (0, 2) = dx;
		T.at<double> (1, 2) = dy;
	}
};

struct Trajectory
{
	Trajectory() {}
	Trajectory(double _x, double _y, double _a)
	{	x = _x;		y = _y;		a = _a;	}
	double x;
	double y;
	double a; // angle
};

// Cummulative Sum of the Optical Flow
std::vector <Trajectory> cumsum(std::vector <TransformParam> &transforms)
{
	// trajectory at all frames
	std::vector <Trajectory> trajectory; 
	// Accumulated frame to frame transform
	double a = 0;
	double x = 0;
	double y = 0;

	for (size_t i = 0; i < transforms.size(); i++)
	{
		x += transforms[i].dx;
		y += transforms[i].dy;
		a += transforms[i].da;

		trajectory.push_back(Trajectory(x, y, a));
	}
	return trajectory;
}

int main() {

	try {
		// open the video from filePath
		VideoCapture cap("./v45_112_Life_Of_Pi_Lying_Actress_tilts_her_head_talking.mp4");
		if (!cap.isOpened()) { CV_Error(CV_StsError, "Can not open Video file"); }
		
		cout << "Total number of frames in Video = " << cap.get(CV_CAP_PROP_FRAME_COUNT) << endl;

		//*********************INITIALIZING FOR VIDEO WRITING****************************
		int n_frames = int(cap.get(CAP_PROP_FRAME_COUNT));		// Get total frame count of video stream
		int w = int(cap.get(CAP_PROP_FRAME_WIDTH));				// Get width of video stream
		int h = int(cap.get(CAP_PROP_FRAME_HEIGHT));			// Get height of video stream
		double fps = cap.get(CV_CAP_PROP_FPS);					// Get frames per second (fps)

		// Setup output video
		VideoWriter out("With_Temporal_Smoothing.avi", CV_FOURCC('M', 'J', 'P', 'G'), fps, Size(w, h));

		//********************FACE DETECTION AND POSE ESTIMATION************************
		frontal_face_detector detector = get_frontal_face_detector();		// Load face detection model
		shape_predictor pose_model;											// Define pose estimation model
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model; // Load pose estimation model

		// Define Dlib's Vector to store Landmark points for all frames
		std::vector<full_object_detection> old_shapes;
		double scale = 1;
		cv::Mat temp;

		while (cap.read(temp))
		{
			// Turn OpenCV's Mat into dlib's BGR image.
			// This just wraps the Mat object, it doesn't copy anything.
			// So cimg is only valid as long as temp is valid.
			cv_image<bgr_pixel> cimg(temp);

			// Detect faces using face detection model
			std::vector<dlib::rectangle> faces = detector(cimg);

			for (unsigned long j = 0; j < faces.size(); ++j)
			{
				// Detect landmarks using pose estimation model
				full_object_detection shape = pose_model(cimg, faces[j]);
				old_shapes.push_back(shape);
			}
		}
		cout << "All landmark points detected !" << endl;
		cout << "Total number of frames saved = " << old_shapes.size() << endl;
		cout << "PRESS ENTER, to begin Temporal Smoothing" << endl;
		getchar();
		
		//****************************TEMPORAL SMOOTHING*******************************
		Mat frame;
		VideoCapture cam("v45_112_Life_Of_Pi_Lying_Actress_tilts_her_head_talking.mp4");
		
		int frame_number = 0;
		while (cam.read(frame))
		{
			if (frame_number == 30) break;
			cout << "Frame Number: " << frame_number << endl;

			// Define transformation-store array
			std::vector<TransformParam>transforms;
 			Mat last_T;
			
			// Looping for 12 neighbor frames
			for (int j = frame_number; j<frame_number+11; j++)
			{
				// landmark points from current and next frame
				std::vector<cv::Point>next_pts, curr_pts;

				// Converting Dlib points to Vector Points
				dlib_point2cv_Point(old_shapes[j+1], next_pts, scale);
				dlib_point2cv_Point(old_shapes[j], curr_pts, scale);

				// Find transformation matrix
				Mat T = estimateRigidTransform(curr_pts,next_pts, false);

				// In rare cases no transform is found. 
				// We'll just use the last known good transform.
				if (T.data == NULL) last_T.copyTo(T);
				T.copyTo(last_T);

				// Extract translation
				double dx = T.at<double> (0, 2);
				double dy = T.at<double> (1, 2);

				// Extract rotation angle
				double da = atan2(T.at< double>(1, 0), T.at<double>(0, 0));

				// Store transformation 
				transforms.push_back(TransformParam(dx, dy, da));
			}
			
			frame_number = frame_number + 1;

			// Cummulative sum of the 2D optical flow
			cumsum(transforms);			
			std::vector<cv::Point>pts;
			std::vector<cv::Point>new_shapes;
			cv::Point store;

			// Current frames Vector points
			dlib_point2cv_Point(old_shapes[frame_number], pts, scale);
			
			// Transforming 12 frame points to current frame points
			// And then adding them to get and average value
			int sum[68][2];
			for (int k = 0; k < transforms.size(); k++)
			{
				double m[2][2];
				double t[2][1];
				double dx = transforms[k].dx;
				double dy = transforms[k].dy;
				double da = transforms[k].da;
				m[0][0] = cos(da);
				m[0][1] = -sin(da);
				m[1][0] = sin(da);
				m[1][1] = cos(da);
				t[0][0] = dx;
				t[1][0] = dy;

				for (int q = 0; q < 68; q++)
				{
					store.x = m[0][0] * pts[q].x + m[0][1] * pts[q].y + t[0][0];
					store.y = m[1][0] * pts[q].x + m[1][1] * pts[q].y + t[1][0];
					new_shapes.push_back(store);
					sum[q][0]= sum[q][0] + new_shapes[q].x;
					sum[q][1] = sum[q][1] + new_shapes[q].y;
				}
			}
			
			// Finding Average of Tranformed landmark points for current frame
			for (int i = 0; i < 68; i++)
			{
				sum[i][0] = sum[i][0] / 12;
				sum[i][1] = sum[i][1] / 12;
				Point P(sum[i][0], sum[i][1]);
				cv::circle(frame, P, 3, (0, 0, 255), -1);
			}

			// Writing image to create video output
			out.write(frame);
		}
		getchar();
	}

	catch (serialization_error& e)
	{
		cout << "You will have to train or use trained dlib's face landmark detection model to identify the landmarks." << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}
}
*/