
//***************Deep Learning implementation for Face Landmark Detection****************//
/*
	1. This example program shows how to find frontal human faces in an image and
	estimate their pose.
	2. The pose takes the form of 68 landmarks.
	3. The face detector used is made using the classic Histogram of Oriented Gradients (HOG)
	feature combined with a linear classifier, an image pyramid, and sliding window detection scheme.
	4. The pose estimator was created by using dlib's implementation of the paper:
	   One Millisecond Face Alignment with an Ensemble of Regression Trees by
	   Vahid Kazemi and Josephine Sullivan, CVPR 2014, 
	and was trained on the iBUG 300-W face landmark dataset.
	5. The trained model file used is: shape_predictor_68_face_landmarks.dat.bz2
*/


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


int main() {
	try {
		// open the video from filePath
		VideoCapture cap("./v45_112_Life_Of_Pi_Lying_Actress_tilts_her_head_talking.mp4");
		if (!cap.isOpened()) { CV_Error(CV_StsError, "Can not open Video file"); }
		
		//*********************INITIALIZING FOR VIDEO WRITING****************************
		int n_frames = int(cap.get(CAP_PROP_FRAME_COUNT));		// Get total frame count of video stream
		int w = int(cap.get(CAP_PROP_FRAME_WIDTH));				// Get width of video stream
		int h = int(cap.get(CAP_PROP_FRAME_HEIGHT));			// Get height of video stream
		double fps = cap.get(CV_CAP_PROP_FPS);					// Get frames per second (fps)

		// Setup output video
		VideoWriter out("Without_Temporal_Smoothing.avi", CV_FOURCC('M', 'J', 'P', 'G'), fps, Size(w, h));

		// define window to visuale the output video
		image_window win;

		//********************FACE DETECTION AND POSE ESTIMATION*************************
		frontal_face_detector detector = get_frontal_face_detector();		// Load face detection model
		shape_predictor pose_model;											// Define pose estimation model
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model; // Load pose estimation model
		
		//Grab and process frames until the main window is closed by user
		while (!win.is_closed())
		{
			double scale = 1;
			cv::Mat temp;
			if (!cap.read(temp)) break;

			// Turn OpenCV's Mat into dlib's BGR image.
			// This just wraps the Mat object, it doesn't copy anything.
			// So cimg is only valid as long as temp is valid.
			cv_image<bgr_pixel> cimg(temp);

			// Detect faces using face detection model
			std::vector<dlib::rectangle> faces = detector(cimg);

			// Define vector to store detected shapes/poses/landmarks of entire video
			std::vector<full_object_detection> shapes;
			std::vector<cv::Point> landmarks;

			for (unsigned long j = 0; j < faces.size(); ++j)
			{
				// Detect landmarks using pose estimation model
				full_object_detection shape = pose_model(cimg, faces[j]);
				
				cout << "" << shape.part(0);
				cout << "" << shape.part(1);
				cout << "number of parts: " << shape.num_parts() << endl;
				cout << "pixel position of first part:  " << shape.part(0) << endl;
				cout << "pixel position of second part: " << shape.part(1) << endl;

				Mat image = toMat(cimg);
				dlib_point2cv_Point(shape, landmarks, scale);

				// Draw the identified landmarks on to frame
				for (int i = 0; i < 68; i++) {
					cv::circle(image, landmarks[i], 3, (0, 0, 255), -1);
				}
				// Writing image to create video output
				out.write(image);
				shapes.push_back(shape);
				}
	
			// Display it all on the screen
			win.clear_overlay();
			win.set_image(cimg);
			win.add_overlay(render_face_detections(shapes));
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

/*
int main()
{
	
	array2d<rgb_pixel> img;
	Mat gray;
	Mat temp;
	
	load_image(img,"./assignment1_frames/025.jpg");
	frontal_face_detector detector = get_frontal_face_detector();
	//imshow("image", img);
	/*
	//---------------------
	CascadeClassifier faceDetector;
	faceDetector.load("haarcascade_frontalface_alt2.xml");
	cvtColor(temp, gray, COLOR_RGB2GRAY);
	std::vector<dlib::rectangle> dets;
	faceDetector.detectMultiScale(gray, dets);

	dlib::assign_image(img, dlib::cv_image<dlib::rgb_pixel>(temp));
	//---------------------
	*/

/*
	shape_predictor sp;
	deserialize("./shape_predictor_68_face_landmarks.dat") >> sp;
	//deserialize("./lbfmodel.yaml") >> sp;

	image_window win, win_faces;
	cout << "Image Loaded";
	//pyramid_up(img);

	std::vector<dlib::rectangle> dets = detector(img);
	cout << "Number of faces detected: " << dets.size() << endl;
	cout<<dets[0]<<endl;
	std::vector<full_object_detection> shapes;
	for (unsigned long j = 0; j < dets.size(); ++j)
	{
		full_object_detection shape = sp(img, dets[j]);
		cout << ""<<shape.part(0);
		cout << "" << shape.part(1);
		cout << "number of parts: " << shape.num_parts() << endl;
		cout << "pixel position of first part:  " << shape.part(0) << endl;
		cout << "pixel position of second part: " << shape.part(1) << endl;
		// You get the idea, you can get all the face part locations if
		// you want them.  Here we just store them in shapes so we can
		// put them on the screen.
		shapes.push_back(shape);
	}

	// Now let's view our face poses on the screen.
	win.clear_overlay();
	win.set_image(img);
	win.add_overlay(render_face_detections(shapes));

	// We can also extract copies of each face that are cropped, rotated upright,
	// and scaled to a standard size as shown here:
	dlib::array<array2d<rgb_pixel> > face_chips;
	extract_image_chips(img, get_face_chip_details(shapes), face_chips);
	win_faces.set_image(tile_images(face_chips));
	
	waitKey(0);
	getchar();
	return 0;

}
*/