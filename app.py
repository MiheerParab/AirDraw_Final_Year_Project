from flask import Flask, render_template, Response
import cv2 as cv
import numpy as np
from hands import HandDetector
from canvas import Canvas

app = Flask(__name__)

# Loading the default webcam of PC.
cap = cv.VideoCapture(0)

# width and height for 2-D grid
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) + 0.5)

# set the default background mode (CAM/BLACK)
background_mode = 'CAM'

# initialize the canvas element and hand-detector program
canvas = Canvas(width, height)
detector = HandDetector(background_mode)


def generate_frames():
    while True:
        # Reading the frame from the camera
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)

        if background_mode == 'BLACK':
            black_frame = np.zeros((height, width, 1), dtype="uint8")
            request = detector.determine_gesture(frame, black_frame)
            frame = black_frame
        else:
            request = detector.determine_gesture(frame, frame)

        gesture = request.get('gesture')
        # if we have a gesture, deal with it
        if gesture is not None:
            idx_finger = request['idx_fing_tip']  # coordinates of tip of index fing
            _, c, r = idx_finger

            data = {'idx_finger': idx_finger}
            rows, cols, _ = frame.shape

            # check the radius of concern
            if (0 < c < cols and 0 < r < rows):
                if gesture == "DRAW":
                    canvas.push_point((r, c))
                elif gesture == "ERASE":
                    # stop the current line
                    canvas.end_line()

                    radius = request['idx_mid_radius']

                    _, mid_r, mid_c = request['mid_fing_tip']
                    canvas.erase_mode((mid_r, mid_c), int(radius * 0.5))

                    # add features for the drawing phase
                    data['mid_fing_tip'] = request['mid_fing_tip']
                    data['radius'] = radius
                elif gesture == "HOVER":
                    canvas.end_line()
                elif gesture == "TRANSLATE":
                    canvas.end_line()

                    idx_position = (r, c)
                    shift = request['shift']
                    radius = request['idx_pinky_radius']
                    radius = int(radius * 0.8)

                    canvas.translate_mode(idx_position, int(radius * 0.5), shift)

                    # add features for the drawing phase
                    data['radius'] = radius

            frame = canvas.draw_dashboard(frame, gesture, data=data)
        else:
            frame = canvas.draw_dashboard(frame)
            canvas.end_line()

        # draw the stack
        frame = canvas.draw_lines(frame)

        _, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gall')
def index2():
    return render_template('gallery.html')


@app.route('/cont')
def index3():
    return render_template('contact.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
