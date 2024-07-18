from modules import *

hol = mp.solutions.holistic
draw = mp.solutions.drawing_utils

DATA = os.path.join('ver2\\data') 
actions = np.array(['Hello', 'Stop', 'Please'])
no_sequences = 10
sequence_length = 10

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA, action, str(sequence)))
        except:
            pass

# A mediapipe function which takes in both an image and a mediapipe holistic model
def detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results

# Functions which draws landmarks by taking in the webcam image and the results
def landmarks(image, results):
    draw.draw_landmarks(image, results.face_landmarks, hol.FACEMESH_TESSELATION)
    draw.draw_landmarks(image, results.pose_landmarks, hol.POSE_CONNECTIONS)
    draw.draw_landmarks(image, results.left_hand_landmarks, hol.HAND_CONNECTIONS)
    draw.draw_landmarks(image, results.right_hand_landmarks, hol.HAND_CONNECTIONS)

def styleLandmarks(image, results):
    draw.draw_landmarks(image, results.face_landmarks, hol.FACEMESH_CONTOURS,
    draw.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    draw.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    
    draw.draw_landmarks(image, results.pose_landmarks, hol.POSE_CONNECTIONS,
    draw.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    draw.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))     
    
    draw.draw_landmarks(image, results.left_hand_landmarks, hol.HAND_CONNECTIONS,
    draw.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    draw.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))  
    
    draw.draw_landmarks(image, results.right_hand_landmarks, hol.HAND_CONNECTIONS,
    draw.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    draw.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 

# A function with four numpy arrays, one for each set of landmarks. Its main function will be to find and export keypoint values
def find_keypoints(results):
    body = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    head = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    leftH = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rightH = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([body, head, leftH, rightH])

cap = cv.VideoCapture(0)
# Statement to set the mediapipe model
with hol.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                image, results = detection(frame, holistic)
                # Draw landmarks
                styleLandmarks(image, results)

                # NEW Apply wait logic
                if frame_num == 0: 
                    cv.putText(image, 'Starting', (120,200), 
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv.LINE_AA)
                    cv.putText(image, 'Collection for {}, Number: {}'.format(action, sequence), (15,12), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

                    cv.waitKey(3000)
                else: 
                    cv.putText(image, 'Collection for {}, Number: {}'.format(action, sequence), (15,12), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                    
                # NEW Export keypoints
                keypoints = find_keypoints(results)
                npy_path = os.path.join(DATA, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                    
                # Show to screen
                cv.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv.waitKey(10) & 0xFF == ('q'):
                    break
    cap.release()
    cv.destroyAllWindows()
    
    body = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    head = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    leftH = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rightH = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)