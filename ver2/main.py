from modules import *

hol = mp.solutions.holistic
draw = mp.solutions.drawing_utils

DATA = os.path.join('ver2\\data') 
acts = np.array(['Hello', 'Stop', 'Please'])
num_sequences = 10
length_sequence = 10

for act in acts: 
    for sequence in range(num_sequences):
        try: 
            os.makedirs(os.path.join(DATA, act, str(sequence)))
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

# Catpure Keypoints
cap = cv.VideoCapture(0)
# Statement to set the mediapipe model
with hol.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for act in acts:
        for sequence in range(num_sequences):
            for frame_num in range(length_sequence):
                ret, frame = cap.read()
                image, results = detection(frame, holistic)
                # Draw landmarks
                styleLandmarks(image, results)
                    
                # Export keypoints
                keypoints = find_keypoints(results)
                npy_path = os.path.join(DATA, act, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

    cap.release()
    cv.destroyAllWindows()

body = []
for res in results.pose_landmarks.landmark:
    test = np.array([res.x, res.y, res.z, res.visibility])
    body.append(test)

result_test = find_keypoints(results)
np.save('0', result_test)

# The creation of a dicrionary that stores each word/index number
labels = {label:num for num, label in enumerate(acts)}

labelling, sequences = [], []

# Reading in data from numpy arrays
for act in acts:
    for sequence in range(num_sequences):
        frames  = []
        for frame_num in range(length_sequence):
            res = np.load(os.path.join(DATA, act, str(sequence), "{}.npy".format(frame_num)))
            frames.append(res)
        sequences.append(frames)
        labelling.append(labels[act])

# Storing sequences/labels in an array called x/y
x = np.array(sequences)
y = to_categorical(labelling).astype(int)

# Preforming a training and testing split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

log_dir = os.path.join('ver2\\Logs')

tb_callback = TensorBoard(log_dir=log_dir)

# The Sequential() API allows for the creation of a multi layer model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(acts.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=50, callbacks=[tb_callback])
model.summary()

res = model.predict(x_test)
acts[np.argmax(res[0])]
acts[np.argmax(y_test[0])]

model.save('act.h5')
model.load_weights('act.h5')

one = model.predict(x_test)
two = np.argmax(y_test, axis=1).tolist()
one = np.argmax(one, axis=1).tolist()
multilabel_confusion_matrix(two, one)

colours = [(245,117,16), (117,245,16), (16,117,245)]
def visibality(res, acts, input_frame, colours):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colours[num], -1)
        cv.putText(output_frame, acts[num], (0, 85+num*40), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)
        
    return output_frame

sequence = []
sentence = []
predictions = []
threshold = 0.4

cap = cv.VideoCapture(0)
# Access and set the MP model where initial detections will be made
with hol.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        # Make detections
        image, results = detection(frame, holistic)
        # Printing results (keypoints/landmarks)
        print(results)
        # Draw landmarks
        styleLandmarks(image, results)
        # 2. Prediction logic
        keypoints = find_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-10:]
        
        if len(sequence) == 10:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(acts[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 

                    if len(sentence) > 0: 
                        if acts[np.argmax(res)] != sentence[-1]:
                            sentence.append(acts[np.argmax(res)])
                    else:
                        sentence.append(acts[np.argmax(res)])    
            if len(sentence) > 5: 
                sentence = sentence[-5:]
            # Visualiation probabilities
            image = visibality(res, acts, image, colours)
            
        cv.rectangle(image, (0,0), (500, 50), (250, 110, 10), -1)
        cv.putText(image, ' '.join(sentence), (3,30), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (225, 225, 225), 2, cv.LINE_AA)
        
        cv.imshow('OpenCV Feed', image)

    cap.release()
    cv.destroyAllWindows()