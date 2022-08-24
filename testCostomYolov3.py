import cv2
import numpy as np

#cap = cv2.VideoCapture(0)

#define label
obj_file ="obj.names"
obj_classes = []
net_config = "cfg\yolov3_training.cfg"
net_weights = "cfg\yolov3_traning_last (2).weights"
blob_size = 320
confidence_threshold = 0.5
nms_threshold = 0.3
with open(obj_file ,"rt")as f:
    #"rt": r yani to halat read , t yani dar halat text bekon agar b bezarim yan halat binery
    obj_classes = f.read().rstrip("\n").split("\n")
#rstrip: momken hast dar enteha file text chand khat khali ijad shdeh bashe ke miyad on ha ro pak mikoneh
#split : be tor pishfarz kalamar ro ba space joda mikoneh val ma inja farz ro \n garar dadim
#make darknet
net = cv2.dnn.readNetFromDarknet(net_config,net_weights)
#AMADEH SAZI SAKHTAFZAR LAZEM
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findobjects(output, img):
    img_h, img_w, img_c = img.shape
    bboxes =[]
    class_ids = []
    confidences = []
    c=[]
    for member in output:
        m=len(member)
        for detect_vector in member :
            scores = detect_vector[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold :
                w,h = int(detect_vector[2] *  img_w ) , int(detect_vector[3] *  img_h)
                x, y =int((detect_vector[0] * img_w)- w/2)  , int((detect_vector[1] * img_h)- h/2)
                bboxes.append([x,y,w,h])
                class_ids.append(class_id)
                confidences.append(float(confidence))   
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, confidence_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        bbox = bboxes[i]
        x,y,w,h =bbox[0], bbox[1], bbox[2], bbox[3]
        if (class_ids[i]) ==0:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, f'{obj_classes[class_ids[i]].upper()}{int(confidences[i]*100)}%',
                    (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(img, f'{obj_classes[class_ids[i]].upper()}{int(confidences[i]*100)}%',
                    (x-70,y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

frame= cv2.imread("test.jpg")
    #scalefactor = baraye normalize kardan estefadeh mishe va chon tasviremon rangi bod va mikhahim normalize konim omadim az 1/255 estefadej kardim
blob = cv2.dnn.blobFromImage(frame , scalefactor = 1/255, size=(blob_size, blob_size), mean =(0,0,0),
                            swapRB = True, crop = False)
    
#mean = mean subtraction , baraye inke tanzimat roshnayee tasvir dorost beshe dar bazi az data seyt ha mannad image net miayan va baraye r,g,b har kodam yek megdar moshakhas mikonan ke beyad az lenhv anha kam konad
#swapRGB= dar opencv tasavier be format bgr hast va agar garag basha mean kam shavad pas bayad in tabdil sorat girad ta eshtebah mafgadir az ham kam nashavand
#crop = agar crop true bashad va size tasv ir ma ba size tasvir corodi ke moshakhas kardim bozrgtar bashad crop miayad va be haman andazeh az tasvireman boresh midahad ama agar false bashad khodam yek kari mikonad , tabdil size ra anjam midahad

net.setInput(blob)
    #adress on 3ta layeh khoroji ro peyda mikonim
out_names = net.getUnconnectedOutLayersNames()
    #hala migim ke beya on  3 ta adress ro begir va bro beszesh
output = net.forward(out_names)
findobjects(output, frame)
#plt.rcParams['figure.figsize'] = (10,10)
#plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#plt.show()
cv2.imshow("Webcam", frame)
cv2.waitKey(0)
#matplotlib inline

'''
while True:
    success,frame =cap.read()
    #scalefactor = baraye normalize kardan estefadeh mishe va chon tasviremon rangi bod va mikhahim normalize konim omadim az 1/255 estefadej kardim
    blob = cv2.dnn.blobFromImage(frame , scalefactor = 1/255, size=(blob_size, blob_size), mean =(0,0,0),
                                 swapRB = True, crop = False)
    
#mean = mean subtraction , baraye inke tanzimat roshnayee tasvir dorost beshe dar bazi az data seyt ha mannad image net miayan va baraye r,g,b har kodam yek megdar moshakhas mikonan ke beyad az lenhv anha kam konad
#swapRGB= dar opencv tasavier be format bgr hast va agar garag basha mean kam shavad pas bayad in tabdil sorat girad ta eshtebah mafgadir az ham kam nashavand
#crop = agar crop true bashad va size tasv ir ma ba size tasvir corodi ke moshakhas kardim bozrgtar bashad crop miayad va be haman andazeh az tasvireman boresh midahad ama agar false bashad khodam yek kari mikonad , tabdil size ra anjam midahad

        
    net.setInput(blob)
    #adress on 3ta layeh khoroji ro peyda mikonim
    out_names = net.getUnconnectedOutLayersNames()
    #hala migim ke beya on  3 ta adress ro begir va bro beszesh
    output = net.forward(out_names)
    findobjects(output, frame)
   
    
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1)== ord('q'):
        break
'''