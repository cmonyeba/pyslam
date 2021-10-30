import cv2



cap = cv2.VideoCapture('../images/drive.mp4')





while(cap.isOpened()):
        
# Capture frame-by-frame
    ret, frame = cap.read()
    # print(ret,frame)
    if ret:
        #gray-scale frame and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print(frame.shape)
        # print(frame)
        cv2.imshow('frame', frame)
    

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('no frame')
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

  