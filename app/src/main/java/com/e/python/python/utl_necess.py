import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imutils
import glob
import os
cv2.ocl.setUseOpenCL(False)


per=20

def remove_oldfiles(dir,rel_path,tester_name):
    import os
    if os.path.exists(rel_path+tester_name+"/"+str(dir)+"/temp_1.png"):
        os.remove(rel_path+tester_name+"/"+str(dir)+'/temp_1.png')
    if os.path.exists(rel_path+tester_name+"/"+str(dir)+'/temp_2.png'):
          os.remove(rel_path+tester_name+"/"+str(dir)+'/temp_2.png')
    if os.path.exists(rel_path+tester_name+"/"+str(dir)+'/temp_3.png'):
          os.remove(rel_path+tester_name+"/"+str(dir)+'/temp_3.png')   
    if os.path.exists(rel_path+tester_name+"/"+str(dir)+'/final_stitched.png'):
          os.remove(rel_path+tester_name+"/"+str(dir)+'/final_stitched.png') 
    print("old results deleted...")


def apply_remove_oldfiles(dir,rel_path,tester_name,dirs):
    for direct in dirs:
        remove_oldfiles(direct,rel_path,tester_name)   
        
        
        
        


def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)
        
    # Match descriptors.
    best_matches = bf.match(featuresA,featuresB)
    
    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    # print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches  
def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"
    
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

 # select the image id (valid values 1,2,3, or 4)
feature_extractor = 'orb' # one of 'sift', 'surf', 'brisk', 'orb'
feature_matching = 'bf'   

  
def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """
    
    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"
    
    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)  
def read_rawimages(path):
    import glob
    images=glob.glob(path+'*.png')

    # images[0:5]
    level0=[]
    level1=[]
    level2=[]  ###NOte :parameterize
    for i in range(0,len(images)):
      # print(images[i].split('/')[-1].split('_')[-1].split('.')[0])
        if images[i].split('/')[-1].split('_')[-1].split('.')[0]=='0':
          # print("yes")
          name=images[i].split('/')[-1]
          level0.append(name)
          print(f"level 0 images:::::::::::::::{level0}")
        elif images[i].split('/')[-1].split('_')[-1].split('.')[0]=='1':
          name=images[i].split('/')[-1]
          level1.append(name)
          print(f"level 1 images:::::::::::::::{level1}")
        else:
          name=images[i].split('/')[-1]
          level2.append(name)
          print(f"level 2 images:::::::::::::::{level2}")
    return level0,level1,level2





import random
import albumentations as A
def view_transform(image):
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(image)
    
# img='/home/siva/Desktop/ivy/stitching_images/ivy_dec16/Isaac Reyes Alarcón -CALZADA LA VIGA/b7f7681e7b8945c5d66b92807f51ba37/b7f7681e7b8945c5d66b92807f51ba37image_0_0.png'    
####old cropping logic
# def cropping(img,per):   
    
#     import numpy as np
#     import cv2
    
#     figure = cv2.imread(img)
#     # figure1 = cv2.imread('/content/gdrive/MyDrive/Image_stitching/test/02162022131446294image_7_0.png')
#     figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
#     # figure1 = cv2.cvtColor(figure1, cv2.COLOR_BGR2RGB)
    
#     # img = cv2.imread('/home/siva/Desktop/ivy/stitching_images/ivy_dec16/Isaac Reyes Alarcón -CALZADA LA VIGA/b7f7681e7b8945c5d66b92807f51ba37/b7f7681e7b8945c5d66b92807f51ba37image_0_0.png')
#     # height, width, channels = figure.shape--commmented
#     # print height, width, channels
   
#     resized = cv2.resize(figure, (1440, 1920), interpolation=cv2.INTER_AREA)
#     height, width =resized.shape[:2]
#     print(f"Height of the Resized image::::::::::::::::{height}")
#     print(f"width of the Resized image::::::::::::::::{width}")
#     # adj_x_max=width-int((per/100)*width)
#     adj_x_min=int((per/100)*width)
    
#     transform = A.Crop(x_min=adj_x_min,y_min=0, x_max=width, y_max=height )
    
    
    
#     random.seed(7)
#     augmented_image = transform(image=figure)['image']
    
#     # cv2.imwrite('/content/gdrive/MyDrive/Image_stitching/test/cropped0.png', augmented_image)
#     view_transform(augmented_image)
    
#     return augmented_image

def cropping(img,per):   
    
    import numpy as np
    import cv2
    
    figure = cv2.imread(img)
    # figure1 = cv2.imread('/content/gdrive/MyDrive/Image_stitching/test/02162022131446294image_7_0.png')
    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
    # figure1 = cv2.cvtColor(figure1, cv2.COLOR_BGR2RGB)
    
    # img = cv2.imread('/home/siva/Desktop/ivy/stitching_images/ivy_dec16/Isaac Reyes Alarcón -CALZADA LA VIGA/b7f7681e7b8945c5d66b92807f51ba37/b7f7681e7b8945c5d66b92807f51ba37image_0_0.png')
    height, width, channels = figure.shape
    print(f"Height of the  image::::::::::::::::{height}")
    print(f"width of the  image::::::::::::::::{width}")
    if  width != 1440 and height!= 1920:
        print("resizing...")
        resized = cv2.resize(figure, (1440, 1920), interpolation=cv2.INTER_AREA)
        height, width =resized.shape[:2]
        # print(f"Height of the Resized image::::::::::::::::{height}")
        # print(f"width of the Resized image::::::::::::::::{width}")
        # adj_x_max=width-int((per/100)*width)
        adj_x_min=int((per/100)*width)
        
        transform = A.Crop(x_min=adj_x_min,y_min=0, x_max=width, y_max=height )
        
        
        
        # random.seed(7)
        augmented_image = transform(image=resized)['image']
        height, width =augmented_image.shape[:2]
        print(f"Height of the Resized image::::::::::::::::{height}")
        print(f"width of the Resized image::::::::::::::::{width}")
        
        
        # cv2.imwrite('/content/gdrive/MyDrive/Image_stitching/test/'+cropped0.png', augmented_image)
        # view_transform(augmented_image)
    else:
        figure = cv2.imread(img)
        # figure1 = cv2.imread('/cont
        figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
        height, width, channels = figure.shape
        adj_x_min=int((per/100)*width)
        
        transform = A.Crop(x_min=adj_x_min,y_min=0, x_max=width, y_max=height )
        augmented_image = transform(image=figure)['image']
        # augmented_image=figure
        
        print("resize not required")
         
    return augmented_image         
    
def my_stitching(path,leftimg,rightimg,level):
  # read images and transform them to grayscale
  # Make sure that the train image is the image that will be transformed
  # trainImg = imageio.imread('/content/gdrive/MyDrive/Image_stitching/test/02162022131446294image_7_0.png')
  trainImg = imageio.imread(path+leftimg)
  # trainImg = cropping(path+leftimg,per)
  
  # trainImg1 = imageio.imread('/content/gdrive/MyDrive/Image_stitching/test/02162022131446294image_1_0.png')
  trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)
  height,width=trainImg.shape[:2]
  # print("Image height and width of trainImg  %(height)s is %(width)s" % {'height': height, 'width': width})
  # queryImg = imageio.imread('/content/gdrive/MyDrive/Image_stitching/test/02162022131446294image_8_0.png')
  queryImg = cropping(path+rightimg,per)
  
  # Opencv defines the color channel in the order BGR. 
  # Transform it to RGB to be compatible to matplotlib
  queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)
  height,width=queryImg_gray.shape[:2]
  # print("Image height and width of queryImg  %(height)s is %(width)s" % {'height': height, 'width': width})
  # fig, (ax2, ax1) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16,9))
  # ax2.imshow(trainImg, cmap="gray")
  # ax2.set_xlabel("Train image (Image to be transformed)", fontsize=14)
  # ax1.imshow(queryImg, cmap="gray")
  # ax1.set_xlabel("Query image", fontsize=14)


  kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
  kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor)

  # display the keypoints and features detected on both images
  # fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
  # ax1.imshow(cv2.drawKeypoints(trainImg_gray,kpsA,None,color=(0,255,0)))
  # ax1.set_xlabel("(a)", fontsize=14)
  # ax2.imshow(cv2.drawKeypoints(queryImg_gray,kpsB,None,color=(0,255,0)))
  # ax2.set_xlabel("(b)", fontsize=14)

  # plt.show()


  # print("Using: {} feature matcher".format(feature_matching))

  fig = plt.figure(figsize=(20,8))

  if feature_matching == 'bf':
      matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
      img3 = cv2.drawMatches(trainImg,kpsA,queryImg,kpsB,matches[:0],
                            None)
  elif feature_matching == 'knn':
      matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
      img3 = cv2.drawMatches(trainImg,kpsA,queryImg,kpsB,np.random.choice(matches,100),
                            None)
  img3=cv2.cvtColor(img3, cv2.COLOR_RGB2BGR )    
  # stitched_images.append(img3)
  # plt.savefig('/content/gdrive/MyDrive/Image_stitching/test/perno_sample_stitch/'+str(i)+"comb.png",img3)
  # stitched_name='/content/gdrive/MyDrive/Image_stitching/test/perno_sample_stitch/'+str(i)+"comb.png"

  # stitch_list.append(stitched_name)
  # cv2.imwrite(stitched_name,img3)
  cv2.imwrite(path+"temp_"+str(level)+".png",img3)
  # return stitched_name
  # plt.imshow(img3)
  # plt.show()



def stitching_levels(level0,level1,level2,path):
    level0.sort()
    level1.sort()
    level2.sort()
    if len(level0)==1:
        # print("Stitchin of level0 started..")
        img=imageio.imread(path+level0[0])
        cv2.imwrite(path+"temp_"+str(1)+".png",img)
        
        
    if len(level1)==1:
        # print("Stitchin of level0 started..")
        img=imageio.imread(path+level1[0])
        cv2.imwrite(path+"temp_"+str(2)+".png",img)
    if len(level0)>1:
        print("Stitchin of level0 started..")
        left_img=0
        right_img=1
        # result=my_stitching(level0[i],level0(int(i+1)))
        print(level0[left_img])
        print(level0[right_img])
        my_stitching(path,level0[left_img],level0[right_img],1)
        for i in range(2,len(level0)):
          # print(i)
          # if i<len(level0):
          print(level0[i])    
          my_stitching(path,"temp_1.png",level0[i],1)
          print("temp_1 is generated..")
    else:
         print("level0 has only one ")
    if len(level1)>1:
        print("Stitchin of level0 started..")
        left_img=0
        right_img=1
        print(level1[left_img])
        print(level1[right_img])
        # result=my_stitching(level0[i],level0(int(i+1)))
        my_stitching(path,level1[left_img],level1[right_img],2)
        for i in range(2,len(level1)):
          # print(i)
          print(level1[i])  
          my_stitching(path,"temp_2.png",level1[i],2)
          print("temp_2 is generated..")
    else:
          print("level 1 doesnt exist")  
 
    if len(level2)>0:
        print("Stitchin of level0 started..")
        left_img=0
        right_img=1
        # result=my_stitching(level0[i],level0(int(i+1)))
        my_stitching(path,level2[left_img],level2[right_img],3)
        for i in range(2,len(level2)):
          # print(i)
          # if i<len(level0):
          my_stitching(path,"temp_3.png",level2[i],3)
        print("temp_3 is generated..")
    else:
          print("level 2 doesnt exist")     




def stitch_verticall(img1,img2,rel_path,dir):
  def vconcat_resize(img_list, interpolation
          = cv2.INTER_CUBIC):
    # take minimum width
    w_min = min(img.shape[1]
          for img in img_list)
    
    # resizing images
    im_list_resize = [cv2.resize(img,
            (w_min, int(img.shape[0] * w_min / img.shape[1])),
                  interpolation = interpolation)
            for img in img_list]
    # return final image
    return cv2.vconcat(im_list_resize)

  # function calling
  img_v_resize = vconcat_resize([img1, img2])
  
  # show the output image
  cv2.imwrite(rel_path+str(dir)+'cropped_'+str(per)+'.png', img_v_resize)

def stitch_verticall3(rel_path,img1,img2,img3,dir):
    def vconcat_resize(img_list, interpolation
                       
            = cv2.INTER_CUBIC):
      # take minimum width
      w_min = min(img.shape[1]
            for img in img_list)
      
      # resizing images
      im_list_resize = [cv2.resize(img,
              (w_min, int(img.shape[0] * w_min / img.shape[1])),
                    interpolation = interpolation)
              for img in img_list]
      # return final image
      return cv2.vconcat(im_list_resize)

    # function calling
    img_v_resize = vconcat_resize([img1, img2,img3])

    # show the output image
    cv2.imwrite(rel_path+str(dir)+'cropped_'+str(per)+'.png', img_v_resize)



def final_stitching(dir,rel_path,tester_name,per):
    
    # for dir in dirs:
    #     print(dir)
    
    path=rel_path+tester_name+"/"+str(dir)+"/"
    print(path)
    level0,level1,level2=read_rawimages(path)
    
    level0.sort()
    level1.sort()
    level2.sort()
    print("checkpoint 1") 
    
    stitching_levels(level0,level1,level2,path)
    
    print("chcek point 2")
    # if len(level1)>0:
    #    stitching_levels(level1,path)
    #    print("stitching level1")
    # else:
    #     print("levelv 1 doesnt exits..")
    # print("chcek point 3")
    
    # if len(level2)>0:
    #    stitching_levels(level2,path)
    #    print("stitching level2")
    # else:
    #     print("levelv 2 doesnt exits..")
    # stitching_levels(level2)
    # stitch_levwise(level)

    import time
    # time.sleep(50)
    
    
    # from google.colab.patches import cv2_imshow
    # if os.path.exists(path+'temp_1.png'):
        # print("adfad")
    
    # import cv2
    # import numpy as np
    if os.path.exists(path+'temp_1.png') and os.path.exists(path+'temp_2.png') and os.path.exists(path+'temp_3.png'):
        img1 = cv2.imread(path+'/temp_1.png')
        img2 = cv2.imread(path+'/temp_2.png')
        img3 = cv2.imread(path+'/temp_3.png')
    
        stitch_verticall3(rel_path,img1,img2,img3,dir)
        print("final_stitched image with 3 images is generated successully..")
        
    
    elif   os.path.exists(path+'temp_1.png') and os.path.exists(path+'temp_2.png'):
            print("all 3 are not existed..")   
            img1 = cv2.imread(path+'/temp_1.png')
            img2 = cv2.imread(path+'/temp_2.png')
            # img3=cv2.imread(path+'/temp_3.png')
            stitch_verticall(img1,img2,rel_path,dir)
            print("final_stitched image with 2 images is generated successully..")
        
    else :
            print("only one file is existed..")
          
            img1 = cv2.imread(rel_path+tester_name+"/"+str(dir)+'/temp_1.png')
            cv2.imwrite(rel_path+str(dir)+'cropped_'+str(per)+'.png',img1)
            print("final_stitched image with one image  generated successully..")
        






        