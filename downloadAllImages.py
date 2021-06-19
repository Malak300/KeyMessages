#This script is for downloading images that are not loaded properly

import urllib.request
import csv
path = 'C:/Users/USER/Desktop/tensor/Data_Download_images .csv'

with open(path) as csv_Data:
    path0 = csv.DictReader(csv_Data)
    i =0    # Number of images downloaded
    for row in path0:
        try:
            print(row['id'])
            if (row['frame_url'][-2]== 'n'):      #if the image type is png
                end= '.png'
            elif (row['frame_url'][-2] == 'p'):   #if the image type is jpg
                    end = '.jpg'

            saveto = "C:/Users/USER/PycharmProjects/yolov3-tf2-master/data/xxx/" + row['id'] + end   #
            urllib.request.urlretrieve(row['frame_url'], saveto)
            i+=1


        except Exception as e:
            pass


print(i)

