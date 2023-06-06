import pickle
from tqdm import tqdm
import os

# read the pickle file
image_urls = pickle.load(open("../utils/image_urls.pkl", "rb"))

home_path = "../data/ADE20K_2021_17_01/images/training"
images_final_required_path = "../data/images_final_required/"
# copy the images to the required folder
for i in tqdm(range(len(image_urls))):
    image_url = image_urls[i]
    
    try:
        image_path = home_path+image_url
        if os.path.exists(image_path):
            os.system("cp "+ image_path + " "+images_final_required_path)
            pass
        else:
            image_name_split = image_url.split("/")
            print(image_name_split)
            image_path = image_name_split[1] +  "/" + "__".join(image_name_split[2:-1]) + "/" + image_name_split[-1]
            image_path = home_path + "/" + image_path 
            os.system("cp "+ image_path + " "+images_final_required_path)
    except:
        print("Error in: ", home_path+image_url)