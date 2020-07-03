import re
import os
import shutil
import copy
import sys

def copy_paste_age_directories(folder, root_dir, dest_dir):
    '''
    This function goes through the folder (in our case either 'crop_part1' or 'UTKFace') and it
    gets the age from the filename and copies and pastes the images into their designated age group.
    
    INPUTS: 
    folder - the folder from the root directory you want to apply the function on
    root_dir - the directory of the folder where the pictures are stored
    dest_dir - the directory you choose to paste the images to
    '''
    folders = ['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65+']
    for f in folders:
        os.mkdir(os.path.join(dest_dir,f))

    folder1 = os.listdir(folder) # get all the files in the folder
    for file in folder1: # for each file get the age and then copy and paste the file if in age range
        age = int(file.split('_')[0])
        if age in [0,1,2,3,4]:
            shutil.copy(root_dir + '/' + folder + '/' + file, dest_dir + '/0-4')
        elif age in [5,6,7,8,9]:
            shutil.copy(root_dir + '/' + folder + '/' + file, dest_dir + '/5-9')
        elif age in [10,11,12,13,14]:
            shutil.copy(root_dir + '/' + folder + '/' + file, dest_dir + '/10-14')
        elif age in [15,16,17,18,19]:
            shutil.copy(root_dir + '/' + folder + '/' + file, dest_dir + '/15-19')
        elif age in [20,21,22,23,24]:
            shutil.copy(root_dir + '/' + folder + '/' + file, dest_dir + '/20-24')
        elif age in [25,26,27,28,29]:
            shutil.copy(root_dir + '/' + folder + '/' + file, dest_dir + '/25-29')
        elif age in [30,31,32,33,34]:
            shutil.copy(root_dir + '/' + folder + '/' + file, dest_dir + '/30-34')
        elif age in [35,36,37,38,39]:
            shutil.copy(root_dir + '/' + folder + '/' + file, dest_dir + '/35-39')
        elif age in [40,41,42,43,44]:
            shutil.copy(root_dir + '/' + folder + '/' + file, dest_dir + '/40-44')
        elif age in [45,46,47,48,49]:
            shutil.copy(root_dir + '/' + folder + '/' + file, dest_dir + '/45-49')
        elif age in [50,51,52,53,54]:
            shutil.copy(root_dir + '/' + folder + '/' + file, dest_dir + '/50-54')
        elif age in [55,56,57,58,59]:
            shutil.copy(root_dir + '/' + folder + '/' + file, dest_dir + '/55-59')
        elif age in [60,61,62,63,64]:
            shutil.copy(root_dir + '/' + folder + '/' + file, dest_dir + '/60-64')
        else:
            shutil.copy(root_dir + '/' + folder + '/' + file, dest_dir + '/65+') 


def categorize_imdb_wiki(folders, imdb = True, root_dir, dest_dir):
    '''
    This function goes through the folder (in this case either 'wiki_crop' or 'imdb_crop') and it
    gets the age from the filename and copies and pastes the images into their designated age group.
    There is a difference in how to extract the age for both of the folders so we need to choose that in our
    parameters
    INPUTS: 
    folders - the folder from the root directory you want to apply the function on
    imdb - if True this extracts the age how it should if going through the imdb folder, if you would like to
    use this function on the wiki folder, then set the parameter to False
    root_dir - the directory of the folder where the pictures are stored
    dest_dir - the directory you choose to paste the images to
    '''
    

    for folder in os.listdir(folders):
        for file in os.listdir(root_dir + '/' + folders + '/' + folder):
            if imdb:
                age = int(int(re.split('\.|_|-| ',file)[5]) - int(re.split('\.|_|-| ',file)[2]))
            else:
                age = int(int(re.split('\.|_|-| ',file)[4]) - int(re.split('\.|_|-| ',file)[1]))
            print(age)
            print(root_dir + '/' + folders + '/' + folder + '/' + file)
            if age in [0,1,2,3,4]:
                shutil.copy(root_dir + '/' + folders + '/' + folder + '/' + file, dest_dir + '/0-4')
            elif age in [5,6,7,8,9]:
                shutil.copy(root_dir + '/' + folders + '/' + folder + '/' + file, dest_dir + '/5-9')
            elif age in [10,11,12,13,14]:
                shutil.copy(root_dir + '/' + folders + '/' + folder + '/' + file, dest_dir + '/10-14')
            elif age in [15,16,17,18,19]:
                shutil.copy(root_dir + '/' + folders + '/' + folder + '/' + file, dest_dir + '/15-19')
            elif age in [20,21,22,23,24]:
                shutil.copy(root_dir + '/' + folders + '/' + folder + '/' + file, dest_dir + '/20-24')
            elif age in [25,26,27,28,29]:
                shutil.copy(root_dir + '/' + folders + '/' + folder + '/' + file, dest_dir + '/25-29')
            elif age in [30,31,32,33,34]:
                shutil.copy(root_dir + '/' + folders + '/' + folder + '/' + file, dest_dir + '/30-34')
            elif age in [35,36,37,38,39]:
                shutil.copy(root_dir + '/' + folders + '/' + folder + '/' + file, dest_dir + '/35-39')
            elif age in [40,41,42,43,44]:
                shutil.copy(root_dir + '/' + folders + '/' + folder + '/' + file, dest_dir + '/40-44')
            elif age in [45,46,47,48,49]:
                shutil.copy(root_dir + '/' + folders + '/' + folder + '/' + file, dest_dir + '/45-49')
            elif age in [50,51,52,53,54]:
                shutil.copy(root_dir + '/' + folders + '/' + folder + '/' + file, dest_dir + '/50-54')
            elif age in [55,56,57,58,59]:
                shutil.copy(root_dir + '/' + folders + '/' + folder + '/' + file, dest_dir + '/55-59')
            elif age in [60,61,62,63,64]:
                shutil.copy(root_dir + '/' + folders + '/' + folder + '/' + file, dest_dir + '/60-64')
            else:
                shutil.copy(root_dir + '/' + folders + '/' + folder + '/' + file, dest_dir + '/65+') 


def main():
    # set root directory
    root_dir = os.getcwd()
    # create directory to put the categorized data
    os.mkdir(os.path.join(os.getcwd(),'Face_Age_Dataset'))
    # update dest directory
    dest_dir = os.path.join(os.getcwd(),'Face_Age_Dataset/')
    print('Copying images...')
    copy_paste_age_directories('crop_part1', root_dir, dest_dir)
    copy_paste_age_directories('UTKFace', root_dir, dest_dir)


    print('Copying imdb and wiki images... This will take a while')
    categorize_imdb_wiki('wiki_crop', False, root_dir, dest_dir)
    categorize_imdb_wiki('imdb_crop', True, root_dir, dest_dir)
    
    print('Finished copying!')


if __name__ == '__main__':
    main()