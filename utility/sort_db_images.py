import os
import shutil

class SortDbImages():
    def __init__(self, 
                 db_path = "data/all-croped", 
                 output_path = "data", 
                 op_folder_name = "sorted_db_test", 
                 jewellery_types = {"RG" : "Ring", 
                           "NK" : "Necklace", 
                           "BR" : "Bracelate", 
                           "ER" : "Earring",
                           "OT" : "Other"}
                ):
        
        self.db_path = db_path
        self.output_path = output_path
        self.op_folder_name = op_folder_name
        self.jewellery_types = jewellery_types
    
    def srt(self):
        self.all_images = os.listdir(self.db_path)

        #make folder with subfolder of all jewellery type
        self.dirpath = os.path.join(self.output_path, self.op_folder_name)
        if os.path.exists(self.dirpath) and os.path.isdir(self.dirpath):
            shutil.rmtree(self.dirpath)

        os.mkdir(os.path.join(self.output_path, self.op_folder_name))

        for i in self.jewellery_types.values():
            os.mkdir(os.path.join(self.output_path, self.op_folder_name, i))

        for i in range(len(self.all_images)):
            try:
                j_type = self.jewellery_types.get(self.all_images[i].split(" ")[2][:2], "Other")

                img_path = os.path.join(self.db_path, self.all_images[i])
                dest_path = os.path.join(self.output_path, self.op_folder_name, j_type)

                shutil.copy(img_path, dest_path)
            
            except:
                print(self.all_images[i])
            
        print("Done!")
