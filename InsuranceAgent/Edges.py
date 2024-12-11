from PIL import Image
def checkAllCars(state):
    """
    Check if the request contains any Image
    """
    if len(state["images"]) == 0:
        return "General"
    else:
        return "CheckDamage"
    
def generalOrImage(state):
    """
    Check if the request contains Images
    Check if the Images are readable
    """
    if len(state["images"]) == 0:
        return "General"
    else:
        counter = 0
        removeInds = []
        for ind,image in enumerate(state["images"]):
            # check if there's any error opening the image file
            try:
                image = Image.open(image)
            except Exception as e:
                counter+=1
                #append images with exceptions to remove
                removeInds.append(ind)
        # remove images which causes exception when opened
        state["images"] = [value for i,value in enumerate(state["images"]) if i not in removeInds]

        # if images are empty return to General
        if len(state["images"])==0:
            return "General"
        return "Images"

