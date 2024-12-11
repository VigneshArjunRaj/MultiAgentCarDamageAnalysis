from InsuranceAgent.Inference import inferParts,inferdamage,inferCar,map_damages_to_parts
from PIL import Image



def checkIfCar(state):
    """ Computer Vision node to infer recognition of car from the set of images"""
    # append results to combined list
    combinedlist = []

    for image in state["images"]:
        try:

            image = Image.open(image)
            # infer image
            carCheck = inferCar(image)
            combinedlist.append(carCheck)
        except Exception as e:
            state["exception"] = str(e)
            print(e)
    
    # check if the image is a car or not using True or False
    state["images"] = [value for i,value in enumerate(state["images"]) if combinedlist[i]]
    print("---------------- Inside CheckIfCar--------------------")
    return {"images": state["images"]}

        
    

def checkDamage(state):
    """ Computer Vision node to infer Car damages detection from the set of images"""
    e = None
    # append results to combined list
    combinedlist = []
    for image in state["images"]:
        try:
            image = Image.open(image)
            # infer image
            damagesResult = inferdamage(image)
            
            combinedlist.append(damagesResult)
        except Exception as e:
            state["exception"] = str(e)
            print(e)
    print("---------------- Inside checkDamage--------------------")
    return {"damagesresult":combinedlist,"images":state["images"]}


def checkParts(state):
    """ Computer Vision node to infer Car parts segmentation from the set of images"""
    e= None
    #print(state["images"])
    # append results to combined list
    combinedlist = []
    for image in state["images"]:
        try:
            image = Image.open(image)
            # infer image
            damagesResult = inferParts(image)
            combinedlist.append(damagesResult)
        except Exception as e:
            state["exception"] = str(e)
    
    #print(state["damagesresult"])
    # map pose segmentation over damage detection using intersection over union, return damage to each part IoU overlap
    mapped_results = map_damages_to_parts(state["damagesresult"], combinedlist)
    print("---------------- Inside checkParts--------------------")
    return {"mappedresult":mapped_results,"partsresult":combinedlist,"damagesresult":state["damagesresult"]}



