from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END,add_messages
from InsuranceAgent.Nodes import checkDamage,checkIfCar,checkParts
from InsuranceAgent.AgentNodesEdges import reviewerBot,checkForHallucinationandCorrectness,generalConversation, Search, generalWebSearch
from InsuranceAgent.Edges import generalOrImage, checkAllCars



class State(TypedDict):
    """
    Stateful class to store state attributes
    """
    images: list # for list of images
    messages: Annotated[list, add_messages] # add messages/inputs/outputs from user or each multi agent 
    damagesresult: dict # YOLO results dict
    partsresult: dict # YOLO results dict
    mappedresult: dict # custom damages to parts mapping dict
    exception: Annotated[list, add_messages] # add exceptions if any
    search: dict # dict for review agent search results
    generalwebsearch: dict # dict for general web search results
    userquery: str # to store user query


# Initialize graph with our State class

graph_b = StateGraph(State)

# Nodes Initialization

# Agent to review car damage results
graph_b.add_node("ReviewerBot",reviewerBot)

# CV tool to infer car damage results
graph_b.add_node("CheckDamage",checkDamage)

# CV tool to infer car parts segmentation results and mapped damage to part results
graph_b.add_node("CheckParts",checkParts)

# Check if ther is a car in the given image
graph_b.add_node("CheckIfCar",checkIfCar)

# general conversation agent
graph_b.add_node("GeneralConversation",generalConversation)

# Agent to check for hallucinations and correctness
graph_b.add_node('HallucinationAndReviewer',checkForHallucinationandCorrectness)

# Agent for web search review queries
graph_b.add_node("WebSearch", Search)

# Agent for web search general queries
graph_b.add_node("GeneralWebSearch", generalWebSearch)

# Edges Initialization

graph_b.add_conditional_edges(START,
                              generalOrImage,
                              {
                                  'General': "GeneralWebSearch",
                                    'Images': "CheckIfCar"
                              }
                              
                              )

graph_b.add_conditional_edges("CheckIfCar",
                              checkAllCars,
                              {
                                  'General': "GeneralWebSearch",
                                    'CheckDamage': "CheckDamage"
                              }
                              
                              )

graph_b.add_edge("CheckDamage","CheckParts")

graph_b.add_edge("CheckParts","WebSearch")




graph_b.add_edge("WebSearch", "ReviewerBot")

#graph_b.add_edge("GeneralConversation", "TAVILYSearch")

graph_b.add_edge("GeneralWebSearch", "GeneralConversation")

graph_b.add_edge("GeneralConversation", END)

graph_b.add_edge("ReviewerBot", 'HallucinationAndReviewer')

graph_b.add_edge('HallucinationAndReviewer',END)



graph = graph_b.compile()