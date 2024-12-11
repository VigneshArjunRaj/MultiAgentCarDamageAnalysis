from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
import yaml
import os
from InsuranceAgent.Prompts import reviewerPrompt,generalConversationPrompt,generalSearchPrompt,HallucinationPrompt,SearchPrompt

config = yaml.safe_load(open("config.yaml"))
os.environ["TAVILY_API_KEY"] = config["TAVILY_API_KEY"]

# If you donot have Tavily API, comment the below tool

tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
)
# uncomment DuckDUckGo search for api free web search
#tool =  DuckDuckGoSearchRun()

llm = ChatOllama(model="llama3.2:3b")



def reviewerBot(state):
    """
    Agent Node to review mapped result analysis for Car parts and damages including websearch
    Args:
         state -> Graph state containing mappedresult, websearch results
    Returns:
        dict: A dict of damage results, parts result and review message for state of the graph

    """

    print("----------------Inside Reviewer---------------")

    print(state["mappedresult"])

    return {"messages":llm.invoke( 
            [SystemMessage(content=reviewerPrompt)]+
            [HumanMessage(content=f" analysed damages dictionary {state['mappedresult']} Websearch result {state['search']} ")]
            ),
            "damagesresult":state["damagesresult"], "partsresult":state["partsresult"]
            }


def checkForHallucinationandCorrectness(state):
    """
    Agent Node to check hallucinations and correctness of reviewer node output

     Args:
         state -> Graph state containing mappedresult, websearch results
    Returns:
        dict: A dict of damage results, parts result and review message for state of the graph
    
    """
    print("----------------Inside Hallucination Check---------------")

    print(state["mappedresult"])

    
    return {"messages":llm.invoke( 
        [SystemMessage(content=HallucinationPrompt)]+
        [HumanMessage(content=f" analysed damages dictionary {state['mappedresult']}, output from the model {state['messages']}")]
        ),
        "damagesresult":state["damagesresult"], "partsresult":state["partsresult"]
        }

def generalConversation(state):
    """
    General Conversation Agent Node for reviewing car parts and insurance related concepts

    ## Parameters
    
    Args:
         state -> Graph state reviewing general web search
    Returns:
        dict: A dict of message and web search result for state of the graph
    """
    #print(state["damagesresult"])
    
    return {"messages":llm.invoke(
        [SystemMessage(content=generalConversationPrompt)]+
        [HumanMessage(content=f"text {state['messages'][-1].content} web search result {state['generalwebsearch']}")])}


def generalWebSearch(state):
    """
    General Web Search Agent Node for processing user text and searching the web
    
    Args:
         state -> Graph state containing general web search
    Returns:
        searchresult (str): web search result 

    """
    print(state['messages'])

    content = llm.invoke( 
            [SystemMessage(content=generalSearchPrompt)]+
            [HumanMessage(content=f"process the following text {state['messages']}")]
            )
    try:
        searchresult = tool.invoke(content.content)
    except Exception:
        searchresult = "No good results found."
    print(content.content)
    print(searchresult)
    

    return {"generalwebsearch": searchresult, 'messages': content}



def Search(state):
    """
    Agent Node for processing user text and searching the web

    Args:
         state -> Graph state containing web search
    Returns:
        searchresult (str): web search result 

    """
    print(state['messages'])

    content = llm.invoke( 
            [SystemMessage(content=SearchPrompt)]+
            [HumanMessage(content=f"process the following text {state['messages'][-1].content}")]
            )
    try:
        searchresult = tool.invoke(content.content)
    except Exception:
        searchresult = "No good results found."
    print(content.content)
    print(searchresult)
    

    return {"search": searchresult, "damagesresult":state["damagesresult"], "partsresult":state["partsresult"]}