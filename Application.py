import streamlit as st
from Graph import graph
from InsuranceAgent.Inference import draw_boxes_with_hover
from PIL import Image
from io import BytesIO
import base64
from Utility import annotationToWeb, generate_pdf
import math
from st_multimodal_chatinput import multimodal_chatinput
import streamlit_js_eval


def writeImages(images):
    """
    Write images uploaded by the user in the chat area in grid format

     Args:
         images (list) -> list of uploaded images
    
    """

    # Create grid with rows then columns

    # calculate number of rows for 4 columns of total images
    rows = math.ceil(len(images)/4)
    counter = 0
    #for each row
    for row in range(rows):
        # define 4 columns
        cols = st.columns(4)
        for col in range(4):
            if counter == len(images):
                return
            #add image to each column
            cols[col].image(images[counter])
            counter+=1


def echo(message:dict):
        """
        Streams the user message into the Langraph orchestration, grabs inferences of images

        Args:
            message (dict) -> User multi modal input message
        Returns:
            list: A list of review summary, web ready annotated damage results and parts result, annotated damage results and parts result
        
        """
        
        # access text and uploaded images into variables
        user_input = message["text"]
        uploaded_files = message["images"]

        #Initializae empty lists to store various data
        total = []
        damages= []
        parts = []
        images= []
        final_annotated = []
        annotated_images_pdf = []

        # stream through the graph
        for event in graph.stream({"messages":("human",user_input),"images":uploaded_files}):
            for value in event.values():
                # If messages are there in value, which means agent output states of the graph
                if "messages" in value and type(value["messages"]) != list:
                    #print(value["messages"] )
                    # apped agent output to total
                    total.append(value["messages"].content)
                
                # if annotated images in state, append to images
                if "images" in value:
                    images=value["images"]
                
                #If image inference in state append to damages and parts
                if "damagesresult" in value and "partsresult" in value:
                    damages,parts=value["damagesresult"],value["partsresult"]
        #Write the final agent response to user
        st.write(total[-1])
        
        print("---------------------------")
        #print(damages,parts)

        
        

        print(len(images),len(damages),len(parts))
        
        # For I in images
        for i in range(len(images)):

            img = Image.open(images[i])

            #draw colored segmentation masks on b/w images and detection boundary boxes
            img = draw_boxes_with_hover(img,damages[i],parts[i])
            #img = Image.fromarray(img).convert('RGB')
            annotated_images_pdf.append(img)
            # annotate interactivity to the images and receive HTML code for the same
            code = annotationToWeb(img,damages[i],parts[i])
            # append html code to final_annotated
            final_annotated.append(code)
                    
        

        print("---------------------------")
        #print(damages,parts)
        return total[-1], final_annotated, annotated_images_pdf



# Set page config
st.set_page_config(
    page_title="Auto Insurance Agent",
    page_icon="🚗",
)


# adding custom styles for Multimodal input and chat area
st.markdown("""<style>
   
            .stHorizontalBlock {
            position: sticky;
       
            }
            
            .stMainBlockContainer { 
            padding: 0 !important;
            padding-top: 3rem !important;

            }
            
      

             </style>
            """,unsafe_allow_html=True)


# get height of the windopw to adjust screen size
height = streamlit_js_eval.streamlit_js_eval(js_expressions='screen.height', key = 'SCR')
if height:
    height = int(height//(1.6))

print(height)


#Initialize chat area
chatarea = st.container(height= height)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chatinput = []
    response = """ Hello """
    #st.session_state.messages.append({"role": "assistant", "content": response, "images": []})


# Display existing messages
for message in st.session_state.messages:
    with chatarea.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"]== "user":
            writeImages(message["images"])
        for image in message["images"]:
            if message["role"]== "assistant":
                #print(image)
                st.components.v1.html(image,height= 400)
        

# create form area
formarea = st.columns([0.5,2,0.5])
with formarea[1]:
    #Initialize multimodal chat area
    chatinput = multimodal_chatinput()


# Analysis over the input data
images = []

# If user inputs data into the chatinput
if chatinput and chatinput not in st.session_state.chatinput:
    #add to session
    st.session_state.chatinput.append(chatinput)

    #seperate text and uploaded files from the chat dict
    prompt = chatinput["text"]
    uploaded_files = [ BytesIO(base64.b64decode(img.split(",")[1])) for img in chatinput["images"]]
    
    # Build the user message content
    message_content = {}

    #check if the input contains only text or uploaded images
    if prompt:
        message_content["text"]= prompt
    else:
        message_content["text"]= ""
    
    if uploaded_files is not None:
        
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            images.append(image)  
        message_content["images"]=uploaded_files
    else:
        message_content["images"]= []
            
    #add processed message dict after reading files
    st.session_state.messages.append({"role": "user", "content": prompt, "images": images, "annotimage": [] })
    with chatarea.chat_message("user"):
        if prompt:
            st.markdown(prompt)
        writeImages(images)
    
    
    
    # Get response from the LLM
   
    # Append assistant's response to messages
   
    with chatarea.chat_message("assistant"):
        
        response, finalannotated, annotated_images = echo(message_content)
      
        for html_code in finalannotated:
            st.components.v1.html(html_code,height= 400)
        st.session_state.messages.append({"role": "assistant", "content": response, "images": finalannotated, "annotimage": annotated_images})
        
        pdf_buffer = generate_pdf(st.session_state.messages, )
        st.download_button(
                    label="Download PDF",
                    data=pdf_buffer,
                    file_name="chat_history.pdf",
                    mime="application/pdf"
                )
    
    
