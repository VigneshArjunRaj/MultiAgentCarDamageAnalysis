import streamlit as st
import numpy as np
import base64
from PIL import Image
import json
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import StringIO
from io import BytesIO
import base64
from textwrap import wrap
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as IMG
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from markdown2 import markdown

def xywh_to_xyxy(xywh):
    """
    Converts a bounding box in xywh format to 4-point (x, y) format.
    
    Parameters:
        xywh (tuple or list): A bounding box in (x, y, w, h) format.

    Returns:
        list: A list of four (x, y) coordinates representing the corners of the rectangle.
    """
    x, y, w, h = xywh
    x1, y1, x2, y2 = int(x),int(y),int(w),int(h)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def draw_multiline_text(pdf, text, x, y, max_width,page_height, line_spacing=15):
        """
        Draws multiline text on the PDF canvas.
        """
        margin = 50
        lines = wrap(text, width=max_width // 7)  # Approximation for text wrapping
        for line in lines:
            if y < margin:  # Check if there's enough space on the page
                pdf.showPage()  # Create a new page
                pdf.setFont("Helvetica", 12)
                y = page_height - margin
            pdf.drawString(x, y, line)
            y -= line_spacing  # Move to the next line
        return y



# Function to generate PDF
def generate_pdf(chat_history):
 
    buffer = BytesIO()
    pdf = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=50, leftMargin=50,
        topMargin=50, bottomMargin=50
    )

    styles = getSampleStyleSheet()

    yourStyle = ParagraphStyle('yourtitle',
                           fontName="Helvetica",
                           fontSize=13,
                           parent=styles['Heading2'],
                           
                           spaceAfter=14)
    story = []

    for msg in chat_history:

        #print(msg["content"])
        
        message_text = markdown(msg["content"].replace("\n", "<br/>"))  # Convert Markdown to HTML
        story.append(Paragraph(message_text, yourStyle))
        story.append(Spacer(1, 0.2 * inch))  # Add a bigger space between messages
        try:
                for img in msg['annotimage']:
                    img_buffer = BytesIO()
                    img.save(img_buffer, format="PNG")
                    img_buffer.seek(0)
                    img = IMG(img_buffer)  # Resize image to fit
                    story.append(img)
                    story.append(Spacer(1, 0.2 * inch))  # Add space after the image
        except Exception as e:
                story.append(Paragraph(f"Error loading image: {e}", styles["BodyText"]))


    pdf.build(story)
    buffer.seek(0)
    return buffer



def annotationToWeb(image, damages, parts):
        
        
        masksd = []
        partsd = []

        for mask in damages['bounding boxes']:
             masksd.append(xywh_to_xyxy(mask[0]))
             
             

        damages_serializable = {
            'Categories': damages['Categories'],
            'Confidence scores': damages['Confidence scores'],
            'masks': masksd
        }

    

        
        # Convert PIL Image to Base64
        buffered = BytesIO()
        image = image.convert('RGB')
        image.save(buffered, format="JPEG")  # Replace "JPEG" with your image format if needed
        img_data = base64.b64encode(buffered.getvalue()).decode()
        w, h  = image.size

        # Create the HTML and JavaScript for the interactive canvas

        damages_json = json.dumps(damages_serializable)
       
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <body>
            <img id="image" src="data:image/jpeg;base64,{img_data}" """+f""" width="{w}" height="{h}" style="position: absolute; z-index: 1;">
            <canvas id="myCanvas" width="{w}" height="{h}" style="border: 1px solid #000000; position: absolute; z-index: 2;"></canvas>
            <script>
            var canvas = document.getElementById("myCanvas"),
                ctx = canvas.getContext("2d");
                

        var damages = {damages_json};
            
        """+"""

        
            function drawPolygon(points, color) {
                ctx.beginPath();
                ctx.moveTo(points[0][0], points[0][1]);
                for (var i = 1; i < points.length; i++) {
                ctx.lineTo(points[i][0], points[i][1]);
                }
                ctx.closePath();
                ctx.fillStyle = color;
                ctx.fill();

            }
            
            function drawAnnotations() {
                // Draw damages
                for (var i = 0; i < damages.masks.length; i++) {
                drawPolygon(damages.masks[i], 'rgba(255, 0, 0, 0.08)');  // Red for damages
                }

             
            }

            drawAnnotations();

            canvas.onmousemove = function(e) {
                var rect = canvas.getBoundingClientRect(),
                    x = e.clientX - rect.left,
                    y = e.clientY - rect.top;

                ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas
                drawAnnotations();  // Redraw annotations

                // Highlight damages on hover
                for (var i = 0; i < damages.masks.length; i++) {
                    ctx.beginPath();
                    drawPolygon(damages.masks[i], 'transparent');
                    if (ctx.isPointInPath(x, y)) {
                    drawPolygon(damages.masks[i], 'rgba(255, 0, 0, 0.2)');  // Brighter red

                    
                    }
                }

                
            };
            </script>
        </body>
        </html>
        """

        return html_code
# Embed the HTML and JavaScript using Streamlit
#st.components.v1.html(html_code, height=500)
