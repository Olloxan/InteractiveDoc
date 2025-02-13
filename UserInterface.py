import gradio as gr
import pandas as pd

from Utils import FileLoader
from Utils import ConfigManager

import matplotlib.pyplot as plt
import numpy as np

# Create some example data for plotting
x = np.linspace(0, 10, 100)
y = np.sin(x)

def plot_function():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x, y)
    ax.set_title("Example Plot")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    
    # Return the figure to display it in Gradio
    return fig

class UserInterface:
    def __init__(self, chat_fn=None) -> None:
        self._chat_function = chat_fn                
        self.fileloader = FileLoader()
        self.config = ConfigManager()
                    
    def render(self):
        try:
            port = 7860
            
            with gr.Blocks(fill_width=True, fill_height=True) as blocks:
                with gr.Row():
                    # Left column: Week Plan Table
                    with gr.Column(scale=1):
                        gr.Markdown("### Week Plan")
                        gr.Plot(plot_function, label="Matplotlib Plot")                        
                    
                    # Middle column: Chatbot
                    with gr.Column(scale=1):
                        gr.Markdown("### Chat Bot")
                        self.chatbot = gr.Chatbot(value=[[None, "Hello you!"]], max_height=1000, min_height=800)                       
                        self.msg = gr.Textbox(
                            label="Chat message",
                            placeholder="Type your message here...",
                            show_label=False
                        )
                        with gr.Row():
                            self.submit = gr.Button("Submit")
                            self.clear = gr.Button("Clear")
                    
                    # Right column: Chain input and document display
                    with gr.Column(scale=1):
                        gr.Markdown("### Document Retrieval")
                                                                        
                        initial_data = pd.DataFrame({
                                        "Nr.": [""],
                                        "Rezept": [""]                                        
                                        })

                        self.data_frame_retrieval = gr.DataFrame(value=initial_data, headers=["Nr.", "Rezept", "Score"], datatype=["number", "str", "number"], wrap=True)                                                                       
                
                # Attach button event handlers
                self._attach_button_events()               
            
            # Launch the interface
            blocks.queue()
            blocks.launch(debug=True, show_api=False, inline=False, inbrowser=True)
            
        except Exception as e:
            print(e)
            raise e
    
###################### attach events ######################
    def _attach_button_events(self):
        """Attach event handlers to buttons."""
            
        # Chat submit and clear buttons
        self.submit.click(fn=self._handle_chat_submit, inputs=[self.msg, self.chatbot], outputs=[self.msg, self.chatbot])
        self.msg.submit(fn=self._handle_chat_submit, inputs=[self.msg, self.chatbot], outputs=[self.msg, self.chatbot])
        self.clear.click(fn=self._handle_chat_clear, inputs=None, outputs=self.chatbot, queue=False)
                                                 

###################### center: Chatbot ######################
    def _handle_chat_submit(self, message:str, chat_history:list[tuple]):
        """Handle chat submit event.
        message: str
        history: [[(user) None, (agent) "Hello you!"]] (List of Lists)
        """        
       
       
        chat_history.append((message, ""))
       
        bot_message = self._chat_function(message, chat_history)
                        
        for bot_message_part in bot_message:
            # Append each part of the bot's response to the history
            chat_history[-1] = (message, bot_message_part)        
            yield None, chat_history                          

    def _handle_chat_clear(self):
        """Handle chat clear button event."""
        return None
