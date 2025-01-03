import streamlit as st
import openai
import os
from dotenv import find_dotenv, load_dotenv
import time
import logging
from datetime import datetime
import requests
import json
import streamlit as st

load_dotenv()


client = openai.OpenAI( api_key=os.environ.get("OPENAI_API_KEY"))
model = "gpt-3.5-turbo"



class AssistantManager:
    thread_id = "thread_G9pToCwIflbHrvcJwRea2ooh"
    assistant_id = "asst_fFcm3KmPEvIyFiukm02QONWa"

    def __init__(self, model: str = model):
        self.client = client
        self.model = model
        self.assistant = None
        self.thread = None
        self.run = None
        self.summary = None

        # Retrieve existing assistant and thread if IDs are already set
        if AssistantManager.assistant_id:
            self.assistant = self.client.beta.assistants.retrieve(
                assistant_id=AssistantManager.assistant_id
            )
        if AssistantManager.thread_id:
            self.thread = self.client.beta.threads.retrieve(
                thread_id=AssistantManager.thread_id
            )

    def create_assistant(self, name, instructions, tools):
        if not self.assistant:
            assistant_obj = self.client.beta.assistants.create(
                name=name, instructions=instructions, tools=tools, model=self.model
            )
            AssistantManager.assistant_id = assistant_obj.id
            self.assistant = assistant_obj
            print(f"AssisID:::: {self.assistant.id}")

    def create_thread(self):
        if not self.thread:
            thread_obj = self.client.beta.threads.create()
            AssistantManager.thread_id = thread_obj.id
            self.thread = thread_obj
            print(f"ThreadID::: {self.thread.id}")

    def add_message_to_thread(self, role, content):
        if self.thread:
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id, role=role, content=content
            )

    def run_assistant(self, instructions):
        if self.thread and self.assistant:
            self.run = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                instructions=instructions,
            )

    def process_message(self):
        if self.thread:
            messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
            summary = []

            last_message = messages.data[0]
            role = last_message.role
            response = last_message.content[0].text.value
            summary.append(response)

            self.summary = "\n".join(summary)
            print(f"SUMMARY-----> {role.capitalize()}: ==> {response}")

            # for msg in messages:
            #     role = msg.role
            #     content = msg.content[0].text.value
            #     print(f"SUMMARY-----> {role.capitalize()}: ==> {content}")

    def call_required_functions(self, required_actions):
        if not self.run:
            return
        tool_outputs = []

        for action in required_actions["tool_calls"]:
            func_name = action["function"]["name"]
            arguments = json.loads(action["function"]["arguments"])

            if func_name == "get_news":
                output = get_news(topic=arguments["topic"])
                print(f"STUFFFFF;;;;{output}")
                final_str = ""
                for item in output:
                    final_str += "".join(item)

                tool_outputs.append({"tool_call_id": action["id"], "output": final_str})
            else:
                raise ValueError(f"Unknown function: {func_name}")

        print("Submitting outputs back to the Assistant...")
        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread.id, run_id=self.run.id, tool_outputs=tool_outputs
        )

    # for streamlit
    def get_summary(self):
        return self.summary
    
    def get_history(self):
        if self.thread:
            messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
            return [
                {"role": msg.role, "content": msg.content[0].text.value}
                for msg in messages.data
            ]
        return []

    def wait_for_completion(self):
        if self.thread and self.run:
            while True:
                time.sleep(5)
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id, run_id=self.run.id
                )
                print(f"RUN STATUS:: {run_status.model_dump_json(indent=4)}")

                if run_status.status == "completed":
                    self.process_message()
                    break
                elif run_status.status == "requires_action":
                    print("FUNCTION CALLING NOW...")
                    self.call_required_functions(
                        required_actions=run_status.required_action.submit_tool_outputs.model_dump()
                    )

    # Run the steps
    def run_steps(self):
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=self.thread.id, run_id=self.run.id
        )
        print(f"Run-Steps::: {run_steps}")
        return run_steps.data






def show_history(manager):
    st.write("### Conversation History")
    history = manager.get_history()
    if history:
        for msg in history:
            role = msg["role"].capitalize()
            content = msg["content"]
            st.write(f"**{role}**: {content}")
    else:
        st.write("No previous conversation available.")

def main():
    # news = get_news("bitcoin")
    # print(news[0])

    manager = AssistantManager()
    

    # Streamlit interface
    st.title("Medical Advisor App")

    with st.form(key="user_input_form"):
        instructions = st.text_input("Enter your enquiry:")
        submit_button = st.form_submit_button(label="Run Enquiry")
        
        
        st.sidebar.header("Options")
        new_conversation = st.sidebar.button("Start New Conversation")

        if new_conversation:
            manager.thread = None  # Reset the thread to start fresh
            st.sidebar.success("Started a new conversation.")



        if submit_button:
            if not manager.thread:
                manager.create_thread()

            # Add the message and run the assistant
            manager.add_message_to_thread(
                role="user", content=f"{instructions}?"
            )
            manager.run_assistant(instructions="Provide expert medical advice.")

            # Wait for completions and process messages
            manager.wait_for_completion()

            summary = manager.get_summary()

            st.write(summary)
            
            # Display full conversation history
            # show_history(manager)

            # st.text("Run Steps:")
            # st.code(manager.run_steps(), line_numbers=True)


if __name__ == "__main__":
    main()
