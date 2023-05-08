import openai
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chat_models import ChatOpenAI

api_key = 'your-key'


def new_chat():
    """
    Clears session state and starts a new chat
    :return:
    """

    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.buffer.clear()


def generate_response(prompt):

    # Create the ConversationChain object with the specified configuration
    Conversation = ConversationChain(
        llm=llm,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=st.session_state.entity_memory
    )

    output = Conversation.run(input=user_input)

    return output


llm = ChatOpenAI(
    temperature=0,
    openai_api_key=api_key,
    model_name='gpt-3.5-turbo',
    verbose=False
)

# Create a ConversationEntityMemory object if not already created
if 'entity_memory' not in st.session_state:
    st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=200)

st.title("  chatBot based by GPT-3.5-turbo")


if 'generated' not in st.session_state:
    st.session_state['generated'] = []

def get_text():
    input_text - st.text_input("You: ","Hello, how are you?", key="input")
    return input_text


btn = st.button('새로운 대화 시작')
if btn:
    new_chat()

user_input = get_text()

if user_input:
    output = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=srt(i) + '_user')