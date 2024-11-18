import os
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage

from agents.agent import Agent


def populate_envs(sender_email, receiver_email, subject):
    os.environ['FROM_EMAIL'] = sender_email
    os.environ['TO_EMAIL'] = receiver_email
    os.environ['EMAIL_SUBJECT'] = subject


def initialize_agent():
    if 'agent' not in st.session_state:
        st.session_state.agent = Agent()


def render_custom_css():
    st.markdown(
        '''
        <style>
        .main-title {
            font-size: 2.5em;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 1em;
            font-weight: bold;
            padding: 10px;
            background: #f1f1f1;
            border-radius: 10px;
        }
        .sub-title {
            font-size: 1.2em;
            color: #333;
            text-align: left;
            margin-bottom: 0.5em;
        }
        .center-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
        }
        .query-box {
            width: 80%;
            max-width: 600px;
            margin-top: 0.5em;
            margin-bottom: 1em;
        }
        .query-container {
            width: 80%;
            max-width: 600px;
            margin: 0 auto;
        }
        @media screen and (max-width: 768px) {
            .main-title {
                font-size: 1.8em;
            }
            .query-container {
                width: 100%;
            }
        }
        .email-form {
            padding: 15px;
            background: #e8f5e9;
            border-radius: 10px;
            margin-top: 1em;
        }
        </style>
        ''', unsafe_allow_html=True)


def render_ui():
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    st.markdown('<div class="main-title">✈️🌍 AI Travel Agent 🏨🗺️</div>', unsafe_allow_html=True)
    st.markdown('<div class="query-container">', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter your travel query and get flight, hotel, and attraction information:</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        'Travel Query',
        height=200,
        key='query',
        placeholder='Type your travel query here, e.g., "Find me flights, hotels, and nearby attractions in Paris."',
        help="Enter your query to find travel details.",
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    return user_input


def process_query(user_input):
    if user_input:
        try:
            thread_id = str(uuid.uuid4())
            st.session_state.thread_id = thread_id

            messages = [HumanMessage(content=user_input)]
            config = {'configurable': {'thread_id': thread_id}}

            with st.spinner('Processing your travel query...'):
                result = st.session_state.agent.graph.invoke({'messages': messages}, config=config)

            st.markdown("## Travel Information")
            st.write(result['messages'][-1].content)

            st.session_state.travel_info = result

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error('Please enter a travel query.')


def render_email_form():
    send_email_option = st.radio('Do you want to send this information via email?', ('No', 'Yes'))
    if send_email_option == 'Yes':
        st.markdown('<div class="email-form">', unsafe_allow_html=True)
        with st.form(key='email_form'):
            sender_email = st.text_input('Sender Email', os.environ.get('FROM_EMAIL', ''))
            receiver_email = st.text_input('Receiver Email')
            subject = st.text_input('Email Subject', 'Travel Information')
            submit_button = st.form_submit_button(label='Send Email')

        if submit_button:
            if sender_email and receiver_email and subject:
                populate_envs(sender_email, receiver_email, subject)
                st.success("Email sent successfully!")
                # Clear session state
                for key in ['travel_info', 'thread_id']:
                    st.session_state.pop(key, None)
            else:
                st.error('Please fill out all email fields.')
        st.markdown('</div>', unsafe_allow_html=True)


def main():
    initialize_agent()
    render_custom_css()
    user_input = render_ui()

    if st.button('Get Travel Information'):
        process_query(user_input)

    if 'travel_info' in st.session_state:
        render_email_form()

    # Add footer
    st.markdown(
        '''
        <footer style="text-align: center; margin-top: 2em;">
            <hr>
            <p style="font-size: 0.9em;">AI Travel Agent © 2024 | Powered by LangChain & Resend API</p>
        </footer>
        ''', unsafe_allow_html=True)


if __name__ == '__main__':
    main()