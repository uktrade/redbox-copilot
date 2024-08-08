import logging
import sys

# Set up logging
logger = logging.getLogger('streamlit_app')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

print('This is actually doing something!!')
logger.info('This is actually doing something!!')

import streamlit as st
from analysis_of_chat_history import ChatHistoryAnalysis

logger.info("Starting the Streamlit application")
print("Starting the Streamlit application")

st.set_page_config(page_title="Redbox Chat Analysis", layout="centered")
st.set_option("deprecation.showPyplotGlobalUse", False)
st.title("Redbox Chat History Dashboard")

logger.info("Configured Streamlit page settings")
print("Configured Streamlit page settings")

user_usage_tab, word_freq_tab, route_tab, topic_tab, prompt_complex = st.tabs(
    [
        "Redbox User Usage",
        "Word frequency",
        "Route Analysis",
        "Topic modelling",
        "Prompt Complexity",
    ]
)

logger.info("Initialized tabs")
print("Initialized tabs")

cha = ChatHistoryAnalysis()

with user_usage_tab:
    logger.info("Rendering Redbox User Usage tab")
    print("Rendering Redbox User Usage tab")
    st.pyplot(cha.user_frequency_analysis())
    st.pyplot(cha.redbox_traffic_analysis())
    st.pyplot(cha.redbox_traffic_by_user())

with word_freq_tab:
    logger.info("Rendering Word frequency tab")
    print("Rendering Word frequency tab")
    st.subheader("User")
    st.pyplot(cha.user_word_frequency_analysis())
    st.subheader("AI")
    st.pyplot(cha.ai_word_frequency_analysis())

with route_tab:
    logger.info("Rendering Route Analysis tab")
    print("Rendering Route Analysis tab")
    st.pyplot(cha.route_analysis())
    st.pyplot(cha.route_transitions())

with topic_tab:
    logger.info("Rendering Topic modelling tab")
    print("Rendering Topic modelling tab")
    with st.spinner("Fitting topic model..."):
        cha.get_topics()

    st.plotly_chart(cha.visualise_topics())
    st.plotly_chart(cha.visualise_topics_over_time())
    st.plotly_chart(cha.visualise_barchart())
    st.plotly_chart(cha.visualise_hierarchy())

with prompt_complex:
    logger.info("Rendering Prompt Complexity tab")
    print("Rendering Prompt Complexity tab")
    # Adding slider for prompt length
    max_outlier = cha.get_prompt_lengths()["no_input_words"].max() + 10
    outlier = st.slider("Please use the slicer if you wish to remove outliers.", 0, max_outlier, 700)
    st.pyplot(cha.visualise_prompt_lengths(outlier_max=outlier))
    st.pyplot(cha.vis_prompt_length_vs_chat_legnth(outlier_max=outlier))

logger.info("Finished rendering all tabs")
print("Finished rendering all tabs")
